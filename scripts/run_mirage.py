import mirage as mi
import numpy as np
import torch
from felix_attn.operators import PF_CAUSAL, PF_GLOBAL, BenchmarkOperator
from felix_attn.ours import attn_plain as pf
from felix_attn.runner import BenchmarkRunner
from felix_attn.utils import profiler_ctx


def setup_op_get_our_runners(input_shape_5d: list[int], mask: bool):
    if mask:
        operator = PF_CAUSAL
        our_runners = pf.NeptuneAttentionRunner.create_flex_from_schedulers(
            operator, mask_cond=pf.causal_mask_cond
        )
    else:
        operator = PF_GLOBAL
        our_runners = pf.NeptuneAttentionRunner.create_flex_from_schedulers(operator)
    batch, n_heads, qs, kvs, h_dim = input_shape_5d
    torch_inputs = operator.create_inputs((batch, n_heads, n_heads, qs, kvs, h_dim))
    # Reduce the input values to avoid overflows in Mirage.
    torch_inputs = [x * 1e-2 for x in torch_inputs]

    avail_runners: list[pf.NeptuneAttentionRunner] = []
    for runner in our_runners:
        runner: pf.NeptuneAttentionRunner
        if not runner.setup(*torch_inputs):
            continue
        avail_runners.append(runner)
    return operator, torch_inputs, avail_runners


class MirageRunner(BenchmarkRunner):
    def __init__(self, operator: BenchmarkOperator, mask: bool):
        super().__init__(operator, "mirage", "mirage")
        self.mask = mask

    def setup(self, *inputs) -> bool:
        q, k, v = inputs
        batch, n_heads, qs, h_dim = q.shape
        kvs = k.shape[2]
        def_graph = self._mirage_define_attention([batch, n_heads, qs, kvs, h_dim], self.mask)
        cygraphs = mi.search(def_graph.cygraph, verbose=True, default_config="attention")
        best_graph = None
        best_perf = float("inf")
        for idx, cygraph in enumerate(cygraphs):
            input_tensors = [
                torch.randn([t.dim(i) for i in range(t.num_dims)], dtype=q.dtype, device=q.device)
                for t in cygraph.get_input_dtensors()
            ]
            kngraph = mi.KNGraph(cygraph)
            kngraph.visualize(f"test/mirage_graph_{idx}")
            cuda_program = kngraph.compile(inputs=input_tensors)
            if cuda_program is None:
                print(f"Compilation failed for muGraph {idx}")
                continue
            perf = self.measure_single_graph(kngraph, input_tensors)
            print("muGraph {}: profiled performance (ms) = {}".format(idx, perf))
            with open(f"test/mirage_program_{idx}.cu", "w") as f:
                f.write(cuda_program["code"])
            if perf < best_perf:
                best_graph, best_perf = kngraph, perf

        if best_graph is None:
            return False
        self.graph = best_graph

        q = q.view(batch * n_heads, qs, h_dim)
        k = k.view(batch * n_heads, kvs, h_dim).transpose(1, 2)
        v = v.view(batch * n_heads, kvs, h_dim)
        self.inputs = [q, k, v]
        if self.mask:
            # Create an additive, lower triangular mask, with -1e20 everywhere else
            # to simulate a causal mask.
            mask_mask = torch.ones(batch, n_heads, qs, kvs, dtype=torch.bool).tril()
            self.inputs.append(torch.where(mask_mask, 0, -1e20).to(torch.float16).to(q.device))
        return True

    def run(self):
        import nvtx

        with nvtx.annotate("mirage-full", color="green"):
            inputs = self.q_div_by_sqrt_d(self.inputs)
            with self.mark_ctx():
                self.output = self.graph(inputs=inputs)

    def get_output(self) -> torch.Tensor:
        assert self.output is not None, "Output not set"
        return self.output[0]

    def measure_single_graph(self, kngraph: mi.KNGraph, inputs: list[torch.Tensor]):
        # Warmup runs
        for _ in range(16):
            kngraph(inputs=inputs)
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(1000):
            kngraph(inputs=inputs)
        ender.record()
        torch.cuda.synchronize()
        return starter.elapsed_time(ender) / 1000

    @staticmethod
    def _mirage_define_attention(input_shape_5d: list[int], mask: bool):
        f16 = mi.float16  # type: ignore
        batch, n_heads, qs, kvs, h_dim = input_shape_5d
        qshape = batch * n_heads, qs, h_dim
        kshape = batch * n_heads, h_dim, kvs
        vshape = batch * n_heads, kvs, h_dim
        score_shape = batch * n_heads, qs, kvs
        graph = mi.new_kernel_graph()
        q = graph.new_input(dims=qshape, dtype=f16)
        k = graph.new_input(dims=kshape, dtype=f16)
        v = graph.new_input(dims=vshape, dtype=f16)
        if mask:
            mask_data = graph.new_input(dims=score_shape, dtype=f16)
        else:
            mask_data = None
        score = graph.matmul(q, k)
        if mask_data is not None:
            score = graph.add(score, mask_data)
        sexp = graph.exp(score)
        ssum = graph.reduction(sexp, 2)
        ssoftmax = graph.div(sexp, ssum)
        output = graph.matmul(ssoftmax, v)
        graph.mark_output(output)
        # graph.visualize("attention")
        return graph

    @staticmethod
    def q_div_by_sqrt_d(inputs: list[torch.Tensor]):
        q = inputs[0]
        inputs[0] = q / np.sqrt(q.shape[-1])
        return inputs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("shape", type=lambda s: [int(x) for x in s.split(",")])
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--mask", action="store_true")
    args = parser.parse_args()

    operator, torch_inputs, our_runners = setup_op_get_our_runners(args.shape, args.mask)
    mirage_runner = MirageRunner(operator, args.mask)
    # If Mirage doesn't support the operator, skip the rest.
    print("Mirage tuning...")
    if not mirage_runner.setup(*torch_inputs):
        return
    print("Running Mirage kernel once for warmup...")
    mirage_runner.run()
    mirage_output = mirage_runner.get_output()
    for runner in our_runners:
        print(f"Running our {runner} once for warmup...")
        runner.run()
        our_output = runner.get_output()
        assert torch.allclose(mirage_output, our_output, atol=1e-2, rtol=0), (
            f"Mirage output {mirage_output} != our output {our_output} (from runner {runner})"
        )
    with profiler_ctx():
        mirage_runner.repeat(args.repeat)
        for runner in our_runners:
            runner.repeat(args.repeat)


if __name__ == "__main__":
    main()
