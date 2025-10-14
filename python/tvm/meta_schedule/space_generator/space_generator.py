# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Meta Schedule design space generators that generates design
space for generation of measure candidates.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

from tvm._ffi import register_object
from tvm.ir import IRModule
from tvm.runtime import Object
from tvm.tir.schedule import Schedule

from .. import _ffi_api

if TYPE_CHECKING:
    from ..mutator import Mutator
    from ..postproc import Postproc
    from ..schedule_rule import ScheduleRule
    from ..tune_context import TuneContext


@register_object("meta_schedule.SpaceGenerator")
class SpaceGenerator(Object):
    """The abstract design space generator interface."""

    ScheduleFnType = (
        Callable[[Schedule], None]  # No output
        | Callable[[Schedule], Schedule]  # Single output
        | Callable[[Schedule], list[Schedule]]  # Multiple outputs
    )

    SpaceGeneratorType: TypeAlias = (
        ScheduleFnType | Literal["post-order-apply", "union"] | "SpaceGenerator"
    )

    postprocs: list["Postproc"] | None
    mutator_probs: dict["Mutator", float] | None

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the design space generator with tuning context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initializing the design space generator.
        """
        _ffi_api.SpaceGeneratorInitializeWithTuneContext(  # type: ignore # pylint: disable=no-member
            self, context
        )

    def generate_design_space(self, mod: IRModule) -> list[Schedule]:
        """Generate design spaces given a module.

        Parameters
        ----------
        mod : IRModule
            The module used for design space generation.

        Returns
        -------
        design_spaces : list[tvm.tir.Schedule]
            The generated design spaces, i.e., schedules.
        """
        return _ffi_api.SpaceGeneratorGenerateDesignSpace(self, mod)  # type: ignore # pylint: disable=no-member

    def clone(self) -> "SpaceGenerator":
        """Clone the design space generator.

        Returns
        -------
        cloned_sg : SpaceGenerator
            The cloned design space generator.
        """
        return _ffi_api.SpaceGeneratorClone(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def create(  # pylint: disable=keyword-arg-before-vararg
        kind: Literal["post-order-apply", "union"] | ScheduleFnType = "post-order-apply",
        *args,
        **kwargs,
    ) -> "SpaceGenerator":
        """Create a design space generator."""
        from . import (  # pylint: disable=import-outside-toplevel
            PostOrderApply,
            ScheduleFn,
            SpaceGeneratorUnion,
        )

        if callable(kind):

            def create_schedule_fn(
                func,
                postprocs=[],
                mutator_probs={},
            ):  # pylint: disable=dangerous-default-value
                return ScheduleFn(func, postprocs, mutator_probs)

            return create_schedule_fn(kind, *args, **kwargs)  # type: ignore
        if kind == "post-order-apply":
            return PostOrderApply(*args, **kwargs)
        if kind == "union":
            return SpaceGeneratorUnion(*args, **kwargs)
        if isinstance(kind, str):
            return PostOrderApply(sch_rules=kind, postprocs=kind, mutator_probs=kind)
        raise ValueError(f"Unknown SpaceGenerator: {kind}")


ScheduleFnType = SpaceGenerator.ScheduleFnType
ScheduleRuleType = (
    Sequence["ScheduleRule"] | Literal["llvm", "cuda", "cuda-tensorcore", "hexagon", "from-target"]
)
PostprocType = (
    Sequence["Postproc"] | Literal["llvm", "cuda", "cuda-tensorcore", "hexagon", "from-target"]
)
MutatorProbType = (
    Mapping["Mutator", float] | Literal["llvm", "cuda", "cuda-tensorcore", "hexagon", "from-target"]
)
create = SpaceGenerator.create  # pylint: disable=invalid-name


def _normalize_rules(
    postprocs: PostprocType,
    mutator_probs: MutatorProbType,
) -> tuple[
    Sequence["Postproc"] | None,
    Mapping["Mutator", float] | None,
]:
    # pylint: disable=import-outside-toplevel
    from ..mutator import Mutator
    from ..postproc import Postproc

    # pylint: enable=import-outside-toplevel
    assert postprocs is not None
    assert mutator_probs is not None

    def create_or_none(arg, cls):
        if not isinstance(arg, str):
            return arg
        if arg == "from-target":
            return None
        return cls.create(arg)

    postprocs_ = create_or_none(postprocs, Postproc)
    mutator_probs_ = create_or_none(mutator_probs, Mutator)
    return postprocs_, mutator_probs_


@register_object("meta_schedule.PySpaceGenerator")
class _PySpaceGenerator(SpaceGenerator):
    """
    A TVM object space generator to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PySpaceGenerator
    """

    def __init__(
        self,
        postprocs: PostprocType = "from-target",
        mutator_probs: MutatorProbType = "from-target",
        f_initialize_with_tune_context: Callable | None = None,
        f_generate_design_space: Callable | None = None,
        f_clone: Callable | None = None,
    ):
        """Constructor."""
        postprocs_, mutator_probs_ = _normalize_rules(postprocs, mutator_probs)

        self.__init_handle_by_constructor__(
            _ffi_api.SpaceGeneratorPySpaceGenerator,  # type: ignore # pylint: disable=no-member
            postprocs_,
            mutator_probs_,
            f_initialize_with_tune_context,
            f_generate_design_space,
            f_clone,
        )


class PySpaceGenerator:
    """
    An abstract space generator with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PySpaceGenerator,
        "fields": ["postprocs", "mutator_probs"],
        "methods": ["_initialize_with_tune_context", "generate_design_space", "clone"],
    }

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the design space generator with tuning context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initializing the design space generator.
        """
        raise NotImplementedError

    def generate_design_space(self, mod: IRModule) -> list[Schedule]:
        """Generate design spaces given a module.

        Parameters
        ----------
        mod : IRModule
            The module used for design space generation.

        Returns
        -------
        design_spaces : list[tvm.tir.Schedule]
            The generated design spaces, i.e., schedules.
        """
        raise NotImplementedError

    def clone(self) -> SpaceGenerator:
        """Clone the design space generator.

        Returns
        -------
        cloned_sg : SpaceGenerator
            The cloned design space generator.
        """
        raise NotImplementedError
