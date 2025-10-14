import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV
headers = ["model", "name", "length", "batch_size", "time"]
df_gemma = pd.read_csv("gemma2_e2e_results.csv", names=headers)
df_mpt = pd.read_csv("mpt_e2e_results.csv", names=headers)
df_llama = pd.read_csv("llama_e2e_results.csv", names=headers)
df_vit = pd.read_csv("vit_e2e_results.csv", names=headers)

attn_name_maps = {
    "felix_attn": "Neptune",
    "flex_attention": "Flex Attn",
    "flash_attention_2": "Flash Attn 2",
    "flash": "Flash Attn 2",
    "torch": "Torch (Non-Flash)",
    "sdpa": "Torch (Non-Flash)",
    "eager": "Torch (Non-Flash)",
}

# Define colors for each attention mechanism
mechanisms = list(set(attn_name_maps.values()))
mechanisms.remove("Neptune")
mechanisms.insert(0, "Neptune")
palette = sns.color_palette("husl", len(mechanisms))
# Move the "green" color (index 2) to the front for Neptune
palette.insert(0, palette.pop(2))


n_models = 4
fig, axs = plt.subplots(1, n_models, figsize=(n_models*5, 5))

def plot_e2e(df, ax, title):
    df = df.copy()
    # Data contains multiple measurements for the same (name, length) pair. Model and batch size are constant.
    # Take mean of time for each (name, length) pair.
    df.drop(columns=["model", "batch_size"], inplace=True)
    df = df.groupby(["name", "length"], as_index=False).agg({"time": ("mean", "sem")})
    df["sem"] = df["time"]["sem"]
    df["mean"] = df["time"]["mean"]
    df.drop(columns=["time"], inplace=True)
    
    # Map attention names to their full names
    df["name"] = df["name"].map(attn_name_maps)
    
    # Normalize time against Neptune. 
    neptune_times = df[df["name"] == "Neptune"][["length", "mean"]]
    neptune_times = dict(zip(neptune_times["length"], neptune_times["mean"]))
    df["norm_mean"] = df.apply(lambda row: row["mean"] / neptune_times[row["length"].item()], axis=1)
    df["norm_sem"] = df.apply(lambda row: row["sem"] / neptune_times[row["length"].item()], axis=1)
    print(title)
    print(df)

    local_hue_order = []
    local_pallete = []
    # Go through hue_order and remove any that are not in df["name"], along with their corresponding colors
    for name in mechanisms:
        if name in df["name"].values:
            local_hue_order.append(name)
            local_pallete.append(palette[mechanisms.index(name)])
    n_drops = len(mechanisms) - len(local_hue_order)
    # Pad with none so that the bar sizes + spacings are consistent across plots
    local_hue_order += [None] * n_drops
    local_pallete += ["#000000"] * n_drops
    sns.barplot(
        data=df,
        x="length",
        y="norm_mean",
        hue="name",
        hue_order=local_hue_order,
        palette=local_pallete,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Relative Latency (vs Neptune)")
    # ax.set_xscale("log")
    ax.set_ylim(0, None)
    # want a single legend for all subplots
    ax.get_legend().remove()


plot_e2e(df_gemma, axs[0], "Gemma2-7b*")
plot_e2e(df_mpt, axs[1], "MPT-7b")
plot_e2e(df_llama, axs[2], "Llama-2-7b")
plot_e2e(df_vit, axs[3], "ViT*")
plt.suptitle("End-to-End Model Latency", fontsize=16)
plt.legend(title="Attention Implementation", loc="lower right")
plt.tight_layout()

plt.savefig("logs/e2e_plot.png")