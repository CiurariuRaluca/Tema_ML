import seaborn as sns
import matplotlib.pyplot as plt
from preprocess_dataset import get_eda_objects

cart_features_with_target, numerical_cols, schnitzel_dataset, target, cart_features = (
    get_eda_objects()
)

# 1) Heatmap - corelații
fig1, ax1 = plt.subplots(figsize=(7, 7))
sns.heatmap(
    cart_features_with_target[numerical_cols + ["target"]].corr(),
    cmap="BrBG",
    fmt=".2f",
    linewidths=2,
    annot=True,
    ax=ax1,
)
ax1.set_title(
    "correlation between numerical features and target", fontsize=14, fontweight="bold"
)
plt.tight_layout()
fig1.savefig("../../plots/heatmap_correlations.pdf", bbox_inches="tight")
plt.show()

# 2) Top 15 produse cu Crazy Schnitzel
fig2, ax2 = plt.subplots(figsize=(10, 7))
product_counts = schnitzel_dataset["retail_product_name"].value_counts().head(15)

ax2.barh(
    range(len(product_counts)),
    product_counts.values,
    color="steelblue",
    edgecolor="black",
    alpha=0.7,
)
ax2.set_yticks(range(len(product_counts)))
ax2.set_yticklabels([name[:30] for name in product_counts.index], fontsize=9)
ax2.set_xlabel("Frequency", fontsize=11)
ax2.set_title(
    "Top 15 Products in carts with Crazy Schnitzel", fontsize=14, fontweight="bold"
)
ax2.invert_yaxis()
ax2.grid(axis="x", alpha=0.3)

plt.tight_layout()
fig2.savefig("../../plots/top_products.pdf", bbox_inches="tight")

plt.show()

# 3) Distribuția targetului
fig3, ax3 = plt.subplots(figsize=(8, 6))
target_dist = target.value_counts().reindex([0, 1])
colors = ["#ff7f0e", "#2ca02c"]

bars = ax3.bar(
    ["Without Crazy Sauce", "With Crazy Sauce"],
    target_dist.values,
    color=colors,
    edgecolor="black",
    alpha=0.7,
)

ax3.set_ylabel("Number of carts", fontsize=11)
ax3.set_title("Target distribution", fontsize=14, fontweight="bold")

for bar, val in zip(bars, target_dist.values):
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 20,
        f"{val}\n({val/len(target)*100:.1f}%)",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

ax3.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig3.savefig("../../plots/target_distribution.pdf", bbox_inches="tight")
plt.show()

# 4) Distribuția mărimii coșului
fig4, ax4 = plt.subplots(figsize=(10, 6))
cart_size_dist = cart_features["cart_size"].value_counts().sort_index()

ax4.bar(
    cart_size_dist.index,
    cart_size_dist.values,
    color="coral",
    edgecolor="black",
    alpha=0.7,
)
ax4.set_xlabel("Number of products in cart", fontsize=11)
ax4.set_ylabel("Frequency", fontsize=11)
ax4.set_title("Cart size distribution", fontsize=14, fontweight="bold")
ax4.grid(axis="y", alpha=0.3)
ax4.set_xlim(0, 12)

plt.tight_layout()
fig4.savefig("../../plots/cart_size_distribution.pdf", bbox_inches="tight")
plt.show()
