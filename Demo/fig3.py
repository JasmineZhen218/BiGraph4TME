import sys
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("./..")


f, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(aspect="equal"))
labels = [
    "No clinical data (N=10)",
    "Less than 500 cells (N=104)",
    "Discovery set (N=379)",
    "Inner validation set (N=200)",
]
sizes = [10, 104, 379, 200]
colors = [
    "grey",
    "lightgrey",
    sns.color_palette("Set2")[1],
    sns.color_palette("Set2")[0],
]
pie = ax.pie(
    sizes, labels=None, startangle=90, colors=colors, wedgeprops=dict(linewidth=1)
)
ax.legend(
    labels,
    loc="upper left",
    bbox_to_anchor=(1, 0.5),
    fontsize=12,
)
# Equal aspect ratio ensures that the pie chart is circular.
plt.axis("equal")
# Show the plot
f.savefig("Results/fig3_a.jpg", dpi=300, bbox_inches="tight")

f, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(aspect="equal"))
labels = ["Less than 500 cells (N=15)", "External validation set (N=266)"]
sizes = [15, 266]
colors = ["lightgrey", sns.color_palette("Set2")[2]]
pie = ax.pie(
    sizes, labels=None, startangle=90, colors=colors, wedgeprops=dict(linewidth=1)
)
ax.legend(
    labels,
    loc="upper left",
    bbox_to_anchor=(1, 0.5),
    fontsize=12,
)

# Equal aspect ratio ensures that the pie chart is circular.
plt.axis("equal")
# Show the plot
f.savefig("Results/fig3_b.jpg", dpi=300, bbox_inches="tight")