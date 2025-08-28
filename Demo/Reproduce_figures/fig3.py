import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("..")
os.makedirs("Results/Fig3", exist_ok=True)

def plot_pie_chart(sizes, colors, filename, figsize=(3, 3)):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"))
    ax.pie(sizes, startangle=90, colors=colors, wedgeprops=dict(linewidth=1))
    plt.axis("equal")
    fig.savefig(f"Results/Fig3/{filename}", dpi=300, bbox_inches="tight")

# Figure 3a
plot_pie_chart(
    sizes=[10, 104, 379, 200],
    colors=["grey", "lightgrey", sns.color_palette("Set2")[1], sns.color_palette("Set2")[0]],
    filename="fig3_a.svg",
    figsize=(5, 5),
)

# Figure 3b
plot_pie_chart(
    sizes=[15, 266],
    colors=["lightgrey", sns.color_palette("Set2")[2]],
    filename="fig3_b.svg"
)

# Figure 3c
plot_pie_chart(
    sizes=[4, 274],
    colors=["lightgrey", sns.color_palette("Set3")[2]],
    filename="fig3_c.svg"
)
