#!/usr/bin/env python3
"""FastScan + Top-R: recall & latency vs R (DEEP100K 实测). 输出 PDF/PNG 供 LaTeX 插图."""
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 数据：同一时间、同机多次运行（R=0 表示未启用 Top-R 精排，仅 FastScan 粗排）
R = [0, 100, 200, 500, 1000]
recall = [0.238301, 0.708851, 0.837004, 0.943554, 0.982302]
latency_us = [582.208, 1712.79, 1851.57, 1848.52, 2512.0]

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

fig, ax1 = plt.subplots(figsize=(7.2, 4.2))

color_r = "#1f77b4"
color_l = "#d62728"
ax1.set_xlabel(r"Top-$R$ 全精度重排候选数 $R$")
ax1.set_ylabel("平均 Recall@10", color=color_r)
ax1.plot(R, recall, "o-", color=color_r, linewidth=2, markersize=8, label="Recall@10")
ax1.tick_params(axis="y", labelcolor=color_r)
ax1.set_ylim(0.15, 1.05)
ax1.set_xticks(R)
ax1.grid(True, linestyle="--", alpha=0.35)

ax2 = ax1.twinx()
ax2.set_ylabel(r"平均查询延迟 / $\mu$s", color=color_l)
ax2.plot(R, latency_us, "s-", color=color_l, linewidth=2, markersize=8, label="Latency")
ax2.tick_params(axis="y", labelcolor=color_l)
ax2.set_ylim(400, 2800)

# 标注 R=500 为推荐工作点
r_star = 500
ax1.axvline(r_star, color="green", linestyle="--", linewidth=1.5, alpha=0.85)
idx = R.index(r_star)
ax1.annotate(
    f"推荐 $R={r_star}$\nRecall≈{recall[idx]:.3f}\n{latency_us[idx]:.0f} μs",
    xy=(r_star, recall[idx]),
    xytext=(r_star + 180, recall[idx] - 0.12),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="green", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="green", alpha=0.95),
)

# 合并图例
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc="center left", bbox_to_anchor=(0.02, 0.35))

plt.title("PQ FastScan + Top-$R$ IP 重排：Recall 与延迟随 $R$ 变化（DEEP100K）")
plt.tight_layout()

out_base = __file__.rsplit(".", 1)[0]
plt.savefig(out_base + ".pdf", bbox_inches="tight")
plt.savefig(out_base + ".png", dpi=200, bbox_inches="tight")
print("written:", out_base + ".pdf", out_base + ".png")
