"""
Figure 1 для статті: rel_err vs j для polynomial MUET vs PATP-MUET
на 5 manufactured solutions. Виводить PDF у paper/fig_rq3.pdf.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq3_demo import run_one  # noqa: E402


scenarios = [
    (r"$0.6\,x^{0.5} + 0.4\,x^{1.5}$",  lambda x: 0.6 * x ** 0.5 + 0.4 * x ** 1.5,  "M1 (NT)"),
    (r"$x^{0.7}$",                       lambda x: x ** 0.7,                          "M2 (S)"),
    (r"$x^{1.7}$",                       lambda x: x ** 1.7,                          "M3 (S)"),
    (r"$\sqrt{x}\,(1 + x^2/3)$",         lambda x: np.sqrt(x) * (1 + x ** 2 / 3),    "M4 (NT)"),
    (r"$x^3$",                           lambda x: x ** 3,                            "M5 (AT)"),
    (r"$\ln(1+x)$",                      lambda x: np.log(1 + x),                     "M6 (H)"),
    (r"$\exp(-x)$",                      lambda x: np.exp(-x),                        "M7 (H)"),
]

print("Збираю дані...")
data = []
for label, f, tag in scenarios:
    r = run_one(f"{tag}", f)
    rel_poly = [r["j_results"][j]["rel_poly"] for j in [1, 2, 3, 4]]
    rel_patp = [max(r["j_results"][j]["rel_patp"], 1e-17) for j in [1, 2, 3, 4]]
    data.append((label, tag, rel_poly, rel_patp))

# --- Малювання: 2×4 сітка (7 панелей + 8-й слот під легенду) ---
fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.0), sharey=True)
axes_flat = axes.flatten()
j_vals = [1, 2, 3, 4]
for idx, (label, tag, rel_poly, rel_patp) in enumerate(data):
    ax = axes_flat[idx]
    ax.semilogy(j_vals, rel_poly, "o-", label="Polynomial MUET", color="#d62728", linewidth=2.0, markersize=8)
    ax.semilogy(j_vals, rel_patp, "s-", label="PATP-MUET",       color="#1f77b4", linewidth=2.0, markersize=8)
    ax.set_title(f"{tag}: {label}", fontsize=13)
    ax.set_xticks(j_vals)
    ax.tick_params(labelsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_ylim(1e-17, 1e0)

# x-підпис на нижній видимій панелі кожного стовпця (M4 — у верхньому ряду, бо під ним порожньо)
for ax in (axes[1, 0], axes[1, 1], axes[1, 2], axes[0, 3]):
    ax.set_xlabel(r"moment order $j$", fontsize=12)
# y-підпис на лівому стовпці обох рядів
axes[0, 0].set_ylabel("relative error", fontsize=12)
axes[1, 0].set_ylabel("relative error", fontsize=12)

# легенда у вільному 8-му слоті
legend_ax = axes_flat[7]
legend_ax.axis("off")
handles, lbls = axes_flat[0].get_legend_handles_labels()
legend_ax.legend(handles, lbls, loc="center", fontsize=13, framealpha=0.95,
                 title="Method", title_fontsize=13)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "outputs", "fig_rq3.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, bbox_inches="tight")
print(f"Figure saved: {out}")
