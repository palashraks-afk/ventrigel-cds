"""
Generate the manuscript figures.

    python make_figures.py

Every figure is drawn from the same modules the numbers come from, so a figure
cannot drift out of step with the text. Output goes to ``results/figures/`` as
both PDF (for the manuscript) and PNG (for the web application and README).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

from ventrigel.economics import CostModel, optimal_enrichment
from ventrigel.power import design, enrichment_curve
from ventrigel.sensitivity import DEFAULT_SHRINKAGE, bootstrap_designs, sweep_assumptions
from ventrigel.trial_data import ENDPOINTS, N_EARLY, N_LATE

FIGURES = Path("results/figures")
PI = N_LATE / (N_EARLY + N_LATE)
ALPHA, POWER, DROPOUT = 0.05, 0.80, 0.10

EARLY_C = "#C0504D"
LATE_C = "#2E6E9E"
TOTAL_C = "#4A4A4A"
ACCENT = "#D98C1F"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# --------------------------------------------------------------------------


def fig1_cancellation() -> None:
    """The central observation: pooled effects are cancellations, not small effects."""
    keys = ["lvesv", "viable_mass", "mlwhfq", "six_min_walk", "lvedv", "ef"]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.0))

    for ax, key in zip(axes.ravel(), keys):
        ep = ENDPOINTS[key]
        t = ep.change_6mo
        sign = -1.0 if ep.lower_is_better else 1.0
        groups = ["early", "late", "total"]
        vals = [sign * t[g].mean for g in groups]
        errs = [t[g].sem for g in groups]
        colors = [EARLY_C, LATE_C, TOTAL_C]

        y = np.arange(3)[::-1]
        ax.barh(y, vals, xerr=errs, color=colors, height=0.6, error_kw={"lw": 1.0, "ecolor": "#333"})
        ax.axvline(0, color="#222", lw=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"early (n={t['early'].n})", f"late (n={t['late'].n})", f"pooled (n={t['total'].n})"]
        )
        ax.set_title(f"{ep.label}  ({ep.unit}, benefit positive)", fontsize=8.5, pad=14)
        ax.tick_params(labelsize=7.5)
        ax.margins(x=0.12)

        # Flag genuine opposition only. A stratum whose effect is a rounding
        # error away from zero is not "moving in the opposite direction", it is
        # not moving, so require the smaller effect to be a real fraction of
        # the larger before calling it a cancellation.
        a, b = t["early"].mean, t["late"].mean
        if a * b < 0 and min(abs(a), abs(b)) > 0.10 * max(abs(a), abs(b)):
            ax.text(
                0.5,
                1.02,
                "strata oppose",
                transform=ax.transAxes,
                ha="center",
                fontsize=7,
                style="italic",
                color=ACCENT,
            )

    fig.suptitle(
        "Six-month change from baseline by treatment timing, VentriGel Phase I\n"
        "Bars are published means, whiskers published SEMs; signs flipped so right = benefit",
        fontsize=9.5,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig1_subgroup_cancellation")


def fig2_enrichment_curves() -> None:
    """Required sample size as enrollment is progressively restricted."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.9))

    styles = {
        "lvesv": (LATE_C, "-", "LVESV"),
        "viable_mass": ("#4E8A5B", "-", "Viable mass"),
        "six_min_walk": (ACCENT, "-", "6-min walk"),
        "mlwhfq": ("#8A6BA8", "--", "MLWHFQ"),
    }
    for key, (c, ls, lab) in styles.items():
        sweep = enrichment_curve(key, PI, alpha=ALPHA, power=POWER, dropout=DROPOUT, n_points=121)
        e = [d.e for d in sweep]
        n = [d.n_total if (d.feasible and d.favors_treatment) else np.nan for d in sweep]
        ax.plot(e, n, color=c, ls=ls, lw=1.8, label=lab)

    ax.set_yscale("log")
    ax.set_xlabel("Fraction of enrolled patients drawn from the late stratum")
    ax.set_ylabel("Total randomized patients (both arms)")
    ax.axvline(PI, color="#888", ls=":", lw=1.2)
    ax.text(PI + 0.01, ax.get_ylim()[1] * 0.5, "unselected\npool", fontsize=7, color="#666")
    ax.set_title("Sample size falls as enrollment is enriched", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Right panel: decomposition of why, for the primary endpoint.
    sweep = enrichment_curve(PRIMARY := "lvesv", PI, alpha=ALPHA, power=POWER, dropout=DROPOUT, n_points=121)
    e = np.array([d.e for d in sweep])
    ax2.plot(e, [d.effect for d in sweep], color=LATE_C, lw=1.8, label="effect $\\Delta$ (mL, benefit +)")
    ax2.plot(e, [d.sd for d in sweep], color=EARLY_C, lw=1.8, label="SD $\\sigma$ (mL)")
    ax2.plot(e, [d.sd_between for d in sweep], color="#888", lw=1.4, ls="--", label="between-stratum SD")
    ax2.axhline(0, color="#222", lw=0.9)
    ax2.axvline(PI, color="#888", ls=":", lw=1.2)
    ax2.set_xlabel("Fraction of enrolled patients drawn from the late stratum")
    ax2.set_ylabel("mL")
    ax2.set_title("Enrichment raises the effect and lowers the variance", fontsize=9)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.tight_layout()
    save(fig, "fig2_enrichment_curves")


def fig3_bootstrap() -> None:
    """Estimation uncertainty in the enriched design."""
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    for ax, key in zip(axes, ("lvesv", "viable_mass", "six_min_walk")):
        b_un = bootstrap_designs(
            key, PI, PI, shrinkage=DEFAULT_SHRINKAGE, alpha=ALPHA, power=POWER, dropout=DROPOUT
        )
        b_en = bootstrap_designs(
            key, 1.0, PI, shrinkage=DEFAULT_SHRINKAGE, alpha=ALPHA, power=POWER, dropout=DROPOUT
        )
        bins = np.logspace(1, 5, 45)
        for b, c, lab in ((b_un, EARLY_C, "unselected"), (b_en, LATE_C, "enriched")):
            finite = b.samples[np.isfinite(b.samples)]
            ax.hist(finite, bins=bins, color=c, alpha=0.55, label=f"{lab} ({b.feasible_fraction:.0%} solvable)")
        ax.set_xscale("log")
        ax.set_xlabel("Total randomized patients")
        ax.set_title(ENDPOINTS[key].label, fontsize=8.5)
        ax.legend(fontsize=6.8, frameon=False)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("Bootstrap draws")
    fig.suptitle(
        f"Estimation uncertainty from the 6-8 patients per stratum "
        f"(effects discounted {1 - DEFAULT_SHRINKAGE:.0%} for winner's curse)",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig3_bootstrap")


def fig4_control_drift() -> None:
    """The dominant unmeasured assumption, mapped rather than asserted."""
    grid = sweep_assumptions(
        "lvesv", PI, shrinkage=DEFAULT_SHRINKAGE, alpha=ALPHA, power=POWER, dropout=DROPOUT
    )
    adv = np.array(grid.advantage, dtype=float)
    # An infinite advantage means the unselected trial has no benefit to detect
    # at any size. That is the strongest case for enrichment, so it must render
    # as the top of the scale rather than as a blank cell.
    plot = np.where(np.isinf(adv), 1e4, adv)
    plot = np.clip(plot, 1.0, 1e4)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(
        plot.T,
        origin="lower",
        aspect="auto",
        cmap="BuPu",
        norm=LogNorm(vmin=1, vmax=1e4),
        extent=(
            grid.control_early_values[0],
            grid.control_early_values[-1],
            grid.control_late_values[0],
            grid.control_late_values[-1],
        ),
    )
    cs = ax.contour(
        grid.control_early_values,
        grid.control_late_values,
        plot.T,
        levels=[2, 5, 10, 50, 500],
        colors="#333",
        linewidths=0.8,
    )
    ax.clabel(cs, fmt=lambda v: f"{v:,.0f}x", fontsize=7)
    ax.axhline(0, color="#fff", lw=0.8, alpha=0.6)
    ax.plot(9.3, 0.0, marker="o", ms=7, mfc="none", mec=ACCENT, mew=2.0)
    ax.annotate(
        "entire early-stratum change\nattributed to natural history",
        xy=(9.3, 0.0),
        xytext=(4.4, 2.6),
        fontsize=7,
        color=ACCENT,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.0},
    )
    ax.set_xlabel("Assumed control-arm LVESV drift, early stratum (mL at 6 months)")
    ax.set_ylabel("Assumed control-arm drift, late stratum (mL)")
    ax.set_title(
        "Enrichment advantage under every control-arm assumption\n"
        "(unselected N / enriched N; the Phase I was single-arm so this is unmeasured)",
        fontsize=9,
    )
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, label="advantage (x)", extend="max")
    cb.ax.text(
        0.5,
        1.04,
        "no pooled\nbenefit at all",
        transform=cb.ax.transAxes,
        ha="center",
        fontsize=6.5,
        color="#444",
    )
    fig.tight_layout()
    save(fig, "fig4_control_drift")


def fig5_economics() -> None:
    """Cost, duration, and where the screening penalty binds."""
    model = CostModel()
    pis = np.linspace(0.03, 0.75, 25)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.9))

    for key, c, lab in (("lvesv", LATE_C, "LVESV"), ("six_min_walk", ACCENT, "6-min walk")):
        costs, months, estar = [], [], []
        for pi in pis:
            sweep = enrichment_curve(key, float(pi), alpha=ALPHA, power=POWER, dropout=DROPOUT, n_points=61)
            best, _ = optimal_enrichment(sweep, model)
            costs.append(best.total_cost / 1e6)
            months.append(best.duration_months)
            estar.append(best.design.e)
        ax.plot(pis, costs, color=c, lw=1.8, label=f"{lab}: cost")
        ax2.plot(pis, estar, color=c, lw=1.8, label=f"{lab}: optimal enrichment")

    ax.set_xlabel("Late-stratum prevalence in the eligible pool")
    ax.set_ylabel("Cost of the cheapest viable design ($M)")
    ax.set_title("Rarer responders cost more to find", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)

    ax2.plot(pis, pis, color="#999", ls=":", lw=1.2, label="unselected (e = $\\pi$)")
    ax2.set_xlabel("Late-stratum prevalence in the eligible pool")
    ax2.set_ylabel("Cost-optimal enrichment level $e^*$")
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_title("When full enrichment stops paying", fontsize=9)
    ax2.legend(fontsize=7.5, frameon=False, loc="center right")
    ax2.text(
        0.045,
        0.62,
        "6-min walk optimum drops to\nunselected: excluded patients\nstill improve, so screening\nfor them is not worth it",
        fontsize=6.8,
        color="#555",
    )

    fig.tight_layout()
    save(fig, "fig5_economics")


def fig6_design_summary() -> None:
    """One panel a trial designer can act on."""
    keys = ["lvesv", "viable_mass", "mlwhfq", "lvedv", "six_min_walk", "ef"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y = np.arange(len(keys))[::-1]

    for i, key in zip(y, keys):
        u = design(key, PI, PI, alpha=ALPHA, power=POWER, dropout=DROPOUT)
        e = design(key, 1.0, PI, alpha=ALPHA, power=POWER, dropout=DROPOUT)
        b = bootstrap_designs(
            key, 1.0, PI, shrinkage=DEFAULT_SHRINKAGE, alpha=ALPHA, power=POWER, dropout=DROPOUT
        )
        if u.favors_treatment and u.feasible:
            ax.plot([u.n_total], [i + 0.18], "s", color=EARLY_C, ms=6)
        else:
            # No amount of enrolment demonstrates benefit here, so there is no
            # point on the axis. Park a marker past the right edge instead of
            # silently omitting the row.
            ax.plot([2.1e5], [i + 0.18], ">", color=EARLY_C, ms=8, clip_on=False)
        if math.isfinite(b.n_total_median):
            ax.plot(
                [b.n_total_q10, b.n_total_q90], [i - 0.18, i - 0.18], color=LATE_C, lw=3.0, alpha=0.40
            )
            ax.plot([b.n_total_median], [i - 0.18], "o", color=LATE_C, ms=6)

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([ENDPOINTS[k].label for k in keys], fontsize=8)
    ax.set_xlabel("Total randomized patients required (log scale)")
    ax.set_xlim(10, 3e5)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.plot([], [], "s", color=EARLY_C, ms=6, label="unselected cohort")
    ax.plot([], [], ">", color=EARLY_C, ms=8, label="unselected: no benefit to detect at any N")
    ax.plot([], [], "o", color=LATE_C, ms=6, label="enriched, discounted (median, 80% interval)")
    ax.legend(
        fontsize=7.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        columnspacing=1.4,
    )
    ax.set_title(
        "What each candidate primary endpoint would cost in patients\n"
        "Enriched estimates carry a 25% effect discount and full estimation uncertainty",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, "fig6_design_summary")


def main() -> None:
    print("Generating figures...")
    fig1_cancellation()
    fig2_enrichment_curves()
    fig3_bootstrap()
    fig4_control_drift()
    fig5_economics()
    fig6_design_summary()
    print(f"Done. {FIGURES.resolve()}")


if __name__ == "__main__":
    main()
