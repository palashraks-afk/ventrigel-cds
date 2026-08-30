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
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from ventrigel.assurance import (
    assurance_ceiling,
    assurance_curve,
    n_for_assurance,
    programme_grid,
)
from ventrigel.inference import all_interaction_tests, multiplicity
from ventrigel.literature import ANCHORS, anchored_control
from ventrigel.power import enrichment_curve, interaction_design
from ventrigel.sensitivity import bootstrap_designs, sweep_assumptions
from ventrigel.trial_data import ENDPOINTS, N_EARLY, N_LATE

FIGURES = Path("results/figures")
PI = N_LATE / (N_EARLY + N_LATE)
ALPHA, POWER, DROPOUT, SHRINK = 0.05, 0.80, 0.10, 0.75
PRIMARY = "lvesv"

EARLY_C = "#C0504D"
LATE_C = "#2E6E9E"
TOTAL_C = "#4A4A4A"
ACCENT = "#D98C1F"
GOOD_C = "#4E8A5B"

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

_LV = anchored_control("lvesv", "early") or (0.0, 0.0)
_LL = anchored_control("lvesv", "late") or (0.0, 0.0)
CE, CL = _LV[0], _LL[0]
SE_L = _LL[1]


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def _controls_for(key: str) -> tuple[float, float]:
    """Anchored control assumptions, per endpoint, or zero where none exists."""
    e = anchored_control(key, "early") or (0.0, 0.0)
    l = anchored_control(key, "late") or (0.0, 0.0)
    return e[0], l[0]


# --------------------------------------------------------------------------


def fig1_cancellation() -> None:
    """Six-month change by stratum: where the pooled effect comes from."""
    keys = ["lvesv", "viable_mass", "mlwhfq", "six_min_walk", "lvedv", "ef"]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.4))

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
        # Two lines, and the long Minnesota name abbreviated. At the width this
        # figure occupies in the two-column manuscript, single-line titles for
        # the wider endpoints run into their neighbours.
        short = {
            "Minnesota Living with Heart Failure score": "MLWHFQ score",
            "6-minute walk distance": "6-min walk distance",
        }.get(ep.label, ep.label)
        ax.set_title(f"{short}\n({ep.unit}, benefit positive)", fontsize=8.0, pad=13)
        ax.tick_params(labelsize=7.5)
        ax.margins(x=0.12)

        a, b = t["early"].mean, t["late"].mean
        if a * b < 0 and min(abs(a), abs(b)) > 0.10 * max(abs(a), abs(b)):
            # Inside the axes, not above it: the two-line titles leave no room
            # overhead, and the lower-right corner is empty in every panel
            # because the pooled bar is the shortest.
            ax.text(
                0.97, 0.05, "strata oppose", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.8, style="italic", color=ACCENT,
            )

    fig.suptitle(
        "Six-month change from baseline by treatment timing, VentriGel Phase I\n"
        "Bars are published means, whiskers published SEMs; signs flipped so right = benefit",
        fontsize=9.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig1_subgroup_cancellation")


def fig2_interaction() -> None:
    """The test nobody ran, with the multiplicity correction beside it."""
    tests = all_interaction_tests()
    mult = {m.endpoint: m for m in multiplicity(tests)}
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.8, 4.2), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    # Left: forest plot of the between-stratum difference, benefit-signed.
    y = np.arange(len(tests))[::-1]
    for i, t in zip(y, tests):
        sign = -1.0 if t.lower_is_better else 1.0
        lo, hi = sorted([sign * t.ci_low, sign * t.ci_high])
        # Standardizing by the standard error lets endpoints on wildly
        # different scales share one axis. The bar is then the fixed 95%
        # reference interval and the marker is the t statistic, so distance
        # from zero reads directly as evidence.
        scale = (hi - lo) / (2 * 1.96) if hi > lo else 1.0
        c = LATE_C if t.nominally_significant else "#9AA5B1"
        ax.plot([lo / scale, hi / scale], [i, i], color=c, lw=2.6, alpha=0.75, solid_capstyle="round")
        ax.plot([sign * t.difference / scale], [i], "o", color=c, ms=7)
        ax.text(
            0.985, i, f"p={t.p_value:.3f}", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=7,
            color="#222" if t.nominally_significant else "#777",
        )
    ax.axvline(0, color="#222", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([t.endpoint for t in tests], fontsize=8)
    ax.set_xlabel("Standardized between-stratum difference $t$ (benefit positive)")
    ax.set_title(
        "Do the strata actually differ?\nWelch two-sample test on the change scores",
        fontsize=9,
    )
    ax.set_xlim(-4.2, 4.6)

    # Right: p-values against both correction thresholds.
    ordered = sorted(tests, key=lambda t: t.p_value)
    ranks = np.arange(1, len(ordered) + 1)
    ps = np.array([t.p_value for t in ordered])
    m = len(ordered)
    ax2.plot(ranks, ps, "o-", color=TOTAL_C, lw=1.4, ms=6, label="observed p")
    ax2.plot(ranks, 0.05 * ranks / m, "--", color=GOOD_C, lw=1.6, label="Benjamini-Hochberg")
    ax2.axhline(0.05 / m, color=EARLY_C, ls=":", lw=1.6, label=f"Bonferroni (0.05/{m})")
    ax2.axhline(0.05, color="#BBB", lw=1.0)
    ax2.text(m * 0.98, 0.052, "nominal 0.05", fontsize=6.8, color="#888", ha="right")
    ax2.annotate(
        f"LVESV p={ps[0]:.3f}\nabove both thresholds",
        xy=(1, ps[0]), xytext=(2.1, 0.006), fontsize=7, color=ACCENT,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.0},
    )
    ax2.set_yscale("log")
    ax2.set_xlabel("Rank of endpoint by p-value")
    ax2.set_ylabel("p")
    ax2.set_title("No endpoint survives multiplicity correction", fontsize=9)
    ax2.legend(fontsize=7, frameon=False, loc="upper left")

    fig.tight_layout()
    save(fig, "fig2_interaction_and_multiplicity")


def fig3_literature_anchors() -> None:
    """External control arms: what untreated patients do."""
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    order = ["preservation_i", "time", "empress_mi", "focus_cctrn", "focus_hf"]  # LV volume anchors only
    y = np.arange(len(order))[::-1]

    for i, key in zip(y, order):
        a = ANCHORS[key]
        c = EARLY_C if a.phase == "acute" else LATE_C
        val = a.absolute_change()
        sd = a.absolute_sd()
        if sd is not None:
            se = sd / math.sqrt(a.n)
            ax.plot([val - 1.96 * se, val + 1.96 * se], [i, i], color=c, lw=2.6, alpha=0.5)
        ax.plot([val], [i], "o", color=c, ms=9)
        ax.text(
            val, i + 0.30, f"{a.trial} ({a.year}), n={a.n}", fontsize=7.5,
            ha="center", color="#333",
        )
        ax.text(
            val, i - 0.34, a.measure, fontsize=6.6, ha="center", color="#888",
        )

    ax.axvline(0, color="#222", lw=1.0)
    # VentriGel's own observed changes, for direct comparison.
    ax.axvline(9.3, color=EARLY_C, ls="--", lw=1.4, alpha=0.85)
    ax.axvline(-7.6, color=LATE_C, ls="--", lw=1.4, alpha=0.85)
    ax.text(9.5, len(order) - 0.55, "VentriGel\nearly (+9.3)", fontsize=7, color=EARLY_C)
    ax.text(-7.4, len(order) - 0.55, "VentriGel\nlate (-7.6)", fontsize=7, color=LATE_C, ha="right")

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{ANCHORS[k].phase}" for k in order], fontsize=8
    )
    ax.set_xlabel("Six-month change in LV volume, control or placebo arm (mL, BSA 1.9 m$^2$)")
    ax.set_title(
        "What happens to untreated patients\n"
        "Acute cohorts disagree in sign; chronic cohorts are stable",
        fontsize=9.5,
    )
    ax.set_ylim(-0.8, len(order) - 0.2)
    fig.tight_layout()
    save(fig, "fig3_literature_anchors")


def fig4_enrichment_curves() -> None:
    """Required sample size as enrollment is progressively restricted."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.9))

    styles = {
        "lvesv": (LATE_C, "-", "LVESV"),
        "viable_mass": (GOOD_C, "--", "Viable mass"),
        "six_min_walk": (ACCENT, "-", "6-min walk"),
        "mlwhfq": ("#8A6BA8", "--", "MLWHFQ"),
    }
    for key, (c, ls, lab) in styles.items():
        c_e, c_l = _controls_for(key)
        sweep = enrichment_curve(
            key, PI, c_e, c_l, ALPHA, POWER, DROPOUT, n_points=121, shrinkage=SHRINK
        )
        e = [d.e for d in sweep]
        n = [d.n_total if (d.feasible and d.favors_treatment) else np.nan for d in sweep]
        ax.plot(e, n, color=c, ls=ls, lw=1.8, label=lab)

    ax.set_yscale("log")
    ax.set_xlabel("Fraction of enrolled patients from the late stratum")
    ax.set_ylabel("Total randomized patients")
    ax.axvline(PI, color="#888", ls=":", lw=1.2)
    ax.text(PI + 0.012, ax.get_ylim()[1] * 0.35, "unselected\npool", fontsize=7, color="#666")
    ax.set_title("Sample size falls as enrollment is enriched", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    sweep = enrichment_curve(
        PRIMARY, PI, CE, CL, ALPHA, POWER, DROPOUT, n_points=121, shrinkage=SHRINK
    )
    e = np.array([d.e for d in sweep])
    ax2.plot(e, [d.effect for d in sweep], color=LATE_C, lw=1.8, label="effect $\\Delta$ (mL, benefit +)")
    ax2.plot(e, [d.sd for d in sweep], color=EARLY_C, lw=1.8, label="SD $\\sigma$ (mL)")
    ax2.plot(e, [d.sd_between for d in sweep], color="#888", lw=1.4, ls="--", label="between-stratum SD")
    ax2.axhline(0, color="#222", lw=0.9)
    ax2.axvline(PI, color="#888", ls=":", lw=1.2)
    ax2.set_xlabel("Fraction of enrolled patients from the late stratum")
    ax2.set_ylabel("mL")
    ax2.set_title("Enrichment raises the effect and lowers the variance", fontsize=9)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.tight_layout()
    save(fig, "fig4_enrichment_curves")


def fig5_assurance() -> None:
    """Power against probability of success, and what the comparator's SE costs.

    This merges what were two separate figures. They were both assurance curves
    against sample size and the reader had to hold one in memory to read the
    other, which is a sign they should have been one figure.
    """
    ns = np.unique(np.round(np.logspace(math.log10(24), math.log10(4000), 30)).astype(int))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    nominal = [r.nominal_power for r in
               assurance_curve(PRIMARY, ns, SHRINK, CL, ALPHA, n_draws=20000, control_se=0.0)]
    exact = assurance_curve(PRIMARY, ns, SHRINK, CL, ALPHA, n_draws=20000, control_se=0.0)
    prop = assurance_curve(PRIMARY, ns, SHRINK, CL, ALPHA, n_draws=20000, control_se=SE_L)

    ax.plot(ns, nominal, color=EARLY_C, lw=1.7, ls="--", label="nominal power")
    ax.plot(ns, [r.assurance for r in exact], color="#9AA5B1", lw=2.0,
            label="assurance, comparator exact")
    ax.plot(ns, [r.assurance for r in prop], color=LATE_C, lw=2.4,
            label=f"assurance, comparator SE {SE_L:.1f} mL")

    for res, c in ((exact, "#9AA5B1"), (prop, LATE_C)):
        se = 0.0 if c == "#9AA5B1" else SE_L
        ceil = assurance_ceiling(PRIMARY, SHRINK, CL, n_draws=20000, control_se=se)
        ax.axhline(ceil, color=c, ls=":", lw=1.1)
        ax.text(ns[0] * 1.04, ceil + 0.014, f"ceiling {ceil:.0%}", fontsize=6.8, color=c)

    ax.axhline(0.80, color=GOOD_C, lw=1.0, alpha=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("Total randomized patients")
    ax.set_ylabel("Probability of a significant result")
    ax.set_ylim(0, 1.03)
    ax.set_title("Nominal power overstates the real chance of success", fontsize=9)
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    targets = [0.5, 0.6, 0.7, 0.8]
    xs = np.arange(len(targets))
    w = 0.38
    for i, (se, c, lab) in enumerate(
        ((0.0, "#9AA5B1", "comparator exact"), (SE_L, LATE_C, "SE propagated"))
    ):
        vals = [n_for_assurance(PRIMARY, t, SHRINK, CL, ALPHA, n_draws=20000, control_se=se)
                for t in targets]
        vals = [v if math.isfinite(v) else np.nan for v in vals]
        ax2.bar(xs + (i - 0.5) * w, vals, width=w, color=c, label=lab)
        for x, v in zip(xs + (i - 0.5) * w, vals):
            if np.isfinite(v):
                ax2.text(x, v * 1.07, f"{v:,.0f}", ha="center", fontsize=7)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{t:.0%}" for t in targets])
    ax2.set_yscale("log")
    ax2.set_xlabel("Target probability of success")
    ax2.set_ylabel("Total randomized patients")
    ax2.set_title("Treating the comparator as exact halves the trial", fontsize=9)
    ax2.legend(fontsize=7.2, frameon=False, loc="upper left")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    save(fig, "fig5_assurance")


def fig6_control_drift() -> None:
    """The assumption space, with the literature-anchored region marked."""
    grid = sweep_assumptions(
        PRIMARY, PI, np.linspace(-16, 14, 31), np.linspace(-11, 11, 23),
        shrinkage=SHRINK, alpha=ALPHA, power=POWER, dropout=DROPOUT,
    )
    adv = np.array(grid.advantage, dtype=float)
    plot = np.where(np.isinf(adv), 1e4, adv)
    plot = np.clip(plot, 1.0, 1e4)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    im = ax.imshow(
        plot.T, origin="lower", aspect="auto", cmap="BuPu", norm=LogNorm(vmin=1, vmax=1e4),
        extent=(
            grid.control_early_values[0], grid.control_early_values[-1],
            grid.control_late_values[0], grid.control_late_values[-1],
        ),
    )
    # Where the enriched design itself fails: late control at or beyond the
    # treatment effect. This is the only region that actually matters.
    fail = ENDPOINTS[PRIMARY].change_6mo["late"].mean  # -7.6
    ax.axhspan(
        grid.control_late_values[0], fail, color="#1B1B1B", alpha=0.58, lw=0, zorder=3,
    )
    ax.axhline(fail, color="#fff", lw=1.3, zorder=4)
    ax.text(
        grid.control_early_values[0] + 0.6, fail - 0.55,
        "enriched design fails below this line:\nnatural history exceeds the treatment effect",
        fontsize=7.4, color="#fff", zorder=5, va="top", linespacing=1.35,
    )

    # The dashed box spans every published anchor for each stratum, so it shows
    # the region the literature actually constrains rather than an assumed range.
    early_vals = [a.absolute_change() for a in ANCHORS.values()
                  if a.endpoint == "lvesv" and a.phase == "acute"]
    late_vals = [a.absolute_change() for a in ANCHORS.values()
                 if a.endpoint == "lvesv" and a.phase == "chronic"]
    x0, x1 = min(early_vals), max(early_vals)
    y0, y1 = min(late_vals), max(late_vals)
    ax.add_patch(
        Rectangle(
            (x0, y0), x1 - x0, max(y1 - y0, 0.4),
            fill=False, edgecolor=ACCENT, lw=2.0, ls="--", zorder=5,
        )
    )
    ax.plot([CE], [CL], marker="*", ms=17, color=ACCENT, mec="#222", mew=0.8, zorder=6)
    ax.annotate(
        "anchored central\n(TIME / FOCUS-CCTRN)",
        xy=(CE, CL), xytext=(CE - 13.0, CL + 6.0), fontsize=7.5, color="#222", zorder=6,
        arrowprops={"arrowstyle": "->", "color": "#222", "lw": 1.0},
    )
    ax.set_xlabel("Assumed control-arm LVESV change, early stratum (mL at 6 months)")
    ax.set_ylabel("Assumed control-arm change, late stratum (mL)")
    ax.set_title(
        "Enrichment advantage across the control-arm assumption space\n"
        "Dashed box is the range spanned by published control arms",
        fontsize=9,
    )
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, label="unselected N / enriched N", extend="max")
    cb.ax.text(0.5, 1.11, "no pooled benefit", transform=cb.ax.transAxes,
               ha="center", fontsize=6.5, color="#444")
    fig.tight_layout()
    save(fig, "fig6_control_drift")


def fig7_bootstrap() -> None:
    """Estimation uncertainty in the enriched design."""
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    for ax, key in zip(axes, ("lvesv", "viable_mass", "six_min_walk")):
        c_e, c_l = _controls_for(key)
        b_un = bootstrap_designs(key, PI, PI, c_e, c_l, SHRINK, ALPHA, POWER, DROPOUT)
        b_en = bootstrap_designs(key, 1.0, PI, c_e, c_l, SHRINK, ALPHA, POWER, DROPOUT)
        bins = np.logspace(1, 5, 45)
        for b, c, lab in ((b_un, EARLY_C, "unselected"), (b_en, LATE_C, "enriched")):
            finite = b.samples[np.isfinite(b.samples)]
            ax.hist(bins=bins, x=finite, color=c, alpha=0.55,
                    label=f"{lab} ({b.feasible_fraction:.0%} solvable)")
        ax.set_xscale("log")
        ax.set_xlabel("Total randomized patients")
        ax.set_title(ENDPOINTS[key].label, fontsize=8.5)
        ax.legend(fontsize=6.8, frameon=False)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("Bootstrap draws")
    fig.suptitle(
        "Estimation uncertainty from the 6-8 patients per stratum (25% effect discount)",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig7_bootstrap")


def fig8_programme_and_confirmation() -> None:
    """Unconditional success, and the cost of confirming the claim itself."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.0))

    ns = np.unique(np.round(np.logspace(math.log10(40), math.log10(3000), 24)).astype(int))
    priors = np.array([0.3, 0.5, 0.7, 1.0])
    grid = programme_grid(PRIMARY, ns, priors, SHRINK, CL, SE_L, ALPHA, n_draws=20000)
    shades = ["#C7D3E0", "#8FAAC4", "#5A82A8", LATE_C]
    for j, (p, c) in enumerate(zip(priors, shades)):
        lab = "effect certain" if p == 1.0 else f"prior {p:.0%}"
        ax.plot(ns, grid[:, j], color=c, lw=2.2, label=lab)
    ax.axhline(0.80, color=GOOD_C, lw=1.0, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Total randomized patients")
    ax.set_ylabel("Unconditional probability of success")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "No sample size beats the prior\n"
        "Probability the subgroup effect is real times assurance",
        fontsize=9,
    )
    ax.legend(fontsize=7.2, frameon=False, loc="upper left")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    scenarios = [
        ("no control\ndrift", 0.0, 0.0),
        ("anchored\ncontrols", CE, CL),
    ]
    labels, enriched, interaction = [], [], []
    for name, c_e, c_l in scenarios:
        d = interaction_design(PRIMARY, c_e, c_l, ALPHA, POWER, DROPOUT, SHRINK)
        labels.append(name)
        enriched.append(d.n_enriched_reference)
        interaction.append(d.n_total)
    xs = np.arange(len(labels))
    w = 0.36
    ax2.bar(xs - w / 2, enriched, width=w, color=LATE_C, label="enriched 2-arm\n(effect in late stratum)")
    ax2.bar(xs + w / 2, interaction, width=w, color=ACCENT, label="2x2 interaction\n(does timing matter?)")
    for x, v in zip(xs - w / 2, enriched):
        ax2.text(x, v * 1.05, f"{v:,.0f}", ha="center", fontsize=7.5)
    for x, v in zip(xs + w / 2, interaction):
        ax2.text(x, v * 1.05, f"{v:,.0f}", ha="center", fontsize=7.5)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Total randomized patients")
    ax2.set_title(
        "Confirming the claim, not just the effect\n"
        "Anchoring halves the contrast and quadruples the trial",
        fontsize=9,
    )
    ax2.legend(fontsize=7, frameon=False, loc="upper left")
    ax2.set_ylim(0, max(interaction) * 1.35)

    fig.tight_layout()
    save(fig, "fig8_programme_and_confirmation")


def main() -> None:
    print("Generating figures...")
    fig1_cancellation()
    fig2_interaction()
    fig3_literature_anchors()
    fig4_enrichment_curves()
    fig5_assurance()
    fig6_control_drift()
    fig7_bootstrap()
    fig8_programme_and_confirmation()
    print(f"Done. {FIGURES.resolve()}")


if __name__ == "__main__":
    main()
