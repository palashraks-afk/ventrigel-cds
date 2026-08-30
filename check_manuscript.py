"""
Check that every number quoted in the manuscript and README matches the
analysis output.

    python check_manuscript.py

A paper is a claim about what the code produced. Nothing enforces that except
a check like this one, and prose drifts from code every time a number is
recomputed. This reads ``results/summary.json`` and the result tables, then
confirms each claim appears in the text where it should.

Exit status is non-zero if any claim fails, so it can gate a release.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PAPER = Path("paper/ventrigel_enrichment.tex")
README = Path("README.md")
RESULTS = Path("results")


def load() -> dict:
    with open(RESULTS / "summary.json") as f:
        return json.load(f)


def norm(text: str) -> str:
    """Strip LaTeX escapes and unify dashes so numbers match plain digits."""
    text = text.replace("\\%", "%").replace("{,}", ",").replace("\\,", " ")
    text = text.replace("-", "-").replace(",", "-").replace("−", "-")
    return text


def main() -> int:
    s = load()
    paper = norm(PAPER.read_text(encoding="utf-8"))
    readme = norm(README.read_text(encoding="utf-8"))
    both = paper + "\n" + readme

    ev = s["evidence"]
    anc = s["anchors"]
    au = s["anchor_uncertainty"]
    exact = au["anchor treated as exact"]
    prop = au["anchor uncertainty propagated"]
    rec = s["programme"]["recommended"]

    checks: list[tuple[str, bool, str]] = []

    def claim(label: str, needle: str, where: str = "both") -> None:
        """Record one claim.

        ``where="both"`` means the value must appear in *each* document, not in
        their concatenation. An earlier version searched the concatenated text,
        which let a corrected number in the README mask a stale one still in the
        paper -- the exact failure this checker exists to prevent, and it passed
        24/24 while the manuscript said 6.6 where the code said 6.5.
        """
        if where == "both":
            missing = [n for n, t in (("paper", paper), ("README", readme)) if needle not in t]
            ok = not missing
            note = needle if ok else f"{needle!r} (missing from: {', '.join(missing)})"
            checks.append((label, ok, note))
            return
        hay = {"paper": paper, "readme": readme}[where]
        checks.append((label, needle in hay, needle))

    # -- validation ---------------------------------------------------------
    claim("p-values checked", f"{s['validation']['p_checks_agree']} ")
    claim("median SD error", f"{s['validation']['sd_reconstruction_median_rel_error'] * 100:.1f}%")

    # -- the interaction ----------------------------------------------------
    claim("interaction p", f"{ev['strongest_p']:.3f}")
    claim("Bonferroni threshold", f"{0.05 / ev['n_tests']:.4f}")
    claim("effective families", f"{ev['effective_n_tests']} ")

    # -- anchors ------------------------------------------------------------
    claim("late anchor SE", f"{anc['control_late_se']:.1f}")
    claim("early anchor value", f"{anc['control_early']:.1f}")

    # -- the anchor-uncertainty correction ----------------------------------
    claim("assurance exact @174", f"{exact['assurance_174'] * 100:.0f}%")
    claim("assurance propagated @174", f"{prop['assurance_174'] * 100:.0f}%")
    claim("ceiling exact", f"{exact['ceiling'] * 100:.1f}%")
    claim("ceiling propagated", f"{prop['ceiling'] * 100:.1f}%")
    claim("N for 80% exact", f"{exact['n_for_80']:,.0f}")
    claim("N for 80% propagated", f"{prop['n_for_80']:,.0f}")

    # -- recommended design -------------------------------------------------
    claim("recommended N", f"{rec['n_total']:,.0f}")
    claim("screens for recommended design", f"{rec['n_screened']:,.0f}", "paper")
    claim("sites for 36 months", f"{rec['sites_for_calendar']['36']} sites", "paper")

    # -- confirmatory design ------------------------------------------------
    for d in s["confirmatory"]["designs"]:
        tag = "naive" if d["scenario"] == "no control drift" else "anchored"
        claim(f"2x2 N ({tag})", f"{d['n_total']:,.0f}")
        claim(f"2x2 contrast ({tag})", f"{d['contrast']:.1f}")

    # -- programme ----------------------------------------------------------
    row = next(r for r in s["programme"]["programme"] if r["n_total"] == rec["n_total"])
    claim("unconditional POS at prior 0.5", f"{row['prior_0.5'] * 100:.0f}%")

    # -- Table 1 sample sizes ----------------------------------------------
    import csv

    with open(RESULTS / "sample_sizes.csv") as f:
        rows = {r["endpoint"]: r for r in csv.DictReader(f)}
    for key in ("lvesv", "six_min_walk", "lvedv"):
        r = rows[key]
        n_enr = float(r["n_enriched"])
        checks.append(
            (f"Table 1 enriched N ({key})", f"{n_enr:,.0f}" in paper, f"{n_enr:,.0f}")
        )

    # -- report -------------------------------------------------------------
    width = max(len(c[0]) for c in checks) + 2
    failed = 0
    print("Verifying manuscript claims against results/\n")
    for label, ok, needle in checks:
        if not ok:
            failed += 1
        print(f"  {'OK  ' if ok else 'FAIL'}  {label:<{width}} {needle if not ok else repr(needle)}")

    print(f"\n{len(checks) - failed}/{len(checks)} claims verified")
    if failed:
        print(
            "\nA failure means the prose and the code disagree. Fix the prose, or "
            "the number, before shipping."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
