"""Apply the drill's exact miner to unordered synthetic season baskets."""
from typing import Sequence

from thilimarketbasket.mine import frequent_itemsets, rules

from app.core import FieldHistory


def mine_baskets(
    baskets: Sequence[frozenset[str]], min_support: float = 0.04,
    min_confidence: float = 0.55, max_size: int = 4,
) -> dict:
    """Wrap thilimarketbasket's support and rule API in stable report records."""
    if not baskets or not 0 < min_support <= 1 or not 0 <= min_confidence <= 1 or max_size < 1:
        raise ValueError("Require baskets, support in (0, 1], confidence in [0, 1], and positive max size.")
    supports = frequent_itemsets(baskets, min_support, max_size=max_size)
    itemsets = [{"items": sorted(items), "support": support, "count": round(support * len(baskets))}
                for items, support in supports.items()]
    itemsets.sort(key=lambda row: (-row["support"], row["items"]))
    associations = [
        {"antecedent": sorted(a), "consequent": sorted(b), "support": support,
         "confidence": confidence, "lift": lift}
        for a, b, support, confidence, lift in rules(supports, min_confidence=min_confidence)
    ]
    associations.sort(key=lambda row: (
        -row["lift"], -row["support"], row["antecedent"], row["consequent"]
    ))
    return {"basket_count": len(baskets), "min_support": min_support,
            "min_confidence": min_confidence, "max_size": max_size,
            "itemsets": itemsets, "rules": associations}


def mine_training(
    fields: Sequence[FieldHistory], min_support: float = 0.04,
    min_confidence: float = 0.55, max_size: int = 4,
) -> dict:
    """Pool distinct unordered windows from the supplied training fields."""
    return mine_baskets(
        [basket for field in fields for basket in field.baskets()],
        min_support, min_confidence, max_size,
    )


def render_patterns(report: dict) -> str:
    """Render the most supported co-occurrences and strongest associations."""
    lines = ["# Patterns mined from training fields", "",
             f'{report["basket_count"]:,} season baskets; minimum support {report["min_support"]:.1%}; '
             f'minimum confidence {report["min_confidence"]:.1%}; itemsets up to size {report["max_size"]}.',
             "", "These are co-occurrences in a synthetic dataset.", "",
             "| Frequent itemset | Count | Support |", "| --- | ---: | ---: |"]
    pairs = [row for row in report["itemsets"] if len(row["items"]) > 1]
    for row in pairs[:12]:
        lines.append(f'| {", ".join(row["items"])} | {row["count"]:,} | {row["support"]:.2%} |')
    lines.extend(["", "| Association | Support | Confidence | Lift |",
                  "| --- | ---: | ---: | ---: |"])
    for row in report["rules"][:12]:
        lines.append(f'| {", ".join(row["antecedent"])} → {", ".join(row["consequent"])} | '
                     f'{row["support"]:.2%} | {row["confidence"]:.2%} | {row["lift"]:.2f} |')
    return "\n".join(lines) + "\n"


def compare_support(
    fields: Sequence[FieldHistory], thresholds: Sequence[float], min_confidence: float = 0.55,
    max_size: int = 4,
) -> dict:
    """Hold the field split and confidence fixed while changing minimum support."""
    rows = []
    for threshold in thresholds:
        report = mine_training(fields, threshold, min_confidence, max_size)
        rules_found = report["rules"]
        rows.append({"min_support": threshold, "itemsets": len(report["itemsets"]),
                     "rules": len(rules_found), "strongest_rule": rules_found[0] if rules_found else None})
    return {"min_confidence": min_confidence, "max_size": max_size, "thresholds": rows}


def render_support(report: dict) -> str:
    """Summarize how counts and the strongest retained rule change."""
    lines = ["# Minimum support sensitivity", "",
             f'Minimum confidence held at {report["min_confidence"]:.0%}.', "",
             "| Min support | Itemsets | Rules | Highest-lift retained rule | Lift |",
             "| ---: | ---: | ---: | --- | ---: |"]
    for row in report["thresholds"]:
        best = row["strongest_rule"]
        pair = f'{", ".join(best["antecedent"])} → {", ".join(best["consequent"])}' if best else "None"
        lift = f'{best["lift"]:.2f}' if best else "–"
        lines.append(f'| {row["min_support"]:.0%} | {row["itemsets"]} | {row["rules"]} | {pair} | {lift} |')
    return "\n".join(lines) + "\n"
