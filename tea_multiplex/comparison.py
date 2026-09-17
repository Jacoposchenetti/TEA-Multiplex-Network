"""Statistical comparison utilities for TEA Multiplex group contrasts."""

import random
import numpy as np
from scipy import stats


def cohens_d(a, b):
    """Pooled-variance Cohen's d for two independent samples."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def effect_label(d):
    """Map |d| to a descriptive label."""
    d = abs(d)
    if d < 0.2:
        return "negl."
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "LARGE"


def match_by_word_count(group_a, group_b, tolerance=0.35):
    """
    Greedy word-count matching between two groups.

    Each text in *group_a* is paired with the closest unused text in
    *group_b* whose word count is within *tolerance* (relative to group_a's
    word count).  Returns two parallel lists of matched dicts.
    """
    b_words = np.array([r["n_words"] for r in group_b])
    matched_a, matched_b = [], []
    indices_a = list(range(len(group_a)))
    random.shuffle(indices_a)
    used_b = set()

    for ia in indices_a:
        wc_a = group_a[ia]["n_words"]
        best_ib, best_diff = None, float("inf")
        for ib in range(len(group_b)):
            if ib in used_b:
                continue
            diff = abs(b_words[ib] - wc_a)
            if diff < best_diff:
                best_diff = diff
                best_ib = ib
        if best_ib is not None and best_diff / max(wc_a, 1) <= tolerance:
            used_b.add(best_ib)
            matched_a.append(group_a[ia])
            matched_b.append(group_b[best_ib])

    return matched_a, matched_b


def compare_groups(name_a, data_a, name_b, data_b, metrics):
    """
    Length-matched Mann–Whitney U comparison between two groups.

    Parameters
    ----------
    name_a, name_b : str
        Group labels.
    data_a, data_b : list[dict]
        Per-text metric dicts (as returned by ``analyze_text``).
    metrics : list[tuple[str, str]]
        ``(column_key, display_label)`` pairs to compare.

    Returns
    -------
    list[dict]
        One row per metric with means, Cohen's d, p-value, effect label.
    """
    matched_a, matched_b = match_by_word_count(data_a, data_b)

    if len(matched_a) < 15:
        return []

    results = []
    for col, label in metrics:
        vals_a = np.array([r[col] for r in matched_a if r.get(col) is not None])
        vals_b = np.array([r[col] for r in matched_b if r.get(col) is not None])

        if len(vals_a) < 10 or len(vals_b) < 10:
            continue

        u, p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        d = cohens_d(vals_a, vals_b)
        eff = effect_label(d)

        results.append({
            "comparison": f"{name_a} vs {name_b}",
            "metric": label,
            "a_mean": round(float(np.mean(vals_a)), 4),
            "b_mean": round(float(np.mean(vals_b)), 4),
            "cohens_d": round(d, 3),
            "p_value": round(p, 6),
            "effect": eff,
            "n_pairs": len(matched_a),
            "significant": p < 0.05,
        })

    return results
