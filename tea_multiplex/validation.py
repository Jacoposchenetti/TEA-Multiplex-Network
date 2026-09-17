"""Statistical validation utilities: Benjamini-Hochberg correction and null models."""

import random
import numpy as np
import networkx as nx
from scipy import stats


def benjamini_hochberg(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction for multiple comparisons.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values.
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    adjusted : list[float]
        Adjusted p-values.
    significant : list[bool]
        Whether each test survives correction.
    """
    n = len(p_values)
    if n == 0:
        return [], []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    significant = [False] * n

    prev_adj = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        adj_p = min(prev_adj, p * n / rank)
        adj_p = min(adj_p, 1.0)
        adjusted[orig_idx] = adj_p
        significant[orig_idx] = adj_p < alpha
        prev_adj = adj_p

    return adjusted, significant


def configuration_model_null(G, n_rewires=100):
    """
    Generate configuration-model null networks preserving the degree sequence.

    Returns mean and std of density, clustering, and LSCC fraction across
    *n_rewires* random realizations, or ``None`` if the graph is too small.
    """
    if G.number_of_nodes() < 5:
        return None

    in_seq = [d for _, d in G.in_degree()]
    out_seq = [d for _, d in G.out_degree()]

    null_metrics = {"density": [], "clustering": [], "lscc_frac": []}

    for _ in range(n_rewires):
        try:
            G_rand = nx.directed_configuration_model(in_seq, out_seq, seed=random.randint(0, 99999))
            G_rand = nx.DiGraph(G_rand)
            G_rand.remove_edges_from(nx.selfloop_edges(G_rand))

            n = G_rand.number_of_nodes()
            null_metrics["density"].append(nx.density(G_rand))

            Gu = G_rand.to_undirected()
            null_metrics["clustering"].append(nx.average_clustering(Gu) if n > 2 else 0)

            sccs = list(nx.strongly_connected_components(G_rand))
            lscc = max(len(c) for c in sccs) if sccs else 0
            null_metrics["lscc_frac"].append(lscc / n if n > 0 else 0)
        except Exception:
            continue

    if not null_metrics["density"]:
        return None

    return {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in null_metrics.items()}
