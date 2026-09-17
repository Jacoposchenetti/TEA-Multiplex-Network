"""Multiplex network metrics for TEA Multiplex layers."""

import numpy as np
import networkx as nx
from scipy import stats


PLUTCHIK = ["fear", "anger", "sadness", "joy", "trust", "disgust", "surprise", "anticipation"]


def compute_layer_dominance(layers):
    """Fraction of edges in each emotional layer relative to total emotional edges."""
    total_emo_edges = sum(layers[e].number_of_edges() for e in PLUTCHIK)
    if total_emo_edges == 0:
        return {e: 0.0 for e in PLUTCHIK}
    return {e: layers[e].number_of_edges() / total_emo_edges for e in PLUTCHIK}


def compute_participation_coefficient(layers):
    """
    Multiplex participation coefficient, averaged over all nodes.

    For each node *i* present in at least one emotional layer:

        P_i = (M / (M-1)) * (1 - sum_alpha (k_i^alpha / o_i)^2)

    where *k_i^alpha* is the weighted degree of *i* in layer *alpha*,
    *o_i* is the total weighted degree across all emotional layers, and
    *M* = 8 (number of Plutchik layers).

    P = 0 when a node is active in one layer only; P -> 1 when its edges are
    evenly distributed across all layers.
    """
    all_nodes = set()
    for emo in PLUTCHIK:
        all_nodes |= set(layers[emo].nodes())

    if not all_nodes:
        return 0.0

    M = len(PLUTCHIK)
    coefficients = []

    for node in all_nodes:
        degrees = []
        for emo in PLUTCHIK:
            G = layers[emo]
            if G.has_node(node):
                degrees.append(G.degree(node, weight="weight"))
            else:
                degrees.append(0)

        o_i = sum(degrees)
        if o_i == 0:
            continue

        sum_sq = sum((k / o_i) ** 2 for k in degrees)
        P_i = (M / (M - 1)) * (1 - sum_sq)
        coefficients.append(P_i)

    return np.mean(coefficients) if coefficients else 0.0


def compute_inter_layer_correlation(layers, layer_a, layer_b):
    """Pearson correlation of node degrees between two layers."""
    nodes_a = set(layers[layer_a].nodes())
    nodes_b = set(layers[layer_b].nodes())
    common = nodes_a & nodes_b

    if len(common) < 5:
        return np.nan

    deg_a = [layers[layer_a].degree(n, weight="weight") for n in common]
    deg_b = [layers[layer_b].degree(n, weight="weight") for n in common]

    if np.std(deg_a) == 0 or np.std(deg_b) == 0:
        return np.nan

    r, _ = stats.pearsonr(deg_a, deg_b)
    return r


def compute_emotional_lscc(layers):
    """Largest Strongly Connected Component fraction for each emotional layer."""
    result = {}
    for emo in PLUTCHIK:
        G = layers[emo]
        n = G.number_of_nodes()
        if n < 2:
            result[emo] = 0.0
            continue
        sccs = list(nx.strongly_connected_components(G))
        lscc = max(len(c) for c in sccs) if sccs else 0
        result[emo] = lscc / n
    return result


def compute_first_person_emotional_profile(triplets):
    """
    For first-person agent triplets (I/me/my/myself/we/us/our),
    compute which emotions dominate.

    Returns a dict ``{"fp_fear": float, "fp_anger": float, ...}`` with
    the fraction of first-person triplets carrying each emotion.
    """
    from tea_multiplex.core import get_triplet_emotions

    fp = {"i", "me", "my", "myself", "we", "us", "our"}
    emo_counts = {e: 0 for e in PLUTCHIK}
    fp_total = 0

    for t in triplets:
        if t["node1"] in fp and t["tea1"] == "Agent":
            fp_total += 1
            emotions = get_triplet_emotions(t["event_text"], t["node1"], t["node2"])
            for emo in emotions:
                emo_counts[emo] += 1

    if fp_total == 0:
        return {f"fp_{e}": 0.0 for e in PLUTCHIK}
    return {f"fp_{e}": emo_counts[e] / fp_total for e in PLUTCHIK}
