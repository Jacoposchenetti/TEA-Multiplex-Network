"""Core TEA Multiplex building blocks: emotion classification, triplet extraction, multiplex construction."""

import numpy as np
import networkx as nx
from nrclex import NRCLex
from scipy import stats

from tea_multiplex.metrics import (
    compute_layer_dominance,
    compute_participation_coefficient,
    compute_inter_layer_correlation,
    compute_emotional_lscc,
    compute_first_person_emotional_profile,
)

PLUTCHIK = ["fear", "anger", "sadness", "joy", "trust", "disgust", "surprise", "anticipation"]

_nrc = NRCLex("init")
NRC_LEX = _nrc.__lexicon__


def get_word_emotions(word):
    """Return the set of Plutchik emotions for a word from NRC EmoLex."""
    emos = NRC_LEX.get(word.lower(), [])
    return set(e for e in emos if e in PLUTCHIK)


def get_triplet_emotions(event_text, node1_text, node2_text):
    """
    Determine which emotional layers a TEA triplet belongs to.

    All three components (Event, Agent, Target) contribute: if any word in any
    component carries a Plutchik emotion according to NRC EmoLex, the triplet is
    assigned to that emotional layer.
    """
    emotions = set()
    for word in event_text.lower().split():
        emotions |= get_word_emotions(word)
    for word in node1_text.lower().split():
        emotions |= get_word_emotions(word)
    for word in node2_text.lower().split():
        emotions |= get_word_emotions(word)
    return emotions


def extract_tea_triplets(text, nlp):
    """Extract SVO triplets from text using the TEA Networks library."""
    from teanets.svo_extraction import extract_svos

    if len(text) > 49000:
        text = text[:49000]
    doc = nlp(text)
    try:
        df = extract_svos(doc)
        if df is None or len(df) == 0:
            return []
        triplets = []
        for _, row in df.iterrows():
            n1 = str(row["Node 1"]).strip().lower()
            tea1 = str(row["TEA"]).strip()
            n2 = str(row["Node 2"]).strip().lower()
            tea2 = str(row["TEA2"]).strip()
            if not n1 or not n2 or n1 == "nan" or n2 == "nan":
                continue
            event_col = row.get("Hypergraph", "")
            event_text = str(event_col) if event_col != "N/A" else ""
            triplets.append({
                "node1": n1, "tea1": tea1,
                "node2": n2, "tea2": tea2,
                "event_text": event_text,
            })
        return triplets
    except Exception:
        return []


def build_multiplex(triplets):
    """
    Build a multiplex network from TEA triplets.

    Returns a dict of ``nx.DiGraph`` keyed by layer name:
    ``{"syntactic": DiGraph, "fear": DiGraph, ..., "anticipation": DiGraph}``.

    The *syntactic* layer contains every edge (like a standard TEA graph).
    Each emotional layer contains only edges whose triplet carries that emotion.
    """
    layers = {"syntactic": nx.DiGraph()}
    for emo in PLUTCHIK:
        layers[emo] = nx.DiGraph()

    for t in triplets:
        n1, n2 = t["node1"], t["node2"]

        if layers["syntactic"].has_edge(n1, n2):
            layers["syntactic"][n1][n2]["weight"] += 1
        else:
            layers["syntactic"].add_edge(n1, n2, weight=1)

        emotions = get_triplet_emotions(t["event_text"], n1, n2)
        for emo in emotions:
            G = layers[emo]
            if G.has_edge(n1, n2):
                G[n1][n2]["weight"] += 1
            else:
                G.add_edge(n1, n2, weight=1)

    return layers


def analyze_text(text, nlp):
    """
    Run the full TEA Multiplex pipeline on a single text.

    Returns a dict with all multiplex metrics, or ``None`` if the text
    yields fewer than 5 triplets.
    """
    triplets = extract_tea_triplets(text, nlp)
    if len(triplets) < 5:
        return None

    layers = build_multiplex(triplets)

    n_syntactic_edges = layers["syntactic"].number_of_edges()
    n_syntactic_nodes = layers["syntactic"].number_of_nodes()
    if n_syntactic_nodes == 0:
        return None

    dominance = compute_layer_dominance(layers)
    participation = compute_participation_coefficient(layers)

    correlations = {}
    for emo in PLUTCHIK:
        r = compute_inter_layer_correlation(layers, "syntactic", emo)
        correlations[f"corr_synt_{emo}"] = r

    emo_lscc = compute_emotional_lscc(layers)
    fp_profile = compute_first_person_emotional_profile(triplets)

    coverage = {}
    for emo in PLUTCHIK:
        emo_nodes = set(layers[emo].nodes())
        synt_nodes = set(layers["syntactic"].nodes())
        coverage[f"coverage_{emo}"] = len(emo_nodes & synt_nodes) / max(len(synt_nodes), 1)

    dom_emo = max(PLUTCHIK, key=lambda e: dominance[e])
    dom_ratio = dominance[dom_emo]
    hhi = sum(v ** 2 for v in dominance.values())

    result = {
        "n_words": len(text.split()),
        "n_triplets": len(triplets),
        "n_nodes": n_syntactic_nodes,
        "n_edges": n_syntactic_edges,
        "participation_coeff": round(participation, 4),
        "dominant_emotion": dom_emo,
        "dominant_ratio": round(dom_ratio, 4),
        "emotional_concentration": round(hhi, 4),
    }

    for emo in PLUTCHIK:
        result[f"dom_{emo}"] = round(dominance[emo], 4)
        result[f"lscc_{emo}"] = round(emo_lscc[emo], 4)
        result[f"coverage_{emo}"] = round(coverage[f"coverage_{emo}"], 4)

    result.update({k: round(v, 4) if not np.isnan(v) else None for k, v in correlations.items()})
    result.update({k: round(v, 4) for k, v in fp_profile.items()})

    return result
