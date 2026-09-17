"""
TEA Multiplex Networks — Multi-emotional multiplex extension of TEA Networks.

Extends Target-Event-Agent (TEA) Networks by treating each Plutchik emotion as
a separate layer in a multiplex network. Each SVO triplet is classified into
emotional layers using the NRC Emotion Lexicon (EmoLex).

Layers:
  - syntactic: all SVO edges (base TEA graph)
  - fear, anger, sadness, joy, trust, disgust, surprise, anticipation:
    edges whose Event / Agent / Target carries that emotion (NRC EmoLex)
"""

from tea_multiplex.core import (
    PLUTCHIK,
    get_word_emotions,
    get_triplet_emotions,
    build_multiplex,
    extract_tea_triplets,
    analyze_text,
)

from tea_multiplex.metrics import (
    compute_layer_dominance,
    compute_participation_coefficient,
    compute_inter_layer_correlation,
    compute_emotional_lscc,
    compute_first_person_emotional_profile,
)

from tea_multiplex.comparison import (
    cohens_d,
    effect_label,
    match_by_word_count,
    compare_groups,
)

from tea_multiplex.validation import (
    benjamini_hochberg,
)

__version__ = "0.1.0"
