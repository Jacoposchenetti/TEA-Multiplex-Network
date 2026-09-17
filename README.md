# TEA Multiplex Networks
![Uploading image.png…]()

**Multi-emotional multiplex extension of Target-Event-Agent (TEA) Networks**

🧠 NLP · 🕸️ Multiplex Networks · 😡😨😢😊 Plutchik Emotions · 📊 Clinical Text Analysis

A Python library that extends [TEA Networks](https://github.com/MassimoStel/TEA_Networks) by treating each Plutchik emotion as a **separate structural layer** in a multiplex network.  Each Agent→Event→Target triplet is classified into emotional layers using the [NRC Emotion Lexicon (EmoLex)](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm), producing a 9-layer multiplex: one syntactic layer (all edges) plus eight emotional layers (fear, anger, sadness, joy, trust, disgust, surprise, anticipation).

---

## Description

Standard TEA Networks extract "who did what" (Agent→Event→Target) from text and build a single directed graph.  Emotional information (via VADER) is used only for *visualization* — node/edge coloring at plot time — but is **not a structural property** of the network.

TEA Multiplex Networks turn emotion into structure:

```
                    ┌──────────── SYNTACTIC LAYER ────────────┐
                    │  All Agent→Event→Target edges            │
                    └──────────────────────────────────────────┘
                    ┌──── FEAR ────┐  ┌──── ANGER ───┐
                    │ edges whose  │  │ edges whose   │   ...  × 8 emotions
                    │ triplet      │  │ triplet       │
                    │ carries fear │  │ carries anger │
                    └──────────────┘  └───────────────┘
```

Each emotional layer is a directed graph (`nx.DiGraph`) where edges exist only if the triplet's Event, Agent, or Target carries that emotion according to NRC EmoLex.  A single triplet can appear in multiple emotional layers simultaneously (e.g., "I **fear** losing **trust**" appears in both the fear and trust layers).

### How is this different from EmoAtlas?

| | **EmoAtlas** | **TEA Multiplex** |
|---|---|---|
| **Network type** | Word co-occurrence | Agent→Event→Target (SVO) |
| **Emotion on** | Nodes (words carry emotion) | Edges (narrative relationships carry emotion) |
| **What it captures** | Which emotions co-occur in text windows | Which emotions mediate narrative actions between actors |
| **Structural role** | Emotion colors/labels nodes | Emotion defines separate network layers |

### How is this different from standard TEA Networks?

| | **Standard TEA** | **TEA Multiplex** |
|---|---|---|
| **Valence** | VADER, 3 categories (pos/neg/neutral) | NRC EmoLex, 8 Plutchik emotions |
| **Usage** | Visualization only (node/edge coloring) | Structural: separate network layers + multiplex metrics |
| **Metrics** | Graph-level (density, LSCC, etc.) | Layer-level + cross-layer (participation, inter-layer correlation, layer dominance, etc.) |

## Multiplex Metrics

| Metric | Description |
|---|---|
| **Layer dominance** | Fraction of total emotional edges in each layer |
| **Participation coefficient** | How evenly a node's edges spread across emotional layers (0 = one layer, 1 = all layers equally) |
| **Inter-layer degree correlation** | Pearson *r* between node degree sequences across layer pairs |
| **Emotional LSCC** | Largest strongly connected component fraction per emotional layer |
| **Emotional coverage** | Fraction of syntactic nodes present in each emotional layer |
| **First-person emotional profile** | Emotion distribution for triplets where the Agent is first-person (I/me/my/we/us/our) |
| **Emotional concentration (HHI)** | Herfindahl index on layer dominance; high values = emotion concentrated in few layers |

## Installation

### Prerequisites

- Python 3.10–3.12
- [TEA Networks](https://github.com/MassimoStel/TEA_Networks) installed (for SVO extraction)
- A spaCy English model (`en_core_web_lg` or `en_core_web_trf`)

### Install

```bash
pip install git+https://github.com/Jacoposchenetti/TEA-Multiplex-Network.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/Jacoposchenetti/TEA-Multiplex-Network.git
cd TEA-Multiplex-Network
pip install -e .
```

Make sure TEA Networks is installed:

```bash
pip install git+https://github.com/MassimoStel/TEA_Networks.git
python -m spacy download en_core_web_lg
```

## Usage

```python
import spacy
import tea_multiplex as tmx

nlp = spacy.load("en_core_web_lg")
nlp.max_length = 50000

text = """
I keep checking the door lock over and over. The fear that something
terrible will happen consumes me. My family tries to help, but the
anxiety returns every time I walk away.
"""

# Full analysis pipeline
metrics = tmx.analyze_text(text, nlp)

print(f"Dominant emotion:      {metrics['dominant_emotion']}")
print(f"Participation coeff:   {metrics['participation_coeff']:.3f}")
print(f"Fear dominance:        {metrics['dom_fear']:.3f}")
print(f"First-person fear:     {metrics['fp_fear']:.3f}")
```

### Step-by-step usage

```python
# 1. Extract TEA triplets
triplets = tmx.extract_tea_triplets(text, nlp)

# 2. Build the multiplex network
layers = tmx.build_multiplex(triplets)

# 3. Compute individual metrics
dominance = tmx.compute_layer_dominance(layers)
participation = tmx.compute_participation_coefficient(layers)
lscc = tmx.compute_emotional_lscc(layers)
fp_profile = tmx.compute_first_person_emotional_profile(triplets)

# 4. Inter-layer correlation (e.g., syntactic vs fear)
r = tmx.compute_inter_layer_correlation(layers, "syntactic", "fear")
```

### Group comparison with statistical validation

```python
from tea_multiplex.comparison import compare_groups
from tea_multiplex.validation import benjamini_hochberg

# compare_groups returns a list of dicts with Cohen's d and p-values
results = compare_groups("Group A", data_a, "Group B", data_b, metrics_list)

# Apply Benjamini-Hochberg FDR correction
p_values = [r["p_value"] for r in results]
adjusted, significant = benjamini_hochberg(p_values)
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a complete working example.

## Project Structure

```
TEA-Multiplex-Network/
├── tea_multiplex/
│   ├── __init__.py          # Public API
│   ├── core.py              # Triplet extraction, multiplex construction, analysis
│   ├── metrics.py           # Multiplex network metrics
│   ├── comparison.py        # Group comparison (Mann-Whitney U, Cohen's d, matching)
│   └── validation.py        # Benjamini-Hochberg correction, configuration model null
├── examples/
│   └── basic_usage.py       # Minimal working example
├── setup.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you use this library, please cite both this work and the original TEA Networks paper:

```
Schenetti, J. (2026). TEA Multiplex Networks: Multi-emotional multiplex extension
of Target-Event-Agent Networks [Software]. GitHub.
https://github.com/Jacoposchenetti/TEA-Multiplex-Network
```

```
Franchini, S., Carrillo, A., De Duro, E. S., Improta, R., Ardebili, A. A.,
& Stella, M. (2026). TEA Nets combine AI and cognitive network science to
model actors, actions, and consequences in text [Preprint]. arXiv.
https://github.com/MassimoStel/TEA_Networks
```

## References

- Stella, M. (2020). Text-mining forma mentis networks reconstruct public perception of the STEM gender gap in social media. *PeerJ Computer Science*, 6, e295.
- Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a word–emotion association lexicon. *Computational Intelligence*, 29(3), 436–465.
- Plutchik, R. (1980). *Emotion: A Psychoevolutionary Synthesis*. Harper & Row.
- Hutto, C., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of the AAAI Conference on Web and Social Media*, 8(1), 216–225.

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
