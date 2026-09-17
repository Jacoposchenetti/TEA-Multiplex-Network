"""
Basic usage example for TEA Multiplex Networks.

This script demonstrates how to:
  1. Extract TEA (Agent-Event-Target) triplets from a text
  2. Build a multiplex network with one layer per Plutchik emotion
  3. Compute multiplex metrics (layer dominance, participation, LSCC, etc.)
"""
import sys
sys.path.insert(0, "..")  # only needed if not pip-installed

import spacy
import tea_multiplex as tmx

# Load a spaCy model (en_core_web_lg recommended; en_core_web_trf for max accuracy)
nlp = spacy.load("en_core_web_lg")
nlp.max_length = 50000

# -------------------------------------------------------------------
# Example text: a short OCD-like narrative
# -------------------------------------------------------------------
text = """
I keep checking the door lock over and over. The fear that something terrible
will happen if I don't check consumes me. My family tries to help, but the
anxiety returns every time I walk away. I feel trapped in a cycle of doubt
and dread. Sometimes I check the stove too, afraid it might cause a fire.
The rituals take hours out of my day. I know it doesn't make sense, but
I can't stop. I anticipate disaster at every turn.
"""

# Step 1: extract SVO triplets (requires teanets on PYTHONPATH)
triplets = tmx.extract_tea_triplets(text, nlp)
print(f"Extracted {len(triplets)} triplets\n")

for t in triplets[:5]:
    print(f"  {t['node1']:15s}  --[{t['tea1']:6s}]--> {t['node2']}")

# Step 2: build the multiplex network
layers = tmx.build_multiplex(triplets)

print(f"\n{'Layer':15s} {'Nodes':>6s} {'Edges':>6s}")
print("-" * 30)
for name in ["syntactic"] + tmx.PLUTCHIK:
    G = layers[name]
    print(f"{name:15s} {G.number_of_nodes():6d} {G.number_of_edges():6d}")

# Step 3: compute all metrics at once
metrics = tmx.analyze_text(text, nlp)

if metrics:
    print(f"\nDominant emotion: {metrics['dominant_emotion']}")
    print(f"Participation coefficient: {metrics['participation_coeff']:.3f}")
    print(f"Emotional concentration (HHI): {metrics['emotional_concentration']:.3f}")

    print(f"\n{'Emotion':15s} {'Dominance':>10s} {'LSCC':>8s} {'Coverage':>10s}")
    print("-" * 48)
    for emo in tmx.PLUTCHIK:
        print(f"{emo:15s} {metrics[f'dom_{emo}']:10.4f} {metrics[f'lscc_{emo}']:8.4f} "
              f"{metrics[f'coverage_{emo}']:10.4f}")

    print("\nFirst-person emotional profile:")
    for emo in tmx.PLUTCHIK:
        val = metrics[f"fp_{emo}"]
        bar = "#" * int(val * 50)
        print(f"  {emo:15s} {val:.3f} {bar}")
