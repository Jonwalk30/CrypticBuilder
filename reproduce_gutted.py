from src.corpus import corpus
from src.step.positional_deletion import PositionalDeletionStep

pd = PositionalDeletionStep()

target = "cat"
step = pd.generate(corpus, target)

print(f"Results for '{target}':")
for c in step.candidates:
    if "gutted" in c.source:
        print(f"  {c.source} -> {c.produced} (Score: {c.score:.3f})")

# Test 'croat' specifically
df = corpus.corpus
croat_row = df[df['entry'] == 'croat']
if not croat_row.empty:
    print("\n'croat' is in corpus.")
else:
    print("\n'croat' is NOT in corpus.")
