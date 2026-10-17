from vecforge import VecForge

db = VecForge(":memory:")

# Add documents about different topics (animals, space, food, tech)
db.add("The quick brown fox jumps over the lazy dog")
db.add("NASA's Artemis program aims to return humans to the Moon")
db.add("Authentic Italian pizza requires a wood-fired oven and San Marzano tomatoes")
db.add("Quantum computing utilizes qubits to perform complex calculations")
db.add("A healthy diet includes plenty of fruits and vegetables")

print("=== Search: space exploration ===")
results1 = db.search("space exploration")
for r in results1:
    print(f"{r.score:.3f}  →  {r.text}")

assert results1[0].text.startswith("NASA's Artemis program"), "Failed: Space query should return space doc"

print("\n=== Search: culinary baking ===")
results2 = db.search("culinary baking")
for r in results2:
    print(f"{r.score:.3f}  →  {r.text}")

assert results2[0].text.startswith("Authentic Italian pizza"), "Failed: Culinary query should return pizza doc"

print("\n=== Search: animal pets ===")
results3 = db.search("animal pets")
for r in results3:
    print(f"{r.score:.3f}  →  {r.text}")

assert results3[0].text.startswith("The quick brown fox"), "Failed: Animal query should return fox doc"

# Verify scores are well-distributed and normalized for the first result set
scores = [r.score for r in results1]
assert len(set(scores)) > 1, "Failed: Scores are identical"
assert 0.0 <= min(scores) and max(scores) <= 1.0, "Failed: Scores are out of bounds [0, 1]"
assert scores == sorted(scores, reverse=True), "Failed: Scores are not sorted descending"

print("\n✅ New varied text search tests passed perfectly!")
