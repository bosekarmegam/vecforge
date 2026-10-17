from vecforge import VecForge

db = VecForge(":memory:")

db.add("The sky is blue and clear today")
db.add("Machine learning powers modern AI")
db.add("Python is great for data science")
db.add("VecForge is built by ArcGX TechLabs")

results = db.search("azure atmosphere")

print("=== Search: azure atmosphere ===")
for r in results:
    print(f"{r.score:.3f}  →  {r.text}")

# These must all pass
assert len(set(r.score for r in results)) > 1, \
    "FAIL: All scores are identical"

assert results[0].text == "The sky is blue and clear today", \
    f"FAIL: Wrong top result: {results[0].text}"

assert results[0].score > 0.5, \
    f"FAIL: Top score too low: {results[0].score}"

print("\n✅ All tests passed!")

# Test 2 - Different query, different top result
results2 = db.search("neural networks deep learning")
assert results2[0].text == "Machine learning powers modern AI", \
    "FAIL: ML doc should top for ML query"
print("✅ Test 2 passed - ML query returns ML doc")

# Test 3 - Scores are between 0 and 1
for r in results:
    assert 0.0 <= r.score <= 1.0, f"FAIL: Score out of range: {r.score}"
print("✅ Test 3 passed - All scores in [0, 1] range")

# Test 4 - Results are sorted highest first
scores = [r.score for r in results]
assert scores == sorted(scores, reverse=True), \
    "FAIL: Results not sorted by score"
print("✅ Test 4 passed - Results sorted correctly")

print("\n🎉 VecForge scoring is working correctly!")
print("Built by Suneel Bose K · ArcGX TechLabs Private Limited")
