"""
=============================================================
STEP 2: EMBEDDING + VECTOR STORE
=============================================================
What this does:
  - Converts each text chunk into a numerical vector (embedding)
  - Stores all vectors so we can search them later

What is an embedding?
  An embedding turns text into a list of numbers that captures
  its *meaning*. Similar sentences end up with similar vectors.
  Example:
    "sick leave needs a doctor's note"  → [0.12, 0.85, 0.03, ...]
    "medical certificate for illness"   → [0.11, 0.83, 0.04, ...]
    "casual leave max 3 days"           → [0.67, 0.02, 0.91, ...]
  The first two are close; the third is far away.

Why TF-IDF instead of sentence-transformers?
  sentence-transformers requires a large model download (~90MB).
  TF-IDF (Term Frequency–Inverse Document Frequency) is a classic
  NLP technique that works well for keyword-rich policy documents
  and has zero external dependencies beyond sklearn.

  TF-IDF score for a word = (how often it appears in this chunk)
                           × (how rare it is across ALL chunks)

  Rare but frequent words in a chunk → high score → strong signal.

Run: python step2_vector_store.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Reuse chunking from Step 1 ─────────────────────────────
def load_and_chunk(filepath, chunk_size=500, overlap=100):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words): break
        start += chunk_size - overlap
    return chunks


class VectorStore:
    """
    TF-IDF based vector store.

    Internally stores a (num_chunks × vocab_size) sparse matrix.
    Each row is one chunk's TF-IDF vector.
    Searching = computing cosine similarity between query vector
                and all chunk vectors, then picking top-k.
    """

    def __init__(self):
        # ngram_range=(1,2) means we use single words AND two-word
        # phrases as features, e.g. "sick leave", "medical certificate"
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.chunks = []
        self.matrix = None   # sparse matrix: shape (n_chunks, vocab_size)

    def build(self, chunks: list) -> None:
        """Fit the vectorizer on all chunks and store the matrix."""
        self.chunks = chunks
        self.matrix = self.vectorizer.fit_transform(chunks)
        vocab_size  = len(self.vectorizer.vocabulary_)
        print(f"[✓] Vector store built")
        print(f"    Chunks indexed : {len(chunks)}")
        print(f"    Vocabulary size: {vocab_size} unique terms/bigrams\n")

    def search(self, query: str, top_k: int = 3) -> list:
        """
        Find the top_k most relevant chunks for a query.

        Steps:
          1. Transform query using the SAME vectorizer (same vocab)
          2. Compute cosine similarity: query_vec · each_chunk_vec
             (cosine similarity = 1 means identical, 0 means unrelated)
          3. Return chunks with highest scores
        """
        query_vec = self.vectorizer.transform([query])
        scores    = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx   = scores.argsort()[::-1][:top_k]   # indices of top scores

        results = []
        for idx in top_idx:
            if scores[idx] > 0:   # ignore completely irrelevant chunks
                results.append({
                    "chunk"     : self.chunks[idx],
                    "score"     : round(float(scores[idx]), 4),
                    "chunk_idx" : int(idx)
                })
        return results


# ── Run this step ──────────────────────────────────────────
if __name__ == "__main__":
    POLICY_FILE = "leave_policy.txt"

    # Build the store
    chunks = load_and_chunk(POLICY_FILE)
    store  = VectorStore()
    store.build(chunks)

    # Test a few queries to see what gets retrieved
    test_queries = [
        "Can I take 5 days casual leave?",
        "Do I need a medical certificate for sick leave?",
        "How many days advance notice for earned leave?",
    ]

    for query in test_queries:
        print(f"Query : {query}")
        print(f"{'─'*60}")
        results = store.search(query, top_k=3)
        for rank, r in enumerate(results, 1):
            preview = r["chunk"][:200].replace("\n", " ")
            print(f"  Rank {rank} | Score {r['score']:.4f} | {preview}...")
        print()

    print("[Step 2 complete] → VectorStore ready to use in Step 3")
