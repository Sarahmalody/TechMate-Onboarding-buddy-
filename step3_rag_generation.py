"""
=============================================================
STEP 3: RETRIEVAL + LLM GENERATION (using Groq)
=============================================================
What changed from Ollama version:
  - No local model installation needed
  - Uses Groq's free API (llama-3.1-8b-instant model)
  - Responses are near-instant (Groq is very fast)
  - Just needs GROQ_API_KEY environment variable

Run: python3 step3_rag_generation.py
"""

import os
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Helpers from previous steps ───────────────────────────

def load_and_chunk(filepath, chunk_size=500, overlap=100):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    words  = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words): break
        start += chunk_size - overlap
    return chunks


class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.chunks = []
        self.matrix = None

    def build(self, chunks):
        self.chunks = chunks
        self.matrix = self.vectorizer.fit_transform(chunks)

    def search(self, query, top_k=3):
        q_vec  = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_idx if scores[i] > 0]


# ── LLM Generation via Groq ───────────────────────────────

def generate_with_llm(query: str, context_chunks: list) -> str:
    """
    Sends prompt to Groq API.
    Groq runs LLaMA3 on their servers — free tier, very fast.
    """
    client  = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    context = "\n\n---\n\n".join(context_chunks)

    print(f"\n[→] Sending to Groq (llama-3.1-8b-instant)...")

    response = client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [
            {
                "role"   : "system",
                "content": (
                    "You are an HR Leave Policy Assistant. "
                    "Answer questions using ONLY the provided policy context. "
                    "Be concise and factual. Do not add information not in the context."
                )
            },
            {
                "role"   : "user",
                "content": f"POLICY CONTEXT:\n{context}\n\nEMPLOYEE QUESTION:\n{query}"
            }
        ],
        max_tokens  = 300,
        temperature = 0.1,   # low temperature = more deterministic, less hallucination
    )

    return response.choices[0].message.content.strip()


# ── Run this step ──────────────────────────────────────────
if __name__ == "__main__":
    POLICY_FILE = "leave_policy.txt"

    print("[*] Building vector store...")
    chunks = load_and_chunk(POLICY_FILE)
    store  = VectorStore()
    store.build(chunks)
    print(f"    {len(chunks)} chunks indexed\n")

    test_queries = [
        "Can I take 5 days of casual leave?",
        "Do I need a medical certificate for sick leave?",
        "Can I combine casual and sick leave together?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        relevant_chunks = store.search(query, top_k=3)
        print(f"[✓] Retrieved {len(relevant_chunks)} relevant chunk(s)")

        answer = generate_with_llm(query, relevant_chunks)
        print(f"\nLLM RAW ANSWER:\n{answer}")

    print("\n[Step 3 complete] → Ready for rule validation in Step 4")