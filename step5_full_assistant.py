"""
=============================================================
STEP 5: FULL SYSTEM — INTERACTIVE TERMINAL CHATBOT (Groq)
=============================================================
Complete pipeline:

  User Query
      │
      ▼
  [Layer 1] Document Processor  → load + chunk policy doc
      │
      ▼
  [Layer 2] Vector Store        → TF-IDF index all chunks
      │
      ▼
  [Layer 3] Retrieval           → top-3 relevant chunks
      │
      ▼
  [Layer 4] LLM Generation      → LLaMA3 via Groq (free API)
      │
      ▼
  [Layer 5] Rule Validator      → validate + override if needed
      │
      ▼
  Structured Terminal Output

Setup:
  pip install groq scikit-learn --break-system-packages
  export GROQ_API_KEY="your-key-here"

Run:
  python3 step5_full_assistant.py
"""

import os
import re
import textwrap
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ════════════════════════════════════════════════════════════
# LAYER 1: DOCUMENT PROCESSING
# ════════════════════════════════════════════════════════════

def load_and_chunk(filepath: str, chunk_size=500, overlap=100) -> list:
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


# ════════════════════════════════════════════════════════════
# LAYER 2 + 3: VECTOR STORE + RETRIEVAL
# ════════════════════════════════════════════════════════════

class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.chunks = []
        self.matrix = None

    def build(self, chunks: list):
        self.chunks = chunks
        self.matrix = self.vectorizer.fit_transform(chunks)

    def search(self, query: str, top_k=3) -> list:
        q_vec   = self.vectorizer.transform([query])
        scores  = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_idx if scores[i] > 0]


# ════════════════════════════════════════════════════════════
# LAYER 4: LLM GENERATION (Groq - free API)
# ════════════════════════════════════════════════════════════

def generate_with_llm(query: str, context_chunks: list, client: Groq) -> str:
    """
    Calls Groq API with retrieved context + user query.
    Model: llama-3.1-8b-instant (free, fast, accurate)
    temperature=0.1 keeps answers factual and consistent.
    """
    context = "\n\n---\n\n".join(context_chunks)

    try:
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
            temperature = 0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq error] {str(e)}"


# ════════════════════════════════════════════════════════════
# LAYER 5: RULE-BASED VALIDATOR
# ════════════════════════════════════════════════════════════

POLICY = {
    "casual"     : {"max_consecutive_days": 3, "annual_days": 12,
                    "advance_notice_days": 1,  "carry_forward": False},
    "sick"       : {"annual_days": 12, "doc_required_after": 2,
                    "carry_forward_max": 30,   "report_by": "10:00 AM on day 1"},
    "earned"     : {"annual_days": 18, "min_days": 3,
                    "advance_notice_days": 7,  "carry_forward_max": 45,
                    "probation_eligible": False},
    "maternity"  : {"total_days": 182, "advance_notice_weeks": 4},
    "paternity"  : {"total_days": 5,   "use_within_days": 30},
    "bereavement": {"immediate_family_days": 5, "extended_family_days": 2},
    "compoff"    : {"use_within_days": 30},
}

APPROVAL_CHAIN = [
    (1, 3,   "Reporting Manager only"),
    (4, 7,   "Reporting Manager + Department Head"),
    (8, 999, "Reporting Manager + Department Head + HR"),
]


def detect_leave_type(text: str):
    t = text.lower()
    if re.search(r'\bcasual\b|\bcl\b', t):                      return "casual"
    if re.search(r'\bsick\b|\bsl\b', t):                        return "sick"
    if re.search(r'\bearned\b|\bprivilege\b|\bel\b|\bpl\b', t): return "earned"
    if re.search(r'\bmaternity\b', t):                          return "maternity"
    if re.search(r'\bpaternity\b', t):                          return "paternity"
    if re.search(r'\bbereavement\b', t):                        return "bereavement"
    if re.search(r'\bcomp.?off\b|\bcompensatory\b', t):         return "compoff"
    return None


def extract_days(text: str):
    m = re.search(r'\b(\d+)\s*days?\b', text, re.IGNORECASE)
    if m: return int(m.group(1))
    for word, num in {"one":1,"two":2,"three":3,"four":4,"five":5,
                      "six":6,"seven":7,"eight":8,"nine":9,"ten":10}.items():
        if re.search(rf'\b{word}\b', text, re.IGNORECASE): return num
    return None


def get_approval_chain(days: int) -> str:
    for lo, hi, chain in APPROVAL_CHAIN:
        if lo <= days <= hi: return chain
    return "HR (contact directly)"


def validate(query: str, llm_answer: str) -> dict:
    leave_type = detect_leave_type(query)
    days       = extract_days(query)
    rules      = POLICY.get(leave_type, {})

    answer, overridden, warnings, policy_ref = llm_answer, False, [], []

    if leave_type == "casual":
        max_d = rules["max_consecutive_days"]
        policy_ref += [
            f"Max consecutive days: {max_d}",
            f"Annual entitlement: {rules['annual_days']} days",
            f"Advance notice: {rules['advance_notice_days']} day(s)",
            "No carry-forward to next year",
        ]
        if days and days > max_d:
            answer     = (f"No, casual leave cannot exceed {max_d} consecutive days. "
                          f"You requested {days} days, which exceeds the limit.")
            overridden = True
            warnings.append(f"{days} days requested > {max_d}-day casual leave maximum.")

    elif leave_type == "sick":
        doc_after = rules["doc_required_after"]
        policy_ref += [
            f"Documentation required after: {doc_after} days",
            f"Report absence by: {rules['report_by']}",
            f"Annual entitlement: {rules['annual_days']} days",
            f"Carry-forward allowed up to: {rules['carry_forward_max']} days",
        ]
        if days and days > doc_after:
            w = f"Medical certificate required (leave > {doc_after} days)."
            warnings.append(w)
            if "certificate" not in llm_answer.lower() and "document" not in llm_answer.lower():
                answer     = llm_answer + f"\n\n  ⚠ Policy note: {w}"
                overridden = True

    elif leave_type == "earned":
        min_d = rules["min_days"]
        policy_ref += [
            f"Minimum days per request: {min_d}",
            f"Advance notice: {rules['advance_notice_days']} days",
            f"Annual entitlement: {rules['annual_days']} days",
            f"Carry-forward up to: {rules['carry_forward_max']} days",
            "Not available during probation (first 6 months)",
        ]
        if days and days < min_d:
            answer     = (f"No, earned leave requires a minimum of {min_d} days per application. "
                          f"You requested {days} day(s). Use casual leave for shorter durations.")
            overridden = True
            warnings.append(f"{days} day(s) is below the {min_d}-day earned leave minimum.")

    # Combination rule
    q = query.lower()
    if "casual" in q and "sick" in q and re.search(r'combin|together|both|and', q):
        answer     = "No, casual leave and sick leave cannot be combined per policy (Section 3.2)."
        overridden = True
        warnings.append("Casual + Sick leave combination is not permitted.")
        policy_ref.append("Section 3.2: Casual and Sick leave cannot be combined.")

    approval = get_approval_chain(days) if days else ""

    return {
        "answer"     : answer,
        "overridden" : overridden,
        "warnings"   : warnings,
        "policy_ref" : policy_ref,
        "approval"   : approval,
        "leave_type" : leave_type,
        "days"       : days,
    }


# ════════════════════════════════════════════════════════════
# OUTPUT FORMATTER
# ════════════════════════════════════════════════════════════

def print_response(query: str, result: dict):
    W   = 62
    tag = "⚠  RULE OVERRIDE" if result["overridden"] else "✓  VALIDATED"
    print(f"\n{'═'*W}")
    print(f"  QUERY  : {query}")
    print(f"{'─'*W}")

    lines = textwrap.wrap(result["answer"], width=W - 12)
    print(f"  Answer : {lines[0]}")
    for line in lines[1:]:
        print(f"           {line}")

    reason = "Rule override applied." if result["overridden"] else "LLM answer passed policy validation."
    print(f"  Reason : {reason}")
    print(f"  Status : {tag}")

    if result["approval"]:
        print(f"  Approve: {result['approval']}")
    for w in result["warnings"]:
        print(f"  ⚠ Warn : {w}")
    if result["policy_ref"]:
        print(f"  Policy :")
        for ref in result["policy_ref"]:
            print(f"    • {ref}")

    print(f"{'═'*W}")


# ════════════════════════════════════════════════════════════
# MAIN: INTERACTIVE LOOP
# ════════════════════════════════════════════════════════════

def main():
    POLICY_FILE = "leave_policy.txt"

    print("\n" + "═"*62)
    print("   🤖  Intelligent Leave Policy Assistant")
    print("       Powered by Groq (LLaMA3) — Free API")
    print("═"*62)

    # Check API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n[!] GROQ_API_KEY not set.")
        print("    Run:  export GROQ_API_KEY='your-key-here'")
        return

    # Build vector store
    print("\n[*] Loading and indexing policy document...")
    try:
        chunks = load_and_chunk(POLICY_FILE)
    except FileNotFoundError:
        print(f"[!] '{POLICY_FILE}' not found. Place it in the same folder as this script.")
        return

    store  = VectorStore()
    store.build(chunks)
    client = Groq(api_key=api_key)
    print(f"[✓] {len(chunks)} chunks indexed. System ready.\n")
    print("  Ask any leave-related question. Type 'exit' to quit.\n")

    # Query loop
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Goodbye]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("[Goodbye]")
            break

        print("  [→] Retrieving relevant policy chunks...")
        context = store.search(query, top_k=3)

        if not context:
            print("  [!] No relevant policy found for that query.\n")
            continue

        print(f"  [→] Generating answer via Groq ({len(context)} chunks)...")
        llm_answer = generate_with_llm(query, context, client)

        print("  [→] Validating against policy rules...")
        result = validate(query, llm_answer)

        print_response(query, result)


if __name__ == "__main__":
    main()