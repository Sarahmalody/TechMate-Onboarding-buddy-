"""
=============================================================
STEP 4: RULE-BASED LOGIC LAYER
=============================================================
What this does:
  - Intercepts the LLM's raw answer
  - Detects leave type and number of days from the query
  - Applies deterministic policy rules to validate the answer
  - Overrides incorrect LLM outputs with correct policy facts
  - Formats everything into a clean structured response

Why do we need this if we already have RAG?
  LLMs are probabilistic — even with good context, they can:
    • Miscount days ("you can take up to 4 casual leave days")
    • Forget a condition ("no documentation needed" — wrong after day 2)
    • Be ambiguous ("approval may be required" vs. exactly WHO must approve)

  The rule layer is deterministic — it always produces the
  correct answer for cases it covers, with zero hallucination risk.

  Think of it as: RAG handles the long tail of open questions,
  rules handle the high-stakes specific scenarios.

Run: python step4_rule_validator.py
"""

import re


# ── Policy Rules (single source of truth) ─────────────────

POLICY = {
    "casual": {
        "max_consecutive_days" : 3,
        "annual_days"          : 12,
        "advance_notice_days"  : 1,
        "carry_forward"        : False,
        "doc_required"         : False,
        "probation_eligible"   : True,
        "combine_with_sick"    : False,
    },
    "sick": {
        "annual_days"          : 12,
        "doc_required_after"   : 2,       # days
        "carry_forward_max"    : 30,
        "probation_eligible"   : True,
        "combine_with_casual"  : False,
        "report_by"            : "10:00 AM on day 1",
    },
    "earned": {
        "annual_days"          : 18,
        "min_days"             : 3,       # minimum per application
        "advance_notice_days"  : 7,
        "carry_forward_max"    : 45,
        "encashment_max"       : 15,
        "probation_eligible"   : False,
    },
    "maternity" : {"total_days": 182, "advance_notice_weeks": 4},
    "paternity" : {"total_days": 5,   "use_within_days": 30},
    "bereavement": {"immediate_family_days": 5, "extended_family_days": 2},
    "compoff"   : {"use_within_days": 30},
}

APPROVAL_CHAIN = [
    (1, 3,   "Reporting Manager only"),
    (4, 7,   "Reporting Manager + Department Head"),
    (8, 999, "Reporting Manager + Department Head + HR"),
]


# ── Extraction helpers ─────────────────────────────────────

def detect_leave_type(text: str) -> str:
    """Return leave type keyword or None."""
    t = text.lower()
    if re.search(r'\bcasual\b|\b\bcl\b', t):                       return "casual"
    if re.search(r'\bsick\b|\bsl\b', t):                           return "sick"
    if re.search(r'\bearned\b|\bprivilege\b|\bel\b|\bpl\b', t):   return "earned"
    if re.search(r'\bmaternity\b', t):                             return "maternity"
    if re.search(r'\bpaternity\b', t):                             return "paternity"
    if re.search(r'\bbereavement\b', t):                           return "bereavement"
    if re.search(r'\bcomp.?off\b|\bcompensatory\b', t):            return "compoff"
    return None


def extract_days(text: str) -> int:
    """Extract number of days mentioned in text. Returns None if not found."""
    m = re.search(r'\b(\d+)\s*days?\b', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    word_map = {"one":1,"two":2,"three":3,"four":4,"five":5,
                "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    for word, num in word_map.items():
        if re.search(rf'\b{word}\b', text, re.IGNORECASE):
            return num
    return None


def get_approval_chain(days: int) -> str:
    for lo, hi, chain in APPROVAL_CHAIN:
        if lo <= days <= hi:
            return chain
    return "HR (contact directly)"


# ── Core validator ─────────────────────────────────────────

def validate_and_format(query: str, llm_answer: str) -> dict:
    """
    Main rule-based validation function.

    Returns a dict with:
      answer     : final answer (overridden if rule violation detected)
      reason     : policy basis for the answer
      warnings   : list of policy flags raised
      policy_ref : relevant policy rules that apply
      approval   : who must approve (if days are mentioned)
      overridden : True if LLM answer was replaced by a rule
    """
    leave_type = detect_leave_type(query)
    days       = extract_days(query)
    rules      = POLICY.get(leave_type, {})

    # Start with LLM's answer as the default
    answer     = llm_answer
    overridden = False
    warnings   = []
    policy_ref = []

    # ── Rule Set 1: Casual Leave ───────────────────────────
    if leave_type == "casual":
        max_days = rules["max_consecutive_days"]
        policy_ref.append(f"Casual leave max: {max_days} consecutive days (Section 1.1)")
        policy_ref.append(f"Annual entitlement: {rules['annual_days']} days")
        policy_ref.append(f"Advance notice: {rules['advance_notice_days']} day(s) prior")
        policy_ref.append("Cannot be carried forward to next year")

        if days is not None and days > max_days:
            # OVERRIDE: LLM might have said "yes" — rule says no
            answer     = (f"No, you cannot take {days} consecutive days of casual leave. "
                          f"The maximum allowed is {max_days} days at a time.")
            overridden = True
            warnings.append(f"Requested {days} days exceeds the {max_days}-day casual leave limit.")

    # ── Rule Set 2: Sick Leave ─────────────────────────────
    elif leave_type == "sick":
        doc_threshold = rules["doc_required_after"]
        policy_ref.append(f"No documentation needed for up to {doc_threshold} days (Section 1.2)")
        policy_ref.append(f"Medical certificate required if sick leave > {doc_threshold} days")
        policy_ref.append(f"Report absence to manager by: {rules['report_by']}")
        policy_ref.append(f"Annual entitlement: {rules['annual_days']} days")

        if days is not None and days > doc_threshold:
            doc_note = f"Medical certificate required (sick leave exceeds {doc_threshold} days)."
            warnings.append(doc_note)
            # Only override if the LLM forgot to mention documentation
            if "certificate" not in llm_answer.lower() and "document" not in llm_answer.lower():
                answer     = llm_answer + f"\n\n⚠ Note: {doc_note}"
                overridden = True

    # ── Rule Set 3: Earned Leave ───────────────────────────
    elif leave_type == "earned":
        min_days = rules["min_days"]
        policy_ref.append(f"Minimum application: {min_days} days at a time (Section 1.3)")
        policy_ref.append(f"Advance notice: {rules['advance_notice_days']} days prior approval")
        policy_ref.append(f"Annual entitlement: {rules['annual_days']} days")
        policy_ref.append("Not available during probation period (first 6 months)")

        if days is not None and days < min_days:
            answer     = (f"No, earned leave requires a minimum of {min_days} days per application. "
                          f"You requested only {days} day(s). Consider applying casual leave instead.")
            overridden = True
            warnings.append(f"Requested {days} day(s) is below the {min_days}-day minimum for earned leave.")

    # ── Rule Set 4: Combination check ─────────────────────
    q_lower = query.lower()
    if "casual" in q_lower and "sick" in q_lower and ("combine" in q_lower or "together" in q_lower or "and" in q_lower):
        answer     = "No, casual leave and sick leave cannot be combined as per company policy (Section 3.2)."
        overridden = True
        warnings.append("Casual leave + sick leave combination is not permitted.")
        policy_ref.append("Combination Rule: Casual + Sick leave cannot be combined (Section 3.2)")

    # ── Approval chain (whenever days are specified) ───────
    approval = ""
    if days is not None:
        approval = get_approval_chain(days)

    reason = (
        "LLM answer was overridden by deterministic policy rule."
        if overridden
        else "LLM answer is consistent with policy rules."
    )

    return {
        "answer"     : answer,
        "reason"     : reason,
        "warnings"   : warnings,
        "policy_ref" : policy_ref,
        "approval"   : approval,
        "overridden" : overridden,
        "leave_type" : leave_type,
        "days_found" : days,
    }


# ── Pretty printer ─────────────────────────────────────────

def print_result(result: dict):
    tag = "⚠ OVERRIDDEN" if result["overridden"] else "✓ VALIDATED"
    print(f"\n┌─ [{tag}] ─────────────────────────────────────")
    print(f"│ Answer   : {result['answer']}")
    print(f"│ Reason   : {result['reason']}")
    if result["approval"]:
        print(f"│ Approval : {result['approval']}")
    if result["warnings"]:
        for w in result["warnings"]:
            print(f"│ ⚠ Warning : {w}")
    if result["policy_ref"]:
        print(f"│ Policy   :")
        for p in result["policy_ref"]:
            print(f"│   • {p}")
    print(f"└{'─'*55}")


# ── Run this step ──────────────────────────────────────────
if __name__ == "__main__":

    # Simulated LLM answers (as if Step 3 returned these)
    test_cases = [
        {
            "query"      : "Can I take 5 days casual leave?",
            "llm_answer" : "Yes, you can take 5 days of casual leave if your manager approves.",
        },
        {
            "query"      : "I need sick leave for 4 days, do I need any documents?",
            "llm_answer" : "You can take sick leave for 4 days.",
        },
        {
            "query"      : "Can I take 1 day of earned leave?",
            "llm_answer" : "Yes, you may apply for earned leave as needed.",
        },
        {
            "query"      : "Can I combine casual leave and sick leave?",
            "llm_answer" : "Combining leaves may be possible with approval.",
        },
        {
            "query"      : "How many days of casual leave can I take in a year?",
            "llm_answer" : "You are entitled to 12 days of casual leave per calendar year.",
        },
    ]

    for case in test_cases:
        print(f"\nQUERY      : {case['query']}")
        print(f"LLM ANSWER : {case['llm_answer']}")
        result = validate_and_format(case["query"], case["llm_answer"])
        print_result(result)

    print("\n[Step 4 complete] → Rule validation working. Ready to assemble in Step 5.")
