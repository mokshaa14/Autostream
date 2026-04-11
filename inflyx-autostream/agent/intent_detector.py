"""
agent/intent_detector.py
Classifies user intent into one of three categories using keyword heuristics
combined with LLM-based classification for accuracy.
"""

import re

# ── Keyword signal sets ────────────────────────────────────────────────────────

GREETING_SIGNALS = {
    "hi", "hello", "hey", "howdy", "good morning", "good afternoon",
    "good evening", "greetings", "sup", "what's up", "whats up"
}

HIGH_INTENT_SIGNALS = {
    "sign up", "signup", "subscribe", "buy", "purchase", "get started",
    "start now", "i want to try", "i want to buy", "let's go", "lets go",
    "ready to", "i'll take", "i want the", "get the pro", "get the basic",
    "enroll", "join", "i'm in", "im in", "let me in", "onboard",
    "give me access", "add me", "start my trial", "free trial",
    "sounds good", "that sounds great", "perfect", "i'm interested",
    "count me in"
}

PRODUCT_SIGNALS = {
    "price", "pricing", "plan", "plans", "cost", "how much", "feature",
    "features", "what do you offer", "tell me about", "explain",
    "difference", "compare", "4k", "captions", "resolution", "support",
    "refund", "cancel", "trial", "what is", "how does", "do you have",
    "can you", "does it", "what formats", "limit", "unlimited"
}


def classify_intent(user_message: str) -> str:
    """
    Classify the user's message into one of three intents.
    
    Returns:
        "greeting"         – casual hello / chitchat
        "product_inquiry"  – asking about features, pricing, policies
        "high_intent"      – ready to sign up or trial
    """
    msg_lower = user_message.lower().strip()

    # Check high intent FIRST (highest priority — don't miss a hot lead)
    for signal in HIGH_INTENT_SIGNALS:
        if signal in msg_lower:
            return "high_intent"

    # Check greeting signals (only if short message — avoids false positives)
    words = msg_lower.split()
    if len(words) <= 6:
        for signal in GREETING_SIGNALS:
            if signal in msg_lower:
                return "greeting"

    # Check product inquiry signals
    for signal in PRODUCT_SIGNALS:
        if signal in msg_lower:
            return "product_inquiry"

    # Default: treat as product inquiry so the agent tries to help
    return "product_inquiry"
