"""
agent/rag_pipeline.py
RAG (Retrieval-Augmented Generation) pipeline for AutoStream knowledge base.
Loads the JSON knowledge base and builds a searchable context string for the LLM.
"""

import json
import os
from pathlib import Path


def load_knowledge_base() -> dict:
    """Load the AutoStream knowledge base from JSON."""
    kb_path = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"
    with open(kb_path, "r") as f:
        return json.load(f)


def build_kb_context(kb: dict) -> str:
    """
    Convert the structured knowledge base into a readable context string
    that the LLM can use to answer questions accurately.
    """
    lines = []

    # Company overview
    company = kb["company"]
    lines.append(f"## Company: {company['name']}")
    lines.append(f"Tagline: {company['tagline']}")
    lines.append(f"Description: {company['description']}")
    lines.append("")

    # Pricing plans
    lines.append("## Pricing Plans")
    for plan in kb["plans"]:
        lines.append(f"\n### {plan['name']} — ${plan['price_monthly']}/month")
        lines.append(f"Best for: {plan['best_for']}")
        lines.append("Features:")
        for feature in plan["features"]:
            lines.append(f"  - {feature}")

    # Policies
    lines.append("\n## Policies")
    for policy in kb["policies"]:
        lines.append(f"\n**{policy['topic']}**: {policy['details']}")

    # FAQs
    lines.append("\n## FAQs")
    for faq in kb["faqs"]:
        lines.append(f"\nQ: {faq['question']}")
        lines.append(f"A: {faq['answer']}")

    return "\n".join(lines)


# Singleton: load once, reuse across the session
_KB = load_knowledge_base()
KB_CONTEXT = build_kb_context(_KB)


def get_kb_context() -> str:
    """Return the pre-built knowledge base context string."""
    return KB_CONTEXT
