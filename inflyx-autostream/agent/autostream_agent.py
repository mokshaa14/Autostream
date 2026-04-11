"""
agent/autostream_agent.py

AutoStream Conversational AI Agent — built with LangGraph.
Implements: Intent Detection → RAG Knowledge Retrieval → Lead Capture Tool.

Architecture:
  - LangGraph StateGraph manages conversation state across all turns
  - Groq powers the LLM reasoning
  - RAG pipeline injects knowledge base context into every LLM call
  - Intent detector routes the flow between greeting / inquiry / lead nodes
"""

import os
import re
from typing import Optional
from typing_extensions import TypedDict

from groq import Groq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.rag_pipeline import get_kb_context
from agent.intent_detector import classify_intent
from tools.lead_capture import mock_lead_capture


# ══════════════════════════════════════════════════════════════════════════════
# State Schema
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    Full conversation state persisted across every graph node invocation.
    LangGraph passes this dict between nodes and preserves it across turns.
    """
    messages: list                  # Full chat history (Human + AI messages)
    intent: str                     # Current classified intent
    collecting_lead: bool           # True when we're in lead-collection mode
    lead_name: Optional[str]        # Collected prospect name
    lead_email: Optional[str]       # Collected prospect email
    lead_platform: Optional[str]    # Collected creator platform
    lead_captured: bool             # True once mock_lead_capture() has fired
    turn_count: int                 # Number of conversation turns


# ══════════════════════════════════════════════════════════════════════════════
# LLM Setup  —  GROQ
# ══════════════════════════════════════════════════════════════════════════════

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)

    class GroqLLM:
        def invoke(self, messages):
            formatted = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    formatted.append({"role": "system", "content": m.content})
                elif isinstance(m, HumanMessage):
                    formatted.append({"role": "user", "content": m.content})
                elif isinstance(m, AIMessage):
                    formatted.append({"role": "assistant", "content": m.content})

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=formatted,
                max_tokens=512,
                temperature=0.4
            )

            return type("obj", (object,), {
                "content": response.choices[0].message.content
            })

    return GroqLLM()

# ══════════════════════════════════════════════════════════════════════════════
# System Prompt Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt() -> str:
    kb = get_kb_context()
    return f"""You are Aria, the friendly and knowledgeable sales assistant for AutoStream — an AI-powered video editing SaaS platform for content creators.

Your job is to:
1. Answer questions accurately using ONLY the knowledge base below
2. Identify when users are ready to sign up and smoothly collect their details
3. Be concise, warm, and enthusiastic — like a helpful human sales rep

== AUTOSTREAM KNOWLEDGE BASE ==
{kb}
== END OF KNOWLEDGE BASE ==

Rules:
- Never make up features, prices, or policies not in the knowledge base
- If you don't know something, say "I'll check with our team and get back to you"
- Keep responses under 120 words unless the user asks for detail
- When collecting lead info, ask for ONE piece of information at a time
- Never ask for name, email, and platform all in the same message
"""


# ══════════════════════════════════════════════════════════════════════════════
# Helper: Extract info from user message
# ══════════════════════════════════════════════════════════════════════════════

def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None


def extract_platform(text: str) -> Optional[str]:
    platforms = ["youtube", "instagram", "tiktok", "twitter", "facebook",
                 "twitch", "linkedin", "snapchat", "pinterest", "x"]
    lower = text.lower()
    for p in platforms:
        if p in lower:
            return p.capitalize()
    # If user typed something else (e.g., "my own website"), capture it
    if len(text.strip()) > 2:
        return text.strip().title()
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Graph Nodes
# ══════════════════════════════════════════════════════════════════════════════

def detect_intent_node(state: AgentState) -> AgentState:
    """Classify the latest user message and update state."""
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    # If already collecting lead info, preserve that context
    if state["collecting_lead"] and not state["lead_captured"]:
        state["intent"] = "high_intent"
    else:
        state["intent"] = classify_intent(last_human)
    return state


def greeting_node(state: AgentState) -> AgentState:
    """Handle casual greetings."""
    llm = get_llm()
    system = SystemMessage(content=build_system_prompt())
    hint = HumanMessage(content="(system hint: this is a greeting — respond warmly and briefly, then invite them to ask about AutoStream)")
    messages = [system] + state["messages"] + [hint]
    response = llm.invoke(messages)
    state["messages"].append(AIMessage(content=response.content))
    state["turn_count"] += 1
    return state


def rag_inquiry_node(state: AgentState) -> AgentState:
    """Handle product/pricing inquiries using RAG context."""
    llm = get_llm()
    system = SystemMessage(content=build_system_prompt())
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    state["messages"].append(AIMessage(content=response.content))
    state["turn_count"] += 1
    return state


def lead_collection_node(state: AgentState) -> AgentState:
    """
    Progressively collect lead info: name → email → platform.
    Calls mock_lead_capture() only when all three are present.
    """
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    # ── Step 1: Collect Name ──────────────────────────────────────────────────
    if not state["lead_name"]:
        prev_ai = next(
            (m.content for m in reversed(state["messages"][:-1]) if isinstance(m, AIMessage)),
            ""
        )
        if "name" in prev_ai.lower() and len(last_human.strip().split()) <= 5:
            state["lead_name"] = last_human.strip().title()
            reply = f"Great to meet you, {state['lead_name']}! 🎉 What's your email address so we can set up your account?"
        else:
            state["collecting_lead"] = True
            reply = "Awesome, let's get you started! 🚀 First, what's your name?"

        state["messages"].append(AIMessage(content=reply))
        state["turn_count"] += 1
        return state

    # ── Step 2: Collect Email ─────────────────────────────────────────────────
    if not state["lead_email"]:
        email = extract_email(last_human)
        if email:
            state["lead_email"] = email
            reply = f"Got it — {email} ✅\nOne last thing: which creator platform are you primarily on? (e.g., YouTube, Instagram, TikTok)"
        else:
            reply = "I didn't catch a valid email there. Could you share your email address? (e.g., you@example.com)"

        state["messages"].append(AIMessage(content=reply))
        state["turn_count"] += 1
        return state

    # ── Step 3: Collect Platform ──────────────────────────────────────────────
    if not state["lead_platform"]:
        platform = extract_platform(last_human)
        if platform:
            state["lead_platform"] = platform
        else:
            state["messages"].append(AIMessage(content="Which platform do you create content on? (YouTube, Instagram, TikTok, etc.)"))
            state["turn_count"] += 1
            return state

    # ── Step 4: Fire the tool — all three collected ───────────────────────────
    if not state["lead_captured"]:
        result = mock_lead_capture(
            name=state["lead_name"],
            email=state["lead_email"],
            platform=state["lead_platform"]
        )
        state["lead_captured"] = True

        reply = (
            f"🎊 You're all set, {state['lead_name']}!\n\n"
            f"We've registered your AutoStream Pro trial and you'll receive a welcome email at **{state['lead_email']}** shortly.\n\n"
            f"As a {state['lead_platform']} creator, you're going to love the AI captions and 4K exports. "
            f"Is there anything else I can help you with before you dive in?"
        )
        state["messages"].append(AIMessage(content=reply))
        state["turn_count"] += 1

    return state


# ══════════════════════════════════════════════════════════════════════════════
# Routing Logic
# ══════════════════════════════════════════════════════════════════════════════

def route_intent(state: AgentState) -> str:
    """Route to the correct node based on classified intent."""
    if state["lead_captured"]:
        return "rag_inquiry"
    if state["collecting_lead"] or state["intent"] == "high_intent":
        return "lead_collection"
    if state["intent"] == "greeting":
        return "greeting"
    return "rag_inquiry"


# ══════════════════════════════════════════════════════════════════════════════
# Graph Construction
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("rag_inquiry", rag_inquiry_node)
    graph.add_node("lead_collection", lead_collection_node)

    graph.set_entry_point("detect_intent")

    graph.add_conditional_edges(
        "detect_intent",
        route_intent,
        {
            "greeting": "greeting",
            "rag_inquiry": "rag_inquiry",
            "lead_collection": "lead_collection"
        }
    )

    graph.add_edge("greeting", END)
    graph.add_edge("rag_inquiry", END)
    graph.add_edge("lead_collection", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Public API: AutoStreamAgent
# ══════════════════════════════════════════════════════════════════════════════

class AutoStreamAgent:
    """
    Stateful wrapper around the LangGraph compiled graph.
    Maintains full conversation state across all turns.
    """

    def __init__(self):
        self.graph = build_graph()
        self.state: AgentState = {
            "messages": [],
            "intent": "",
            "collecting_lead": False,
            "lead_name": None,
            "lead_email": None,
            "lead_platform": None,
            "lead_captured": False,
            "turn_count": 0
        }

    def chat(self, user_message: str) -> str:
        """Process one user turn and return the agent's reply."""
        self.state["messages"].append(HumanMessage(content=user_message))
        self.state = self.graph.invoke(self.state)
        for msg in reversed(self.state["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content
        return "I'm sorry, I encountered an issue. Please try again."

    def reset(self):
        """Reset the conversation (new session)."""
        self.__init__()
