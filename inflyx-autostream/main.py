"""
main.py
Entry point for the AutoStream AI Agent (CLI mode).
Run: python main.py
"""

import os
from dotenv import load_dotenv
from agent.autostream_agent import AutoStreamAgent

load_dotenv()  # Loads ANTHROPIC_API_KEY from .env if present


BANNER = """
╔══════════════════════════════════════════════════════════╗
║          AutoStream AI Agent  — Powered by Inflx         ║
║         Social-to-Lead Conversational AI System          ║
╚══════════════════════════════════════════════════════════╝
  Type your message and press Enter. Type 'quit' to exit.
  Type 'reset' to start a new conversation.
──────────────────────────────────────────────────────────
"""


def main():
    print(BANNER)
    agent = AutoStreamAgent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye! 👋")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("\n[Conversation reset — starting fresh]\n")
            continue

        response = agent.chat(user_input)
        print(f"\nAria (AutoStream): {response}\n")


if __name__ == "__main__":
    main()
