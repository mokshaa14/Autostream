#  AutoStream Conversational AI Agent

##  How to Run the Project Locally

```bash
git clone https://github.com/mokshaa14/Autostream.git
cd inflx-autostream

python -m venv venv
.\venv\Scripts\Activate.ps1   # For Windows

pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

Run the agent:

```bash
python main.py
```

---

## 🧠 Architecture Explanation

This project uses LangGraph to build a structured, stateful conversational AI agent. LangGraph was chosen over simpler frameworks because it enables clear workflow orchestration using nodes and conditional routing, making it ideal for multi-step interactions like intent detection, knowledge retrieval, and lead collection.

The agent maintains a shared state (`AgentState`) that persists across all conversation turns. This state includes message history, detected user intent, and lead collection progress (name, email, platform). Each user input is appended to the state and passed through a graph of nodes such as intent detection, RAG-based response generation, and lead handling.

To ensure efficiency and relevance, the agent uses a sliding memory window of the last 5–6 conversation turns when interacting with the LLM, while still maintaining the full history internally. This balances contextual understanding with performance.

Additionally, a RAG pipeline injects knowledge base content into each response, ensuring factual accuracy and preventing hallucinations.

---

## 📲 WhatsApp Integration (Webhook Design)

To integrate this agent with WhatsApp, we would use the WhatsApp Business API with a webhook-based architecture.

1. A backend server (built using Flask or FastAPI) exposes a webhook endpoint to receive incoming messages from WhatsApp.
2. When a user sends a message, WhatsApp forwards it to the webhook.
3. The server extracts the message and user ID, and passes it to the `AutoStreamAgent.chat()` function.
4. The agent processes the message using LangGraph and returns a response.
5. The server sends this response back to the user via the WhatsApp API.

Each user is associated with a unique session, allowing the agent to maintain stateful conversations across multiple messages. This enables seamless multi-turn interactions such as lead collection and follow-ups.

This architecture ensures scalability, real-time communication, and easy deployment in production environments.
