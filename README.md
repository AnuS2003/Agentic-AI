# 🧠 Agentic AI Research Assistant

A lightweight, fast, and local agentic AI that can:
- Rewrite prompts for better understanding
- Plan subtasks when queries are complex
- Query multiple local LLMs (like `tinyllama`, `phi3`)
- Rank and evaluate responses using semantic similarity
- Self-improve based on critique
- Show alternative answers on demand

Built with:
- [Ollama](https://ollama.com/) for running local LLMs
- `sentence-transformers` for semantic ranking
- `concurrent.futures` for fast parallel model querying

---

## 🚀 Features

| Feature         | Description |
|----------------|-------------|
| 🔁 Prompt Rewriting | Improves vague or unclear input |
| 🔍 Subtask Planning | Breaks down complex queries |
| 🤖 Multi-LLM Querying | Runs queries through multiple local models |
| 🧠 Semantic Ranking | Uses embeddings to rank responses |
| 📋 Critique + Improve | LLM evaluates and improves its own response |
| 📚 Show More | Allows viewing responses from other models |

---

## 🧰 Requirements

- Python 3.8+
- [`Ollama`](https://ollama.com/) installed and running
- Required Python packages:
  
## Workflow
  [USER] --> Rewrite --> Plan Subtasks --> Query LLMs --> Rank --> Critique --> Improve or [Show more] --> [Final Answer]


