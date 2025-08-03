import gradio as gr
from agent import agent_response, show_other_responses

# Persistent chat history across turns
chat_history = []

def chat(message, history):
    global chat_history
    if message.strip().lower() == "show more" and chat_history:
        # Respond with other model outputs
        response = show_other_responses(chat_history[-1])
    else:
        response = agent_response(message)
        chat_history.append(message)

    history = history or []
    history.append((message, response))
    return "", history

# Build Gradio UI
with gr.Blocks(title="Simple Research Assistant") as demo:
    gr.Markdown("### 🤖 Simple Research Assistant")
    gr.Markdown("""
    This agent can:
    - 📄 Summarize text
    - 💬 Query multiple LLMs (Ollama) Eg: Phi3, tinyllama
    - 🧠 Rank the most accurate response

    Try:
    - "Search AI in agriculture"
    - "Summarize: Deep learning is changing..."
    - "Question: What is reinforcement learning?"
    """)

    chatbot = gr.Chatbot(label="💬 Conversation", render_markdown=True)
    msg = gr.Textbox(placeholder="Ask your question here...", scale=8)
    clear = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
