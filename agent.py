import ollama
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Global memory
chat_history = []
last_ranked_responses = []

# Lightweight models
MODEL_LIST = ["tinyllama", "phi3"]

# Query local LLMs in parallel
def query_llms(prompt, models=MODEL_LIST, max_tokens=100):
    def call_model(model):
        try:
            result = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": max_tokens}
            )
            return (model, result["message"]["content"])
        except Exception as e:
            return (model, f"[Error from {model}]: {str(e)}")

    with ThreadPoolExecutor() as executor:
        return list(executor.map(call_model, models))

# Rank LLM responses using cosine similarity
def rank_responses(prompt, responses):
    query_embedding = embedder.encode(prompt, convert_to_tensor=True)
    scored = []
    for model, resp in responses:
        resp_embedding = embedder.encode(resp, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, resp_embedding).item()
        scored.append((model, resp, round(score, 4)))
    return sorted(scored, key=lambda x: x[2], reverse=True)

# Rewrite prompt using phi3
def rewrite_prompt(prompt):
    try:
        res = ollama.chat(model="phi3", messages=[{
            "role": "user", "content": f"Rewrite this prompt for clarity:\n{prompt}"
        }], options={"num_predict": 100})
        return res["message"]["content"].strip()
    except:
        return prompt

# Detect subtasks if needed
def detect_and_plan_subtasks(prompt):
    if any(k in prompt.lower() for k in ["compare", "difference", "how to", "steps", "advantages", "benefits"]):
        try:
            res = ollama.chat(model="phi3", messages=[{
                "role": "user", "content": f"Break this into subtasks:\n{prompt}"
            }], options={"num_predict": 100})
            return [line.strip("-•* ") for line in res["message"]["content"].splitlines() if line.strip()]
        except:
            return [prompt]
    return [prompt]

# Evaluate response with LLM
def evaluate_with_llm(question, answer):
    prompt = f"Evaluate this answer:\n\nQ: {question}\nA: {answer}\n\nIs this accurate and complete?"
    res = ollama.chat(model="phi3", messages=[{"role": "user", "content": prompt}], options={"num_predict": 80})
    return res["message"]["content"].strip()

# Improve the response
def improve_response(question, answer):
    prompt = f"Improve this answer:\n\nQ: {question}\nA: {answer}"
    res = ollama.chat(model="phi3", messages=[{"role": "user", "content": prompt}], options={"num_predict": 100})
    return res["message"]["content"].strip()

# Main agent logic
def agent_response(user_input):
    global chat_history, last_ranked_responses

    try:
        chat_history.append(("user", user_input))

        rewritten = rewrite_prompt(user_input)
        subtasks = detect_and_plan_subtasks(rewritten)
        final_output = ""
        all_ranked = []

        for i, task in enumerate(subtasks, 1):
            responses = query_llms(task)
            ranked = rank_responses(task, responses)
            all_ranked.extend(ranked)

            top_model, top_resp, _ = ranked[0]
            critique = evaluate_with_llm(task, top_resp)

            if any(w in critique.lower() for w in ["improve", "incomplete", "partial", "unclear"]):
                top_resp = improve_response(task, top_resp)
                top_resp += f"\n\n📝 *Improved after critique:* {critique}"

            final_output += f"### 🔹 Task {i}: `{task}`\n**Best Model:** `{top_model}`\n\n{top_resp}\n\n"

        last_ranked_responses = all_ranked
        final_output += "\n🤔 Want to see more responses? Type `show more`."
        chat_history.append(("assistant", final_output))
        return final_output

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Show other responses — now safely accepts a dummy argument
def show_other_responses(_=None):
    if not last_ranked_responses:
        return "⚠️ No previous results available."

    top_model = last_ranked_responses[0][0]
    filtered = [entry for entry in last_ranked_responses if entry[0] != top_model]

    if not filtered:
        return "✅ All responses already shown."

    result = "**📚 Other Model Responses:**\n\n"
    for i, (model, resp, score) in enumerate(filtered, 1):
        result += f"### {i}. `{model}` (Score: {score})\n{resp}\n\n"

    return result
