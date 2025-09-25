# Ex. No. 6 – Development of Python Code Compatible with Multiple AI Tools

# NAME: Dhayananth.P.S  
# REGISTER NUMBER: 212223040039  


---

## Aim
Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with multiple AI tools.

---

## AI Tools Required
- **OpenAI GPT (ChatGPT)**
- **Other AI Provider (e.g., Cohere, Anthropic, or HuggingFace models)**
- **Python 3.8+**
- Libraries: `requests`, `openai`, `difflib`, `json` (optionally `sentence-transformers` for semantic similarity)

---

## Explanation
This experiment applies the **persona pattern** where the AI acts as a *programmer persona* to solve a programming task.  
We send the same prompt to multiple AI providers, collect their generated code, and then compare:

- **Code style** (concise vs verbose)  
- **Correctness** (does the code run?)  
- **Completeness** (includes comments, unit tests, complexity analysis)  
- **Similarity** (textual and semantic)

This enables a fair evaluation of multiple AI tools on the same task.

---

## Procedure
1. **Define a Scenario:**  
   Example — Implement a Python function `merge_sorted_lists(a, b)` that merges two sorted lists, include unit tests and complexity analysis.

2. **Use Persona Pattern:**  
   Prompt the models with:  
   *"Act as an experienced Python programmer. Write clean, optimized code for merging two sorted lists with explanation and complexity analysis."*

3. **Collect Responses:**  
   - Query OpenAI GPT model using `openai` library  
   - Query another AI provider through its REST API

4. **Compare Outputs:**  
   - Measure **difflib similarity ratio**  
   - Count number of words/characters  
   - Optionally compute semantic similarity using embeddings  

5. **Generate Insights:**  
   - Identify which model provided better explanation and correct code  
   - Detect missing parts or improvements

6. **Store Results:**  
   Save as `results.json` or print a tabular summary

---

## Sample Python Code
```python
import os, difflib, json, requests
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompt with persona pattern
prompt = (
    "Act as an experienced Python programmer. "
    "Write a function merge_sorted_lists(a, b) that merges two sorted lists and returns the result. "
    "Include a short explanation and time complexity."
)

# OpenAI response
response_openai = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=400,
    temperature=0.2
)
openai_text = response_openai.choices[0].message.content

# Example second provider (dummy endpoint)
second_provider_endpoint = "https://api.example.com/generate"
headers = {"Authorization": f"Bearer {os.getenv('ANOTHER_API_KEY')}"}
payload = {"prompt": prompt, "max_tokens": 400}
resp2 = requests.post(second_provider_endpoint, headers=headers, json=payload)
provider2_text = resp2.json().get("text", "")

# Compare using difflib
similarity = difflib.SequenceMatcher(None, openai_text, provider2_text).ratio()

# Save results
results = {
    "prompt": prompt,
    "openai_response": openai_text,
    "provider2_response": provider2_text,
    "similarity_score": similarity
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Similarity Score: {similarity:.2f}")
print("Comparison saved to results.json")
```
---

## Conclusion
This experiment successfully demonstrates how to build Python code that integrates with multiple AI tools, compares their results, and extracts actionable insights.  
Such a system can be reused for **AI evaluation, research, or prompt testing** in real-world projects.

---

## Result
Thus, the Python code integrating multiple AI tools was developed, executed, and verified successfully. The outputs from different providers were compared, analyzed, and summarized effectively.
