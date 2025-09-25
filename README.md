# Ex. No. 6 – Development of Python Code Compatible with Multiple AI Tools

**NAME:** Dhayananth.P.S  
**REGISTER NUMBER:** 212223040039

---

## Aim
Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights.

---

## AI Tools Required
- Any two (or more) LLM providers / AI services. Example providers used in the code:
  - OpenAI (ChatGPT / GPT models)
  - Anthropic (Claude) OR Cohere OR any other provider that exposes an HTTP API
- Python 3.8+
- Optional: sentence-transformers for semantic similarity (recommended but optional)

---

## Explanation
This experiment demonstrates the "persona pattern" where prompts are issued as a specific persona (here: *programmer persona*) and the outputs from multiple AI tools are collected, compared, and summarized. The Python code provides a provider-agnostic adapter interface, collects responses, computes similarity and difference metrics, and writes a small comparative report (JSON + human-readable summary).

---

## Design and Procedure

1. Define the persona and the prompt(s) for a programming task (for example: "Implement a function to merge two sorted lists in Python — persona: experienced Python developer, concise code with explanation and complexity analysis").
2. Submit the same prompt to multiple AI providers via adapters.
3. Collect outputs and compute comparison metrics:
   - Text similarity (difflib ratio)
   - Jaccard token overlap
   - Optionally, embedding cosine similarity (if `sentence-transformers` is available)
4. Generate actionable insights:
   - Which provider gave the most concise solution?
   - Which provider included complexity analysis?
   - Where outputs disagree (bugs, omissions)?
5. Save results to `results.json` and display a human-readable summary in console.

---

## Code
Save the following as `multi_ai_compare.py`.

```python
"""
multi_ai_compare.py

Example multi-AI integration & comparison framework.

- Provides a ProviderAdapter interface
- Implements an OpenAI adapter (example)
- Shows how to plug in a second generic HTTP adapter (e.g., Anthropic/Cohere)
- Compares outputs using difflib ratio, Jaccard similarity, and (optional) embedding cosine
- Writes results to results.json and prints a short summary

Requirements:
    pip install openai requests numpy scikit-learn sentence-transformers

Notes:
- Set environment variables for API keys:
    OPENAI_API_KEY, ANTHROPIC_API_KEY (or other provider key)
- You can disable embedding similarity by not installing sentence-transformers.
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

import requests
import difflib
import math

# Optional imports for embeddings
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDING_AVAILABLE = True
except Exception:
    EMBEDDING_AVAILABLE = False

# ----------------------------
# Adapter Interface
# ----------------------------
class ProviderAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Send the prompt to the provider and return a standardized dict:
        {
            "provider": "openai",
            "model": "gpt-4o",
            "response": "the text output",
            "raw": {...}          # provider raw response if useful
        }
        """
        pass

# ----------------------------
# OpenAI Adapter (example)
# ----------------------------
class OpenAIAdapter(ProviderAdapter):
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package required. Install with `pip install openai`.")
        self.openai = openai
        self.openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, **kwargs):
        resp = self.openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        # Extract a clean text
        text = resp.choices[0].message.content.strip()
        return {
            "provider": "openai",
            "model": self.model,
            "response": text,
            "raw": resp
        }

# ----------------------------
# Generic HTTP Adapter (example placeholder)
# Replace with real endpoints & parameters for Anthropic/Cohere/etc.
# ----------------------------
class GenericHTTPAdapter(ProviderAdapter):
    def __init__(self, name: str, endpoint: str, api_key_env: str, extra_headers: Dict[str,str]=None):
        self.name = name
        self.endpoint = endpoint
        self.api_key = os.getenv(api_key_env)
        self.extra_headers = extra_headers or {}

    def generate(self, prompt: str, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers
        }
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.2),
        }
        r = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        j = r.json()
        # Adjust this depending on provider JSON structure:
        text = j.get("text") or j.get("completion") or j.get("response") or json.dumps(j)
        return {
            "provider": self.name,
            "model": j.get("model", "unknown"),
            "response": text.strip() if isinstance(text, str) else str(text),
            "raw": j
        }

# ----------------------------
# Comparison Utilities
# ----------------------------
def difflib_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def jaccard_token_similarity(a: str, b: str) -> float:
    a_tokens = set([t.lower() for t in a.split()])
    b_tokens = set([t.lower() for t in b.split()])
    if not a_tokens and not b_tokens:
        return 1.0
    intersection = a_tokens.intersection(b_tokens)
    union = a_tokens.union(b_tokens)
    return len(intersection) / len(union) if union else 0.0

def embedding_cosine_similarity(a: str, b: str) -> float:
    if not EMBEDDING_AVAILABLE:
        return None
    emb_a = EMBEDDING_MODEL.encode(a, convert_to_numpy=True).reshape(1, -1)
    emb_b = EMBEDDING_MODEL.encode(b, convert_to_numpy=True).reshape(1, -1)
    return float(cosine_similarity(emb_a, emb_b)[0][0])

def summarize_text_lengths(text: str) -> Dict[str,int]:
    return {"chars": len(text), "words": len(text.split())}

# ----------------------------
# Runner & Report Generator
# ----------------------------
def compare_responses(responses: List[Dict[str,Any]]) -> Dict[str,Any]:
    # responses: list of {"provider","model","response",...}
    results = []
    n = len(responses)
    for i in range(n):
        for j in range(i+1, n):
            a = responses[i]
            b = responses[j]
            ratio = difflib_ratio(a["response"], b["response"])
            jacc = jaccard_token_similarity(a["response"], b["response"])
            emb_cos = embedding_cosine_similarity(a["response"], b["response"]) if EMBEDDING_AVAILABLE else None
            results.append({
                "pair": (a["provider"], b["provider"]),
                "model_pair": (a.get("model"), b.get("model")),
                "difflib_ratio": ratio,
                "jaccard": jacc,
                "embedding_cosine": emb_cos,
                "len_a": summarize_text_lengths(a["response"]),
                "len_b": summarize_text_lengths(b["response"]),
            })
    return {"comparisons": results}

def run_experiment(prompt: str, adapters: List[ProviderAdapter], persona: str = None, **kwargs):
    if persona:
        prompt_with_persona = f"Respond as a {persona}.\n\n{prompt}"
    else:
        prompt_with_persona = prompt

    responses = []
    for adapter in adapters:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Querying {adapter.__class__.__name__} ({getattr(adapter,'name',getattr(adapter,'model',adapter.__class__.__name__))})...")
            res = adapter.generate(prompt_with_persona, **kwargs)
            responses.append(res)
            print(f"  -> {res['provider']} returned {len(res['response'])} chars.")
        except Exception as e:
            print(f"  !! Error querying {adapter}: {e}")
            responses.append({
                "provider": getattr(adapter, "name", "unknown"),
                "model": getattr(adapter, "model", "unknown"),
                "response": f"__error__ {e}",
                "raw": {}
            })
    comparison = compare_responses(responses)
    report = {
        "prompt": prompt,
        "prompt_with_persona": prompt_with_persona,
        "responses": responses,
        "comparison": comparison,
        "timestamp": time.time()
    }
    # Save report
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report

# ----------------------------
# Example usage (main)
# ----------------------------
if __name__ == "__main__":
    # Example persona and prompt
    persona = "experienced Python developer who writes concise, production-ready code and explains time/space complexity"
    prompt = (
        "Implement a Python function `merge_sorted_lists(a, b)` that merges two sorted lists of integers "
        "and returns a new sorted list. Provide well-documented code, short unit test examples, and time/space complexity analysis."
    )

    # Initialize adapters
    adapters = []
    # OpenAI adapter (requires OPENAI_API_KEY in env)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            adapters.append(OpenAIAdapter(api_key=openai_key, model=os.getenv("OPENAI_MODEL", "gpt-4o")))
        except Exception as e:
            print("OpenAI adapter could not be initialized:", e)
    else:
        print("OPENAI_API_KEY not set; skipping OpenAI adapter.")

    # Generic HTTP adapter example (set ANOTHER_API_ENDPOINT and ANOTHER_API_KEY env variables if available)
    another_endpoint = os.getenv("ANOTHER_API_ENDPOINT")
    if another_endpoint and os.getenv("ANOTHER_API_KEY"):
        adapters.append(GenericHTTPAdapter(name="other_provider", endpoint=another_endpoint, api_key_env="ANOTHER_API_KEY"))
    else:
        print("Other provider endpoint/key not set; skipping generic adapter.")

    if not adapters:
        print("No adapters configured. Please set OPENAI_API_KEY or ANOTHER_API_ENDPOINT + ANOTHER_API_KEY environment variables.")
        exit(1)

    report = run_experiment(prompt, adapters, persona=persona, max_tokens=512, temperature=0.1)

    # Print summary
    print("\n=== SUMMARY ===")
    for r in report["responses"]:
        print(f"{r['provider']} ({r.get('model')}): {summarize_text_lengths(r['response'])['words']} words")
    print("\nComparisons:")
    for c in report["comparison"]["comparisons"]:
        print(f"{c['pair'][0]} vs {c['pair'][1]}: difflib={c['difflib_ratio']:.3f}, jaccard={c['jaccard']:.3f}, emb_cos={c['embedding_cosine']}")
    print("\nFull report saved to results.json")
