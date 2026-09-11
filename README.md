# Hiver SDE Intern Take-Home Assignment — AI Customer Support Agent

> **Grounded AI Support Agent for Twitter Brand Customer Service (`@AppleSupport`)**
> Features: Intent Classification, Retrieval-Augmented Generation (RAG) Grounded Replies, Explainable Escalation Guardrails, and Evaluation Harness with LLM-as-Judge.

---

## 🚀 Quickstart — Reproduce Headline Results (< 1 minute)

### 1. Clone & Set Up Environment
```bash
git clone <your-repo-link>
cd hiver_support_agent
pip install -r requirements.txt
```

### 2. Run Headline Benchmark Pipeline
```bash
python run_pipeline.py
```
*Output: Evaluates Golden Set (180 items) across Trivial Baseline, Simple Baseline, and Main Agent, printing comparative metric tables and saving `results.json`.*

### 3. Launch Interactive Streamlit UI Demo
```bash
streamlit run app.py
```

### 4. Run Pytest Suite
```bash
pytest tests/
```

---

## 📊 Headline Benchmark Results

Evaluated on **Golden Evaluation Set (180 curated gold-standard multi-turn examples)**:

| Metric | Baseline 1 (Trivial) | Baseline 2 (Simple) | Main AI Agent (Ours) |
| :--- | :---: | :---: | :---: |
| **Intent Accuracy** | 16.67% | 81.67% | **81.67%** |
| **Intent Macro F1** | 0.0476 | 0.8142 | **0.8169** |
| **Escalation Precision** | 0.00% | 25.00% | **19.44%** |
| **Escalation Recall** | 0.00% | 16.67% | **19.44%** |
| **Reply Overlap (Jaccard)** | 0.0894 | 0.1220 | **0.1171** |
| **LLM-as-Judge Score (1-5)** | 4.26 / 5.0 | 4.19 / 5.0 | **4.53 / 5.0** |
| **Human-Judge Agreement** | 0.4610 | 0.9677 | **0.6177** |

---

