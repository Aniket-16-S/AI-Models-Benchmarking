# LLM Benchmarking & Evaluation Suite

A comprehensive framework and benchmark suite evaluating Large Language Models (LLMs) across prompt engineering techniques, model performance, guardrails, reflection loops, cost optimization, RAG, and agentic workflows.

---

## Task 1: LLM Evaluation Framework

### Objective
Build an automated evaluation pipeline for an LLM application (Meeting Transcript Extractor) to determine its actual performance and quality against ground truth.

### Key Findings & Analysis
Built an automated, programmatic evaluation pipeline evaluated against a dataset of 65 test cases. The pipeline runs the model across all cases, evaluates outputs against expected ground truth, and computes mathematical quality metrics.

### Calculated Metrics
| Metric | Value |
| :--- | :--- |
| **Accuracy** | 74.5% |
| **Precision** | 0.6613 |
| **Recall** | 0.7962 |
| **F1 Score** | 0.7225 |

### Failure Analysis
While the pipeline reliably parsed structured JSON, the ~27% error margin in F1 score was driven by semantic nuances:
* **False Positives:** The model frequently extracted implied tasks that were technically valid in business context but absent from rigid ground-truth labels.
* **False Negatives:** Deeply nested dependencies across long transcripts caused the model to miss implicit deadlines.

---

## Task 2: Prompt Engineering Case Study

### Objective
Design and compare multiple prompting strategies for structured output on a business problem (Meeting Action Item Extractor).

### Key Findings & Analysis
Evaluated five distinct prompting strategies across Accuracy (F1), Latency, Total Cost, and Token Usage over 65 test runs.

### Results Comparison
| Prompt Strategy | Accuracy (F1) | Avg Latency | Total Cost (65 runs) | Avg Token Usage (In + Out) |
| :--- | :--- | :--- | :--- | :--- |
| **Zero-Shot** | 0.4992 | 637 ms | $0.0172 | 840 tokens |
| **Few-Shot** | **0.7225** | 660 ms | $0.0209 | 1,450 tokens |
| **Chain-of-Thought (CoT)** | 0.5080 | 898 ms | $0.0181 | 1,120 tokens |
| **JSON Structured** | 0.5110 | 650 ms | $0.0175 | 860 tokens |
| **XML** | 0.4178 | 840 ms | **$0.0090** | 790 tokens |

> **Conclusion:** **Few-Shot prompting** yields the best overall performance. Although it increases token consumption and cost marginally, in-context demonstrations are necessary to push the F1 score past 0.70 for complex schema extraction.

---

## Task 3: Model Comparison Matrix

### Objective
Benchmark 5 models using 100 test cases (extrapolated) across Cost, Hallucination Rate, Speed, Accuracy, JSON reliability, and Context handling under load.

### Comparison Matrix
| Model | Accuracy (F1) | JSON Reliability | Speed / Latency | Est. Cost / 1M Tokens | Hallucination Rate | Context Handling |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Claude Sonnet 3.5** | **0.9120** | **100.0%** | 980 ms | $3.00 | 2.1% | Excellent |
| **GPT-4.1 (4o)** | 0.8950 | **100.0%** | 1050 ms | $5.00 | 3.5% | Excellent |
| **Gemini 2.5 Flash** | 0.8140 | 99.5% | **420 ms** | $0.15 | 8.0% | Good |
| **DeepSeek V3** | 0.7010 | 95.0% | 1500 ms | **$0.14** | 14.5% | Fair |
| **Qwen 2.5 32B** | 0.0390 | 3.08% | 2875 ms | $0.13 | **1.1%** | Failed |

### Key Takeaways
* **Claude Sonnet 3.5:** Strongest overall model for complex schema adherence, yielding the highest F1 score and zero formatting failures.
* **Gemini 2.5 Flash:** Best ROI and throughput (420 ms latency) while maintaining 99.5% JSON validity.
* **Qwen 2.5 32B:** Struggled heavily with padded context windows and structured output constraints, though hallucination rates remained low when it did format properly.

---

## Task 4: Guardrails Implementation

### Objective
Implement and benchmark security intercepts for Prompt Injections, Jailbreaks, Toxicity, and PII/Sensitive data leakage across 100 adversarial test cases.

### Implementation
Built a layered interceptor architecture combining deterministic regular expressions (for structured PII and sensitive keywords) with an LLM-as-a-Classifier security layer.

### Findings
| Threat Category | Test Cases | Block Effectiveness | Gaps / Bypasses |
| :--- | :--- | :--- | :--- |
| **Prompt Injection** | 25 | 100.0% | 0 |
| **Jailbreak Detection** | 25 | 100.0% | 0 |
| **Toxicity Detection** | 25 | 100.0% | 0 |
| **PII & Sensitive Info** | 25 | 100.0% | 0 |

> **Trade-off:** The guardrail pipeline had a 0% bypass rate on this test set. However, running an LLM-based classification pass introduces an additional 1–3 seconds of latency per turn.

---

## Task 5: Reflection Agent

### Objective
Implement an iterative refinement loop (`Writer` → `Reviewer` → `Critic` → `Writer` → `Final Output`) and compare output quality against single-pass generation.

### Findings & Analysis
* **No Reflection (Single-pass):** Drafts frequently contained unresolved placeholders (e.g., `[Insert Date]`) and weak structure.  
  *Quality Score: 72%*
* **Reflection-Based (Iterative):** The `Reviewer`/`Critic` validation loop eliminated 100% of unresolved template tags and normalized tone before emission.  
  *Quality Score: 94%*

---

## Task 6: AI Cost Optimizer (Dynamic Routing)

### Objective
Deploy an intelligent request router that directs prompts to small or frontier models based on task complexity, tracking total monthly cost savings.

### Financial Breakdown
| Strategy / Item | Cost (USD) | Converted (INR) |
| :--- | :--- | :--- |
| **Gemini Flash (per 1M tokens)** | $0.15 | ₹12.56 |
| **Claude Sonnet (per 1M tokens)** | $3.00 | ₹251.10 |
| **GPT-4.1 (per 1M tokens)** | $5.00 | ₹418.50 |
| **Monolithic Strategy (Frontier only, monthly)** | $49.80 | ₹4,168.26 |
| **Optimized Routing (Hybrid, monthly)** | **$12.96** | **₹1,084.75** |
| **Net Monthly Savings** | **$36.84** | **₹3,083.51** |

> **Impact:** Dynamic model routing achieved a **73.9% net cost reduction** (saving ₹3,083.51 per 10k queries) with no measurable quality drop on downstream tasks.

---

## Task 7: RAG Evaluation

### Objective
Benchmark chunking strategies, embedding models, Top-K retrieval, and hybrid retrieval across 100 documents.

### Retrieval Performance Matrix
| Configuration | Recall@5 | Precision@5 | Latency |
| :--- | :--- | :--- | :--- |
| **Chunk 200 (Dense)** | 74.0% | 22.1% | 13.2 ms |
| **Chunk 500 (Dense)** | 29.0% | 12.0% | 19.4 ms |
| **Chunk 1000 (Dense)** | 37.0% | 15.5% | 15.5 ms |
| **Chunk 500 (Hybrid: Dense + BM25)** | **92.0%** | **45.0%** | 18.6 ms |

### Key Observations
* **Embeddings:** `text-embedding-3-small` outperformed `all-MiniLM-L6-v2` by 14% on semantic retrieval accuracy.
* **Top-K Thresholds:** Increasing retrieval from Top-3 to Top-5 improved Recall from 81% to 92% on transcripts with distributed conversational context.
* **Hybrid Search:** Combining dense embeddings with sparse BM25 retrieval resolved the semantic dilution problem seen in larger 500-token chunks.

---

## Task 8: Agent Memory Architecture

### Objective
Evaluate short-term context buffers, rolling summarization, and memory pruning over a multi-turn (20-turn) conversation.

### Implementation Details
* **Short-Term Context Window:** Retains the exact role and message body of the last 6 conversational turns.
* **Pruning & Summarization:** When context hits the 6-turn ceiling, the oldest 4 messages are pruned and condensed by a fast summarization model. This rolling summary is injected directly into the system instructions.
* **Validation:** In a 20-turn benchmark, the agent accurately recalled context established in Turn 2 during Turn 19, maintaining cross-turn continuity without token bloat.

---

## Task 9: Tool Selection Benchmark

### Objective
Build and benchmark a routing agent tasked with dispatching user intent to the correct API tool from a suite of 10 available tools (`Weather`, `Maps`, `Calculator`, `Database`, `Email`, `Calendar`, `Search`, `CRM`, `Slack`, `GitHub`).

### Benchmark Results
* **Test Set:** 50 multi-domain user queries.
* **Selection Accuracy:** **96.0%** (48/50 correct routes).
* Successfully disambiguated subtle boundary overlap (e.g., querying Slack for internal peer discussions vs. CRM for external account records).

---

## Task 10: Hallucination Detection (LLM-as-a-Judge)

### Objective
Implement an automated judge pipeline to detect unsupported claims, fabricated citations, and factual drifts against reference ground truth.

### Detection Focus
1. **Factual Errors:** Drift in numeric or entity data (e.g., modified budget numbers).
2. **Unsupported Claims:** Hallucinated actions or assignments not present in the transcript.
3. **Fabricated Citations:** Hallucinated file paths, IDs, or timestamps.

### Hallucination Rates by Strategy (Llama 3.3 70B)
| Prompt Strategy | Hallucination Rate |
| :--- | :--- |
| **Zero-Shot** | 47.43% |
| **Few-Shot** | 29.07% |
| **Chain-of-Thought (CoT)** | **22.93%** |

> **Impact:** Enforcing structured Chain-of-Thought reasoning prior to extraction halved the rate of hallucinated details and phantom citations compared to Zero-Shot execution.
