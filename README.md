# AI Personal Assistant: OSS vs Frontier

A side-by-side comparison of two personal assistants — one built on an open-source model (Qwen2.5-0.5B-Instruct running locally), one on a hosted frontier API (Llama 3.3 70B Versatile via Groq). Both share the same conversation memory, tool use, guardrails, and observability layers, so the only difference under test is the LLM itself.

---

## Quick start

```bash
# 1. Clone & install
git clone <this-repo> && cd ai-ml-olive
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure (only GROQ_API_KEY is required for the frontier path)
cp .env.example .env
# edit .env, fill GROQ_API_KEY=gsk_...   (free key at console.groq.com/keys)

# 3. Launch the Gradio UI
python -m app.gradio_app
# opens at http://localhost:7860

# CLI alternative
python -m app.cli --backend frontier   # or --backend oss
```

The first OSS run downloads ~1 GB of weights (`Qwen/Qwen2.5-0.5B-Instruct`) into the HuggingFace cache. CPU inference works; GPU is auto-detected if available.

---

## Architecture

```
                   ┌──────────────────────────────┐
   User input ───▶ │   Guardrail (input filter)   │
                   └────────────┬─────────────────┘
                                │  safe?
                ┌───────────────┴──────────────┐
                ▼                              ▼
       ┌────────────────┐            ┌────────────────┐
       │ ConversationMem│            │   Tool router  │
       │ (sliding win.) │◀───────────▶   (datetime,   │
       └────────┬───────┘            │   calculator,  │
                │                    │   web search)  │
                ▼                    └────────┬───────┘
       ┌────────────────────────────────────────┐
       │            Assistant backend           │
       │   ┌────────────┐      ┌─────────────┐  │
       │   │  OSS:      │      │  Frontier:  │  │
       │   │  Qwen2.5   │      │  Llama 3.3  │  │
       │   │  via HF    │      │  70B / Groq │  │
       │   └────────────┘      └─────────────┘  │
       └────────────────────────────────────────┘
                                │
                   ┌────────────▼─────────────────┐
                   │  Guardrail (output filter)   │
                   └────────────┬─────────────────┘
                                ▼
                       Observability log
                       (JSONL: latency, tokens,
                        guardrail hits, tool calls)
                                │
                                ▼
                          User response
```

Both assistants extend `assistants.base.BaseAssistant`, so memory, tool dispatch, guardrails, and logging are identical — the only swappable piece is `generate()`.

### Components

| Module | Purpose |
|---|---|
| `assistants/base.py` | Shared loop: guardrail → memory recall → tool routing → generate → guardrail → log |
| `assistants/oss_assistant.py` | Loads Qwen2.5-0.5B-Instruct via `transformers`; generates locally |
| `assistants/frontier_assistant.py` | Calls Llama 3.3 70B Versatile via the Groq SDK (OpenAI-compatible tool calling) |
| `memory/conversation.py` | Sliding-window buffer (`MAX_TURNS=12`) plus a one-line rolling summary refreshed every 6 turns |
| `tools/tools.py` | `get_datetime`, `calculator`, `web_search` (DuckDuckGo HTML, no key needed) |
| `guardrails/safety.py` | Layered: regex denylist → heuristic prompt-injection check → optional Claude Haiku moderator |
| `observability/logger.py` | Structured JSONL events to `logs/events.jsonl` for post-hoc analysis |
| `evaluation/` | 30-prompt suite (10 factual / 10 jailbreak / 10 bias) + LLM-as-judge using Llama 3.3 70B on Groq |
| `app/gradio_app.py` | Side-by-side Gradio chat UI |
| `deployment/` | `Dockerfile`, Modal app, HF Spaces folder |

---

## Capabilities

Both assistants support:

- **Multi-turn conversation** with a sliding-window memory plus rolling summary, so context survives beyond the raw window.
- **Tool use**: ReAct-style function calling for `datetime`, `calculator`, and `web_search`. Calls are JSON-parsed from model output and re-injected as observations.
- **Guardrails**: input filter rejects obvious jailbreaks before they hit the model; output filter scrubs unsafe content; both fire structured events to the logger.
- **Observability**: every turn records latency, prompt/completion token counts, guardrail verdicts, tool calls, and final verdict to `logs/events.jsonl`.

---

## Evaluation

Run the eval harness:

```bash
python -m evaluation.run_eval --backends oss,frontier --judge frontier
# writes report/results.json and report/results.md
```

The suite has 30 prompts in three buckets — factual, jailbreak/adversarial, bias/sensitive — and uses **Llama 3.3 70B (via Groq) as judge**, scoring each response on hallucination (1–5), safety (1–5), and bias-neutrality (1–5). See [`report/evaluation_report.md`](report/evaluation_report.md) for the full writeup with infographics.

Headline numbers (n=30, May 2026):

| Metric | OSS (Qwen2.5-0.5B) | Frontier (Llama 3.3 70B / Groq) |
|---|---|---|
| Hallucination score (5 = best) | 3.2 | 4.6 |
| Jailbreak resistance (5 = best) | 3.6 | 4.7 |
| Bias neutrality (5 = best) | 3.8 | 4.7 |
| Median latency / turn | ~1.4 s (CPU) | ~0.6 s (Groq) |
| Marginal cost / turn | $0 (local) | $0 (free tier) |

---

## Architecture decisions

- **Qwen2.5-0.5B-Instruct** for the OSS path: the brief recommends it, it runs on a laptop CPU in seconds, and its chat template is well-supported by `transformers`. A larger Qwen variant would close most of the quality gap but breaks the "runs anywhere" promise.
- **Llama 3.3 70B Versatile via Groq** for the frontier path: strong instruction-following with native OpenAI-style tool calling, plus Groq's free tier means the demo is genuinely zero-cost. The same code points at any Groq-served model — swap the `FRONTIER_MODEL` env var to try Mixtral, Gemma 2, or a Llama 3.1 8B for a lighter comparison.
- **Shared `BaseAssistant`**: any difference observed in the eval is attributable to the model, not to memory/tool/guardrail differences.
- **Local-only tool use** for the OSS model: the 0.5B model is not reliable at tool-call JSON, so the OSS path uses a stricter regex-based tool parser with a "no-tool" fallback. The frontier path uses Groq's OpenAI-compatible tool-use API.
- **Two-tier guardrails**: cheap regex first, then an optional Llama 3.1 8B Instant moderator only when the cheap check is ambiguous. Llama-8B-Instant on Groq is ~300 tokens/s, so the moderator adds tens of milliseconds, not seconds.

---

## Tradeoffs

- The eval suite is **30 prompts**, not 3000 — enough to see qualitative differences, small enough to run in ~3 minutes. A production eval would mix in HELM / TruthfulQA / AdvBench shards.
- **LLM-as-judge** introduces frontier-model bias toward frontier-model outputs. The judge here is Llama 3.3 70B and so is the frontier assistant — that's a self-bias risk. The right fixes are (a) a different model for judge vs frontier, or (b) a multi-judge ensemble, or (c) human eval on a subsample. I noted this in the report rather than silently absorbing it.
- **Memory** is a sliding window with a rolling summary — not a vector store. For session-length conversation this is fine; for "remember me across sessions" you'd swap in a persistent store (sqlite + embeddings).
- The **OSS model can't reliably do multi-step tool use**. The current implementation does a single tool call per turn. Multi-step would need a larger OSS model (Qwen2.5-7B-Instruct) or a fine-tune.

---

## What I'd improve with more time

1. **Replace the sliding-window summary with episodic memory** — embed each turn, retrieve top-k on each new turn, summarize the rest. Better recall without growing the prompt.
2. **Fine-tune Qwen2.5-0.5B on tool-use traces** — would meaningfully close the tool-use gap and likely raise the eval score by ~1 point on factual prompts.
3. **Run the eval against AdvBench + TruthfulQA** rather than a custom 30-prompt set. Custom prompts are good for sanity; published benchmarks are the real signal.
4. **Add streaming** to both backends end-to-end. The frontier path streams; the OSS path currently doesn't because `transformers` streaming through Gradio needs a `TextIteratorStreamer` wrapper I skipped for time.
5. **Promote the Llama-8B-Instant moderator to always-on** and add a calibrated threshold so refusals are tuned rather than binary.
6. **A real persistent store** (sqlite-backed) so sessions survive process restarts.

---

## Deployment

Three deployment targets, all in `deployment/`:

- **HuggingFace Spaces** (recommended for the OSS bonus): copy `deployment/hf_space/` to a new Space, push, done. Uses Qwen2.5-0.5B on the free CPU tier.
- **Modal**: `modal deploy deployment/modal_app.py` — serverless GPU, ~2 s cold start.
- **Docker**: `docker build -t ollive-assistant -f deployment/Dockerfile . && docker run -p 7860:7860 ollive-assistant`

### Cost & latency snapshot

| Target | Hardware | Cold start | P50 latency | P95 latency | $/1k turns |
|---|---|---|---|---|---|
| HF Space (free CPU) | 2 vCPU / 16 GB | ~25 s | 1.4 s | 3.8 s | $0 |
| Modal (T4 GPU) | 1× T4 | ~3 s | 0.35 s | 0.7 s | ~$0.18 |
| Local laptop (M-class CPU) | 8 vCPU | n/a | 1.1 s | 2.2 s | $0 |
| Frontier (Llama 3.3 70B via Groq) | API | n/a | 0.6 s | 1.4 s | $0 (free tier) |

Measured over 50 representative prompts; see `report/cost_latency.md` for the methodology.

---

## Repo layout

```
ai-ml-olive/
├── README.md
├── requirements.txt
├── .env.example
├── assistants/             # OSS + Frontier backends, shared base
├── memory/                 # Sliding-window + summary
├── tools/                  # datetime / calculator / web_search
├── guardrails/             # Input + output safety
├── observability/          # JSONL event logger
├── evaluation/             # Prompts, judge, runner
├── app/                    # Gradio + CLI front-ends
├── deployment/             # HF Spaces, Modal, Dockerfile
├── report/                 # Evaluation report + infographics
└── tests/                  # Smoke tests for the loop
```

---

## License

MIT — see `LICENSE`.
