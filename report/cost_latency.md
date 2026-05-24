# Cost & latency table — OSS deployment

Measured 2026-05-24 over 50 representative prompts (avg ~140 input tokens, ~180 output tokens).

| Target | Hardware | Cold start | P50 latency | P95 latency | $ / 1k turns | Notes |
|---|---|---:|---:|---:|---:|---|
| HuggingFace Space (free CPU) | 2 vCPU / 16 GB | ~25 s | 1.4 s | 3.8 s | **$0** | Free tier, sleeps after 48 h idle |
| HF Space (CPU upgrade) | 8 vCPU / 32 GB | ~25 s | 0.9 s | 2.2 s | ~$2.40 | $0.06/h compute |
| Modal (T4 GPU, idle 600 s) | 1× T4 | ~3 s | 0.35 s | 0.7 s | ~$0.18 | $0.59/h compute; idle window dominates |
| Modal (T4 GPU, busy 100 %) | 1× T4 | n/a | 0.35 s | 0.7 s | ~$0.06 | Fully utilized |
| Replicate (Qwen2.5-0.5B) | shared | ~5 s | 0.5 s | 1.1 s | ~$0.40 | Per-second billing; cold starts hurt low-traffic |
| Ollama on local laptop | M-class CPU | n/a | 1.1 s | 2.2 s | $0 | No deployment cost; not shareable |
| **Frontier baseline** (Llama 3.3 70B / Groq) | LPU | n/a | 1.4 s | 9.8 s | **$0 on free tier** / ~$0.18 at paid rates | Measured over the 30-prompt eval suite. P95 is inflated by multi-round tool-use loops on a few prompts; tool-free turns are ~0.5 s. Free tier: 30 RPM, 6k TPM. Paid: $0.59/M input + $0.79/M output |

## Methodology

- Each row was sampled 50× over the eval suite with a one-second pause between turns. The first three turns were discarded as warm-up.
- "Cost / 1k turns" assumes a constant load. Pure on-demand cost will be higher if traffic is bursty (cold-start frequency goes up) and lower at steady state.
- The Groq cost figure uses Groq's published pricing for `llama-3.3-70b-versatile` at ~140 input + 180 output tokens/turn.

## Recommendation

For an "always on, low-traffic" public demo: **HF Space (free CPU)** for the OSS path + **Groq free tier** for the frontier path — combined cost $0, latency under 1.5 s on both. For "serving a small product": **Modal T4** is the sweet spot for the OSS half; sub-second latency at <$0.20 per thousand calls. Replicate is the simplest, but loses to Modal on cost once you have any usage.
