---
title: Ollive OSS Assistant
emoji: 🟢
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Ollive OSS Assistant — Qwen2.5-0.5B-Instruct

Lightweight personal assistant running entirely on the free CPU tier.
Supports multi-turn chat, short-term memory, simple tool use (datetime,
calculator, web search), and a regex-based safety guardrail.

Deploy:

```
huggingface-cli repo create ollive-oss-assistant --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<you>/ollive-oss-assistant && cd $_
# copy everything from this folder + the assistants/, memory/, tools/,
# guardrails/, observability/ packages from the parent repo
git add . && git commit -m "init" && git push
```

The Space takes ~25 s to cold-start while it downloads the Qwen weights.
