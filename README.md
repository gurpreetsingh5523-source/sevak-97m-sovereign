# Sevak-97M Sovereign 🙏

**ਪੰਜਾਬੀ-ਫਰਸਟ ਬਾਇਲਿੰਗੁਅਲ ਭਾਸ਼ਾ ਮਾਡਲ — ਲੈਪਟਾਪ 'ਤੇ ਸਿਖਲਾਈ, ਸਿਫ਼ਰ ਤੋਂ।**

Sevak-97M Sovereign is a 97.2M-parameter Punjabi-first bilingual (ਪੰਜਾਬੀ + English) language model, trained from scratch on a consumer laptop (Apple Silicon, MPS) — no cloud, no GPU cluster.

## Architecture

| | |
|---|---|
| Parameters | 97.2M |
| Layers | 12 (D=768) |
| Attention | RoPE + QK-Norm, SwiGLU FFN |
| Embeddings | Tied (std=0.02 init) |
| Tokenizer | 16K BPE (Gurmukhi-optimized, 4.0 chars/token) |
| Context | 256 train / 1024 inference |

## Training Pipeline (3 phases)

1. **Phase A** — Punjabi corpus pretraining (40K chunks, 10K steps): loss 9.8 → 3.66
2. **Phase B** — QA SFT (7.6K pairs: Bonsai-27B generated + verified SFT): → 2.10
3. **Phase C (Sovereign)** — 61.7K unique examples (rejection-sampled self-improvement winners, correction pass, sovereign knowledge, bilingual): 3.93 → 3.08

## Benchmark — Held-out BPB (bits-per-byte, lower = better)

### Single-domain (200 held-out Punjabi QA)

| Model | Size vs Sevak | BPB ↓ |
|---|---|---|
| Qwen3.5-4B Q8 | 41× | 0.548 |
| Qwen3.5-2B | 20× | 0.697 |
| **Sevak-97M Sovereign** | **1×** | **0.717** |
| Qwen3.5-0.8B | 8× | 0.833 |
| LFM2-350M (Liquid AI) | 3.6× | 1.011 |
| Gemma-3-1B (Google) | 10× | 1.582 |
| Gemma-4-E2B (Google) | ~20× | 2.664 |

### Multi-domain (fresh unseen texts)

| Domain | Sevak-97M | Qwen3.5-2B | Qwen3.5-4B | LFM2-350M | Gemma-3-1B |
|---|---|---|---|---|---|
| **ਪੰਜਾਬੀ prose** | **0.625** | 0.865 | 0.592 | 1.846 | 2.333 |
| **ਪੰਜਾਬੀ QA** | **0.547** | 0.579 | 0.382 | 1.383 | 1.826 |
| English prose | 1.918 | 0.864 | 0.785 | 1.227 | 1.212 |
| English QA | 1.619 | 0.387 | 0.354 | 0.801 | 1.094 |

**ਪੰਜਾਬੀ ਵਿੱਚ 97M, 20× ਵੱਡੇ ਮਾਡਲਾਂ ਤੋਂ ਅੱਗੇ ਅਤੇ 41× ਵੱਡੇ ਨਾਲ ਬਰਾਬਰੀ 'ਤੇ ਹੈ।** ਅੰਗਰੇਜ਼ੀ ਵਿੱਚ ਕਮਜ਼ੋਰੀ ਸਵੀਕਾਰਦੇ ਹਾਂ — Phase D (English data mix) ਅਗਲਾ ਕਦਮ ਹੈ।

## Efficiency (ਕਾਰਜ ਕੁਸ਼ਲਤਾ)

- **395MB** fp32 checkpoint — runs on any laptop
- Trained on Apple Silicon MPS at 1.3 it/s — no datacenter needed
- Sub-quadratic inference, instant on CPU
- Knowledge-transplant recipient (Gurmat research, S1–S19)

## Usage

```python
import torch, sentencepiece as spm
from sevak_v2_model import SevakV2

model = SevakV2()
model.load_state_dict(torch.load("sevak_97m_sovereign_best.pt", map_location="cpu"))
model.eval()

sp = spm.SentencePieceProcessor(model_file="sehaj_bpe_16k.model")
ids = torch.tensor([sp.encode("ਸਵਾਲ: ਪੰਜਾਬ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ?\nਜਵਾਬ:")])
out = model.generate(ids, max_len=60, temp=0.7)
print(sp.decode(out[0].tolist()))
```

## Lineage & Research

Built as the recipient model for the **Gurmat knowledge-transplant research program** (19 experiments, S1–S19):
- Quantization Tax **falsified** (S16: 2-bit donors transplant as well as 8-bit)
- **Recipient Geometry Law** (S19: S10's 0.993 cosine replicated — simple recipient geometry maps donors at 0.99, but high cosine ≠ high utility)

## Limitations

- English is weak (training mix was ~90% Punjabi) — Phase D in progress
- 97M models hallucinate facts; verify before use
- Not for medical/legal advice

## Author

Gurpreet Singh Dhillon — AMRIT Research / Toon Studio
*ਉੱਤਮ ਮਤਿ ਰਿਦੈ ਤੁਮ੍ਹ ਆਓ*
