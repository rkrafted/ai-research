# AI Research Pipeline — "most intelligence per GPU" (plan, 2026-10-01)

Owner: the owner. Planner/verifier: Claude. Builders: DeepSeek (cloud lane, off-peak) + local models (3090).

## 1. Why this exists

Every week there are hundreds of videos and papers about models, training tricks, tools and theory. Almost
none of it is written for someone with **one 24 GB GPU, 30 GB RAM and a $2.70/day API budget**. The pipeline
answers one question, continuously:

> **What gives the most capability per unit of compute for people with limited hardware — and does it hold
> up when we actually run it on our 3090?**

It feeds four things we already run:
- **Distill** (move paid LLM calls to small local models with proof): new techniques to try.
- **merge-lab / agent-distill / looped-model experiments**: the experiment queue.
- **ORBIT** (the YouTube channel): every verified result becomes an episode with real numbers.
- **The runner itself**: tools and methods that make the factory cheaper or smarter.

## 2. The lens: "gains under limits"

Every item (video, paper, repo) is scored on what matters to a compute-limited builder, not on hype:

| Axis | Question | Example high score |
|---|---|---|
| **Capability per parameter** | Does it make a small model smarter at fixed size? | Looped/recurrent-depth models, distillation, better data |
| **Capability per FLOP / per VRAM** | Can it run or train on 24 GB? | QLoRA, quantization, MoE offload, KV-cache tricks |
| **Test-time compute** | Can we trade time for quality on our own GPU? | Latent reasoning loops, self-consistency, verifiers |
| **Data efficiency** | Fewer/cheaper examples for the same gain? | Synthetic data, rejection sampling, DAgger |
| **Cost to try** | GPU-hours and $ to reproduce one claim | < 2 GPU-h and < $1 is ideal |
| **Evidence quality** | Paper with code + numbers > blog > opinion video | Reproduced by others |
| **Transfer to our work** | Does it help Distill, sales-brain, ORBIT, the runner? | Directly testable on our tasks |

**Priority score** = expected gain × transfer ÷ cost to try, discounted by evidence quality. The top of the
list becomes experiments; the rest stays searchable.

## 3. The first deep-dive: looped language models (latent reasoning)

The paper the owner named, *Scaling Latent Reasoning via Looped Language Models* (the "Ouro" models,
2025), is the poster child for this lens. From my prior knowledge (DeepSeek must confirm every point from
the paper itself before it is used):
- Small looped models (≈1.4B and 2.6B) reuse the same layers several times per token ("thinking in latent
  space" instead of writing out chain-of-thought), with a learned exit so easy tokens loop less.
- Reported: they match much larger standard models on reasoning benchmarks; the gain comes from better
  *use* of knowledge (manipulation) rather than storing more facts.
- Why it matters to us: **more reasoning per parameter and per byte of VRAM** — exactly our constraint.

Deliverables for this deep-dive (DeepSeek reads; GPU verifies):
1. A faithful summary of the paper and the neuro-dump video (captions), with every linked paper listed and
   summarised the same way (claim, evidence, code, compute needed).
2. A reproduction plan sized to the 3090: can we run the released checkpoints (if any) in our
   employee-shootout harness and in sales-brain extraction? Loop count vs quality vs latency curve.
3. If checkpoints exist and fit: run them (GPU job, results via the provenance ledger). If not: the
   smallest faithful experiment (e.g. loop an existing small model's layers at inference and measure).

## 4. Sources

### 4.1 Seed channels (the owner's picks — the quality bar)
IBM Technology · AI Search · Aperture · 3Blue1Brown · Perimeter Institute for Theoretical Physics ·
neuro-dump (the looped-LM video).

### 4.2 Candidate channels to evaluate (DeepSeek verifies each exists, is active, and is substantive)
- Paper explainers: Yannic Kilcher, Two Minute Papers, bycloud, AI Coffee Break with Letitia, Gabriel
  Mongaraz, hu-po, Tunadorable, Julia Turc, Algorithmic Simplicity.
- Engineering / training: Andrej Karpathy, Umar Jamil, Trelis Research, Sam Witteveen, sentdex, Hamel Husain,
  Jeremy Howard (fast.ai), Unsloth / Daniel Han talks, Evan Armstrong-style finetuning channels.
- Intuition / maths: Welch Labs, StatQuest, Serrano.Academy, Mutual Information, Steve Brunton, Artem
  Kirsanov, Computerphile.
- Lectures: Stanford Online (CS25, CS336, CS229), MIT OpenCourseWare, Simons Institute, IPAM.
- Interviews / frontier context: Machine Learning Street Talk, Dwarkesh Patel, Latent Space, The Cognitive
  Revolution, Gradient Dissent, TWIML.
- News-level (lower weight, high recall): AI Explained, Matthew Berman, Wes Roth, Fireship, 1littlecoder.

The seed channels define "good": captions with real technical content, titles that name the method, and
papers linked in descriptions. A channel is kept if a sample of its last 20 videos scores like the seeds.

### 4.3 Expansion (so "literally every video about AI" is reachable, not hand-listed)
- Follow links: papers in descriptions → arXiv → other videos that cite the same arXiv IDs.
- Channel graph: channels that cover the same papers within 30 days are candidates; score their samples.
- Topic search sweeps per term (quantization, distillation, MoE, looped/recurrent depth, test-time compute,
  speculative decoding, LoRA/QLoRA, merging, synthetic data, small models, agents, RAG, speech, vision).
- arXiv + Papers with Code + Hugging Face daily papers as the paper side of the same graph.

## 5. Pipeline stages

```
discover ─► fetch captions + metadata ─► extract claims/methods/papers ─► score (lens §2)
   ▲                                                                           │
   └──── expansion (§4.3) ◄──────── knowledge base (SQLite + vectors) ◄────────┘
                                                │
                    top-scored ─► experiment card ─► GPU job on 3090 ─► verified result
                                                                          │
                                         ORBIT episode outline ◄──────────┤
                                         Distill technique backlog ◄──────┘
```

1. **Discover**: channel list + expansion; store channel and video metadata (title, date, description,
   links, duration, views).
2. **Fetch captions**: YouTube auto/manual captions (yt-dlp, no video download). Where captions are missing
   and the video scores high on metadata, transcribe audio locally with Whisper on the 3090.
3. **Extract** (local 27B via local batch for bulk; DeepSeek for spot checks and hard cases): per video —
   methods named, claims with numbers, papers/repos cited, hardware/compute mentioned, "who is it for".
4. **Score** with the §2 lens. Deterministic formula over extracted fields; no LLM in the final score.
5. **Knowledge base**: SQLite (videos, claims, papers, methods, links) + embeddings (local nomic-embed) for
   search; every claim keeps its source timestamp/URL.
6. **Experiment cards** for the top items: hypothesis, the smallest test on our hardware, the metric,
   GPU-hours, pass/fail rule. Become queue items with real `done_when` and `results:` (provenance ledger).
7. **Outputs**: weekly digest ("what's worth it on a 24 GB GPU this week"), ORBIT outlines from verified
   results only, Distill backlog.

## 6. Honesty and quality rules
- A claim from a video is a *claim*; only our GPU run makes it a *result*. Results follow the provenance
  ledger (result files must be written by a runner GPU job).
- Store our own summaries and links, never re-publish captions or transcripts.
- Every number in a digest or episode cites its source (video timestamp, paper table, or our run).
- Hype channels get lower evidence weight; their value is recall (finding things early), not truth.

## 7. Milestones (queue items; DeepSeek builds, local models do bulk text, Claude verifies)
- **R0 — Deep-dive looped LMs**: paper + neuro-dump video + its linked papers summarised; reproduction plan
  for the 3090; run released checkpoints if they fit. Done = summaries with sources, plan, and either a
  real GPU result or a written reason it cannot fit.
- **R1 — Channel registry**: seeds + ≥ 60 candidate channels evaluated on a 20-video sample; kept list with
  scores and reasons.
- **R2 — Ingest**: captions + metadata for every video of kept channels from the last 24 months (target
  ≥ 5,000 videos), extraction via local batch, knowledge base built.
- **R3 — Score + digest**: the lens score over everything; top-50 "gains under limits" list with sources.
- **R4 — Experiment loop**: top 5 cards run as GPU jobs; verified results; 3 ORBIT outlines.
- Later: continuous mode (new videos daily), expansion graph, public digest.

## 8. Budget
- DeepSeek: research and code only, off-peak; bulk extraction runs on the local 27B (free).
- Estimated R0–R3 cloud cost: a few dollars over several days within the $2.70/day cap.
- GPU: extraction batches overnight; experiments ≤ 2 GPU-hours each unless approved.
