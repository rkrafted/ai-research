# Gains under limits — top 50 for one RTX 3090

Deterministic lens (PLAN §2) over `data/kb.sqlite`: **100 eligible method clusters** from 2469 method–video links; no LLM and no randomness in the score. Full component breakdown: `data/lens.json`.

Every item is one canonical method cluster with its evidence links (YouTube timestamp + arXiv). The score is `gain × transfer ÷ cost × evidence`; the axis values are in the component table at the end.

## Reason more with test-time compute

- **Chain of Thought** — let the model write steps; the cheapest reasoning-quality lever we have `priority 1170.0` · evidence: [video](https://www.youtube.com/watch?v=-FSt-8aiMfU&t=182s) · [arXiv 1807.03819](https://arxiv.org/abs/1807.03819) · [arXiv 2001.08361](https://arxiv.org/abs/2001.08361)
- **Self-Consistency** — majority-vote over samples buys accuracy with no training at all `priority 671.3` · evidence: [video](https://www.youtube.com/watch?v=70Pl0R8R9dk&t=364s) · [arXiv 2207.05221](https://arxiv.org/abs/2207.05221) · [arXiv 2305.04091](https://arxiv.org/abs/2305.04091)
- **Test-Time Compute** — spend inference FLOPs (samples, search, revision) instead of parameters `priority 664.8` · evidence: [video](https://www.youtube.com/watch?v=-FSt-8aiMfU&t=182s) · [arXiv 2207.05221](https://arxiv.org/abs/2207.05221) · [arXiv 2305.04091](https://arxiv.org/abs/2305.04091)
- **LLM-as-a-Judge** — a local judge scores outputs for filtering/sampling without human labels `priority 648.6` · evidence: [video](https://www.youtube.com/watch?v=4o_ic83U1Kw) · [arXiv 2401.10020](https://arxiv.org/abs/2401.10020) · [arXiv 2404.12272](https://arxiv.org/abs/2404.12272)
- **Beam search** — keep several hypotheses to raise exact-match quality at fixed weights `priority 508.5` · evidence: [video](https://www.youtube.com/watch?v=2hcsmtkSzIw) · [arXiv 2201.11903](https://arxiv.org/abs/2201.11903) · [arXiv 2305.20050](https://arxiv.org/abs/2305.20050)
- **Reflection** — self-critique and retry is a near-zero-cost quality bump on extraction tasks `priority 485.0` · evidence: [video](https://www.youtube.com/watch?v=8t65bss7U74&t=196s) · [arXiv 2304.03442](https://arxiv.org/abs/2304.03442)
- **Tree of Thoughts** — branch and prune reasoning paths for hard problems on a single GPU `priority 405.9` · evidence: [video](https://www.youtube.com/watch?v=8t65bss7U74&t=196s) · [arXiv 2401.12345](https://arxiv.org/abs/2401.12345)

## Retrieval and memory

- **Prompt Engineering** — zero-GPU quality gains; the first thing to exhaust before any training run `priority 1130.0` · evidence: [video](https://www.youtube.com/watch?v=0-CbsNB9tdk) · [arXiv 1908.10084](https://arxiv.org/abs/1908.10084) · [arXiv 2207.05221](https://arxiv.org/abs/2207.05221)
- **RAG** — external knowledge at inference: no retraining, cut hallucination, small model stays small `priority 862.7` · evidence: [video](https://www.youtube.com/watch?v=-wGzSnhQKPM&t=951s) · [arXiv 1908.10084](https://arxiv.org/abs/1908.10084) · [arXiv 2207.05221](https://arxiv.org/abs/2207.05221)
- **Embeddings** — a local embedding model powers search, dedup and clustering for free `priority 731.9` · evidence: [video](https://www.youtube.com/watch?v=2NQzCbPvUMs) · [arXiv 2310.06816](https://arxiv.org/abs/2310.06816) · [arXiv 2404.10981](https://arxiv.org/abs/2404.10981)
- **context engineering** — what goes in the window beats most training tricks at this scale `priority 681.3` · evidence: [video](https://www.youtube.com/watch?v=4loPnxvWWhg&t=241s)
- **Retrieval** — retrieve-then-read is the cheapest way to add facts to a tiny model `priority 619.4` · evidence: [video](https://www.youtube.com/watch?v=74Du4Ej_-yM&t=1867s) · [arXiv 2304.03442](https://arxiv.org/abs/2304.03442)
- **Vector Search** — approximate nearest neighbours make local retrieval cheap enough for one box `priority 616.0` · evidence: [video](https://www.youtube.com/watch?v=0fJ3TcY8bO8&t=687s) · [arXiv 2404.10981](https://arxiv.org/abs/2404.10981) · [arXiv 2408.04948](https://arxiv.org/abs/2408.04948)
- **BM25** — lexical search is a zero-GPU baseline that is often hard to beat `priority 510.6` · evidence: [video](https://www.youtube.com/watch?v=-wGzSnhQKPM&t=951s)
- **Vector Databases** — a local index is the memory layer for an agent that cannot fine-tune daily `priority 492.6` · evidence: [video](https://www.youtube.com/watch?v=9ozwh9EDGhU&t=2s) · [arXiv 2508.21038](https://arxiv.org/abs/2508.21038)

## Small models, efficient by design

- **In-Context Learning** — examples in the prompt substitute for fine-tuning when the task is simple `priority 1061.7` · evidence: [video](https://www.youtube.com/watch?v=CPkUCqcrULg&t=10s) · [arXiv 2401.04088](https://arxiv.org/abs/2401.04088) · [arXiv 2401.09417](https://arxiv.org/abs/2401.09417)
- **Small Language Models (SLMs)** — the whole game: pick/raise a 1-8B model that runs entirely on the 3090 `priority 439.7` · evidence: [video](https://www.youtube.com/watch?v=CEBhBTbn1BI)
- **Mixture of Experts** — activate a fraction of parameters per token: more capacity at the same FLOPs `priority 356.5` · evidence: [video](https://www.youtube.com/watch?v=-FSt-8aiMfU&t=182s) · [arXiv 1701.06538](https://arxiv.org/abs/1701.06538) · [arXiv 1706.03762](https://arxiv.org/abs/1706.03762)
- **Mamba** — state-space mixing gives linear-time sequence modelling at constant memory `priority 345.0` · evidence: [video](https://www.youtube.com/watch?v=26PVzD707a8&t=47s) · [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) · [arXiv 2111.00396](https://arxiv.org/abs/2111.00396)
- **Transformer** — understanding the baseline block tells you which efficiency change is worth it `priority 343.2` · evidence: [video](https://www.youtube.com/watch?v=-BelefxpVc8&t=1s) · [arXiv 1301.3781](https://arxiv.org/abs/1301.3781) · [arXiv 1706.03762](https://arxiv.org/abs/1706.03762)
- **Sliding Window Attention** — local attention caps KV growth so long contexts fit in 24 GB `priority 329.3` · evidence: [video](https://www.youtube.com/watch?v=AU21wwYIt9E) · [arXiv 2309.17453](https://arxiv.org/abs/2309.17453) · [arXiv 2310.06825](https://arxiv.org/abs/2310.06825)

## Agents and tooling

- **Tool Use** — offload arithmetic, search and code to tools instead of model capacity `priority 765.1` · evidence: [video](https://www.youtube.com/watch?v=1TU9ZrZhqw0) · [arXiv 2310.08560](https://arxiv.org/abs/2310.08560) · [arXiv 2404.01744](https://arxiv.org/abs/2404.01744)
- **DSPy** — compile prompts against a metric, replacing hand-tuning with a few local calls `priority 621.5` · evidence: [video](https://www.youtube.com/watch?v=0gYMqFYRtDI)
- **AI agents** — an agent converts a small model plus tools into work a bigger model would do `priority 577.5` · evidence: [video](https://www.youtube.com/watch?v=--SB_qJw9sg) · [arXiv 2308.10848](https://arxiv.org/abs/2308.10848) · [arXiv 2310.08560](https://arxiv.org/abs/2310.08560)
- **Planning** — decompose before acting; a small model plans well when the steps are small `priority 570.3` · evidence: [video](https://www.youtube.com/watch?v=ETq3ZTqxFlo&t=68s) · [arXiv 2304.03442](https://arxiv.org/abs/2304.03442) · [arXiv 2308.01404](https://arxiv.org/abs/2308.01404)
- **ReAct** — reason-act interleaving is the cheap scaffold that makes tool use reliable `priority 499.3` · evidence: [video](https://www.youtube.com/watch?v=5_L_BpL5Whs)
- **Prompt Injection** — the safety failure mode of every tool-using local agent; cheap to test `priority 448.9` · evidence: [video](https://www.youtube.com/watch?v=4lXks428C9o&t=16s) · [arXiv 2212.06470](https://arxiv.org/abs/2212.06470) · [arXiv 2307.10490](https://arxiv.org/abs/2307.10490)
- **LangGraph** — explicit state graphs make agent loops debuggable and cheap to retry `priority 396.4` · evidence: [video](https://www.youtube.com/watch?v=3tZVGJX9GMw&t=1s)
- **MCP** — a standard tool protocol lets one local model drive many utilities `priority 375.5` · evidence: [video](https://www.youtube.com/watch?v=-rSbvS0qLqY&t=75s)

## Training theory and scaling

- **evals** — a small task-specific eval is the only way to know a technique paid off `priority 640.3` · evidence: [video](https://www.youtube.com/watch?v=3ZMUiFaQ3qg&t=93s) · [arXiv 2404.12272](https://arxiv.org/abs/2404.12272)
- **Scaling Laws** — tells you whether more data, parameters or steps is the cheapest gain `priority 634.4` · evidence: [video](https://www.youtube.com/watch?v=5vBM3D0tZ5U) · [arXiv 1011.3245](https://arxiv.org/abs/1011.3245) · [arXiv 1807.03819](https://arxiv.org/abs/1807.03819)

## Run bigger models on 24 GB

- **Quantization** — 4/8-bit weights cut VRAM ~2-4x so a 7B-13B model fits and decodes faster on one 3090 `priority 611.5` · evidence: [video](https://www.youtube.com/watch?v=0VdNflU08yA) · [arXiv 2210.17323](https://arxiv.org/abs/2210.17323) · [arXiv 2306.00978](https://arxiv.org/abs/2306.00978)
- **KV Cache** — shrink or page the KV cache to raise batch size and context length at fixed VRAM `priority 506.9` · evidence: [video](https://www.youtube.com/watch?v=Hrbq66XqtCo) · [arXiv 2309.17453](https://arxiv.org/abs/2309.17453) · [arXiv 2501.00663](https://arxiv.org/abs/2501.00663)
- **LoRA** — trains <1% of weights; adapters are swappable and cheap to store `priority 445.0` · evidence: [video](https://www.youtube.com/watch?v=1N76mJ1pMro&t=231s) · [arXiv 2210.17323](https://arxiv.org/abs/2210.17323) · [arXiv 2306.00978](https://arxiv.org/abs/2306.00978)
- **Model routing** — send easy inputs to a small local model and hard ones to the big one `priority 424.9` · evidence: [video](https://www.youtube.com/watch?v=35ZWesLrv5A&t=496s)
- **Flash Attention** — fused exact attention cuts activation memory and speeds prefill on one GPU `priority 421.3` · evidence: [video](https://www.youtube.com/watch?v=0-MA3jgYMMg&t=510s) · [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) · [arXiv 2111.00396](https://arxiv.org/abs/2111.00396)
- **Tokenization** — a better tokenizer packs more text per token, cutting both context and FLOPs `priority 417.7` · evidence: [video](https://www.youtube.com/watch?v=Hf7XY38raIs) · [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) · [arXiv 2312.00752](https://arxiv.org/abs/2312.00752)
- **QLoRA** — 4-bit frozen base + LoRA adapters fine-tunes 7B-13B (even 30B) in 24 GB `priority 366.6` · evidence: [video](https://www.youtube.com/watch?v=1N76mJ1pMro&t=231s) · [arXiv 2402.15116](https://arxiv.org/abs/2402.15116)
- **Unsloth** — fused, memory-lean fine-tuning kernels cut QLoRA VRAM and time on consumer cards `priority 328.5` · evidence: [video](https://www.youtube.com/watch?v=3iqvBEGS2So) · [arXiv 2310.05914](https://arxiv.org/abs/2310.05914) · [arXiv 2402.09353](https://arxiv.org/abs/2402.09353)

## Interpretability and steering

- **Sparse Autoencoders** — understand which features drive behaviour before spending GPU-hours on training `priority 544.7` · evidence: [video](https://www.youtube.com/watch?v=5-1lKFvV1i0&t=317s) · [arXiv 2303.08001](https://arxiv.org/abs/2303.08001) · [arXiv 2304.14997](https://arxiv.org/abs/2304.14997)
- **Logit Lens** — read the residual stream at every layer for near-zero compute `priority 492.4` · evidence: [video](https://www.youtube.com/watch?v=1ONwQzauqkc&t=382s) · [arXiv 2004.12605](https://arxiv.org/abs/2004.12605) · [arXiv 2303.08001](https://arxiv.org/abs/2303.08001)

## Vision and video

- **CLIP** — free zero-shot labels and image-text retrieval from a small contrastive model `priority 482.5` · evidence: [video](https://www.youtube.com/watch?v=2NmRJxaUwyE&t=1s) · [arXiv 2006.11239](https://arxiv.org/abs/2006.11239) · [arXiv 2010.02502](https://arxiv.org/abs/2010.02502)

## Fit training on one GPU

- **Synthetic data generation** — a local teacher mints the training set, so no labelled data is bought `priority 479.6` · evidence: [video](https://www.youtube.com/watch?v=-_x7lhhZK7M) · [arXiv 2408.16333](https://arxiv.org/abs/2408.16333) · [arXiv 2506.10943](https://arxiv.org/abs/2506.10943)
- **Distillation** — teacher-to-student transfer is the highest-leverage way to shrink a model `priority 389.7` · evidence: [video](https://www.youtube.com/watch?v=1xDVbu-WaFo&t=3s) · [arXiv 2311.00430](https://arxiv.org/abs/2311.00430) · [arXiv 2311.17713](https://arxiv.org/abs/2311.17713)
- **Model merging** — merge fine-tuned checkpoints without any gradient step to combine skills `priority 367.6` · evidence: [video](https://www.youtube.com/watch?v=8uq0e_i17FE) · [arXiv 2403.07816](https://arxiv.org/abs/2403.07816) · [arXiv 2410.03617](https://arxiv.org/abs/2410.03617)
- **Data Curation** — filtering/deduping a small corpus beats scaling a noisy one `priority 323.2` · evidence: [video](https://www.youtube.com/watch?v=DcvgPEApHT8&t=36s)

## Speech and audio

- **Whisper** — local ASR turns audio into training text for free `priority 375.5` · evidence: [video](https://www.youtube.com/watch?v=0GV5cYKz7Rc&t=56s)
- **Voice cloning** — open voice models run locally; no per-character API cost `priority 347.3` · evidence: [video](https://www.youtube.com/watch?v=3iqvBEGS2So) · [arXiv 2310.00704](https://arxiv.org/abs/2310.00704)
- **speech-to-text** — transcription is a solved local-model task and a data source for everything else `priority 329.2` · evidence: [video](https://www.youtube.com/watch?v=IDNfAZVKvPE&t=214s)

## Post-train with preference / RL

- **DPO** — preference optimisation without a reward model or PPO loop; fits one GPU `priority 323.8` · evidence: [video](https://www.youtube.com/watch?v=3WU6fl7DHj0) · [arXiv 2305.18290](https://arxiv.org/abs/2305.18290) · [arXiv 2401.10020](https://arxiv.org/abs/2401.10020)

## Paper index (cited by the items above)

- [Extrapolating GPT-N Performance](https://arxiv.org/abs/1011.3245) — arXiv:1011.3245
- [An early paper on how directions in embedding spaces have meaning](https://arxiv.org/abs/1301.3781) — arXiv:1301.3781
- [Sparse MoE](https://arxiv.org/abs/1701.06538) — arXiv:1701.06538
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — arXiv:1706.03762
- [Universal Transformers](https://arxiv.org/abs/1807.03819) — arXiv:1807.03819
- [BERT: Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805) — arXiv:1810.04805
- [Sentence BERT](https://arxiv.org/abs/1908.10084) — arXiv:1908.10084
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — arXiv:2001.08361
- [Towards Monosemanticity: Decomposing Language Models Into Interpretable Features](https://arxiv.org/abs/2004.12605) — arXiv:2004.12605
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — arXiv:2006.11239
- [Gshard](https://arxiv.org/abs/2006.16668) — arXiv:2006.16668
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — arXiv:2010.02502
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — arXiv:2103.00020
- [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407) — arXiv:2107.05407
- [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) — arXiv:2111.00396
- [Chain of Thought Paper](https://arxiv.org/abs/2201.11903) — arXiv:2201.11903
- [Flash Attention](https://arxiv.org/abs/2205.14135) — arXiv:2205.14135
- [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) — arXiv:2207.05221
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — arXiv:2207.12598
- [MoE paper](https://arxiv.org/abs/2208.02813) — arXiv:2208.02813
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — arXiv:2210.17323
- [Will we run out of data? Limits of LLM scaling based on human-generated data](https://arxiv.org/abs/2211.04325) — arXiv:2211.04325
- [Position: Considerations for Differentially Private Learning with Large-Scale Public Pretraining](https://arxiv.org/abs/2212.06470) — arXiv:2212.06470
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) — arXiv:2212.09748
- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://arxiv.org/abs/2303.08001) — arXiv:2303.08001
- [Towards Monosemanticity: Decomposing Language Models Into Interpretable Features](https://arxiv.org/abs/2303.08112) — arXiv:2303.08112
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — arXiv:2304.03442
- [ACT Paper](https://arxiv.org/abs/2304.13705) — arXiv:2304.13705
- [Interpretability in the Wild: A Circuit for Statistical Induction in GPT-2](https://arxiv.org/abs/2304.14997) — arXiv:2304.14997
- [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091) — arXiv:2305.04091
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — arXiv:2305.13245
- [Direct preference optimization: Your language model is secretly a reward model](https://arxiv.org/abs/2305.18290) — arXiv:2305.18290
- [Let's verify step by step](https://arxiv.org/abs/2305.20050) — arXiv:2305.20050
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — arXiv:2306.00978
- [StyleTTS2](https://arxiv.org/abs/2306.07691) — arXiv:2306.07691
- [A Comprehensive Overview of Large Language Models](https://arxiv.org/abs/2307.06435) — arXiv:2307.06435
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) — arXiv:2307.08691
- [Abusing Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs](https://arxiv.org/abs/2307.10490) — arXiv:2307.10490
- [WebVoyager](https://arxiv.org/abs/2308.01404) — arXiv:2308.01404
- [AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors](https://arxiv.org/abs/2308.10848) — arXiv:2308.10848
- [Binary-tree/FFF paper](https://arxiv.org/abs/2308.14711) — arXiv:2308.14711
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — arXiv:2309.06180
- [QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2309.14717) — arXiv:2309.14717
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) — arXiv:2309.17453
- [UniAudio: An Audio Foundation Model Toward Universal Audio Generation](https://arxiv.org/abs/2310.00704) — arXiv:2310.00704
- [Think Before You Speak](https://arxiv.org/abs/2310.02226) — arXiv:2310.02226
- [NEFT](https://arxiv.org/abs/2310.05914) — arXiv:2310.05914
- [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816) — arXiv:2310.06816
- [Mistral 7B](https://arxiv.org/abs/2310.06825) — arXiv:2310.06825
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) — arXiv:2310.08560
- [Eureka: Human-Level Reward Design via Large Language Models](https://arxiv.org/abs/2310.12931) — arXiv:2310.12931
- [A Picture is Worth a Thousand Words: Principled Recaptioning Improves Image Generation](https://arxiv.org/abs/2310.16656) — arXiv:2310.16656
- [Deja vu: Contextual sparsity for efficient llms at inference time](https://arxiv.org/abs/2310.17157) — arXiv:2310.17157
- [Distil-Whisper](https://arxiv.org/abs/2311.00430) — arXiv:2311.00430
- [Future Lens](https://arxiv.org/abs/2311.04897) — arXiv:2311.04897
- [Florence](https://arxiv.org/abs/2311.06242) — arXiv:2311.06242
- [On measuring faithfulness of natural language explanations](https://arxiv.org/abs/2311.07466) — arXiv:2311.07466
- [MoWE](https://arxiv.org/abs/2311.10768) — arXiv:2311.10768
- [Scalable Extraction of Training Data from (Production) Language Models](https://arxiv.org/abs/2311.17035) — arXiv:2311.17035
- [ORCA: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2311.17713) — arXiv:2311.17713
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — arXiv:2312.00752
- [MatterGen: A Generative Model for Materials Design](https://arxiv.org/abs/2312.03687) — arXiv:2312.03687
- [AgentCoder](https://arxiv.org/abs/2312.13010) — arXiv:2312.13010
- [MoE-Mamba](https://arxiv.org/abs/2401.04081) — arXiv:2401.04081
- [MoE](https://arxiv.org/abs/2401.04088) — arXiv:2401.04088
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — arXiv:2401.06066
- [VisionMamba](https://arxiv.org/abs/2401.09417) — arXiv:2401.09417
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) — arXiv:2401.10020
- [Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — arXiv:2401.10774
- [The Illusion of Thinking](https://arxiv.org/abs/2401.12345) — arXiv:2401.12345
- [AgentBoard](https://arxiv.org/abs/2401.13178) — arXiv:2401.13178
- [WebVoyager](https://arxiv.org/abs/2401.13919) — arXiv:2401.13919
- [Deepseek-coder](https://arxiv.org/abs/2401.14196) — arXiv:2401.14196
- [KIVI](https://arxiv.org/abs/2402.02750) — arXiv:2402.02750
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — arXiv:2402.03300
- [MambaFormer](https://arxiv.org/abs/2402.04248) — arXiv:2402.04248
- [More Agents Is All You Need](https://arxiv.org/abs/2402.05120) — arXiv:2402.05120
- [An Interactive Agent Foundation Model](https://arxiv.org/abs/2402.05929) — arXiv:2402.05929
- [LLM survey](https://arxiv.org/abs/2402.06196) — arXiv:2402.06196
- [A Tale of Tails: Model Collapse as a Change of Scaling Laws](https://arxiv.org/abs/2402.07043) — arXiv:2402.07043
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) — arXiv:2402.09353
- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200) — arXiv:2402.10200
- [Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context](https://arxiv.org/abs/2402.10376) — arXiv:2402.10376
- [LoRA+](https://arxiv.org/abs/2402.12354) — arXiv:2402.12354
- [Grok-1](https://arxiv.org/abs/2402.15116) — arXiv:2402.15116
- [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15352) — arXiv:2402.15352
- [Berkeley / Halawi Paper](https://arxiv.org/abs/2402.18563) — arXiv:2402.18563
- [Are more LLM calls all you need?](https://arxiv.org/abs/2403.02419) — arXiv:2403.02419
- [NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models](https://arxiv.org/abs/2403.03100) — arXiv:2403.03100
- [Stable Diffusion 3: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — arXiv:2403.03206
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) — arXiv:2403.03507
- [Gemini 1.5](https://arxiv.org/abs/2403.05530) — arXiv:2403.05530
- [Stealing part of a production language model](https://arxiv.org/abs/2403.06634) — arXiv:2403.06634
- [VideoMamba](https://arxiv.org/abs/2403.06977) — arXiv:2403.06977
- [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815) — arXiv:2403.07815
- [2403.07816](https://arxiv.org/abs/2403.07816) — arXiv:2403.07816
- [Gemma](https://arxiv.org/abs/2403.08295) — arXiv:2403.08295
- [Octopus v2: On-device language model for super agent](https://arxiv.org/abs/2404.01744) — arXiv:2404.01744
- [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258) — arXiv:2404.02258
- [OS World](https://arxiv.org/abs/2404.07972) — arXiv:2404.07972
- [RAG Survey](https://arxiv.org/abs/2404.10981) — arXiv:2404.10981
- [More notes](https://arxiv.org/abs/2404.12272) — arXiv:2404.12272
- [Phi-3](https://arxiv.org/abs/2404.14219) — arXiv:2404.14219
- [Let's Think Dot by Dot: Hidden Computation in Transformer Language Models](https://arxiv.org/abs/2404.15758) — arXiv:2404.15758
- [Do Vision & Language Decoders use Images and Text equally? How Self-consistent are their Explanations?](https://arxiv.org/abs/2404.18624) — arXiv:2404.18624
- [KAN](https://arxiv.org/abs/2404.19756) — arXiv:2404.19756
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — arXiv:2405.04434
- [xLSTM (2024)](https://arxiv.org/abs/2405.04517) — arXiv:2405.04517
- [RabitQ](https://arxiv.org/abs/2405.12497) — arXiv:2405.12497
- [ICL](https://arxiv.org/abs/2405.19592) — arXiv:2405.19592
- [Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools](https://arxiv.org/abs/2405.20362) — arXiv:2405.20362
- [Mamba-2](https://arxiv.org/abs/2405.21060) — arXiv:2405.21060
- [LLM for code gen survey](https://arxiv.org/abs/2406.00515) — arXiv:2406.00515
- [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528) — arXiv:2406.02528
- [QJL](https://arxiv.org/abs/2406.03482) — arXiv:2406.03482
- [An Introduction to Diffusion Models](https://arxiv.org/abs/2406.08929) — arXiv:2406.08929
- [On the Representational Capacity of Neural Language Models with Chain-of-Thought Reasoning](https://arxiv.org/abs/2406.14197) — arXiv:2406.14197
- [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](https://arxiv.org/abs/2407.01082) — arXiv:2407.01082
- [AI Agents That Matter](https://arxiv.org/abs/2407.01502) — arXiv:2407.01502
- [Mixture of A Million Experts](https://arxiv.org/abs/2407.04153) — arXiv:2407.04153
- [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620) — arXiv:2407.04620
- [Qwen-2](https://arxiv.org/abs/2407.10671) — arXiv:2407.10671
- [Llama-3](https://arxiv.org/abs/2407.21783) — arXiv:2407.21783
- [Large Language Monkeys](https://arxiv.org/abs/2407.21787) — arXiv:2407.21787
- [Gemma 2](https://arxiv.org/abs/2408.00118) — arXiv:2408.00118
- [Test Time Compute by DeepMind](https://arxiv.org/abs/2408.03314) — arXiv:2408.03314
- [Vector + KG RAG](https://arxiv.org/abs/2408.04948) — arXiv:2408.04948
- [Gemma Scope](https://arxiv.org/abs/2408.05147) — arXiv:2408.05147
- [AI co-scientist: a multi-agent framework for accelerating scientific discovery](https://arxiv.org/abs/2408.06292) — arXiv:2408.06292
- [Web + RAG](https://arxiv.org/abs/2408.07611) — arXiv:2408.07611
- [Auxiliary-loss-free load balancing](https://arxiv.org/abs/2408.15664) — arXiv:2408.15664
- [Self-Improving Diffusion Models with Synthetic Data](https://arxiv.org/abs/2408.16333) — arXiv:2408.16333
- [OmniGen](https://arxiv.org/abs/2409.11340) — arXiv:2409.11340
- [CORE-Bench](https://arxiv.org/abs/2409.11363) — arXiv:2409.11363
- [NVLM](https://arxiv.org/abs/2409.11402) — arXiv:2409.11402
- [To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning](https://arxiv.org/abs/2409.12183) — arXiv:2409.12183
- [Qwen-2-VL](https://arxiv.org/abs/2409.12191) — arXiv:2409.12191
- [Language Models Learn to Mislead Humans via RLHF](https://arxiv.org/abs/2409.12822) — arXiv:2409.12822
- [Pixmo](https://arxiv.org/abs/2409.17146) — arXiv:2409.17146
- [Moshi](https://arxiv.org/abs/2410.00037) — arXiv:2410.00037
- [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201) — arXiv:2410.01201
- [Contextual document embeddings](https://arxiv.org/abs/2410.02525) — arXiv:2410.02525
- [Intelligence at the Edge of Chaos](https://arxiv.org/abs/2410.02536) — arXiv:2410.02536
- [2410.03617](https://arxiv.org/abs/2410.03617) — arXiv:2410.03617
- [Differential Transformer](https://arxiv.org/abs/2410.05258) — arXiv:2410.05258
- [Emergent properties with repeated examples](https://arxiv.org/abs/2410.07041) — arXiv:2410.07041
- [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095) — arXiv:2410.07095
- [Upcycling Large Language Models into Mixture of Experts](https://arxiv.org/abs/2410.07524) — arXiv:2410.07524
- [Decomposing the dark matter of sparse autoencoders](https://arxiv.org/abs/2410.14670) — arXiv:2410.14670
- [2410.15661](https://arxiv.org/abs/2410.15661) — arXiv:2410.15661
- [Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514) — arXiv:2410.18514
- [O1 Replication Journey: A Strategic Progress Report -- Part 1](https://arxiv.org/abs/2410.18982) — arXiv:2410.18982
- [pi0](https://arxiv.org/abs/2410.24164) — arXiv:2410.24164
- [Q Star 2.0](https://arxiv.org/abs/2411.07279) — arXiv:2411.07279
- [LlaVa-CoT](https://arxiv.org/abs/2411.10440) — arXiv:2411.10440
- [Marco-o1](https://arxiv.org/abs/2411.14405) — arXiv:2411.14405
- [LLMs Do Not Think Step-by-step in Implicit Reasoning](https://arxiv.org/abs/2411.15862) — arXiv:2411.15862
- [O1 Replication Journey -- Part 2: Surpassing O1-preview through Simple Distillation, Big Progress or Bitter Lesson?](https://arxiv.org/abs/2411.16489) — arXiv:2411.16489
- [FlashAttention on a Napkin: A Diagrammatic Approach to Deep Learning IO-Awareness](https://arxiv.org/abs/2412.03317) — arXiv:2412.03317
- [Apollo study of scheming AI models](https://arxiv.org/abs/2412.04984) — arXiv:2412.04984
- [CoCoNuT: Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) — arXiv:2412.06769
- [Memory Layers at Scale](https://arxiv.org/abs/2412.09764) — arXiv:2412.09764
- [Chinese Researchers Just CRACKED OpenAI's AGI Secrets!](https://arxiv.org/abs/2412.14135) — arXiv:2412.14135
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — arXiv:2412.19437
- [Titans](https://arxiv.org/abs/2501.00663) — arXiv:2501.00663
- [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575) — arXiv:2501.03575
- [rStar-Math: Small LLMs Can Master Math Too with Self-Evolving Deep Thinking](https://arxiv.org/abs/2501.04519) — arXiv:2501.04519
- [MiniMax [Text-01]](https://arxiv.org/abs/2501.08313) — arXiv:2501.08313
- [Kimi k1.5](https://arxiv.org/abs/2501.12599) — arXiv:2501.12599
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — arXiv:2501.12948
- [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) — arXiv:2501.19393
- [PolarQuant](https://arxiv.org/abs/2502.02617) — arXiv:2502.02617
- [Less is more for reasoning](https://arxiv.org/abs/2502.03387) — arXiv:2502.03387
- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171) — arXiv:2502.05171
- [The Danger of Overthinking](https://arxiv.org/abs/2502.08235) — arXiv:2502.08235
- [Less is more for RL scaling](https://arxiv.org/abs/2502.11886) — arXiv:2502.11886
- [Muon Is Scalable For Pre-training](https://arxiv.org/abs/2502.16982) — arXiv:2502.16982
- [AlphaEvolve: A coding agent for advanced algorithm discovery](https://arxiv.org/abs/2502.19398) — arXiv:2502.19398
- [The First Few Tokens Are All You Need: An Efficient and Effective Unsupervised Prefix Fine-Tuning Method for Reasoning Models](https://arxiv.org/abs/2503.02875) — arXiv:2503.02875
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.10494) — arXiv:2503.10494
- [GR00T N1](https://arxiv.org/abs/2503.14734) — arXiv:2503.14734
- [Why do LLMs attend to the first token?](https://arxiv.org/abs/2504.02732) — arXiv:2504.02732
- [Sleep-Time Compute](https://arxiv.org/abs/2504.13171) — arXiv:2504.13171
- [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) — arXiv:2504.13837
- [TurboQuant](https://arxiv.org/abs/2504.19874) — arXiv:2504.19874
- [Model Merging in Pre-training of Large Language Models](https://arxiv.org/abs/2505.12082) — arXiv:2505.12082
- [Harnessing the Universal Geometry of Embeddings](https://arxiv.org/abs/2505.12540) — arXiv:2505.12540
- [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939) — arXiv:2506.01939
- [Self-Adapting Language Models](https://arxiv.org/abs/2506.10943) — arXiv:2506.10943
- [Fast and Simplex: 2-Simplicial Attention in Triton](https://arxiv.org/abs/2507.02754) — arXiv:2507.02754
- [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524) — arXiv:2507.10524
- [Kimi K2](https://arxiv.org/abs/2507.20534) — arXiv:2507.20534
- [GPT-OSS](https://arxiv.org/abs/2508.10925) — arXiv:2508.10925
- [On the Theoretical Limitations of Embedding-Based Retrieval](https://arxiv.org/abs/2508.21038) — arXiv:2508.21038
- [Ling 2.0 Report](https://arxiv.org/abs/2510.22115) — arXiv:2510.22115
- [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) — arXiv:2510.25741
- [Kimi Linear](https://arxiv.org/abs/2510.26692) — arXiv:2510.26692
- [LeJEPA](https://arxiv.org/abs/2511.08544) — arXiv:2511.08544
- [H-Neurons](https://arxiv.org/abs/2512.01797) — arXiv:2512.01797
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) — arXiv:2512.02556
- [TTT-E2E](https://arxiv.org/abs/2512.23675) — arXiv:2512.23675
- [Recursive Language Models](https://arxiv.org/abs/2512.24601) — arXiv:2512.24601
- [DeepSeek Engram](https://arxiv.org/abs/2601.07372) — arXiv:2601.07372
- [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](https://arxiv.org/abs/2601.16163) — arXiv:2601.16163
- [Kimi K2.5](https://arxiv.org/abs/2602.02276) — arXiv:2602.02276
- [Attention Residuals](https://arxiv.org/abs/2603.15031) — arXiv:2603.15031
- [Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers](https://arxiv.org/abs/2604.07822) — arXiv:2604.07822
- [A Mechanistic Analysis of Looped RLMs](https://arxiv.org/abs/2604.11791) — arXiv:2604.11791
- [Parcae: Scaling Laws For Stable Looped Language Models](https://arxiv.org/abs/2604.12946) — arXiv:2604.12946
- [Kimi K3 tech report](https://arxiv.org/abs/2607.24653) — arXiv:2607.24653
- [Stealing Reasoning Traces from Proprietary LLM APIs](https://arxiv.org/abs/2608.09867) — arXiv:2608.09867

## Component table

`param` = capability per parameter, `flop` = per FLOP/VRAM, `ttc` = test-time compute, `data` = data efficiency, `ev` = evidence quality, `tr` = transfer, `cost` = 3090 GPU-hours.

| # | cluster | param | flop | ttc | data | ev | tr | cost | gain | priority |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | Chain of Thought | 40 | 70 | 90 | 20 | 100 | 100 | 0.25 | 58.5 | 1170.0 |
| 2 | Prompt Engineering | 25 | 85 | 55 | 65 | 100 | 100 | 0.25 | 56.5 | 1130.0 |
| 3 | In-Context Learning | 20 | 80 | 60 | 70 | 90 | 100 | 0.25 | 55.5 | 1061.7 |
| 4 | RAG | 55 | 85 | 40 | 60 | 100 | 100 | 0.50 | 61.0 | 862.7 |
| 5 | Tool Use | 40 | 82 | 55 | 45 | 88 | 100 | 0.50 | 57.1 | 765.1 |
| 6 | Embeddings | 40 | 80 | 30 | 55 | 100 | 100 | 0.50 | 51.8 | 731.9 |
| 7 | context engineering | 35 | 80 | 55 | 70 | 60 | 100 | 0.50 | 58.8 | 681.3 |
| 8 | Self-Consistency | 30 | 60 | 95 | 10 | 80 | 100 | 0.50 | 52.2 | 671.3 |
| 9 | Test-Time Compute | 75 | 75 | 95 | 20 | 84 | 100 | 1.00 | 71.8 | 664.8 |
| 10 | LLM-as-a-Judge | 30 | 55 | 80 | 40 | 76 | 100 | 0.50 | 51.5 | 648.6 |
| 11 | evals | 40 | 80 | 40 | 50 | 66 | 100 | 0.50 | 53.5 | 640.3 |
| 12 | Scaling Laws | 60 | 55 | 20 | 60 | 83 | 100 | 0.50 | 48.5 | 634.4 |
| 13 | DSPy | 40 | 80 | 55 | 75 | 43 | 95 | 0.50 | 61.0 | 621.5 |
| 14 | Retrieval | 50 | 85 | 40 | 60 | 64 | 80 | 0.50 | 59.5 | 619.4 |
| 15 | Vector Search | 35 | 80 | 30 | 50 | 100 | 80 | 0.50 | 49.5 | 616.0 |
| 16 | Quantization | 95 | 98 | 10 | 5 | 100 | 100 | 1.00 | 61.1 | 611.5 |
| 17 | AI agents | 45 | 75 | 60 | 45 | 100 | 100 | 1.00 | 57.8 | 577.5 |
| 18 | Planning | 45 | 75 | 60 | 40 | 70 | 70 | 0.50 | 57.0 | 570.3 |
| 19 | Sparse Autoencoders | 55 | 70 | 40 | 60 | 92 | 100 | 1.00 | 56.5 | 544.7 |
| 20 | BM25 | 20 | 95 | 20 | 40 | 42 | 60 | 0.25 | 45.5 | 510.6 |
| 21 | Beam search | 20 | 70 | 80 | 5 | 82 | 70 | 0.50 | 47.8 | 508.5 |
| 22 | KV Cache | 20 | 95 | 30 | 5 | 90 | 80 | 0.50 | 42.8 | 506.9 |
| 23 | ReAct | 40 | 80 | 65 | 40 | 31 | 80 | 0.50 | 58.2 | 499.3 |
| 24 | Vector Databases | 30 | 80 | 25 | 45 | 69 | 80 | 0.50 | 46.0 | 492.6 |
| 25 | Logit Lens | 35 | 80 | 30 | 50 | 62 | 75 | 0.50 | 49.5 | 492.4 |
| 26 | Reflection | 35 | 60 | 90 | 20 | 44 | 75 | 0.50 | 54.0 | 485.0 |
| 27 | CLIP | 45 | 85 | 30 | 60 | 97 | 80 | 1.00 | 55.5 | 482.5 |
| 28 | Synthetic data generation | 70 | 75 | 20 | 90 | 88 | 100 | 1.50 | 62.0 | 479.6 |
| 29 | Prompt Injection | 20 | 75 | 30 | 20 | 83 | 80 | 0.50 | 39.0 | 448.9 |
| 30 | LoRA | 85 | 85 | 5 | 15 | 100 | 100 | 1.50 | 54.5 | 445.0 |
| 31 | Small Language Models (SLMs) | 95 | 90 | 30 | 30 | 27 | 95 | 1.00 | 67.5 | 439.7 |
| 32 | Model routing | 40 | 90 | 25 | 10 | 25 | 95 | 0.50 | 46.8 | 424.9 |
| 33 | Flash Attention | 10 | 92 | 10 | 5 | 100 | 80 | 0.50 | 33.9 | 421.3 |
| 34 | Tokenization | 15 | 70 | 10 | 20 | 90 | 100 | 0.50 | 31.0 | 417.7 |
| 35 | Tree of Thoughts | 45 | 55 | 95 | 15 | 17 | 70 | 0.50 | 56.0 | 405.9 |
| 36 | LangGraph | 30 | 80 | 45 | 35 | 43 | 60 | 0.50 | 49.5 | 396.4 |
| 37 | Distillation | 95 | 90 | 15 | 55 | 100 | 100 | 3.00 | 67.5 | 389.7 |
| 38 | Whisper | 55 | 80 | 25 | 45 | 55 | 80 | 1.00 | 53.5 | 375.5 |
| 39 | MCP | 35 | 80 | 50 | 40 | 57 | 80 | 1.00 | 53.0 | 375.5 |
| 40 | Model merging | 90 | 90 | 10 | 20 | 66 | 55 | 1.00 | 59.5 | 367.6 |
| 41 | QLoRA | 95 | 95 | 10 | 10 | 67 | 100 | 2.00 | 61.0 | 366.6 |
| 42 | Mixture of Experts | 90 | 85 | 25 | 20 | 100 | 100 | 3.00 | 61.8 | 356.5 |
| 43 | Voice cloning | 60 | 75 | 30 | 40 | 77 | 80 | 1.50 | 54.0 | 347.3 |
| 44 | Mamba | 85 | 90 | 20 | 15 | 100 | 100 | 3.00 | 59.8 | 345.0 |
| 45 | Transformer | 40 | 60 | 30 | 10 | 100 | 80 | 1.00 | 39.0 | 343.2 |
| 46 | Sliding Window Attention | 60 | 85 | 10 | 10 | 92 | 80 | 1.50 | 47.5 | 329.3 |
| 47 | speech-to-text | 55 | 80 | 25 | 45 | 39 | 75 | 1.00 | 53.5 | 329.2 |
| 48 | Unsloth | 80 | 88 | 5 | 10 | 76 | 75 | 1.50 | 53.1 | 328.5 |
| 49 | DPO | 80 | 70 | 40 | 65 | 100 | 100 | 4.00 | 64.8 | 323.8 |
| 50 | Data Curation | 65 | 70 | 10 | 90 | 9 | 95 | 1.00 | 56.5 | 323.2 |

## How the score is computed

`gain = 0.30·parameter + 0.30·FLOP/VRAM + 0.25·test-time + 0.15·data`;
`priority = gain × (0.40 + 0.60·transfer/100) ÷ √max(cost_hours, 0.25) × (0.55 + 0.45·evidence/100) × 10`.

The square root keeps cheap-to-try techniques ahead without letting a 0.25 h prompt tweak outrank a 2 h QLoRA run by 24x.

Each profiled technique gets its four gain axes and its cost from the visible taxonomy in `airesearch/taxonomy.py` (change one integer and rebuild); an unknown family falls back to counting curated signals in its method strings, saturated at a declared cap. Evidence is always counted from the KB: arXiv papers, GitHub/Hugging Face links, numeric claims and channel diversity.

Caveat: the R2 extractor is noisy (see `docs/spot-checks.md`); several claims attached to a cluster are only co-located with it. The digest is a recall-first reading list, not a verified result.
