# Codacus spot-check — local 9B extraction

Deterministic audit of all **30** videos (100 %, the batch is small). `python3 -m codacus.spotcheck` reads `data/codacus/<id>.json`, the transcript and the video description; it never writes the JSON.

Checks (cheap screen): at least one numeric token in a claim appears in the transcript; every link appears in the transcript/description or is the video URL; the mechanism has 3-6 steps; at least one goal. A claim whose numbers do not ground at all is flagged; one grounded number can mask one invented number, so the manual review below is the backstop.

- videos: **30**, claims: **167**, links: **150**
- videos with at least one finding: **2/30**

| video | claims | links | mechanism | findings |
|---|---:|---:|---:|---|
| [0AqpaFm11oI](https://www.youtube.com/watch?v=0AqpaFm11oI) | 5 | 4 | 6 | — |
| [3mX_dc9ZNRA](https://www.youtube.com/watch?v=3mX_dc9ZNRA) | 6 | 6 | 6 | — |
| [6WLBmP-tZ0Q](https://www.youtube.com/watch?v=6WLBmP-tZ0Q) | 6 | 8 | 6 | — |
| [8F_5pdcD3HY](https://www.youtube.com/watch?v=8F_5pdcD3HY) | 6 | 3 | 6 | — |
| [9vY4-Z-tkHs](https://www.youtube.com/watch?v=9vY4-Z-tkHs) | 5 | 6 | 6 | — |
| [FIi2arm_DWM](https://www.youtube.com/watch?v=FIi2arm_DWM) | 6 | 9 | 6 | — |
| [Gunp20UgDFE](https://www.youtube.com/watch?v=Gunp20UgDFE) | 3 | 6 | 4 | — |
| [IH8XmxiwliQ](https://www.youtube.com/watch?v=IH8XmxiwliQ) | 6 | 3 | 6 | — |
| [IwN-eK1s8og](https://www.youtube.com/watch?v=IwN-eK1s8og) | 4 | 5 | 6 | — |
| [NIxCj-3fFNk](https://www.youtube.com/watch?v=NIxCj-3fFNk) | 6 | 3 | 6 | claim number not in transcript: 2.7 |
| [Oq0UdRy1LcM](https://www.youtube.com/watch?v=Oq0UdRy1LcM) | 6 | 1 | 4 | — |
| [SsUKTFSQoGM](https://www.youtube.com/watch?v=SsUKTFSQoGM) | 4 | 2 | 5 | — |
| [Tffg78mOgAk](https://www.youtube.com/watch?v=Tffg78mOgAk) | 6 | 13 | 6 | — |
| [VytSYCDhWQ0](https://www.youtube.com/watch?v=VytSYCDhWQ0) | 6 | 1 | 6 | — |
| [WkLdJ_1TKdY](https://www.youtube.com/watch?v=WkLdJ_1TKdY) | 6 | 4 | 6 | — |
| [Xuh6PzhqrgQ](https://www.youtube.com/watch?v=Xuh6PzhqrgQ) | 6 | 3 | 6 | — |
| [aYBgCl6ynGk](https://www.youtube.com/watch?v=aYBgCl6ynGk) | 6 | 6 | 6 | — |
| [akXT0DRvvK4](https://www.youtube.com/watch?v=akXT0DRvvK4) | 6 | 2 | 6 | — |
| [bcGO7xre46o](https://www.youtube.com/watch?v=bcGO7xre46o) | 6 | 4 | 6 | claim number not in transcript: 306, 3415; claim number not in transcript: 6 |
| [dypHkOKOLt8](https://www.youtube.com/watch?v=dypHkOKOLt8) | 6 | 3 | 6 | — |
| [fpvF4n32lsE](https://www.youtube.com/watch?v=fpvF4n32lsE) | 6 | 4 | 6 | — |
| [g_JG1Xbnz0g](https://www.youtube.com/watch?v=g_JG1Xbnz0g) | 6 | 8 | 6 | — |
| [k_LostFpatg](https://www.youtube.com/watch?v=k_LostFpatg) | 6 | 1 | 6 | — |
| [kpMLDIj6Ls8](https://www.youtube.com/watch?v=kpMLDIj6Ls8) | 4 | 4 | 6 | — |
| [oh50KFF8A_0](https://www.youtube.com/watch?v=oh50KFF8A_0) | 6 | 11 | 6 | — |
| [pIN-2oVJpyU](https://www.youtube.com/watch?v=pIN-2oVJpyU) | 6 | 11 | 6 | — |
| [rBLWDJrXCp0](https://www.youtube.com/watch?v=rBLWDJrXCp0) | 6 | 3 | 6 | — |
| [shJLycXjhVY](https://www.youtube.com/watch?v=shJLycXjhVY) | 6 | 8 | 6 | — |
| [xFEcAGB5kyg](https://www.youtube.com/watch?v=xFEcAGB5kyg) | 4 | 5 | 6 | — |
| [xbedfuqYQYA](https://www.youtube.com/watch?v=xbedfuqYQYA) | 6 | 3 | 6 | — |

<!-- manual -->
## Manual deep review

The deterministic pass above checks tokens, not meaning. I read all 30 transcripts
while doing the source pass and checked five in full against their JSON (17 %,
above the 10 % floor): `pIN-2oVJpyU`, `bcGO7xre46o`, `NIxCj-3fFNk`,
`kpMLDIj6Ls8`, `3mX_dc9ZNRA`.

| video | verdict | what the model got right | miss |
|---|---|---|---|
| `pIN-2oVJpyU` Colibrì | 5/6 | 43 layers x 6 experts + 1 shared = 258 fetches/token; ~12.5 MB per expert; 3.2 GB/token; 71 % cache hit; 162 GB model on 61 GB RAM; `-nr`/`--no-repack` is the flag that makes llama.cpp open it | conflates GLM 5.2's **16 GB minimum** with the 9.9 GB **resident** banner; omits that llama.cpp only failed because repacking allocates 147 GB, the single actionable finding |
| `bcGO7xre46o` Jev | 4/6 | one token per schema field, score only the allowed-token logits, KV cache holds the schema so fields batch, confidence per field, weights unchanged | says "single pass" for the whole object (it is one pass per batch of field questions); invents **306 ms / 3,415 ms** (transcript: "300 milliseconds against 3 and 1/2 seconds"); the one repo link was 404 and is now the description's real fork |
| `NIxCj-3fFNk` Laguna | 5/6 | 118B total / 8B active, `NCPU-MoE-46`, IQ4XS 57.6 GB, 12 GB VRAM + 62 GB usable RAM, 80K context | invents **"2.7x compute per token"** (not in the video) and **"49 GB system RAM"** (video says 62 GB usable) |
| `kpMLDIj6Ls8` MTP | 6/6 | 1.65x dense MTP with 92.8 % acceptance, 1.23x MoE with 84.4 %, ~30 % prefill cost, ~900 MB KV for the head | none |
| `3mX_dc9ZNRA` TurboQuant | 6/6 | CSA 4x + HCA 128x + 128-token window, Walsh-Hadamard + 1-bit correction, 3.5 bits, 4-5x, 10 % KV / 27 % compute | "4TB -> 500MB" is a rounded restatement; both numbers occur in the transcript |

Overall: **167 claims**, 2 videos with a numeric miss (one is a word-form false
positive: "six times" written as "6"). The systematic miss is not numbers but
**actionability**: the 9B names the mechanism and drops the exact flag or
precondition that makes it work (`--no-repack`, the real fork branch, batching
per field). `9vY4-Z-tkHs` is the worst mechanism miss: it says "MoE models fail"
but the video measured 17 tok/s on the 35B MoE, then 6 tok/s **with DFlash on**
because the target was offloaded; DFlash only wins when the target is fully
resident (the 8B went 12 -> 40 tok/s, 3.3x). The link table was worse before
assembly: the 9B invented one project URL and re-derived two correct ones;
`codacus.assemble.ground_links` now keeps only URLs from the transcript or
description, so 150 links across 30 JSONs are real.
