# Strata-safe — can we run Qwen3.8-Flash-Next (DavidAU GGUF) on the 3090?

**Status:** research + offline front-end build done; the CUDA engine build needs a
toolchain this sandbox does not have (exact command in `RUNNER_TODO.md`). **No model
weights were downloaded.** All model facts below come from the Hugging Face API
metadata, the model cards, and the ISTA RCO allocation text files — never from a
`.gguf` payload.

**Short answer.** The DavidAU repo is one IQ3_S file set (83.86 GB) with **no MTP
draft head and no Strata pack**, so the safe (offline) Strata fork cannot run it as
published. The runnable safe path is **IQ3_XXS (75.8 GB) + the MTP head + a pack
rebuilt from the GGUF**, which needs **62 GB of RAM** and gives roughly **59–77
tok/s** decode and **1,200–1,600 tok/s** prefill on this 3090. At the current
**30 GB** the full-size Strata quants are below Strata's own RAM floor; the box can
still run IQ2_XS at ~48–57 tok/s with llama.cpp. **+32 GB is worth it for Strata,
not for llama.cpp.** A 22 GB alternative (Nemotron 3.5 Bolt 74B-A9B) fits the 24 GB
card today and is the cheaper experiment.

---

## 1. Model card facts

### 1.1 The DavidAU repo

`DavidAU/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF`
(last modified 2026-10-17, 2,565 downloads, 27 likes).

| Fact | Value | Source |
|---|---|---|
| Base model (declared) | `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF` | HF `cardData.base_model` |
| Base of the base | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` → `Qwen/Qwen3.8-Flash-Next` | SC117 card, ISTA card |
| Total parameters | **180B** in the card's "FIREPOWER" line; the page header says **120B**; the architecture is **125B core + 51B n-gram + 4B MTP** | DavidAU card; Qwen card |
| Active parameters | **6B** activated (10 routed + 1 shared expert) | Qwen card |
| Experts | **512 routed per layer, 48 layers, 10 routed + 1 shared per token**, expert intermediate dim 640 | Qwen card, ISTA card |
| Quantizations in the repo | **only IQ3_S**, split: `...-IQ3_S-00001-of-00002.gguf` **55.06 GB** + `...-00002-of-00002.gguf` **28.80 GB** (total **83.86 GB**) + `mmproj-...-BF16.gguf` **0.91 GB** | HF file list |
| MTP draft head | **absent** | HF file list |
| Strata pack | **absent** | HF file list |
| Context length | 262,144 native, extensible to 1,000,000 (official model); the DavidAU card does not add a different number | Qwen card |
| Chat template | Qwen's ChatML-style template, 8,952 chars in `tokenizer_config.json`: `<|im_start|>`/`<|im_end|>`, thinking blocks, tool calls, vision tokens (`<|vision_start|><|image_pad|><|vision_end|>`) | official `tokenizer_config.json` |
| Licence declared by DavidAU | `apache-2.0` | HF `cardData.license` |
| Actual base licence | **Qwen Community License 1.0** (`license: other`, `license_name: qwen-community-1.0`) | `Qwen/Qwen3.8-Flash-Next` card |
| Who made the abliteration | **SC117** transplanted 144 residual-writing tensors (all 48 layers) from `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` into ISTA's already-quantized GGUF; 95 changed quant type (to Q8_0), 49 kept the upstream type; per-tensor blake2b verified | SC117 card |
| DavidAU's own contribution | repackaging SC117's IQ3_S + the **"Turbo Brilliance" prompt/construct system** (27 reasoning + 27 instruct modes, 15 constructs) + the `mmproj`; **no weight training** is claimed | DavidAU card |

**Marketing claims, marked unverified:** "50x reasoning", "Turbo Brilliance",
"27 reasoning modes", "15 AI Constructs", "SOTA BF16 performance in IQ3_S",
"17 t/s on 46 GB across 2 GPUs + RAM", "60–90 t/s at full GPU offload",
"150 t/s on an RTX 6000", "STRATA will be even faster / can be used on 12 GB cards".
None were measured here; the 3090 numbers in §3 are anchored on independent 3090
measurements, not on this card.

The model card is internally inconsistent on size (120B vs 180B). The consistent
reading is **125B core + 51B n-gram table + 4B MTP = 180B total, 6B active**;
"177B" in the videos counts 125 + 51 + 1 and ignores the rest of the draft head.
The "120B" header is wrong.

### 1.2 The official Qwen release

`Qwen/Qwen3.8-Flash-Next` exists (2.46M downloads, 6.3k likes, 2026-08-27):

| Fact | Value |
|---|---|
| Parameters | 125B, **6B activated**, plus 51B n-gram embedding and 4B MTP |
| Experts | 512 routed, 10 routed + 1 shared per token, 48 layers, intermediate 640 |
| Context | **262,144 native**, 1,000,000 with YaRN |
| Licence | **Qwen Community License 1.0** (custom; `license: other`) |
| Weights | 131 safetensors shards, ~360 GB BF16, `mmproj`-style vision tower |
| Chat template | Qwen ChatML + thinking/vision/tool blocks (8,952 chars) |

### 1.3 What "GSQ" and "RCO" actually are

They are **quantization methods, not custom file formats**. The files are ordinary
GGUF v3 and run in stock `llama.cpp` / Ollama per ISTA:

- **GSQ** (Gumbel-Softmax Quantization, arXiv 2604.18556): learns per-coordinate
  grid assignments and per-group scales; output is standard scalar GGUF types.
- **RCO** (Riemannian Constrained Optimization, arXiv 2605.00649): chooses one
  standard GGUF type per tensor under a size budget. GGUF cannot express per-expert
  types, so the choice is per layer.

Evidence without opening a `.gguf`: ISTA publishes
`tensor-allocation/...-IQ3_XXS-...rco-allocation.txt`. Its histogram is entirely
standard types:

```
BF16=484, F16=1, F32=292, IQ2_S=20, IQ2_XS=20, IQ2_XXS=18,
IQ3_S=82, IQ3_XXS=12, IQ4_NL=58, IQ4_XS=67, Q2_0=38, Q4_K=45,
Q5_K=15, Q6_K=71          (1,223 tensors; IQ3_XXS build)
```

The 28.80 GB second shard is the `per_layer_token_embd` table at IQ4_NL (51.2B
parameters, a lookup, not a matmul). It is the same file in all four ISTA tiers and
in SC117. **This is why the 28.8 GB shard can live on the SSD**: one row is read per
token, and reads do not wear the drive.

### 1.4 The four official quants (and the abliterated copies)

| Build | Download | Shard 1 (weights) | Shard 2 (n-gram) | Experts in RAM arena | Upstream RAM floor |
|---|---:|---:|---:|---:|---:|
| `Q2_0` | 66.4 GB | 37.62 GB | 28.80 GB | 34.0 GB | 48 GB |
| `IQ2_XS` | 68.0 GB | 39.23 GB | 28.80 GB | 35.5 GB | 48 GB |
| `IQ3_XXS` | 75.8 GB | 47.04 GB | 28.80 GB | 42.9 GB | **60 GB** |
| `IQ3_S` | 83.6 GB | 54.82 GB | 28.80 GB | 50.3 GB | **62 GB** |
| SC117 `IQ3_XXS` (abliterated) | 76.14 GB | 47.34 GB | 28.80 GB | ~43.2 GB | 60 GB |
| SC117 `IQ3_S` (abliterated) | 83.86 GB | 55.06 GB | 28.80 GB | ~50.5 GB | 62 GB |
| DavidAU `IQ3_S` | 83.86 GB | 55.06 GB | 28.80 GB | ~50.5 GB | 62 GB |

SC117 adds `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (4.14 GB), `mtp-q2_0.gguf` (0.89 GB)
and `strata/rt/{experts.bin,dense.bin}` (0.83 GB). DavidAU adds neither. The MTP
head is **stock base, not abliterated** (SC117 card).

---

## 2. Our box, measured

| Item | Measured value | How |
|---|---|---|
| GPU | RTX 3090 24 GB (GA102, `01:00.0`) | `lspci` |
| PCIe link | **16.0 GT/s ×16 = PCIe 4.0 x16** (~64 GB/s raw, ~55 GB/s practical) | `/sys/bus/pci/devices/0000:01:00.0/current_link_{speed,width}` |
| RAM now | **30 GB** (31,986,000 kB = 30.5 GiB); 13 GB used, 12 GB free, 5 GB cache | `/proc/meminfo`, `free -g` |
| RAM after +32 GB | **62 GB usable** (64 GB installed) | purchase target |
| CPU | Ryzen 9 7950X, 32 threads | `lscpu`, `nproc` |
| Disk | Samsung 980 PRO 1 TB, ext4, **335 GB free** | `df -h ~` |
| NVMe sequential | **4.0 GB/s** (2 GiB `dd iflag=direct`) | `dd` |
| NVMe 1 MiB random | **3.97 GB/s** (300 reads) | `O_DIRECT` `preadv` |
| NVMe 64 KiB random | **614 MB/s**, 9,370 IOPS | same |
| NVMe 4 KiB random | **51.8 MB/s**, 12,648 IOPS | same |
| Sidecars to reserve | ~1.5 GB VRAM (Laya/xlmr) → **22.5 GB VRAM usable** | project convention |
| OS/services reserve | ~8 GB RAM → **22 GB usable now, 54 GB after** | project convention |

The expert blobs are ~1.4–2.1 MB each, so the relevant NVMe number is the **1 MiB
random** figure (~4 GB/s), not 4K. That is the number used in §3.

---

## 3. Fit math for 30 GB and 62 GB

### 3.1 Formula

For quant *q* with expert-arena size `E_q` (GB), 24,576 routed experts total
(48 layers × 512), 10 routed per layer per token:

```
expert bytes touched per token      e_q = E_q × 10/512 = E_q / 51.2      [GB/token]

VRAM expert cache                   C = 24 − 1.5 (sidecars) − 4.2 (dense+KV+session)
                                      = 18.3 GB
cache fraction                      f_q = C / E_q
hit rate                            h_q = f_q^0.12    (clamped to 0.97)

bytes/token from RAM (CPU misses)   B_ram  = 0.75 × (1−h_q) × e_q
bytes/token over PCIe (GPU misses)  B_pcie = 0.25 × (1−h_q) × e_q
bytes/token from NVMe (30 GB)       B_nvme = (1−h_q) × max(0, 1 − R/E_q) × e_q
                                      R = 22 GB usable RAM now, 54 GB after

decode ceiling                      T ≤ min( BW_ram/B_ram, BW_pcie/B_pcie,
                                             BW_nvme/B_nvme, measured compute anchor )
BW_ram ≈ 60 GB/s (DDR5 dual-channel spec, not measured here; the RAM
ceiling is far above the observed tok/s, so its exact value does not change the
verdict), BW_pcie ≈ 55 GB/s,
BW_nvme = 3.97 GB/s (measured 1 MiB random)
```

The `0.12` exponent is calibrated on two published points: the Strata explainer's
12 GB card (2,860 slots = 11.6% of 24,576 → 66% hit) and llama.cpp PR #29887's
24 GB/5090 runs (17–28% of experts in cache → 77–89% hit). It gives:

| Quant | `E_q` | cache fraction | `h_q` | `e_q` (GB/token) |
|---|---:|---:|---:|---:|
| Q2_0 | 34.0 GB | 53.8% | 92.8% | 0.664 |
| IQ2_XS | 35.5 GB | 51.5% | 92.4% | 0.693 |
| IQ3_XXS | 42.9 GB | 42.7% | 90.3% | 0.838 |
| IQ3_S | 50.3 GB | 36.4% | 88.6% | 0.982 |

At these hit rates the RAM and PCIe tiers are **not** the limit at 60–100 tok/s:
for IQ3_XXS at 60 tok/s, RAM carries ~5.6 GB/s and PCIe ~1.9 GB/s, well under 60 and
55 GB/s. Decode is limited by CPU expert-math, kernel efficiency and the miss path,
which is why the table uses **independent 3090 measurements** as anchors: 87 tok/s
for IQ2_XS and 59 tok/s for IQ3_S (alesha-pro's Strata pack card, one RTX 3090,
32K prompt), interpolated for IQ3_XXS and Q2_0. These are measurements of the same
engine family, not of this exact box; treat them as ±20%.

### 3.2 The table

`Strata fits?` means the expert arena fits the **pinned-RAM** floor (`E_q ≤ R`);
`llama.cpp fits?` means the experts fit in **VRAM cache + RAM** (`E_q ≤ C + R`),
which is how `--n-cpu-moe`/`-ot` and `--moe-cache-mib` split them.

| Quant | Download | Experts `E_q` | Strata at 30 GB? | Strata at **62 GB**? | llama.cpp at 30 GB? | Est. decode 30 GB (llama.cpp) | Est. decode **62 GB** (Strata) | Est. prefill **62 GB** (Strata, 32K) |
|---|---:|---:|---|---|---|---:|---:|---:|
| Q2_0 | 66.4 GB | 34.0 | no (floor 48) | **yes**, all in RAM | **yes** (34.0 < 39.3) | 55–65 tok/s | **90–105 tok/s** | 1,500–2,000 tok/s |
| IQ2_XS | 68.0 GB | 35.5 | no (floor 48) | **yes**, all in RAM | **yes** (35.5 < 39.3) | 48–57 tok/s | **78–91 tok/s** | 1,300–1,800 tok/s |
| IQ3_XXS | 75.8 GB | 42.9 | no (floor 60) | **yes**, 42.9 < 54 (floor 60) | no (3.6 GB spill) | 33–40 tok/s | **59–77 tok/s** | 1,200–1,600 tok/s |
| IQ3_S | 83.6 GB | 50.3 | no (floor 62) | **edge**, 50.3 < 54 but 62 is the stated minimum | no (11.3 GB spill) | 22–32 tok/s | **46–62 tok/s** | 1,000–1,400 tok/s |
| DavidAU IQ3_S | 83.86 GB | ~50.5 | no | edge, but **safe fork won't accept it** (no MTP, IQ3_S) | no | 22–32 tok/s | — | — |

**Forced low-RAM Strata (not supported):** if the file tier is forced at 30 GB,
the NVMe ceilings are ~236 tok/s (Q2_0), ~197 (IQ2_XS), ~100 (IQ3_XXS), ~63
(IQ3_S); real decode would be far lower because of CPU work and queueing. Estimate
20–35 tok/s for IQ3_XXS and 15–25 for IQ3_S, with an unstable start. Do not plan on
this; it is below Strata's own RAM floor for every full-size quant.

**One sizing conflict to resolve on the first run.** UnbiasedStrata's README says
Qwen3.8-Flash-Next needs **128 GB RAM**; upstream's own `MODELS` table says
**60 GB** for IQ3_XXS and **62 GB** for IQ3_S, and the 12 GB/64 GB video
demonstrates the model running in 64 GB. The engine is upstream's planner, so the
128 GB line looks conservative, but it is untested here. Rehearse with
`--ram-cap 54` and read the startup plan / `/health` before relying on 62 GB. If
the fork refuses, the 62 GB fallback is llama.cpp IQ3_XXS at ~40–48 tok/s.

### 3.3 Prefill

Strata's prefill trick is a big chunk (6,144 tokens on the 12 GB card; more VRAM
allows 8,192+) so non-resident experts cross PCIe once per chunk instead of once per
512-token batch. The ceiling for a 32K prompt is:

```
chunks = ceil(32768 / 8192) = 4
PCIe bytes = 4 × E_q
IQ3_XXS:   4 × 42.9 = 171.6 GB / 55 GB/s = 3.1 s  → 10,500 tok/s ceiling
```

The measured 1,200–1,600 tok/s is the compute/queueing limit, not the bus. From
NVMe at 30 GB the same 171.6 GB / 3.97 GB/s = 43 s → ~760 tok/s ceiling, so
prefill would be 3–6x slower; that is the strongest argument for +32 GB.

### 3.4 SSD wear

Reads do not wear an NVMe drive. Inference here is read-only on the model: the
28.8 GB n-gram shard is memory-mapped (`-lm mmap --lazy-mode on` in llama.cpp;
`--ple-io direct` in Strata) and one row is read per token. Writes are: the
one-time model download (76–84 GB), the one-time pack build (~1.5 GB), the MTP pack
(~0.8 GB), and logs. Strata pins RAM with `mlock`, so it does not swap the model;
keep swap off or small. No engine in this comparison writes model data during
inference.

---

## 4. Strata, safely

### 4.1 Upstream: `Niko1221/Strata`

MIT, 26,807 stars, created 2026-09-24, 2,815 forks (API, 2026-10-17). It is a real,
fast single-model engine — and an installer that assumes network and trust.

| Check | Finding | Evidence |
|---|---|---|
| Licence | MIT | `LICENSE`, GitHub API `spdx_id: MIT` |
| Install method | README tells you to **paste a prompt into an AI coding agent** | `README.md`, "Let your AI set it up" |
| Network at install | downloads a standalone Python with `curl` | `setup.sh:35-37` (`python-build-standalone` release) |
| Network at install | `sudo apt/dnf/pacman` to install Python/pip | `setup.sh:47-52` |
| Network at install | `pip install numpy jinja2 regex pyyaml tqdm requests cmake ninja pillow psutil` into `.venv` | `setup.py:199` (`PY_PACKAGES`) |
| Network at install | downloads a **prebuilt engine** from GitHub `releases/latest` | `setup.py:175` (`PREBUILT_URL`) |
| Network at install | downloads the pinned llama.cpp source zip as a fallback | `setup.py:168` (`LLAMA_CPP_ZIP`) |
| Network at install | downloads the model from Hugging Face / ModelScope | `setup.py:75-133`, `MODELS` table |
| Windows first run | PowerShell `Invoke-WebRequest` downloads `python-3.12.10-amd64.exe` | `START-HERE.bat:20` |
| Update | `update.sh` does `git pull`; `UPDATE.bat` the same; **no background updater found**, but `START-HERE` re-runs `setup.py` and `PREBUILT_URL` is `releases/latest`, so starting it can silently upgrade the engine | `update.sh:2-11`, `UPDATE.bat`, `setup.py:175` |
| Prebuilt binaries **in the tree** | none committed; the installer fetches them | `find` found scripts only |
| Telemetry | no external telemetry endpoint found; `serve/telemetry.py` samples the local GPU for `GET /metrics` | `serve/server.py:3657-3669` |
| Runtime network | local OpenAI/Anthropic server on `127.0.0.1:8080`; no outbound call in the serving path | README, `serve/server.py` |

**Verdict on upstream: do not run it.** The build is not reproducible offline, the
installer downloads and executes a prebuilt binary, and the model download is a
separate 76–84 GB fetch. The code may be fine; the install path is the problem.

### 4.2 The reviewer's offline fork: `robertkarl/UnbiasedStrata`

The review video `i18vpg0T-mk` names it and links it. Pinned here at
`b124b59ce48e4d53abd14d1fd08979350fc27358` (2026-10-13), based on upstream
`6f32ec07` (engine 0.1.39), with llama.cpp submodule
`3cf03257f219afbe7334045ff7c6a06ac68c627d`.

| Check | Finding | Evidence |
|---|---|---|
| Licence | MIT, upstream copyright retained | `LICENSE` |
| Build network | **none**: CMake uses the local submodule; the only `patch` is local | `Makefile`, `CMakeLists.txt:1148-1167` |
| Runtime network | none; one binary serves on `127.0.0.1:8080` | `frontend/main.cpp` |
| Python | removed; C++ tokenizer, chat template, HTTP front end | `UPSTREAM.md` |
| Prebuilt binaries | none; `make build` compiles everything | `Makefile` |
| Auto-update | none | `grep` over the tree |
| Telemetry | none | `grep` over `src/`, `frontend/` |
| One network tool remains | `tools/mtp_fetch.py` uses `urllib` to fetch the **MTP head ranges** from the official checkpoint; it is a one-time prep tool, **not** called by build or run | `tools/mtp_fetch.py:26-35`, `UPSTREAM.md` |
| Patch to llama.cpp | one patch, `patches/llama.cpp-qwen3-coder-any-argument-order.patch`: the Qwen3-Coder tool-call grammar accepts arguments in any order. Applied at configure/build time with `patch`, not fetched | `patches/`, `UPSTREAM.md` |
| Engine changes vs upstream | three, all for slow/remote stores: `robust_read.hpp` (retrying positional reads), `expert_source.cpp` (progress/errors), `ngram.cpp` (`--ple-io ram`) | `UPSTREAM.md` |

**Verdict: use the fork.** It is the same MIT engine without the installer, the
Python layer, the prebuilt binary, the auto-update and the network build.

### 4.3 What we vendored and what we removed

`vendor/strata/` is the fork at the commit above, 49 MB, built from a local clone.
**No source code was changed** except one build line (below). The patch list:

1. **Fork's own removal** (already in UnbiasedStrata): Python installer and
   launchers, Python server and web app, Python tools, Intel SYCL port, Docker
   files, benchmark results, test suite, translations and all docs except
   `docs/upstream/DETAILS.md` and the paper. This is the fork's stated diff against
   upstream, not ours.
2. **Our build fix:** `Makefile` applied the llama.cpp patch with `git -C
   third_party/llama.cpp apply`. A vendored snapshot has no `.git`, so that would
   fail. Replaced with the equivalent `patch -p1 -d third_party/llama.cpp -i
   patches/...` (the CMake path already used `patch`). One line; no behaviour change.
3. **Removed for the repo rules:** 19 tiny tokenizer `.gguf` test fixtures under
   `third_party/llama.cpp/models/` (not weights), the llama.cpp docs/models/
   examples/tests/benches/scripts/media/conversion dirs, `.github`, `.devops`,
   `AUTHORS`, `pyproject.toml`, and local-path-bearing port docs. None is used by
   `make build`.
4. **Added:** `VENDORED.md` (provenance), `BUILD-OFFLINE.sh` (the bwrap build),
   `.gitignore` for `build/`. No network call, no prebuilt binary, no `.gguf`.

### 4.4 Build status

The front end and its protocol tests build **offline** in this sandbox:

```
$ bwrap --unshare-net --ro-bind / / --bind "$PWD" "$PWD" --tmpfs /tmp \
        --dev /dev --proc /proc --chdir "$PWD" make test-frontend
...
PASS: normal exit and watchdog-style abort reaped and diagnosed
PASS: formatter exceptions and client cancellation drain the old request before the next request
PASS: large responses cannot fill an abandoned pipe or leak into the next request
exit=0
```

The CUDA engine could not be compiled **here**: this sandbox has no `cmake` and no
`nvcc`/CUDA toolkit (`cmake: command not found`, `nvcc: command not found`). The
source is complete; run `vendor/strata/BUILD-OFFLINE.sh` on a host with CMake ≥3.24
and CUDA 12.x (sm_86 for the 3090). The exact command is in `RUNNER_TODO.md`.

### 4.5 How to run it (after the build)

```sh
# one-time, from the model dir (network; the only download step)
#   SC117 IQ3_XXS shards 1+2, mtp-q2_0.gguf, strata/rt/*, mmproj
# one-time pack build (no network; Python 3 + numpy)
STRATA_GGUF_PY=vendor/strata/third_party/llama.cpp/gguf-py \
python3 vendor/strata/tools/iq_pack.py \
  --gguf Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf \
  --out pack-sc117-iq3-xxs
cp vendor/strata/data/expert-profile.bin pack-sc117-iq3-xxs/
# serve
vendor/strata/build/unbiased-strata \
  --model Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf \
  --pack pack-sc117-iq3-xxs --mtp mtp \
  --mmproj mmproj-Qwen3.8-Flash-Next-BF16.gguf \
  --ctx 32768 --port 8080 -- --kv q4_0 --ple-io direct
```

The engine reads shard 2 beside shard 1; `--ple-io direct` streams the 28.8 GB
n-gram table from the SSD. `--ram-cap`/`--vram-cap` let you rehearse a smaller box.

---

## 5. Does the safe Strata accept this model?

**The converter.** `tools/iq_pack.py` reads standard GGUF, supports `Q2_0`,
`IQ2_XS` and `IQ3_XXS` (the fork's documented set), stores each layer's actual quant
types, and skips the 28.8 GB `per_layer_token_embd` (the engine reads it from the
GGUF). It records source hashes in `conversions.json` but **does not compare them to
a known model hash**, so a finetune with the same tensor names/shapes is accepted.
The abliterated SC117 IQ3_XXS has exactly the upstream tensor names and types
(SC117 replaced 144 tensors in place and restored the upstream type assignment), so
the pack build should work. The fork's README says "no other model, quantization or
pruned variant is supported": that is a support statement, not a hard check, and
SC117's own card documents the pack rebuild for Strata.

**The blockers for the DavidAU file specifically:**

1. **Quant:** DavidAU ships **IQ3_S**. The safe fork documents and tests only
   **IQ3_XXS** (its `README.md`). Upstream Strata supports IQ3_S; the fork does not.
2. **MTP:** Strata **requires** the MTP draft head ("The draft (MTP) head is
   required", fork README). DavidAU's repo has **no MTP file**. SC117's repo has
   `mtp-q2_0.gguf` + `strata/rt/`; the official route is `tools/mtp_fetch.py`
   (network, one-time, ~5 GB of ranges) → `mtp_pack.py` → `mtp_rt.py`.
3. **Pack:** even for SC117 IQ3_XXS you must rebuild the pack; the official pack's
   offsets do not match the transplanted file (SC117 card, `ArenaExpertSource: the
   pack's native_experts.txt does not match the model`).

**Therefore:** the DavidAU 83.86 GB file is **not runnable on the safe fork as
published**. The file that is runnable is:

- **SC117 `IQ3_XXS`** (76.14 GB: 47.34 + 28.80) + `mtp-q2_0.gguf` (0.89 GB) +
  `strata/rt/` (0.83 GB) + `mmproj` (0.91 GB) + a rebuilt pack; or
- **ISTA `IQ3_XXS`** (75.8 GB) + an MTP head built from the official checkpoint +
  a rebuilt pack (the official, non-abliterated weights).

If you specifically want the DavidAU **prompt system**, that is text: keep the
official/SC117 weights and copy the system prompts. The weights are not DavidAU's
work. If you specifically want IQ3_S on the safe fork, that needs a fork patch and
a new pack — not validated here, and 62 GB leaves no headroom.

---

## 6. Alternatives with llama.cpp (pinned `f3363593`) and ik_llama.cpp

### 6.1 The pin already has the MoE expert cache

Our pinned llama.cpp is `f3363593f457169cdfbdfe22ea399b1cc139ba18`, committed
**2026-10-16**. It is **217 commits ahead** of the merge commit of **PR #29887,
"llama : add a GPU cache for MoE experts kept in host memory"** (`d6cf9ac`,
merged 2026-10-07). So the pin **contains** the expert cache. It also knows the
architecture: `LLM_ARCH_QWEN4EXP` / `"qwen4exp"` is in `src/llama-arch.cpp` at the
pin. The GSQ-RCO files are standard GGUF, so no conversion is needed for llama.cpp.

Enable the cache with `--moe-cache-mib N` (PR body: about 10% of the expert size is
a good start). The PR's own 24 GB numbers, Qwen3.8 Q4_0 (65.4 GB of experts):

| GPU / config | experts in VRAM | decode | hit rate |
|---|---|---:|---:|
| RTX 4090, `--fit` (no cache) | 16.5 GB whole layers | 25.0 tok/s | — |
| RTX 4090, `--fit --moe-cache-mib 6544` | 10.4 + 6.4 GB | 39.4 tok/s | 72% |
| RTX 4090, `-cmoe --moe-cache-mib 11000` | 0 + 10.7 GB | 40.7 tok/s | 77% |
| RTX 5090, `-cmoe --moe-cache-mib 19000` | 0 + 18.6 GB | 67.8 tok/s | 89% |

### 6.2 Fit math with llama.cpp

llama.cpp can put experts in **VRAM cache + RAM**, not only pinned RAM, so it runs
on the 30 GB box where Strata cannot:

```
fits  ⇔  E_q ≤ (22.5 − 4.2) + (RAM_total − 8)
30 GB:  34.0 (Q2_0) and 35.5 (IQ2_XS) fit; 42.9 (IQ3_XXS) spills 3.6 GB;
        50.3 (IQ3_S) spills 11.3 GB.
62 GB:  all four fit with room.
```

A tested 30 GB-class recipe is the SSD video `_TCRH725hAM`: 24 GB VRAM (8 + 16 GB)
+ 32 GB RAM, IQ4_XS 85 GB (45.8 GB core + 39.1 GB n-gram), `--n-cpu-moe 32`,
4-bit KV, 100K context, **22 tok/s** decode and 14 tok/s at 76K. It streams the
n-gram table from NVMe with `-lm mmap --lazy-mode on` and `load-mode none`. That is
the proof that the SSD tier works; it is not fast.

For **our** box the best 30 GB llama.cpp pick is **IQ2_XS** (core 39.67 GB fits in
22.5 GB VRAM + 22 GB RAM; n-gram on SSD): estimated **48–57 tok/s** decode,
150–300 tok/s prefill (stock llama.cpp lacks Strata's sparse-attention prefill and
MTP; the review's hand port reached 620 tok/s on a 17K prompt, but that port is not
in our pin). IQ3_XXS at 30 GB spills 3.6 GB and gives ~33–40 tok/s. At 62 GB,
llama.cpp IQ3_XXS gives maybe 40–48 tok/s — Strata at the same RAM gives 59–77,
which is the real reason to use Strata.

### 6.3 ik_llama.cpp

`ikawrakow/ik_llama.cpp` HEAD (pushed 2026-10-17) knows `qwen4exp` in
`src/llama-arch.cpp`, but its expert-offload interface is the older `-ot` /
`--n-cpu-moe` style; no `--moe-cache-mib` equivalent was found. It is a fallback,
not the primary: it would need its own build and A/B, and it does not have Strata's
prefill kernels.

### 6.4 Engine comparison

| | Strata (safe fork) | llama.cpp pin f3363593 | ik_llama.cpp HEAD | Colibri |
|---|---|---|---|---|
| Licence | MIT | MIT | MIT (fork) | Apache-2.0 |
| Models | Qwen3.8-Flash-Next IQ3_XXS (fork) / four quants (upstream); Qwen3.6-35B-A3B | many, incl. qwen4exp and Nemotron-H | many, incl. qwen4exp | Qwen3.8 family / 284B-class |
| Expert placement | hot-expert VRAM cache + pinned RAM + 3-way split (GPU hits, PCIe misses, CPU misses); no batching | whole layers + `--moe-cache-mib` LRU expert cache | whole layers / `-ot` | mmap + page faults + expert budget |
| Prefill | big chunks (6K–8K), sparse QSA, MTP | stock dense QSA mask, U=512/2048; no MTP for this arch | stock | mmap-bound |
| Measured anchor | 87 tok/s IQ2_XS, 59 tok/s IQ3_S (3090, 32K) | 40.7 tok/s Q4_0 on 4090 (PR) | none | ~1 tok/s for 284B on 61 GB (codacus) |
| Network to build/run | **none** (fork) | none after source | none after source | none after source |
| Fit at 30 GB | **no** full quant | IQ2_XS/Q2_0 yes | likely yes | yes, slow |

**Colibri was not built in this session.** It is a different engine
(`JustVugg/colibri`, Apache-2.0) and the codacus summary already gives its shape
(mmap streaming, expert budget, 71% hit rate, 284B on 61 GB at ~1 tok/s). It does
not help the 62 GB Qwen path, and one engine vetted end-to-end is enough for this
session. The next session can clone/vet/build it if the 30 GB fallback matters.

---

## 7. Provenance, licence and parser risk

### 7.1 Is a third-party abliterated merge acceptable?

For **our internal, offline use**: yes, with conditions. The chain is traceable:
Qwen official BF16 → ISTA GSQ-RCO quant (Apache-2.0 tooling, standard GGUF) →
SC117 in-place transplant of 144 tensors from orcarouter's abliterated release →
DavidAU repackage + prompt system. SC117 documents the 144 tensors and verifies
1,223 per-tensor blake2b digests; the quantized weights are not re-quantized. This
is better provenance than a typical "uncensored" merge.

The conditions:

- **Licence:** the official base is **Qwen Community License 1.0**, not Apache-2.0.
  SC117, DavidAU and several mirrors declare `apache-2.0`; that declaration is
  wrong and cannot override the base licence. Treat the abliterated files as
  Qwen-Community-1.0 and read its terms before any redistribution or commercial
  serving. Internal research use is the low-risk case.
- **Safety:** the model has no refusal behaviour. Do not put it in front of end
  users, do not connect it to tools with real side effects without a moderation
  layer and an allow-list, and do not use it for anything the law forbids.
- **Integrity:** GGUF is not pickle, but it is a binary parser input. Verify the
  SHA-256 of every shard against ISTA's checksums (SC117 publishes `SHA256SUMS`).
  Do not run a random re-upload.
- **Quality:** the abliteration costs ~0.1% decode and is within noise on the
  published spot checks; the MTP head is stock, not abliterated. Do not assume the
  "Turbo Brilliance" prompt system improves reasoning; that is unverified marketing.

### 7.2 Recent llama.cpp GGUF-parser CVEs

The pin is from 2026-10-16. In the NVD/GHSA data read on 2026-10-17, **no published CVE is newer than
the pin**; these are the recent parser-relevant ones, all fixed before it:

| CVE | Issue | Fixed in |
|---|---|---|
| CVE-2026-52131 | reachable assertion in `gguf_reader::read` (DoS) | affects b5693 and earlier; fixed after b5693 |
| CVE-2026-33298 | integer overflow in `ggml_nbytes` during GGUF tensor parsing → heap overflow | b7824 |
| CVE-2026-27940 | integer overflow in `gguf_init_from_file_impl` `mem_size` → heap overflow (bypass of the 2025 fix) | b8146 |
| CVE-2025-53630 | integer overflow in `gguf_init_from_file_impl` | 2025 |
| CVE-2025-49847 / CVE-2025-52566 | vocabulary/tokenizer heap overflow via a malicious GGUF | b5662 / b5721 |

Caveat: a CVE disclosed **after** 2026-10-16 would affect the pin. Re-check before
running a new model file, and prefer a pin that is current at run time. The parser
risk here is a malicious GGUF, not the model's outputs; checksums close most of it.

---

## 8. Second candidate: Nemotron 3.5 Bolt 74B-A9B (video `DfbaiU73--w`)

The video is about **NVIDIA Nemotron 3.5 Bolt**, not "Reflection Beam". The
official GGUF exists and is a much better fit for the **current** 30 GB box:

| Fact | Value |
|---|---|
| Repo | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF` |
| Parameters | **74B total, 9B active** (MoE), native **2-bit QAT** (`IQ2`) |
| File | `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` **22.02 GB** + `mmproj` **1.73 GB** |
| Context | up to **256K** |
| Licence | **OpenMDW-1.1** |
| llama.cpp at our pin | architecture `nemotron_h` / `nemotron_h_moe` is present |
| Fit | 22.02 GB weights in 24 GB VRAM: fits, but tight with the 1.5 GB sidecars; use a small context / q4_0 KV, or offload a little |
| Speed | no 3090 measurement here; the video's ceiling is ~300 tok/s and a 3090 has ~3x the bandwidth of the RTX Spark it cites → **estimate 100–200 tok/s** decode, unverified |
| Risk | quality at 2 bits is the open question; it is a different model family and a different licence |

**What this means:** if the goal is "a frontier-class local model on the 3090 this
month", Bolt is the cheapest experiment (22 GB, no RAM purchase, llama.cpp already
supports it). If the goal is specifically **Qwen3.8-Flash-Next**, Bolt does not
replace it; it is a parallel option. The project's brief is Qwen/Strata, so the
verdict below stays Qwen-first.

---

## 9. Verdict

**Download which file?**

- **Do not download the DavidAU 83.86 GB IQ3_S file** for this box as the first
  move. It is the largest file, it has no MTP head and no Strata pack, and the safe
  fork does not accept IQ3_S. It also needs 62 GB of RAM and leaves no headroom.
- **With +32 GB (62 GB), the safe pick is SC117 `IQ3_XXS`** (76.14 GB: 47.34 +
  28.80) + `mtp-q2_0.gguf` (0.89 GB) + `strata/rt/` (0.83 GB) + `mmproj`
  (0.91 GB), or ISTA `IQ3_XXS` (75.8 GB) + a built MTP head. Rebuild the pack and
  run `vendor/strata/build/unbiased-strata`. Expect **59–77 tok/s** decode and
  **1,200–1,600 tok/s** prefill at 32K. If you insist on IQ3_S, use upstream
  Strata (unsafe) or patch the fork; 46–62 tok/s, no headroom.
- **Without +32 GB (30 GB), do not use Strata for this model.** Use the pinned
  llama.cpp with **IQ2_XS** (68 GB) and the n-gram shard on the SSD:
  `-ngl 99 -cmoe --moe-cache-mib 8000 -lm mmap --lazy-mode on -c 32768`, expect
  **48–57 tok/s** decode / 150–300 tok/s prefill. Or test **Bolt** (22 GB) first.

**Run with which engine?** Safe Strata fork for the 62 GB path; pinned llama.cpp
for the 30 GB path. Upstream Strata only if you accept a network installer and a
prebuilt binary.

**Is +32 GB worth it?** **Yes for Strata**, **no for llama.cpp alone**.

- At 30 GB, Strata's own RAM floor (48/60/62 GB) excludes every full-size quant;
  a forced file tier is unsupported and slow. llama.cpp can run IQ2_XS at ~48–57
  tok/s, but prefill is 150–300 tok/s.
- At 62 GB, Strata runs IQ3_XXS with all experts in RAM: ~59–77 tok/s and
  1,200–1,600 tok/s prefill. Versus llama.cpp IQ3_XXS at 30 GB (~33–40 tok/s), that
  is **~2–3x decode and ~5–8x prefill**, plus better quality than IQ2_XS.
- If Bolt's 2-bit quality is acceptable, it fits today and the RAM purchase buys
  nothing for it. Test Bolt before buying RAM if a 22 GB download is acceptable.

**Exact next steps** (in order):

1. Run `vendor/strata/BUILD-OFFLINE.sh` on a host with CMake ≥3.24 + CUDA 12.x
   (sm_86); it builds inside `bwrap --unshare-net`. Verify
   `vendor/strata/build/unbiased-strata --help`.
2. Decide the RAM purchase. If yes, order the +32 GB (total 62 GB) and download
   SC117 IQ3_XXS + MTP + `strata/rt` (76.1+0.89+0.83 GB) or ISTA IQ3_XXS (75.8 GB).
   Verify SHA-256; 335 GB free is enough.
3. Build the pack with `tools/iq_pack.py`, copy `data/expert-profile.bin`, run the
   engine with `--ctx 32768 -- --kv q4_0 --ple-io direct`; measure decode and
   prefill with the same 20 prompts used in the 3090 anchors.
4. If the RAM purchase is no, A/B llama.cpp IQ2_XS at 30 GB vs Bolt 22 GB on the
   same 20 prompts; keep whichever is better on our extraction task.

---

## Appendix: commands used (all offline unless marked)

```sh
# model facts (network, metadata only; no .gguf payload)
curl -sS 'https://huggingface.co/api/models/DavidAU/Qwen3.8-Flash-Next-...-GGUF?blobs=true'
curl -sS 'https://huggingface.co/api/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF?blobs=true'
curl -sSL 'https://huggingface.co/.../raw/main/tensor-allocation/...rco-allocation.txt'

# hardware
free -g; nproc; cat /proc/meminfo | grep MemTotal
cat /sys/bus/pci/devices/0000:01:00.0/current_link_speed   # 16.0 GT/s
cat /sys/bus/pci/devices/0000:01:00.0/current_link_width   # 16
dd if=<5.3GB file> of=/dev/null bs=1M count=2048 iflag=direct   # 4.0 GB/s
python3 .scratch/nvme_read.py                                   # 4K/64K/1M random

# offline build
bwrap --unshare-net --ro-bind / / --bind "$PWD" "$PWD" --tmpfs /tmp \
      --dev /dev --proc /proc --chdir "$PWD" make test-frontend   # exit 0

# the CUDA engine (needs cmake+nvcc; not available in this sandbox)
vendor/strata/BUILD-OFFLINE.sh
```
