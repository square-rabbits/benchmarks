# Qwen3.8-Flash-Next on a single RTX 3090

## Long-context inference with 24 GiB VRAM and 96 GB system memory

Square Rabbits · Experimental results, 4–7 October 2026 · Published 7 October 2026

[Interactive report](https://square-rabbits.github.io/benchmarks/) · [Measurement dataset](data/measurements.csv) · [Configurations and timing records](data/benchmark-data.json)

## Abstract

This study examines the throughput, memory use and practical integration of Qwen3.8-Flash-Next on an EVGA GeForce RTX 3090 OC FTW3 Ultra, an AMD Ryzen 9 8945HX and nominal 96 GB of system memory. The experiments cover Unsloth UD-Q4_K_XL and UD-IQ3_XXS weights, five inference implementations, and inputs approaching a 262,144-token context allocation. The performance criterion is prompt processing above 1,000 tokens/s and generation at or above 30 tokens/s; 40 tokens/s is a secondary target.

At 260,000 fresh input tokens, Q4 running in Strata achieved median **1,230.04 prompt tokens/s** and **30.40 generated tokens/s** over three capped 512-token reasoning runs. Sampled process VRAM peaked at **11.883 GiB**, while the active cgroup reported **74.060 GiB** peak memory. One run generated at 29.29 tokens/s. Reducing the cgroup limit from 80 to 48 GiB and the resident expert budget from 62 to 24 GiB reduced throughput to **773.13/9.30 tokens/s** in one completed run.

For Q3 in upstream llama.cpp, a 235,929-token throughput request measured **905.76/27.39 tokens/s**, and a separate natural-retrieval request passed its exact-value checks. At 8,192 tokens, the thecodacus implementation reached **1,197.90/35.97 tokens/s**, as the median of five runs. Increasing actual input length reduced throughput in the measured long-context configurations. A matched upstream comparison found substantially faster long-context generation with f16 KV than q8_0 KV.

The Strata observations establish near-full-context throughput, not completed-task quality: all three requests exhausted the output budget during reasoning. Comparisons between Strata and llama.cpp also change quantization, KV handling, speculative decoding and protocol. This report separates within-configuration observations from cross-system comparisons and distinguishes throughput, retrieval, API transport and voice measurements.

## 1. Research questions and principal results

The experiments address four questions:

1. Can this hardware sustain more than 1,000 prompt tokens/s and at least 30 generated tokens/s with a large, genuinely populated context?
2. How do offload placement, expert caching, batch size, KV format and speculative decoding affect that performance envelope?
3. What is the cost of approximately 12 GiB process VRAM and a smaller system-memory budget?
4. How do engine timings translate into API, tool-use and speech latency?

| Configuration | Context allocation | Actual input | Prefill, tok/s | Decode, tok/s | Experimental basis |
| --- | ---: | ---: | ---: | ---: | --- |
| Q3 · thecodacus · CPU MOE 43 · MTP2 · f16 KV | 32,768 | 8,192 | 1,197.90 | 35.97 | Median of 5 throughput requests; separate retrieval tests |
| Q3 · upstream · CPU MOE 43 · no MTP · f16 KV | 32,768 | 28,672 | 1,253.31 | 31.42 | 1 throughput request; separate retrieval pass |
| Q3 · upstream · CPU MOE 46 · microbatch 4,096 · f16 KV | 262,144 | 235,929 | 905.76 | 27.39 | 1 throughput request; separate retrieval pass |
| Q4 · Strata · 12 GiB VRAM reserve · 80 GiB cgroup limit | 262,144 | 260,000 | 1,230.04 | 30.40 | Median of 3 throughput requests; reasoning-only output |
| Q4 · Strata · 12 GiB VRAM reserve · 48 GiB cgroup limit | 262,144 | 260,000 | 773.13 | 9.30 | 1 completed throughput request |
| Q3 · upstream · voice-headroom serving profile | 262,144 | 230,100 | 1,067.78 | 26.56 | 1 API transport test; repetitive synthetic history |

The largest measured input meeting both lower criteria **in the median** was 260,000 tokens in Strata. It did not meet 30 tok/s in every repetition. No accepted near-full-context series reached 40 tok/s. Q3/upstream separately passed retrieval at 235,929 tokens, but its throughput request fell below both lower targets.

## 2. Experimental platform

| Component | Specification |
| --- | --- |
| GPU | EVGA GeForce RTX 3090 OC FTW3 Ultra; 24,576 MiB VRAM |
| NVIDIA driver | 595.91.07 |
| CPU | AMD Ryzen 9 8945HX; 16 physical cores, 32 logical CPUs |
| System memory | Nominal 96 GB; Linux MemTotal 94,397,108 KiB, approximately 90.02 GiB |
| Storage | Kingston KC3000, SKC3000D4096G; nominal 4 TB NVMe |
| Operating system / kernel | Ubuntu 26.04.1 LTS / 7.0.0-34-generic |
| Build environment | NVIDIA SM86; llama.cpp CUDA 12.8.1 build environment; Strata CUDA 13 libraries |
| Display | Radeon; RTX reserved for compute |

The driver verifies the GPU model and capacity; the EVGA board variant is owner-supplied. The recorded topology maps the first 16 logical CPUs, selected by mask `0xffff`, to separate physical cores. Clock, power-limit and thermal settings are not consistently recorded across the campaign, so the factory “OC” designation is not evidence of a controlled manual-overclock experiment. No isolated NVMe throughput benchmark accompanies the dataset.

### 2.1 Weights and implementations

| Implementation | Pinned version | Experimental role |
| --- | --- | --- |
| thecodacus/llama.cpp | `27c54b4bbcefadedcec6397477cc2e866c1db716` | Q4/Q3 offload, hot-expert caching and draft-MTP |
| llama.cpp upstream | `8345f333951c661d166b00e6f9362e553768f292` | Q3 long-context measurements and serving |
| ik_llama.cpp | `3d27f5bb6348c4d79edded63b03e9a93efb2f80e` | Independent MTP screening |
| OptLlama | `167742d9dfa01306786e7d3cfd31fcb8a8b185dc` | Expert cache, workspace, prefetch and overlap experiment |
| Strata | Tag `v0.1.40.1`; Python `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`; engine `0.1.40` | Q4 native pack, streamed KV and bounded residency |
| LiteLLM / Codex CLI | 1.104.0 / 0.160.1 | API transport and client integration |
| Chat template | Froggeric v22.5, `855bffc49448e299789730ff92c9b8d834d6cc14` | Tools and late developer/system messages |

The Unsloth revision is `38bb39ee97821de2c9009abb7e93950eec396e66`. Four UD-Q4_K_XL shards total **111,334,654,784 bytes**; the native payload metric `model_size` is 111,323,630,080 bytes. Three UD-IQ3_XXS shards contain 10,946,624, 49,567,921,344 and 32,382,955,968 bytes, totaling **81,961,823,936 bytes**.

Q3 and Q4 refer to these specific mixed-quantization variants, not a uniform bit width for every tensor. Strata's Q4 pack promotes 195 projections to BF16, another difference from direct GGUF execution. The [artifact manifest](data/artifact-integrity.json) identifies weights, projector, MTP and binaries. The template SHA256 is `e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2`.

Implementation references: [Unsloth weights](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), [thecodacus pinned README](https://github.com/thecodacus/llama.cpp/blob/27c54b4bbcefadedcec6397477cc2e866c1db716/README.md), [Strata](https://github.com/Niko1221/Strata), [llama-swap](https://github.com/mostlygeek/llama-swap). Throughput below comes from this campaign, not project documentation.

## 3. Methods

### 3.1 Experimental units

The dataset contains **406 measurement rows in 85 series**. The historical component contains 339 server-response records and 56 native llama-bench rows from 32 invocations, with three samples per row, totaling 168 native samples. Later additions include six Strata requests (two warmups) and five API/client timing records. The historical inventory includes three profiling sets.

| Protocol | Rows | Purpose |
| --- | ---: | --- |
| Synthetic throughput | 240 | Fresh, repeatable input; prompt and generation speed |
| Natural retrieval | 103 | Exact-value retrieval, valid JSON and completed reasoning |
| Native llama-bench | 56 | Random-token prompt/generation microbenchmarks |
| Warmup | 2 | Readiness and initialization |
| Functional transport | 5 | API, client, compaction and tool continuation |

A server row represents one request; a native row reports its native samples. These are not interchangeable experimental units. Failures before a completed request contribute no throughput row, rather than a zero-token/s observation.

### 3.2 Context occupancy and cache state

**Context allocation** is slot capacity; **actual input** is request length. A short prompt in a 262,144-token slot is not a near-full-context measurement. Inputs of 235,929 and 260,000 occupy approximately 90% and 99.18% of that capacity.

Fresh and reused tokens are separate. A continuation with 99 new tokens and a 48,882-token reused prefix is not a 48,981-token fresh-prefill result. Fresh means no KV-prefix reuse, not a cold file cache. File-cache flushing was not routine; restarting a process does not establish cold NVMe behavior.

### 3.3 Timing, aggregation and quality

- **Prefill / PP:** engine-reported prompt-processing rate.
- **Decode / TG:** engine-reported generation rate, retaining its native denominator.
- **First token / frame:** latency to the first model-generated token frame, excluding transport keepalives.
- **First answer:** first final-answer text where separable from reasoning.
- **Wall time:** complete request latency, including protocol overhead.

Throughput requests often use a 512-token cap. Retrieval requests instead require natural completion, exact expected values, valid JSON and closed reasoning. Capped reasoning can establish speed without establishing completed-task quality. Native random-token benchmarks likewise do not validate natural-language retrieval or tool use.

Medians combine only identical series, case, protocol, actual input and cache state. Screening uses three repetitions and selected finalists five; several long-context results have one. This staged study is not a randomized factorial experiment. A full 16k–256k matrix with all fill levels was not completed for one invariant configuration. Temperature, file cache and test order can influence observations.

In configuration labels, “CPU MoE N” or “CPUN” denotes the `--n-cpu-moe` setting, not the CPU thread count. The shorthand `t`, `b` and `u` denotes threads, batch size and microbatch size, respectively. Full arguments are provided in Appendix C.

## 4. Offload and expert caching

### 4.1 Initial Q4 results

Initial thecodacus Q4 used CPU MoE (`--n-cpu-moe 99`), mmap/lazy, q8_0 KV and profiled hot-expert caching. Screening covered 4/6/8/12/14/16 threads, cache sizes 32/40/48, synchronous/asynchronous CPU scheduling and batch/microbatch variants. Later screens extended cache to 64/80.

A 131,072-token slot, six threads and 32 cache slots measured **135.97 PP / 18.92 TG** for 8,192 fresh tokens. A microbatch-1,024 variant measured **219.65/19.71**. Loading the model did not establish efficient prefill. Native PP512 and HTTP exercise different workloads; native results remain in Appendix A rather than substituting for large-input measurements.

### 4.2 Q3 tuning results

These medians use 8,192-token fresh input. Full settings are linked in the [Q3 comparison](data/comparison-iq3-20261005.csv) and Appendix C.

| Configuration family | PP, tok/s | TG, tok/s | Requests | Interpretation |
| --- | ---: | ---: | ---: | --- |
| thecodacus · CPU99 control | 183.07 | 26.73 | 3 | mmap; q8_0 KV |
| Cache96 control | 192.70 | 29.88 | 3 | Larger cache did not resolve low prefill |
| Cache96 · t14 · u4096 | 594.30 | 31.86 | 3 | Joint thread/microbatch change |
| Cache80 · batch8192 | 746.44 | 30.83 | 3 | Screening |
| Cache80 finalist | 742.57 | 31.15 | 5 | Repeated finalist |
| No expert cache · CPU99 | 1,085.49 | 28.18 | 3 | Higher prefill; decode below criterion |
| Hybrid placement · CPU38 | 1,006.27 | 30.88 | 3 | Both lower criteria met in median |
| Hybrid placement · CPU42 · MTP2 | 1,049.74 | 38.10 | 3 | Hybrid expert placement and speculation |
| CPU42 · MTP2 · pinned | 1,201.53 | 38.25 | 3 | Timing records valid; series failed NVML cleanup |
| CPU43 · MTP2 · q8_0 finalist | 1,197.67 | 32.88 | 5 | Separate KV control |
| CPU43 · MTP2 · f16 finalist | 1,197.90 | 35.97 | 5 | Separate retrieval at 8,192/16,384/28,672 |
| CPU43 · MTP1 · f16 finalist | 1,197.67 | 34.13 | 5 | Lower decode than MTP2 in this series |

Hybrid placement produced a more favorable envelope than the initial fully CPU-offloaded cached path. Several steps also change thread count, batch size or memory handling, preventing single-flag attribution. Recorded thinking settings are retained; headline speed is not obtained by secretly disabling reasoning.

## 5. Scaling actual input length

### 5.1 thecodacus

| Allocation | Input | CPU MOE | Microbatch | PP, tok/s | TG, tok/s | Requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 8,192 | 43 | 8,192 | 1,197.28 | 33.05 | 3 |
| 32,768 | 16,384 | 43 | 8,192 | 1,157.29 | 34.89 | 3 |
| 32,768 | 24,576 | 43 | 8,192 | 1,119.91 | 32.58 | 3 |
| 32,768 | 29,491 | 43 | 8,192 | 1,074.89 | 31.55 | 3 |
| 262,144 | 8,192 | 48 | 2,048 | 665.81 | 31.80 | 1 |
| 262,144 | 65,536 | 48 | 2,048 | 608.12 | 27.07 | 1 |
| 262,144 | 131,072 | 48 | 2,048 | 509.26 | 22.00 | 1 |
| 262,144 | 196,608 | 48 | 2,048 | 434.20 | 18.09 | 1 |
| 262,144 | 235,929 | 48 | 2,048 | 402.45 | 15.80 | 1 |

Both series use q8_0 KV and MTP2. Within the larger-allocation series, increasing input reduces prefill and decode. Between series, CPU MoE and microbatch also change; allocation size is not isolated. Large-slot u8192 failed a CUDA check; u1024 was interrupted.

In a separate f16 90%-fill screen, a 45,056-token slot with 40,550 input tokens measured **1,027.03/30.81**, while a 47,104-token slot with 42,393 input tokens measured **995.06/28.13**. This threshold belongs to that configuration, not to the hardware or model universally.

### 5.2 Upstream llama.cpp

| KV / CPU MoE / microbatch | Allocation | Input | PP, tok/s | TG, tok/s | Separate retrieval |
| --- | ---: | ---: | ---: | ---: | --- |
| f16 / 43 / 8,192 | 32,768 | 8,192 | 1,334.01 | 32.14 | Pass |
| f16 / 43 / 8,192 | 32,768 | 28,672 | 1,253.31 | 31.42 | Pass |
| q8_0 / 48 / 4,096 | 262,144 | 8,192 | 1,046.51 | 28.91 | Pass |
| q8_0 / 48 / 4,096 | 262,144 | 131,072 | 960.72 | 20.85 | Pass |
| q8_0 / 48 / 4,096 | 262,144 | 235,929 | 894.82 | 16.77 | Pass |
| f16 / 48 / 4,096 | 262,144 | 8,192 | 1,046.63 | 29.55 | Pass |
| f16 / 48 / 4,096 | 262,144 | 131,072 | 962.91 | 28.06 | Pass |
| f16 / 48 / 4,096 | 262,144 | 235,929 | 898.37 | 26.78 | Pass |
| f16 / 43 / 2,048 | 262,144 | 235,929 | 695.04 | 28.30 | Pass |
| f16 / 46 / 4,096 | 262,144 | 235,929 | 905.76 | 27.39 | Pass |

Each throughput row is one request, with retrieval tested separately; speculation is disabled. At 235,929 tokens, the matched CPU48/u4096 comparison gives **59.72% faster decode** with f16 than q8_0, from 16.77 to 26.78 tok/s, while prefill remains approximately unchanged (`M0334`, `M0328`). Lower KV memory consumption does not imply faster generation in this workload.

CPU46/u4096 measured **1,066.83/30.45** at 8,192 tokens and **905.76/27.39** at 235,929: **15.10% lower PP** and **10.03% lower TG** (`M0322`, `M0320`; `C053`). Both lower criteria are missed at depth. Every displayed PP/TG pair belongs to one request; maxima from different trials are not combined.

## 6. Speculation and alternative implementations

### MTP

thecodacus recorded working draft-MTP counters. Its five-run f16 finalists reached 35.97 TG with MTP2 and 34.13 with MTP1. Equivalent flag names do not establish equivalent support in other implementations.

ik_llama.cpp's control, with a 32,768-token slot and 8,192-token input, measured **1,223.08/30.94**. MTP1 with main microbatch 4,096 and draft microbatch 512 measured **911.13/33.88**. At 28,672 input tokens, the respective values were **1,089.84/30.73** and **851.70/34.26**. Generation improved while prefill slowed; two other MTP configurations failed before accepted results. The recorded ik version also exhibited `ignore_eos` persistence between requests, without a local patch.

Upstream serving uses `--spec-type none`. Strata's native MTP uses up to four draft tokens and a q2_0 component. Their acceptance counters are not pooled with thecodacus counters.

### OptLlama

One configuration combined a 4,096 MiB expert cache, phase-aware/live-context workspace, PLE prefetch, backend sampling and decode overlap. It used Q3, a 262,144-token slot, CPU MoE 48, f16 KV, batch 8,192, microbatch 4,096, 14 threads and no MTP. A natural-retrieval request at 8,192 tokens passed with **1,287.75/35.38**. The 235,929-token trial had no accepted full response: VRAM headroom reached **197 MiB**, below the 2,048 MiB guard threshold, and its process was stopped. This identifies insufficient headroom for that configuration, not a general failure of the implementation. Individual optimizations cannot be attributed from this combined trial.

## 7. Strata: near-full-context Q4 inference

### 7.1 Configuration and residency

Strata separates logical context from GPU-resident KV. The experiment used a 262,144-token context, **32,768 resident KV cells per layer**, int8 KV, automatic prefill/expert caching and native MTP. The resident expert budget was **62 GiB**, with an **80 GiB cgroup memory limit** and swap disabled. `--vram-reserve-mib 12288` is an allocator reservation, not a hard driver quota.

The runtime reported 15 workers and one host thread, with affinity on CPU 0 for the host and CPUs 1–15 for the worker pool. It used 1,407 GPU expert slots (4.09 GiB), 7,424-token prompt chunks, 1,185 borrowed slots and 3.09 GiB pinned host KV. Speculation used four draft tokens and minimum probability 0.5. Requests used thinking HIGH, temperature zero, a 512-token cap and no reused prefix. Complete settings are in `C094`.

### 7.2 Throughput at 260,000 fresh tokens

| Request | Input / reused | Output | PP, tok/s | TG, tok/s | First token, s | Wall time, s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `M0397` | 260,000 / 0 | 512 | 1,230.04 | 30.40 | 211.671 | 228.481 |
| `M0398` | 260,000 / 0 | 512 | 1,224.69 | 30.55 | 212.612 | 229.330 |
| `M0399` | 260,000 / 0 | 512 | 1,247.39 | 29.29 | 208.745 | 226.180 |
| Median | | | **1,230.04** | **30.40** | **211.671** | **228.481** |

All three stopped with `finish=length` during reasoning, with no final answer. Two met the lower decode criterion, one did not. The medians meet the joint lower target but do not establish a guaranteed minimum or task quality at this depth.

Sampled process VRAM peaked at **12,168 MiB (11.883 GiB)**; the whole-board peak was 12,364 MiB. Active-cgroup `MemoryPeak` was **79,521,480,704 bytes (74.060 GiB)**, swap peak was zero, and minimum host-available memory was 10.544 GiB. NVML's approximately two-second sampling can miss shorter peaks. The roughly 211-second first-token latency is consistent with processing 260,000 fresh tokens at 1,230 tok/s: fast decode does not remove large-input cost.

### 7.3 Lower memory budget

The second condition retained Q4, the 12 GiB VRAM reservation and 260,000-token input, but used a **48 GiB cgroup limit** and **24 GiB resident expert budget** (`C095`). One completed request (`M0401`) measured **773.13/9.30**, with first token at 336.625 s and wall time 391.599 s. Peak cgroup memory was 39.476 GiB; process VRAM peak was 11.883 GiB. The next repetition was interrupted.

Relative to the three-run baseline median, prefill decreased **37.15%** and decode **69.40%**. This changes memory limit and cache budget together, not RAM capacity alone. The physical 96 GB host and system-wide cache also remain; a cgroup cap is not a perfect smaller-machine simulation.

| Resource / result | 80 GiB condition | 48 GiB condition |
| --- | ---: | ---: |
| Completed throughput requests | 3 | 1 |
| Resident expert budget | 62 GiB | 24 GiB |
| Cgroup peak memory | 74.060 GiB | 39.476 GiB |
| Sampled process VRAM peak | 11.883 GiB | 11.883 GiB |
| Swap peak | 0 | 0 |
| Prefill | Median 1,230.04 tok/s | 773.13 tok/s |
| Decode | Median 30.40 tok/s | 9.30 tok/s |

### 7.4 Interpretation

Explicit residency management enabled near-full logical context with approximately 12 GiB process VRAM. Adequate host residency proved important, especially for decode. However, this is **not an engine-only comparison with llama.cpp**: quantization, packing, KV/MTP, prompt and stop conditions differ. Existing Q3 weights were not tested in Strata, and this Q4 experiment did not validate natural retrieval, vision, tools or concurrent speech. The integrated serving profile remained Q3/upstream.

## 8. Serving and application latency

### 8.1 Integrated Q3 profile

The integrated profile leaves Chatterbox headroom through CPU MoE 48 and microbatch 2,048, retaining a 262,144-token context and a CPU vision projector:

```text
llama-server --model MODEL_UD-IQ3_XXS_FIRST_SHARD.gguf \
  --spec-type none --gpu-layers 99 --n-cpu-moe 48 \
  --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 \
  --load-mode none --lazy-mode on --fit off --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 \
  --ctx-size 262144 --parallel 1 --cache-reuse 256 --cache-prompt \
  --batch-size 8192 --ubatch-size 2048 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --metrics --slots --jinja --chat-template-file VENDOR_TEMPLATE.jinja \
  --chat-template-kwargs '{"reasoning_effort":"xhigh"}' \
  --offline --no-context-shift --timeout 7200 \
  --mmproj MMPROJ_F16.gguf --no-mmproj-offload
```

The path is HTTPS → Nginx → LiteLLM → llama-swap → llama.cpp. llama-swap manages LLM processes, not a durable common queue with ComfyUI. CPU48/u2048 must not be assigned the earlier CPU46/u4096 benchmark results. Version-specific commands and environments appear in Appendix C.

### 8.2 Long-history API and client

A Responses request with **230,100 tokens** returned headers at **15.251 s**, emitted 15 keepalives, delivered the first model event at **216.519 s**, and completed at **219.514 s**. Keepalives maintain transport, not generation progress.

Unmodified Codex CLI processed **232,796 tokens**, within an effective window of 249,036, in **251.740 s**. Native compaction took **262.022 s**; the next turn took **79.845 s** in the same session with `apply_patch`, fileChange and three diff events. Exact file contents and the end marker were verified. A 1 MiB oversized fixture was rejected client-side before GPU submission. Earlier concurrent file-edit turns took 237.535/280.157 s and are not isolated latency benchmarks.

| Phase | Total / fresh / reused | Output | PP, tok/s | TG, tok/s |
| --- | --- | ---: | ---: | ---: |
| Long Responses | 230,100 / 230,100 / 0 | 80 | 1,067.78 | 26.56 |
| Codex long turn | 232,796 / 226,582 / 6,214 | 34 | 917.53 | 26.93 |
| Compaction | 230,530 / 230,530 / 0 | 216 | 919.76 | 26.77 |
| Post-compaction tool | 48,886 / 48,886 / 0 | 353 | 862.41 | 28.46 |
| Tool-result continuation | 48,981 / 99 / 48,882 | 69 | 101.71 | 28.99 |

These are single functional observations with repetitive synthetic history, not quality benchmarks. Audio was inactive during these five records; no accompanying per-request memory/NVMe trace was collected.

### 8.3 Voice and agent tools

Twenty synthetic Chatterbox TTS → Qwen ASR cycles yielded full-WAV latency **0.829 s p50 / 2.410 s p95**. The first cold trial took **42.304 s** while overlapping LLM work; minimum free VRAM was 4,317 MiB. A separate warm joint first-response test without thinking for speech measured **5.877 s TTS / 1.872 s ASR**, with minimum free VRAM 5,243 MiB. A portal turn took 14.753 s; reply TTS took 0.922 s.

These are server-stage/complete-WAV times, not end-of-human-speech to first-audible-response latency. Two early voice series completed no phrases. Later cycles produced audio but failed strict transcription: a 4.48 s WAV took 2.661 s TTS plus 1.982 s ASR, with one word wrong. The synthetic loop cannot isolate synthesis from recognition errors; live microphone, listening and interruption quality are outside its scope.

The Pithagoras audit exercised **20 of 32 enabled tools**: files, edits, bash, terminal, canvas, Chromium, three searches, two page reads, file output, continuation, cross-session memory, a 32k subagent and images. An 18-call research task took 413.906 s, without establishing completion of the separate 100-restaurant benchmark. One subagent reasoned in English; a separate bash check returned exit 1. Understory write/read completed in 87.787/66.888 s after increasing the documented timeout beyond 60 s. A 35 s HTTP/1.1 test addressed partial HTTP/2 SSE behavior in Firefox, not live microphone quality.

## 9. Memory placement and stability

Weight-file size is not a requirement for full simultaneous residency. NVMe holds weights/backing files; RAM supplies CPU pages, resident experts and file cache; VRAM holds selected experts, compute and engine-specific KV. This permits operating Q4 files larger than nominal host memory.

thecodacus used native `llama-moe-trace` chat/code/tool profiles, hot-expert cache variants and hybrid paths without that cache. One unconfirmed-cache trial failed its activation check. Strata used automated caching, prefill slot borrowing and streamed KV. Its `file_MB` represents file-layer reads, potentially served by RAM cache, not measured physical NVMe bandwidth. Host diskstats and faults are not isolated process counters. No RAM-disk A/B trial was performed.

Recorded failures include CUDA assertions at large microbatches, occupied ports, connection resets, interrupted runs, cache activation failures, NVML timeouts, failed profiling, debugger output instead of JSONL and low-VRAM guard termination. Some series failed cleanup after valid request timings. Appendix B separates request-level measurements from series-level failure; incomplete streams are not final-response results.

## 10. Conclusions

1. **Near-full-context throughput is feasible on this platform.** Strata Q4 reached median 1,230.04 PP / 30.40 TG at 260,000 fresh tokens with 11.883 GiB sampled process VRAM. One repetition fell below 30 TG; all outputs were capped in reasoning.
2. **Host residency is a central constraint.** Jointly reducing the memory limit and expert budget caused a much larger relative loss in decode than prefill. This is not an isolated RAM-capacity law.
3. **KV format needs workload-specific measurement.** Matched upstream f16 KV increased deep-context decode 59.72% over q8_0 despite its less compact format.
4. **Actual occupancy affects speed.** Upstream CPU46/u4096 lost 15.10% prefill and 10.03% decode between 8,192 and 235,929 input tokens. A 256k allocation does not preserve short-input speed automatically.
5. **Throughput and useful agent behavior require different evidence.** Q3 retrieval was validated separately. Strata's capped speed runs do not establish retrieval, tool or speech quality.
6. **No configuration demonstrated every target simultaneously.** Sustained 40 TG near full context, a guaranteed joint threshold across repeats, broad natural-task quality at 260k and live-speech interruption latency are not established by this dataset.

Explicit residency management extends the long-context performance envelope. End-to-end utility additionally depends on completed answers, retrieval accuracy, tool integration and audio headroom. The integrated Q3 profile and experimental Strata Q4 configuration serve different demonstrated purposes; the campaign is not a controlled ranking of engines.

## 11. Supplementary data and reproducibility

The interactive report separates its research narrative from a measurement explorer. Filters select engine, weights, protocol, input and series status. Point/row selections expose timing records and corresponding configurations. Medians never combine different cases. Appendices provide commands, environments, runtime-effective parameters and recorded build recipes.

Reproduction requires pinned weights/engine/template, matching actual input and allocation, output/thinking/sampling settings, KV/MTP and prefix state. `<PATH>/filename` denotes a local installation path. Records are identified by source path, SHA256 and zero-based index. Original diagnostic strings remain verbatim in the data.

The release includes selected timing/configuration metadata and an original-artifact manifest, not complete raw input fixtures or private application records. It therefore supports verification of published observations and configuration identity, but not byte-for-byte reconstruction of every original request. Missing clock/thermal controls, randomized ordering, isolated physical I/O and an invariant full context matrix constrain generalization. [JSON](data/benchmark-data.json), [CSV](data/measurements.csv), [artifact hashes](data/artifact-integrity.json) and [release checksums](SHA256SUMS) accompany this report.

## Appendix A. Measurement register

Groups share the same series, case, protocol, actual input and cache state. Server values are within-group medians; native-random rows contain the native sample mean, not repeated HTTP requests. Individual records are available in [measurements.csv](data/measurements.csv) and [benchmark-data.json](data/benchmark-data.json).

| ID | Series | Case | Protocol | Allocation | Input | Rows | PP tok/s | TG tok/s | Retrieval passes | Series status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0001 | context176-262144-cpu48-q8-u2048-20261005 | ctx262144-cpu48-q8_0-b8192-u2048 | synthetic-speed | 262144 | 8192 | 1 | 665.81 | 31.80 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0002 | context176-262144-cpu48-q8-u2048-20261005 | ctx262144-cpu48-q8_0-b8192-u2048 | synthetic-speed | 262144 | 65536 | 1 | 608.12 | 27.07 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0003 | context176-262144-cpu48-q8-u2048-20261005 | ctx262144-cpu48-q8_0-b8192-u2048 | context-quality | 262144 | 65536 | 1 | 606.60 | 28.49 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0004 | context176-262144-cpu48-q8-u2048-20261005 | ctx262144-cpu48-q8_0-b8192-u2048 | synthetic-speed | 262144 | 131072 | 1 | 509.26 | 22.00 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0005 | context176-262144-cpu48-q8-u2048-20261005 | ctx262144-cpu48-q8_0-b8192-u2048 | context-quality | 262144 | 131072 | 1 | 508.09 | 23.17 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0006 | context176-262144-cpu48-q8-u2048-20261005 | ctx262144-cpu48-q8_0-b8192-u2048 | synthetic-speed | 262144 | 196608 | 1 | 434.20 | 18.09 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0007 | context176-262144-cpu48-q8-u2048-20261005 | ctx262144-cpu48-q8_0-b8192-u2048 | context-quality | 262144 | 196608 | 1 | 433.62 | 19.96 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0008 | context176-262144-cpu48-q8-u2048-20261005 | ctx262144-cpu48-q8_0-b8192-u2048 | synthetic-speed | 262144 | 235929 | 1 | 402.45 | 15.80 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0009 | context176-262144-cpu48-q8-u2048-20261005 | ctx262144-cpu48-q8_0-b8192-u2048 | context-quality | 262144 | 235929 | 1 | 402.11 | 19.38 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0010… | context176-32768-cpu43-q8-20261005 | ctx32768-cpu43-q8_0-b8192 | synthetic-speed | 32768 | 8192 | 3 | 1197.28 | 33.05 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0013 | context176-32768-cpu43-q8-20261005 | ctx32768-cpu43-q8_0-b8192 | context-quality | 32768 | 8192 | 1 | 1182.42 | 37.77 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0014… | context176-32768-cpu43-q8-20261005 | ctx32768-cpu43-q8_0-b8192 | synthetic-speed | 32768 | 16384 | 3 | 1157.29 | 34.89 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0017 | context176-32768-cpu43-q8-20261005 | ctx32768-cpu43-q8_0-b8192 | context-quality | 32768 | 16384 | 1 | 1150.07 | 40.12 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0018… | context176-32768-cpu43-q8-20261005 | ctx32768-cpu43-q8_0-b8192 | synthetic-speed | 32768 | 24576 | 3 | 1119.91 | 32.58 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0021 | context176-32768-cpu43-q8-20261005 | ctx32768-cpu43-q8_0-b8192 | context-quality | 32768 | 24576 | 1 | 1113.29 | 37.85 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0022… | context176-32768-cpu43-q8-20261005 | ctx32768-cpu43-q8_0-b8192 | synthetic-speed | 32768 | 29491 | 3 | 1074.89 | 31.55 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0025 | context176-32768-cpu43-q8-20261005 | ctx32768-cpu43-q8_0-b8192 | context-quality | 32768 | 29491 | 1 | 1070.55 | 36.44 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0026… | context176-32768-cpu48-q8-control-20261005 | ctx32768-cpu48-q8_0-b8192 | synthetic-speed | 32768 | 8192 | 3 | 1164.93 | 32.45 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0029 | context176-32768-cpu48-q8-control-20261005 | ctx32768-cpu48-q8_0-b8192 | context-quality | 32768 | 8192 | 1 | 1150.73 | 35.54 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0030… | context176-32768-cpu48-q8-u1024-control-20261005 | ctx32768-cpu48-q8_0-b8192-u1024 | synthetic-speed | 32768 | 8192 | 3 | 498.88 | 32.95 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0033 | context176-32768-cpu48-q8-u1024-control-20261005 | ctx32768-cpu48-q8_0-b8192-u1024 | context-quality | 32768 | 8192 | 1 | 490.16 | 38.28 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0034 | context176-32768-cpu48-q8-u2048-control-20261005 | ctx32768-cpu48-q8_0-b8192-u2048 | synthetic-speed | 32768 | 8192 | 1 | 667.57 | 32.02 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0035 | context176-32768-cpu48-q8-u2048-control-20261005 | ctx32768-cpu48-q8_0-b8192-u2048 | context-quality | 32768 | 8192 | 1 | 715.85 | 35.65 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0036 | context176-40960-cpu44-f16-u8192-screen-20261005 | ctx40960-cpu44-f16-b8192-u8192 | synthetic-speed | 40960 | 8192 | 1 | 1040.98 | 33.20 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0037 | context176-40960-cpu44-f16-u8192-screen-20261005 | ctx40960-cpu44-f16-b8192-u8192 | synthetic-speed | 40960 | 36864 | 1 | 1033.96 | 28.03 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0038 | context176-40960-cpu44-f16-u8192-screen-20261005 | ctx40960-cpu44-f16-b8192-u8192 | context-quality | 40960 | 36864 | 1 | 1032.78 | 33.44 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0039 | context176-45056-cpu45-f16-u8192-screen-20261005 | ctx45056-cpu45-f16-b8192-u8192 | synthetic-speed | 45056 | 8192 | 1 | 1031.72 | 31.55 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0040 | context176-45056-cpu45-f16-u8192-screen-20261005 | ctx45056-cpu45-f16-b8192-u8192 | synthetic-speed | 45056 | 40550 | 1 | 1027.03 | 30.81 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0041 | context176-45056-cpu45-f16-u8192-screen-20261005 | ctx45056-cpu45-f16-b8192-u8192 | context-quality | 45056 | 40550 | 1 | 1027.04 | 34.87 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0042 | context176-47104-cpu45-f16-u8192-screen-20261005 | ctx47104-cpu45-f16-b8192-u8192 | synthetic-speed | 47104 | 8192 | 1 | 1032.76 | 31.80 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0043 | context176-47104-cpu45-f16-u8192-screen-20261005 | ctx47104-cpu45-f16-b8192-u8192 | synthetic-speed | 47104 | 42393 | 1 | 995.06 | 28.13 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0044 | context176-47104-cpu45-f16-u8192-screen-20261005 | ctx47104-cpu45-f16-b8192-u8192 | context-quality | 47104 | 42393 | 1 | 992.43 | 33.02 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0045 | context176-49152-cpu46-f16-u8192-screen-20261005 | ctx49152-cpu46-f16-b8192-u8192 | synthetic-speed | 49152 | 8192 | 1 | 1028.96 | 30.27 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0046 | context176-49152-cpu46-f16-u8192-screen-20261005 | ctx49152-cpu46-f16-b8192-u8192 | synthetic-speed | 49152 | 44236 | 1 | 989.93 | 28.18 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0047 | context176-49152-cpu46-f16-u8192-screen-20261005 | ctx49152-cpu46-f16-b8192-u8192 | context-quality | 49152 | 44236 | 1 | 989.83 | 30.35 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0048 | context176-65536-cpu48-f16-u8192-screen-20261005 | ctx65536-cpu48-f16-b8192-u8192 | synthetic-speed | 65536 | 8192 | 1 | 1019.96 | 32.83 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0049 | context176-65536-cpu48-f16-u8192-screen-20261005 | ctx65536-cpu48-f16-b8192-u8192 | synthetic-speed | 65536 | 58982 | 1 | 920.82 | 28.27 | 0 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0050 | context176-65536-cpu48-f16-u8192-screen-20261005 | ctx65536-cpu48-f16-b8192-u8192 | context-quality | 65536 | 58982 | 1 | 920.70 | 28.91 | 1 | PASS_CONTEXT_PROTOCOL_MATRIX_NOT_PRODUCTION_ACCEPTANCE |
| M0051 | guided-speed175-20261005 | baseline96 | synthetic-speed | 32768 | 8192 | 1 | 582.21 | 31.97 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0052 | guided-speed175-20261005 | baseline96 | synthetic-speed | 32768 | 8192 | 1 | 33.12 | 32.15 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0053 | guided-speed175-20261005 | baseline96 | context-quality | 32768 | 8192 | 1 | 594.29 | 33.05 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0054 | guided-speed175-20261005 | batch8192-cache96 | synthetic-speed | 32768 | 8192 | 1 | 722.29 | 30.62 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0055 | guided-speed175-20261005 | batch8192-cache96 | synthetic-speed | 32768 | 8192 | 1 | 33.20 | 30.68 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0056 | guided-speed175-20261005 | batch8192-cache96 | context-quality | 32768 | 8192 | 1 | 744.77 | 32.88 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0057 | guided-speed175-20261005 | overlap-cache112 | synthetic-speed | 32768 | 8192 | 1 | 580.62 | 26.77 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0058 | guided-speed175-20261005 | overlap-cache112 | synthetic-speed | 32768 | 8192 | 1 | 32.49 | 26.85 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0059 | guided-speed175-20261005 | overlap-cache112 | context-quality | 32768 | 8192 | 1 | 592.81 | 27.25 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0060 | guided-speed175-20261005 | gpu-mtp2-cache96 | synthetic-speed | 32768 | 8192 | 1 | 550.10 | 36.49 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0061 | guided-speed175-20261005 | gpu-mtp2-cache96 | synthetic-speed | 32768 | 8192 | 1 | 29.98 | 38.03 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0062 | guided-speed175-20261005 | gpu-mtp2-cache96 | context-quality | 32768 | 8192 | 1 | 568.40 | 41.43 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0063… | guided-speed175-finalist-20261005 | batch8192-cache80 | synthetic-speed | 32768 | 8192 | 5 | 742.57 | 31.15 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0064… | guided-speed175-finalist-20261005 | batch8192-cache80 | synthetic-speed | 32768 | 8192 | 5 | 32.15 | 31.22 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0073 | guided-speed175-finalist-20261005 | batch8192-cache80 | context-quality | 32768 | 8192 | 1 | 733.98 | 31.85 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0074 | guided-speed175-finalist-20261005 | batch8192-cache80 | context-quality | 32768 | 16384 | 1 | 721.85 | 30.21 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0075 | guided-speed175-finalist-20261005 | batch8192-cache80 | context-quality | 32768 | 28672 | 1 | 678.76 | 27.81 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0076… | guided-speed175-followup-20261005 | batch8192-cache80 | synthetic-speed | 32768 | 8192 | 3 | 746.44 | 30.83 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0077… | guided-speed175-followup-20261005 | batch8192-cache80 | synthetic-speed | 32768 | 8192 | 3 | 32.15 | 30.79 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0082 | guided-speed175-followup-20261005 | batch8192-cache80 | context-quality | 32768 | 8192 | 1 | 730.14 | 31.20 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0083 | guided-speed175-followup-20261005 | batch8192-cache80 | context-quality | 32768 | 16384 | 1 | 729.49 | 30.28 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0084 | guided-speed175-followup-20261005 | batch8192-cache80 | context-quality | 32768 | 28672 | 1 | 683.81 | 28.37 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0085… | guided-speed175-followup-20261005 | gpu-mtp2-cache64 | synthetic-speed | 32768 | 8192 | 3 | 562.74 | 36.17 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0086… | guided-speed175-followup-20261005 | gpu-mtp2-cache64 | synthetic-speed | 32768 | 8192 | 3 | 29.46 | 36.07 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0091 | guided-speed175-followup-20261005 | gpu-mtp2-cache64 | context-quality | 32768 | 8192 | 1 | 559.89 | 41.31 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0092 | guided-speed175-followup-20261005 | gpu-mtp2-cache64 | context-quality | 32768 | 16384 | 1 | 554.93 | 40.00 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0093 | guided-speed175-followup-20261005 | gpu-mtp2-cache64 | context-quality | 32768 | 28672 | 1 | 541.33 | 34.60 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0094… | guided-speed175-followup-20261005 | hybrid-mtp2-b8192-cache72 | synthetic-speed | 32768 | 8192 | 3 | 674.43 | 29.53 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0095… | guided-speed175-followup-20261005 | hybrid-mtp2-b8192-cache72 | synthetic-speed | 32768 | 8192 | 3 | 28.11 | 29.68 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0100 | guided-speed175-followup-20261005 | hybrid-mtp2-b8192-cache72 | context-quality | 32768 | 8192 | 1 | 668.88 | 31.33 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0101 | guided-speed175-followup-20261005 | hybrid-mtp2-b8192-cache72 | context-quality | 32768 | 16384 | 1 | 655.12 | 28.31 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0102 | guided-speed175-followup-20261005 | hybrid-mtp2-b8192-cache72 | context-quality | 32768 | 28672 | 1 | 619.25 | 21.92 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0103… | guided-speed175-native-profile-20261005 | batch8192-cache80 | synthetic-speed | 32768 | 16384 | 3 | 733.17 | 29.20 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0104… | guided-speed175-native-profile-20261005 | batch8192-cache80 | synthetic-speed | 32768 | 16384 | 3 | 30.38 | 29.16 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0109 | guided-speed175-native-profile-20261005 | batch8192-cache80 | context-quality | 32768 | 8192 | 1 | 741.78 | 31.67 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0110 | guided-speed175-native-profile-20261005 | batch8192-cache80 | context-quality | 32768 | 16384 | 1 | 729.00 | 30.20 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0111 | guided-speed175-native-profile-20261005 | batch8192-cache80 | context-quality | 32768 | 28672 | 1 | 683.13 | 27.80 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0112… | guided-speed175-native-profile-20261005 | batch8192-cache80-t16 | synthetic-speed | 32768 | 16384 | 3 | 732.27 | 29.05 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0113… | guided-speed175-native-profile-20261005 | batch8192-cache80-t16 | synthetic-speed | 32768 | 16384 | 3 | 31.23 | 29.05 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0118 | guided-speed175-native-profile-20261005 | batch8192-cache80-t16 | context-quality | 32768 | 8192 | 1 | 740.74 | 31.80 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0119 | guided-speed175-native-profile-20261005 | batch8192-cache80-t16 | context-quality | 32768 | 16384 | 1 | 728.60 | 30.16 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0120 | guided-speed175-native-profile-20261005 | batch8192-cache80-t16 | context-quality | 32768 | 28672 | 1 | 681.84 | 27.69 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0121… | guided-speed175-native-profile-20261005 | batch16384-cache16-t16 | synthetic-speed | 32768 | 16384 | 3 | 859.39 | 26.64 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0122… | guided-speed175-native-profile-20261005 | batch16384-cache16-t16 | synthetic-speed | 32768 | 16384 | 3 | 29.96 | 26.62 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0127 | guided-speed175-native-profile-20261005 | batch16384-cache16-t16 | context-quality | 32768 | 8192 | 1 | 737.46 | 28.17 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0128 | guided-speed175-native-profile-20261005 | batch16384-cache16-t16 | context-quality | 32768 | 16384 | 1 | 860.55 | 27.01 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0129 | guided-speed175-native-profile-20261005 | batch16384-cache16-t16 | context-quality | 32768 | 28672 | 1 | 803.71 | 25.14 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0130… | http-baseline6-cache32-vision-20261004 | http-baseline6-cache32-vision-20261004 | synthetic-speed | 131072 | 8192 | 3 | 137.87 | 19.13 | 0 | NO_SUMMARY_STATUS |
| M0131… | http-baseline6-cache32-vision-20261004 | http-baseline6-cache32-vision-20261004 | synthetic-speed | 131072 | 8192 | 3 | 24.88 | 19.37 | 0 | NO_SUMMARY_STATUS |
| M0136… | http-baseline6-cache32-vision-20261004 | http-baseline6-cache32-vision-20261004 | context-quality | 131072 | 8192 | 3 | 137.09 | 20.02 | 3 | NO_SUMMARY_STATUS |
| M0139… | http-batch1024-cache32-vision-20261005 | http-batch1024-cache32-vision-20261005 | synthetic-speed | 131072 | 8192 | 3 | 224.31 | 19.80 | 0 | NO_SUMMARY_STATUS |
| M0140… | http-batch1024-cache32-vision-20261005 | http-batch1024-cache32-vision-20261005 | synthetic-speed | 131072 | 8192 | 3 | 28.05 | 20.13 | 0 | NO_SUMMARY_STATUS |
| M0145… | http-context32-cache32-u1024-20261005 | http-context32-cache32-u1024-20261005 | synthetic-speed | 32768 | 8192 | 3 | 224.38 | 20.09 | 0 | NO_SUMMARY_STATUS |
| M0146… | http-context32-cache32-u1024-20261005 | http-context32-cache32-u1024-20261005 | synthetic-speed | 32768 | 8192 | 3 | 28.38 | 20.10 | 0 | NO_SUMMARY_STATUS |
| M0151… | http-threads14-cache32-vision-20261004 | http-threads14-cache32-vision-20261004 | synthetic-speed | 131072 | 8192 | 3 | 138.02 | 19.76 | 0 | NO_SUMMARY_STATUS |
| M0152… | http-threads14-cache32-vision-20261004 | http-threads14-cache32-vision-20261004 | synthetic-speed | 131072 | 8192 | 3 | 28.66 | 19.73 | 0 | NO_SUMMARY_STATUS |
| M0157 | ik178/results/control32-noMTP | ik-ctx32768-cpu43-f16-u8192-MTP0 | context-quality | 32768 | 8192 | 1 | 1093.24 | 31.06 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0158 | ik178/results/control32-noMTP | ik-ctx32768-cpu43-f16-u8192-MTP0 | context-quality | 32768 | 28672 | 1 | 1089.75 | 30.91 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0159 | ik178/results/control32-noMTP | ik-ctx32768-cpu43-f16-u8192-MTP0 | synthetic-speed | 32768 | 8192 | 1 | 1223.08 | 30.94 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0160 | ik178/results/control32-noMTP | ik-ctx32768-cpu43-f16-u8192-MTP0 | synthetic-speed | 32768 | 28672 | 1 | 1089.84 | 30.73 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0161 | ik178/results/mtp32-n1-mainU4096-draftU512 | ik-ctx32768-cpu43-f16-u4096-MTP1 | context-quality | 32768 | 8192 | 1 | 690.51 | 37.55 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0162 | ik178/results/mtp32-n1-mainU4096-draftU512 | ik-ctx32768-cpu43-f16-u4096-MTP1 | context-quality | 32768 | 28672 | 1 | 786.11 | 33.96 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0163 | ik178/results/mtp32-n1-mainU4096-draftU512 | ik-ctx32768-cpu43-f16-u4096-MTP1 | synthetic-speed | 32768 | 8192 | 1 | 911.13 | 33.88 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0164 | ik178/results/mtp32-n1-mainU4096-draftU512 | ik-ctx32768-cpu43-f16-u4096-MTP1 | synthetic-speed | 32768 | 28672 | 1 | 851.70 | 34.26 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0165… | iq3-author32-accept-20261005 | author-control | synthetic-speed | 32768 | 8192 | 3 | 183.07 | 26.73 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0166… | iq3-author32-accept-20261005 | author-control | synthetic-speed | 32768 | 8192 | 3 | 24.68 | 26.76 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0171 | iq3-author32-accept-20261005 | author-control | context-quality | 32768 | 8192 | 1 | 181.48 | 27.02 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0172 | iq3-author32-accept-20261005 | author-control | context-quality | 32768 | 16384 | 1 | 177.49 | 26.10 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0173… | iq3-author32-accept-20261005 | author-cpu-mtp1 | synthetic-speed | 32768 | 8192 | 3 | 174.23 | 25.35 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0174… | iq3-author32-accept-20261005 | author-cpu-mtp1 | synthetic-speed | 32768 | 8192 | 3 | 22.10 | 25.17 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0179 | iq3-author32-accept-20261005 | author-cpu-mtp1 | context-quality | 32768 | 8192 | 1 | 173.36 | 26.25 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0180 | iq3-author32-accept-20261005 | author-cpu-mtp1 | context-quality | 32768 | 16384 | 1 | 169.15 | 21.60 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0181… | iq3-cache96-accept-20261005 | cache96-control | synthetic-speed | 32768 | 8192 | 3 | 192.70 | 29.88 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0182… | iq3-cache96-accept-20261005 | cache96-control | synthetic-speed | 32768 | 8192 | 3 | 27.05 | 29.84 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0187 | iq3-cache96-accept-20261005 | cache96-control | context-quality | 32768 | 8192 | 1 | 190.63 | 30.15 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0188 | iq3-cache96-accept-20261005 | cache96-control | context-quality | 32768 | 16384 | 1 | 186.25 | 28.50 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0189… | iq3-cache96-t14-u4096-accept-r1-20261005 | cache96-t14-u4096-control | synthetic-speed | 32768 | 8192 | 3 | 594.30 | 31.86 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0190… | iq3-cache96-t14-u4096-accept-r1-20261005 | cache96-t14-u4096-control | synthetic-speed | 32768 | 8192 | 3 | 31.62 | 31.81 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0195 | iq3-cache96-t14-u4096-accept-r1-20261005 | cache96-t14-u4096-control | context-quality | 32768 | 8192 | 1 | 590.48 | 32.86 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0196 | iq3-cache96-t14-u4096-accept-r1-20261005 | cache96-t14-u4096-control | context-quality | 32768 | 16384 | 1 | 584.80 | 30.97 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0197 | opt181/results/opt256-cache4g-ple-overlap-workspace | opt256-cache4g-ple-overlap-workspace | context-quality | 262144 | 8192 | 1 | 1241.43 | 32.80 | 1 | FAIL_NO_RETRY |
| M0198 | opt181/results/opt256-cache4g-ple-overlap-workspace | opt256-cache4g-ple-overlap-workspace | synthetic-speed | 262144 | 8192 | 1 | 1287.75 | 35.38 | 0 | FAIL_NO_RETRY |
| M0199… | placement-speed175-20261005 | placement-control-mmap | synthetic-speed | 32768 | 8192 | 3 | 745.52 | 31.20 | 0 | FAIL_NO_RETRY |
| M0200… | placement-speed175-20261005 | placement-control-mmap | synthetic-speed | 32768 | 8192 | 3 | 32.30 | 31.16 | 0 | FAIL_NO_RETRY |
| M0205 | placement-speed175-20261005 | placement-control-mmap | context-quality | 32768 | 8192 | 1 | 742.69 | 31.70 | 1 | FAIL_NO_RETRY |
| M0206 | placement-speed175-20261005 | placement-control-mmap | context-quality | 32768 | 16384 | 1 | 729.63 | 30.02 | 1 | FAIL_NO_RETRY |
| M0207 | placement-speed175-20261005 | placement-control-mmap | context-quality | 32768 | 28672 | 1 | 684.31 | 27.77 | 1 | FAIL_NO_RETRY |
| M0208… | placement-speed175-r2-20261005 | placement-none-lazy | synthetic-speed | 32768 | 8192 | 3 | 792.41 | 30.27 | 0 | FAIL_NO_RETRY |
| M0209… | placement-speed175-r2-20261005 | placement-none-lazy | synthetic-speed | 32768 | 8192 | 3 | 32.66 | 31.03 | 0 | FAIL_NO_RETRY |
| M0214 | placement-speed175-r2-20261005 | placement-none-lazy | context-quality | 32768 | 8192 | 1 | 813.96 | 31.29 | 1 | FAIL_NO_RETRY |
| M0215 | placement-speed175-r2-20261005 | placement-none-lazy | context-quality | 32768 | 16384 | 1 | 800.49 | 30.12 | 1 | FAIL_NO_RETRY |
| M0216 | placement-speed175-r2-20261005 | placement-none-lazy | context-quality | 32768 | 28672 | 1 | 754.59 | 27.76 | 1 | FAIL_NO_RETRY |
| M0217… | prefill-dma175-20261005 | uncached-hybrid42-mtp2-pinned | synthetic-speed | 32768 | 8192 | 3 | 1201.53 | 38.25 | 0 | FAIL_NO_RETRY |
| M0218… | prefill-dma175-20261005 | uncached-hybrid42-mtp2-pinned | synthetic-speed | 32768 | 8192 | 3 | 29.36 | 38.29 | 0 | FAIL_NO_RETRY |
| M0223 | prefill-dma175-20261005 | uncached-hybrid42-mtp2-pinned | context-quality | 32768 | 8192 | 1 | 1181.96 | 37.85 | 1 | FAIL_NO_RETRY |
| M0224 | prefill-dma175-20261005 | uncached-hybrid42-mtp2-pinned | context-quality | 32768 | 16384 | 1 | 1147.50 | 38.41 | 1 | FAIL_NO_RETRY |
| M0225 | prefill-dma175-20261005 | uncached-hybrid42-mtp2-pinned | context-quality | 32768 | 28672 | 1 | 1074.35 | 36.49 | 1 | FAIL_NO_RETRY |
| M0226… | prefill-finalist175-20261005 | pinned-hybrid43-mtp2-q8 | synthetic-speed | 32768 | 8192 | 5 | 1197.67 | 32.88 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0227… | prefill-finalist175-20261005 | pinned-hybrid43-mtp2-q8 | synthetic-speed | 32768 | 8192 | 5 | 28.71 | 33.07 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0236 | prefill-finalist175-20261005 | pinned-hybrid43-mtp2-q8 | context-quality | 32768 | 8192 | 1 | 1191.41 | 37.62 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0237 | prefill-finalist175-20261005 | pinned-hybrid43-mtp2-q8 | context-quality | 32768 | 16384 | 1 | 1149.71 | 39.58 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0238 | prefill-finalist175-20261005 | pinned-hybrid43-mtp2-q8 | context-quality | 32768 | 28672 | 1 | 1068.14 | 34.72 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0239… | prefill-finalist175-20261005 | pinned-hybrid43-mtp2-f16 | synthetic-speed | 32768 | 8192 | 5 | 1197.90 | 35.97 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0240… | prefill-finalist175-20261005 | pinned-hybrid43-mtp2-f16 | synthetic-speed | 32768 | 8192 | 5 | 28.80 | 35.76 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0249 | prefill-finalist175-20261005 | pinned-hybrid43-mtp2-f16 | context-quality | 32768 | 8192 | 1 | 1181.64 | 37.14 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0250 | prefill-finalist175-20261005 | pinned-hybrid43-mtp2-f16 | context-quality | 32768 | 16384 | 1 | 1140.96 | 37.46 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0251 | prefill-finalist175-20261005 | pinned-hybrid43-mtp2-f16 | context-quality | 32768 | 28672 | 1 | 1068.97 | 37.43 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0252… | prefill-hybrid175-20261005 | uncached-hybrid38 | synthetic-speed | 32768 | 8192 | 3 | 1006.27 | 30.88 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0253… | prefill-hybrid175-20261005 | uncached-hybrid38 | synthetic-speed | 32768 | 8192 | 3 | 33.18 | 31.12 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0258 | prefill-hybrid175-20261005 | uncached-hybrid38 | context-quality | 32768 | 8192 | 1 | 1135.69 | 30.59 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0259 | prefill-hybrid175-20261005 | uncached-hybrid38 | context-quality | 32768 | 16384 | 1 | 1117.55 | 29.32 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0260 | prefill-hybrid175-20261005 | uncached-hybrid38 | context-quality | 32768 | 28672 | 1 | 1034.20 | 27.07 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0261… | prefill-hybrid175-20261005 | uncached-hybrid42-mtp2 | synthetic-speed | 32768 | 8192 | 3 | 1049.74 | 38.10 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0262… | prefill-hybrid175-20261005 | uncached-hybrid42-mtp2 | synthetic-speed | 32768 | 8192 | 3 | 28.99 | 37.80 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0267 | prefill-hybrid175-20261005 | uncached-hybrid42-mtp2 | context-quality | 32768 | 8192 | 1 | 1056.39 | 38.46 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0268 | prefill-hybrid175-20261005 | uncached-hybrid42-mtp2 | context-quality | 32768 | 16384 | 1 | 1025.59 | 37.85 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0269 | prefill-hybrid175-20261005 | uncached-hybrid42-mtp2 | context-quality | 32768 | 28672 | 1 | 946.02 | 37.25 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0270… | prefill-kv175-20261005 | pinned-hybrid43-mtp3-q8 | synthetic-speed | 32768 | 8192 | 3 | 1174.12 | 30.47 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0271… | prefill-kv175-20261005 | pinned-hybrid43-mtp3-q8 | synthetic-speed | 32768 | 8192 | 3 | 28.49 | 30.55 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0276 | prefill-kv175-20261005 | pinned-hybrid43-mtp3-q8 | context-quality | 32768 | 8192 | 1 | 1179.96 | 41.73 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0277 | prefill-kv175-20261005 | pinned-hybrid43-mtp3-q8 | context-quality | 32768 | 16384 | 1 | 1144.29 | 34.76 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0278 | prefill-kv175-20261005 | pinned-hybrid43-mtp3-q8 | context-quality | 32768 | 28672 | 1 | 1069.97 | 36.43 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0279… | prefill-kv175-20261005 | pinned-hybrid43-mtp3-f16 | synthetic-speed | 32768 | 8192 | 3 | 1198.38 | 32.57 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0280… | prefill-kv175-20261005 | pinned-hybrid43-mtp3-f16 | synthetic-speed | 32768 | 8192 | 3 | 28.61 | 32.33 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0285 | prefill-kv175-20261005 | pinned-hybrid43-mtp3-f16 | context-quality | 32768 | 8192 | 1 | 1179.17 | 35.71 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0286 | prefill-kv175-20261005 | pinned-hybrid43-mtp3-f16 | context-quality | 32768 | 16384 | 1 | 1144.26 | 39.41 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0287 | prefill-kv175-20261005 | pinned-hybrid43-mtp3-f16 | context-quality | 32768 | 28672 | 1 | 1071.24 | 38.33 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0288… | prefill-mtp1-finalist175-20261005 | pinned-hybrid43-mtp1-f16 | synthetic-speed | 32768 | 8192 | 5 | 1197.67 | 34.13 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0289… | prefill-mtp1-finalist175-20261005 | pinned-hybrid43-mtp1-f16 | synthetic-speed | 32768 | 8192 | 5 | 29.52 | 33.98 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0298 | prefill-mtp1-finalist175-20261005 | pinned-hybrid43-mtp1-f16 | context-quality | 32768 | 8192 | 1 | 1182.24 | 35.54 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0299 | prefill-mtp1-finalist175-20261005 | pinned-hybrid43-mtp1-f16 | context-quality | 32768 | 16384 | 1 | 1145.49 | 34.53 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0300 | prefill-mtp1-finalist175-20261005 | pinned-hybrid43-mtp1-f16 | context-quality | 32768 | 28672 | 1 | 1069.83 | 32.75 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0301 | prefill-observe175-20261005/benchmark | benchmark | synthetic-speed | 32768 | 16384 | 1 | 739.40 | 29.34 | 0 | NO_SUMMARY_STATUS |
| M0302 | prefill-observe175-20261005/benchmark | benchmark | synthetic-speed | 32768 | 16384 | 1 | 30.57 | 29.36 | 0 | NO_SUMMARY_STATUS |
| M0303… | prefill-path175-20261005 | uncached-cpu99 | synthetic-speed | 32768 | 8192 | 3 | 1085.49 | 28.18 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0304… | prefill-path175-20261005 | uncached-cpu99 | synthetic-speed | 32768 | 8192 | 3 | 29.81 | 28.23 | 0 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0309 | prefill-path175-20261005 | uncached-cpu99 | context-quality | 32768 | 8192 | 1 | 1078.93 | 28.18 | 1 | PASS_QA_NOT_PRODUCTION_ACCEPTANCE |
| M0310 | upstream178/results/control32-noMTP | codacus-ctx32768-cpu43-f16-u8192-noMTP | synthetic-speed | 32768 | 8192 | 1 | 1156.42 | 30.14 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0311 | upstream178/results/control32-noMTP | codacus-ctx32768-cpu43-f16-u8192-noMTP | context-quality | 32768 | 8192 | 1 | 1282.31 | 29.93 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0312 | upstream178/results/control32-noMTP | codacus-ctx32768-cpu43-f16-u8192-noMTP | synthetic-speed | 32768 | 28672 | 1 | 1154.12 | 27.58 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0313 | upstream178/results/control32-noMTP | codacus-ctx32768-cpu43-f16-u8192-noMTP | context-quality | 32768 | 28672 | 1 | 1153.50 | 27.68 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0314 | upstream178/results/upstream256-cpu43-u2048-noMTP | upstream-ctx262144-cpu43-f16-u2048-noMTP | synthetic-speed | 262144 | 8192 | 1 | 818.48 | 31.89 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0315 | upstream178/results/upstream256-cpu43-u2048-noMTP | upstream-ctx262144-cpu43-f16-u2048-noMTP | context-quality | 262144 | 8192 | 1 | 822.70 | 32.08 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0316 | upstream178/results/upstream256-cpu43-u2048-noMTP | upstream-ctx262144-cpu43-f16-u2048-noMTP | synthetic-speed | 262144 | 131072 | 1 | 738.97 | 29.78 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0317 | upstream178/results/upstream256-cpu43-u2048-noMTP | upstream-ctx262144-cpu43-f16-u2048-noMTP | context-quality | 262144 | 131072 | 1 | 736.71 | 29.71 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0318 | upstream178/results/upstream256-cpu43-u2048-noMTP | upstream-ctx262144-cpu43-f16-u2048-noMTP | synthetic-speed | 262144 | 235929 | 1 | 695.04 | 28.30 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0319 | upstream178/results/upstream256-cpu43-u2048-noMTP | upstream-ctx262144-cpu43-f16-u2048-noMTP | context-quality | 262144 | 235929 | 1 | 694.27 | 28.32 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0320 | upstream178/results/upstream256-cpu46-u4096-deep179 | upstream-ctx262144-cpu46-f16-u4096-noMTP | synthetic-speed | 262144 | 235929 | 1 | 905.76 | 27.39 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0321 | upstream178/results/upstream256-cpu46-u4096-deep179 | upstream-ctx262144-cpu46-f16-u4096-noMTP | context-quality | 262144 | 235929 | 1 | 905.25 | 27.46 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0322 | upstream178/results/upstream256-cpu46-u4096-short179 | upstream-ctx262144-cpu46-f16-u4096-noMTP | synthetic-speed | 262144 | 8192 | 1 | 1066.83 | 30.45 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0323 | upstream178/results/upstream256-cpu46-u4096-short179 | upstream-ctx262144-cpu46-f16-u4096-noMTP | context-quality | 262144 | 8192 | 1 | 1082.61 | 30.36 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0324 | upstream178/results/upstream256-f16-noMTP | upstream-ctx262144-cpu48-f16-u4096-noMTP | synthetic-speed | 262144 | 8192 | 1 | 1046.63 | 29.55 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0325 | upstream178/results/upstream256-f16-noMTP | upstream-ctx262144-cpu48-f16-u4096-noMTP | context-quality | 262144 | 8192 | 1 | 1063.23 | 29.63 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0326 | upstream178/results/upstream256-f16-noMTP | upstream-ctx262144-cpu48-f16-u4096-noMTP | synthetic-speed | 262144 | 131072 | 1 | 962.91 | 28.06 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0327 | upstream178/results/upstream256-f16-noMTP | upstream-ctx262144-cpu48-f16-u4096-noMTP | context-quality | 262144 | 131072 | 1 | 960.81 | 27.90 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0328 | upstream178/results/upstream256-f16-noMTP | upstream-ctx262144-cpu48-f16-u4096-noMTP | synthetic-speed | 262144 | 235929 | 1 | 898.37 | 26.78 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0329 | upstream178/results/upstream256-f16-noMTP | upstream-ctx262144-cpu48-f16-u4096-noMTP | context-quality | 262144 | 235929 | 1 | 896.98 | 26.80 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0330 | upstream178/results/upstream256-noMTP | upstream-ctx262144-cpu48-q8_0-u4096-noMTP | synthetic-speed | 262144 | 8192 | 1 | 1046.51 | 28.91 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0331 | upstream178/results/upstream256-noMTP | upstream-ctx262144-cpu48-q8_0-u4096-noMTP | context-quality | 262144 | 8192 | 1 | 1063.28 | 29.45 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0332 | upstream178/results/upstream256-noMTP | upstream-ctx262144-cpu48-q8_0-u4096-noMTP | synthetic-speed | 262144 | 131072 | 1 | 960.72 | 20.85 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0333 | upstream178/results/upstream256-noMTP | upstream-ctx262144-cpu48-q8_0-u4096-noMTP | context-quality | 262144 | 131072 | 1 | 957.25 | 20.88 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0334 | upstream178/results/upstream256-noMTP | upstream-ctx262144-cpu48-q8_0-u4096-noMTP | synthetic-speed | 262144 | 235929 | 1 | 894.82 | 16.77 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0335 | upstream178/results/upstream256-noMTP | upstream-ctx262144-cpu48-q8_0-u4096-noMTP | context-quality | 262144 | 235929 | 1 | 895.03 | 16.77 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0336 | upstream178/results/upstream32-noMTP | upstream-ctx32768-cpu43-f16-u8192-noMTP | synthetic-speed | 32768 | 8192 | 1 | 1334.01 | 32.14 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0337 | upstream178/results/upstream32-noMTP | upstream-ctx32768-cpu43-f16-u8192-noMTP | context-quality | 32768 | 8192 | 1 | 1368.90 | 32.15 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0338 | upstream178/results/upstream32-noMTP | upstream-ctx32768-cpu43-f16-u8192-noMTP | synthetic-speed | 32768 | 28672 | 1 | 1253.31 | 31.42 | 0 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0339 | upstream178/results/upstream32-noMTP | upstream-ctx32768-cpu43-f16-u8192-noMTP | context-quality | 32768 | 28672 | 1 | 1245.47 | 31.38 | 1 | PASS_FINITE_PROTOCOL_AND_RETRIEVAL_NOT_PRODUCTION_ACCEPTANCE |
| M0340 | historical180/benchmarks-20261004/cache-c32-t14-a1-20261004T214318Z-578888 | t14-b2048-u512-PP512-TG0 | native-random | — | 512 | 1 | 209.55 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0341 | historical180/benchmarks-20261004/cache-c32-t14-a1-20261004T214318Z-578888 | t14-b2048-u512-PP0-TG256 | native-random | — | 0 | 1 | — | 19.67 | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0342 | historical180/benchmarks-20261004/cache-c40-t14-a1-20261004T214501Z-581674 | t14-b2048-u512-PP512-TG0 | native-random | — | 512 | 1 | 211.05 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0343 | historical180/benchmarks-20261004/cache-c40-t14-a1-20261004T214501Z-581674 | t14-b2048-u512-PP0-TG256 | native-random | — | 0 | 1 | — | 19.81 | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0344 | historical180/benchmarks-20261004/cache-c48-t14-a1-20261004T214653Z-584724 | t14-b2048-u512-PP512-TG0 | native-random | — | 512 | 1 | 216.51 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0345 | historical180/benchmarks-20261004/cache-c48-t14-a1-20261004T214653Z-584724 | t14-b2048-u512-PP0-TG256 | native-random | — | 0 | 1 | — | 19.98 | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0346 | historical180/benchmarks-20261004/overlap-c48-t14-a0-20261004T214842Z-587507 | t14-b2048-u512-PP512-TG0 | native-random | — | 512 | 1 | 214.00 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0347 | historical180/benchmarks-20261004/overlap-c48-t14-a0-20261004T214842Z-587507 | t14-b2048-u512-PP0-TG256 | native-random | — | 0 | 1 | — | 22.96 | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0348 | historical180/benchmarks-20261004/overlap-c48-t14-a1-20261004T215031Z-590180 | t14-b2048-u512-PP512-TG0 | native-random | — | 512 | 1 | 214.31 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0349 | historical180/benchmarks-20261004/overlap-c48-t14-a1-20261004T215031Z-590180 | t14-b2048-u512-PP0-TG256 | native-random | — | 0 | 1 | — | 19.84 | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0350 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b1024-u1024-pf0-20261004T223734Z-660149 | t14-b1024-u1024-PP4096-TG0 | native-random | — | 4096 | 1 | 318.05 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0351 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b1024-u128-pf0-20261004T223048Z-649423 | t14-b1024-u128-PP4096-TG0 | native-random | — | 4096 | 1 | 86.02 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0352 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b1024-u256-pf0-20261004T223403Z-654270 | t14-b1024-u256-PP4096-TG0 | native-random | — | 4096 | 1 | 133.10 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0353 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b1024-u512-pf0-20261004T223610Z-657788 | t14-b1024-u512-PP4096-TG0 | native-random | — | 4096 | 1 | 206.96 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0354 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u1024-pf0-20261004T224515Z-669667 | t14-b2048-u1024-PP4096-TG0 | native-random | — | 4096 | 1 | 318.36 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0355 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u128-pf0-20261004T223830Z-661175 | t14-b2048-u128-PP4096-TG0 | native-random | — | 4096 | 1 | 86.54 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0356 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u256-pf0-20261004T224144Z-665751 | t14-b2048-u256-PP4096-TG0 | native-random | — | 4096 | 1 | 133.79 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0357 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u512-pf0-20261004T224351Z-668259 | t14-b2048-u512-PP4096-TG0 | native-random | — | 4096 | 1 | 207.36 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0358 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b4096-u1024-pf0-20261004T225256Z-677442 | t14-b4096-u1024-PP4096-TG0 | native-random | — | 4096 | 1 | 317.18 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0359 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b4096-u128-pf0-20261004T224611Z-670693 | t14-b4096-u128-PP4096-TG0 | native-random | — | 4096 | 1 | 86.40 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0360 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b4096-u256-pf0-20261004T224925Z-673871 | t14-b4096-u256-PP4096-TG0 | native-random | — | 4096 | 1 | 133.58 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0361 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b4096-u512-pf0-20261004T225132Z-676022 | t14-b4096-u512-PP4096-TG0 | native-random | — | 4096 | 1 | 206.93 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0362 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b512-u128-pf0-20261004T222319Z-634105 | t14-b512-u128-PP4096-TG0 | native-random | — | 4096 | 1 | 79.32 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0363 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b512-u256-pf0-20261004T222716Z-642697 | t14-b512-u256-PP4096-TG0 | native-random | — | 4096 | 1 | 132.75 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0364 | historical180/benchmarks-20261004/prefill-c32-t14-a1-b512-u512-pf0-20261004T222924Z-646700 | t14-b512-u512-PP4096-TG0 | native-random | — | 4096 | 1 | 207.25 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0365 | historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469 | t4-b2048-u512-PP512-TG0 | native-random | — | 512 | 1 | 164.03 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0366 | historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469 | t4-b2048-u512-PP0-TG256 | native-random | — | 0 | 1 | — | 18.68 | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0367 | historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469 | t6-b2048-u512-PP512-TG0 | native-random | — | 512 | 1 | 167.81 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0368 | historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469 | t6-b2048-u512-PP0-TG256 | native-random | — | 0 | 1 | — | 18.83 | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0369 | historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469 | t8-b2048-u512-PP512-TG0 | native-random | — | 512 | 1 | 170.04 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0370 | historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469 | t8-b2048-u512-PP0-TG256 | native-random | — | 0 | 1 | — | 18.47 | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0371 | historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469 | t12-b2048-u512-PP512-TG0 | native-random | — | 512 | 1 | 171.98 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0372 | historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469 | t12-b2048-u512-PP0-TG256 | native-random | — | 0 | 1 | — | 19.13 | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0373 | historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469 | t14-b2048-u512-PP512-TG0 | native-random | — | 512 | 1 | 170.45 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0374 | historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469 | t14-b2048-u512-PP0-TG256 | native-random | — | 0 | 1 | — | 19.27 | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0375 | historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469 | t16-b2048-u512-PP512-TG0 | native-random | — | 512 | 1 | 171.98 | — | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0376 | historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469 | t16-b2048-u512-PP0-TG256 | native-random | — | 0 | 1 | — | 19.10 | 0 | PROCESS_PASS_REQUIRES_RESULT_REVIEW |
| M0377 | historical180/benchmarks-pp8192-20261005/pp8192-c48-t14-a0-b8192-u8192-20261005T073938Z-1096850 | t14-b8192-u8192-PP8192-TG0 | native-random | — | 8192 | 1 | 573.18 | — | 0 | VALIDATED_STOCK_PP8192_REQUIRES_OPERATOR_REVIEW |
| M0378 | historical180/benchmarks-speed-20261005/speed-c48-t14-a0-b2048-u2048-nopo0-20261005T064446Z-1028912 | t14-b2048-u2048-PP4096-TG0 | native-random | — | 4096 | 1 | 314.15 | — | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0379 | historical180/benchmarks-speed-20261005/speed-c48-t14-a0-b2048-u2048-nopo0-20261005T064446Z-1028912 | t14-b2048-u2048-PP0-TG256 | native-random | — | 0 | 1 | — | 22.46 | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0380 | historical180/benchmarks-speed-20261005/speed-c48-t14-a0-b4096-u2048-nopo0-20261005T064617Z-1031775 | t14-b4096-u2048-PP4096-TG0 | native-random | — | 4096 | 1 | 468.64 | — | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0381 | historical180/benchmarks-speed-20261005/speed-c48-t14-a0-b4096-u2048-nopo0-20261005T064617Z-1031775 | t14-b4096-u2048-PP0-TG256 | native-random | — | 0 | 1 | — | 22.91 | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0382 | historical180/benchmarks-speed-20261005/speed-c48-t14-a0-b4096-u4096-nopo0-20261005T064730Z-1034279 | t14-b4096-u4096-PP4096-TG0 | native-random | — | 4096 | 1 | 642.70 | — | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0383 | historical180/benchmarks-speed-20261005/speed-c48-t14-a0-b4096-u4096-nopo0-20261005T064730Z-1034279 | t14-b4096-u4096-PP0-TG256 | native-random | — | 0 | 1 | — | 22.94 | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0384 | historical180/benchmarks-speed-20261005/speed-c64-t14-a0-b2048-u2048-nopo0-20261005T064835Z-1036566 | t14-b2048-u2048-PP4096-TG0 | native-random | — | 4096 | 1 | 476.87 | — | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0385 | historical180/benchmarks-speed-20261005/speed-c64-t14-a0-b2048-u2048-nopo0-20261005T064835Z-1036566 | t14-b2048-u2048-PP0-TG256 | native-random | — | 0 | 1 | — | 23.25 | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0386 | historical180/benchmarks-speed-20261005/speed-c64-t14-a0-b4096-u2048-nopo0-20261005T064948Z-1038971 | t14-b4096-u2048-PP4096-TG0 | native-random | — | 4096 | 1 | 475.83 | — | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0387 | historical180/benchmarks-speed-20261005/speed-c64-t14-a0-b4096-u2048-nopo0-20261005T064948Z-1038971 | t14-b4096-u2048-PP0-TG256 | native-random | — | 0 | 1 | — | 23.24 | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0388 | historical180/benchmarks-speed-20261005/speed-c64-t14-a0-b4096-u4096-nopo0-20261005T065101Z-1041105 | t14-b4096-u4096-PP4096-TG0 | native-random | — | 4096 | 1 | 646.80 | — | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0389 | historical180/benchmarks-speed-20261005/speed-c64-t14-a0-b4096-u4096-nopo0-20261005T065101Z-1041105 | t14-b4096-u4096-PP0-TG256 | native-random | — | 0 | 1 | — | 23.24 | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0390 | historical180/benchmarks-speed-20261005/speed-c80-t14-a0-b2048-u2048-nopo0-20261005T065205Z-1042879 | t14-b2048-u2048-PP4096-TG0 | native-random | — | 4096 | 1 | 483.61 | — | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0391 | historical180/benchmarks-speed-20261005/speed-c80-t14-a0-b2048-u2048-nopo0-20261005T065205Z-1042879 | t14-b2048-u2048-PP0-TG256 | native-random | — | 0 | 1 | — | 23.72 | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0392 | historical180/benchmarks-speed-20261005/speed-c80-t14-a0-b4096-u2048-nopo0-20261005T065318Z-1044457 | t14-b4096-u2048-PP4096-TG0 | native-random | — | 4096 | 1 | 485.11 | — | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0393 | historical180/benchmarks-speed-20261005/speed-c80-t14-a0-b4096-u2048-nopo0-20261005T065318Z-1044457 | t14-b4096-u2048-PP0-TG256 | native-random | — | 0 | 1 | — | 23.63 | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0394 | historical180/benchmarks-speed-20261005/speed-c80-t14-a0-b4096-u4096-nopo0-20261005T065430Z-1047277 | t14-b4096-u4096-PP4096-TG0 | native-random | — | 4096 | 1 | 649.26 | — | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0395 | historical180/benchmarks-speed-20261005/speed-c80-t14-a0-b4096-u4096-nopo0-20261005T065430Z-1047277 | t14-b4096-u4096-PP0-TG256 | native-random | — | 0 | 1 | — | 23.61 | 0 | VALIDATED_STOCK_SCREEN_REQUIRES_OPERATOR_REVIEW |
| M0396 | strata188 | strata188 | warmup | 262144 | 65 | 1 | 46.56 | 27.25 | 0 | MAX_CONTEXT_THROUGHPUT_MEASURED_PRODUCTION_RESTORED |
| M0397… | strata188 | strata188 | synthetic-speed | 262144 | 260000 | 3 | 1230.04 | 30.40 | 0 | MAX_CONTEXT_THROUGHPUT_MEASURED_PRODUCTION_RESTORED |
| M0400 | strata188-ram48 | strata188-ram48 | warmup | 262144 | 65 | 1 | 9.41 | 7.00 | 0 | OWNER_INTERRUPTED |
| M0401 | strata188-ram48 | strata188-ram48 | synthetic-speed | 262144 | 260000 | 1 | 773.13 | 9.30 | 0 | OWNER_INTERRUPTED |
| M0402 | functional195 | public Responses long-input keepalive | functional-transport | 262144 | 230100 | 1 | 1067.78 | 26.56 | 0 | PASS_TRANSPORT_NOT_BENCHMARK |
| M0403 | functional195 | stock Codex long-input turn | functional-transport | 262144 | 232796 | 1 | 917.53 | 26.93 | 0 | PASS_TRANSPORT_NOT_BENCHMARK |
| M0404 | functional195 | stock native compaction | functional-transport | 262144 | 230530 | 1 | 919.76 | 26.77 | 0 | PASS_TRANSPORT_NOT_BENCHMARK |
| M0405 | functional195 | after-compaction file tool call | functional-transport | 262144 | 48886 | 1 | 862.41 | 28.46 | 0 | PASS_TRANSPORT_NOT_BENCHMARK |
| M0406 | functional195 | after-compaction tool result continuation | functional-transport | 262144 | 48981 | 1 | 101.71 | 28.99 | 0 | PASS_TRANSPORT_NOT_BENCHMARK |

## Appendix B. Excluded trials, series failures and profiling

| Series | Type | Rows | Status | Recorded diagnostic / scope |
| --- | --- | --- | --- | --- |
| context176-262144-cpu48-q8-20261005 | server-requests | 0 | FAIL_NO_RETRY | {'category': 'QAError', 'message': 'QAError: CUDA/assert error in own child log'} |
| context176-262144-cpu48-q8-u1024-20261005 | server-requests | 0 | FAIL_NO_RETRY | {'category': 'QAError', 'message': 'QAError: Finite QA signal SIGTERM'} |
| context176-65536-cpu43-q8-20261005 | server-requests | 0 | FAIL_NO_RETRY | {'category': 'OSError', 'message': '[Errno 98] Address already in use'} |
| historical180/mtp-screen32-20261005 | server-requests | 0 | FAIL_MATRIX_ABORTED_NO_RETRY | {'type': 'QAError', 'message': 'native expert-cache64 engagement not proved'} |
| ik178/results/mtp32-n1 | server-requests | 0 | FAIL_NO_RETRY | No throughput record; see series metadata for profiling or startup scope |
| ik178/results/mtp32-n1-draftU512 | server-requests | 0 | FAIL_NO_RETRY | No throughput record; see series metadata for profiling or startup scope |
| opt181/results/opt256-cache4g-ple-overlap-workspace | server-requests | 2 | FAIL_NO_RETRY | No throughput record; see series metadata for profiling or startup scope |
| placement-speed175-20261005 | server-requests | 9 | FAIL_NO_RETRY | {'category': 'ConnectionResetError', 'message': '[Errno 104] Connection reset by peer'} |
| placement-speed175-r1-20261005 | server-requests | 0 | FAIL_NO_RETRY | {'category': 'QAError', 'message': "TimeoutExpired: Command '['/usr/bin/nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']' timed out after 8 seconds"} |
| placement-speed175-r2-20261005 | server-requests | 9 | FAIL_NO_RETRY | {'category': 'QAError', 'message': 'QA server did not stop gracefully'} |
| prefill-dma175-20261005 | server-requests | 9 | FAIL_NO_RETRY | {'category': 'QAError', 'message': "TimeoutExpired: Command '['/usr/bin/nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']' timed out after 0.7 seconds"} |
| historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u1024-pf2-20261005T060634Z-977668 | native-bench | 0 | BENCH_OR_MONITOR_FAILED | No throughput record; see series metadata for profiling or startup scope |
| profile-iq3-175-r1-20261005 | profiling-artifacts | 0 | PASS_STOCK_IQ3_PROFILE_NOT_DEPLOYED | No throughput record; see series metadata for profiling or startup scope |
| historical180/profile-iq3-175-20261005 | profiling-artifacts | 0 | FAIL_NO_RETRY | {'category': 'QAError', 'message': 'Native trace failed; no inference retry'} |
| historical180/profiles-20261004 | profiling-artifacts | 0 | NO_SUMMARY_STATUS | No throughput record; see series metadata for profiling or startup scope |
| strata188-ram48 | server-requests | 2 | OWNER_INTERRUPTED | No throughput record; see series metadata for profiling or startup scope |

## Appendix C. Configuration reference

Commands are specific to the pinned implementation. Substitute local paths for `<PATH>/filename`. A configuration record without an associated measurement is not evidence of a completed trial.

### C001 — context176-262144-cpu48-q8-20261005

Source: `context176-262144-cpu48-q8-20261005/ctx262144-cpu48-q8_0-b8192/configuration.json`, SHA256 `2d7c52b17b7a4b0a5182ea8b4189b12d74e36636a0e9f4a319195fa1fc300251`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14027 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 48 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 262144 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C002 — context176-262144-cpu48-q8-u1024-20261005

Source: `context176-262144-cpu48-q8-u1024-20261005/ctx262144-cpu48-q8_0-b8192-u1024/configuration.json`, SHA256 `e2cb8a7ff9f14b47988a588a4b56e9f69541b1fec30f6ef8bc37a24265b03139`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14027 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 48 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 262144 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 1024 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C003 — context176-262144-cpu48-q8-u2048-20261005

Source: `context176-262144-cpu48-q8-u2048-20261005/ctx262144-cpu48-q8_0-b8192-u2048/configuration.json`, SHA256 `eb0739664396eaa29e7e1196ccfe94a8e203ab42d109add7f99a0280d6c4b4f6`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14027 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 48 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 262144 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 2048 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C004 — context176-32768-cpu43-q8-20261005

Source: `context176-32768-cpu43-q8-20261005/ctx32768-cpu43-q8_0-b8192/configuration.json`, SHA256 `bfbc368e38b5ef2bb4a85a4d66565236e05d7dedcfc41fce73c27f1a03e0d771`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14027 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 43 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C005 — context176-32768-cpu48-q8-control-20261005

Source: `context176-32768-cpu48-q8-control-20261005/ctx32768-cpu48-q8_0-b8192/configuration.json`, SHA256 `5c7c5471de304f68ca306df546ec5590a1ed9ef63bd3e493a50d35ab8219d045`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14027 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 48 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C006 — context176-32768-cpu48-q8-u1024-control-20261005

Source: `context176-32768-cpu48-q8-u1024-control-20261005/ctx32768-cpu48-q8_0-b8192-u1024/configuration.json`, SHA256 `43566c28a2aad347a8e5366fe4772e14c128dbff6ee99de7a6eff188162e357c`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14027 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 48 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 1024 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C007 — context176-32768-cpu48-q8-u2048-control-20261005

Source: `context176-32768-cpu48-q8-u2048-control-20261005/ctx32768-cpu48-q8_0-b8192-u2048/configuration.json`, SHA256 `480938ce5744ef74cc07f20bae6df84fa65c0a178e94f25ae882b4ee12f7c611`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14027 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 48 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 2048 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C008 — context176-40960-cpu44-f16-u8192-screen-20261005

Source: `context176-40960-cpu44-f16-u8192-screen-20261005/ctx40960-cpu44-f16-b8192-u8192/configuration.json`, SHA256 `491e906764b52c775796c720a618cbe18d53c4b0c9522f7732eee01d1a956a24`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14027 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 44 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 40960 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C009 — context176-45056-cpu45-f16-u8192-screen-20261005

Source: `context176-45056-cpu45-f16-u8192-screen-20261005/ctx45056-cpu45-f16-b8192-u8192/configuration.json`, SHA256 `1b5c0f89909e03f461d8901faa280f36c78c6c5d5bae7d880e652ea3a99259a3`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14027 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 45 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 45056 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C010 — context176-47104-cpu45-f16-u8192-screen-20261005

Source: `context176-47104-cpu45-f16-u8192-screen-20261005/ctx47104-cpu45-f16-b8192-u8192/configuration.json`, SHA256 `9b8f25602bf689fd026f00bffb2ce94dd0e09c61987c42833db4af0b7ff02e68`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14027 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 45 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 47104 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C011 — context176-49152-cpu46-f16-u8192-screen-20261005

Source: `context176-49152-cpu46-f16-u8192-screen-20261005/ctx49152-cpu46-f16-b8192-u8192/configuration.json`, SHA256 `ffee4c421befbc2f9bb01336a5a00e386a02ef6cba5be48f523ab29641bab8bc`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14027 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 46 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 49152 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C012 — context176-65536-cpu48-f16-u8192-screen-20261005

Source: `context176-65536-cpu48-f16-u8192-screen-20261005/ctx65536-cpu48-f16-b8192-u8192/configuration.json`, SHA256 `0c02056115b63d74a5cbb37a2f75254dda81f84e2d59424a8c8e002a6bed2def`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14027 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 48 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 65536 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C013 — guided-speed175-20261005

Source: `guided-speed175-20261005/baseline96/configuration.json`, SHA256 `f5f2a2fdad1dfa2182df479de95d4d672536266e6b3577edace51c083646f840`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 96 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 4096 --ubatch-size 4096 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C014 — guided-speed175-20261005

Source: `guided-speed175-20261005/batch8192-cache96/configuration.json`, SHA256 `087814832b2ab620091c561a4a98892c9bb19f07e6ad8c34d8a0167e031f4cd4`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 96 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C015 — guided-speed175-20261005

Source: `guided-speed175-20261005/gpu-mtp2-cache96/configuration.json`, SHA256 `a485cd73d5fda457a43f623ec7551a264d41533b6c75a30ea5094713144b20d4`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 96 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 99 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 4096 --ubatch-size 4096 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C016 — guided-speed175-20261005

Source: `guided-speed175-20261005/overlap-cache112/configuration.json`, SHA256 `741ea1c27bf242763b06503282eaf126c7d27035a18d3c65e991f6145ab75b84`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 112 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 4096 --ubatch-size 4096 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C017 — guided-speed175-finalist-20261005

Source: `guided-speed175-finalist-20261005/batch8192-cache80/configuration.json`, SHA256 `fb9c4186871f22d7bce37aad37f53dcfa9fbf0c4bda8f66d124bee4c38e959db`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 80 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C018 — guided-speed175-followup-20261005

Source: `guided-speed175-followup-20261005/batch8192-cache80/configuration.json`, SHA256 `6b8f16237300e7125def1b702d0931a9956af3fde08bcb92f027eb4c27765a71`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 80 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C019 — guided-speed175-followup-20261005

Source: `guided-speed175-followup-20261005/gpu-mtp2-cache64/configuration.json`, SHA256 `f737871efe0e520a93facd3131f74a9e684b9f0a970904afb9ec7b8bf343a661`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 64 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 99 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 4096 --ubatch-size 4096 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C020 — guided-speed175-followup-20261005

Source: `guided-speed175-followup-20261005/hybrid-mtp2-b8192-cache72/configuration.json`, SHA256 `59e4fe17c871bf55f6b827a75c62acbb3e7b68e5c06bc39d8c82d88822679adc`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 72 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 99 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 0 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C021 — guided-speed175-native-profile-20261005

Source: `guided-speed175-native-profile-20261005/batch16384-cache16-t16/configuration.json`, SHA256 `1db4a961601d87d10a75d3b22ff824e977d6c7ba9224914d67bb18e2f174fb5d`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 16 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 16 --threads-batch 16 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 16384 --ubatch-size 16384 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C022 — guided-speed175-native-profile-20261005

Source: `guided-speed175-native-profile-20261005/batch8192-cache80/configuration.json`, SHA256 `fb9c4186871f22d7bce37aad37f53dcfa9fbf0c4bda8f66d124bee4c38e959db`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 80 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C023 — guided-speed175-native-profile-20261005

Source: `guided-speed175-native-profile-20261005/batch8192-cache80-t16/configuration.json`, SHA256 `6e1cdb8f05539023e9ae6f1adf3918a4f9dc7dca5470ad551fa13ac88b716853`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 80 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 16 --threads-batch 16 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C024 — historical180/mtp-screen32-20261005

Source: `historical180/mtp-screen32-20261005/control/configuration.json`, SHA256 `b15bb1ecb94ff5b10bd8fad553f478c17de1775ef37f7ef94af1cff4c04fb150`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14024 --ctx-size 32768 --parallel 1 --gpu-layers 99 --n-cpu-moe 99 --fit off --spec-type none --load-mode mmap --lazy-mode on --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --threads 14 --threads-batch 14 --batch-size 4096 --ubatch-size 4096 --no-sched-async-cpu --cpu-mask 0xffff --cpu-strict 1 --cache-prompt --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 64 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --offline
```

```json
{
  "LANG": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C025 — http-baseline6-cache32-vision-20261004

Source: `http-baseline6-cache32-vision-20261004/configuration.json`, SHA256 `8c8829a8ee0b31c41962533e19d9b0a63b15fdba5e9d67fe24a53c1eb3236f03`.

```text
Command unavailable in the recorded artifact
```

### C026 — http-batch1024-cache32-vision-20261005

Source: `http-batch1024-cache32-vision-20261005/configuration.json`, SHA256 `3d371913a816640ec1877fc7cbdc45bc7460396dbb7a37f4be456c5518e7fd2b`.

```text
Command unavailable in the recorded artifact
```

### C027 — http-context32-cache32-u1024-20261005

Source: `http-context32-cache32-u1024-20261005/configuration.json`, SHA256 `53b018526e9b800739dce80320aab9d1f4873f549440c1609ff274e79c6756cd`.

```text
Command unavailable in the recorded artifact
```

### C028 — http-threads14-cache32-vision-20261004

Source: `http-threads14-cache32-vision-20261004/configuration.json`, SHA256 `f2a316eff1d06613abb33796f06a6809a7fe92b1fe3a51c45acee4758447681e`.

```text
Command unavailable in the recorded artifact
```

### C029 — ik178/results/control32-noMTP

Source: `ik178/results/control32-noMTP/summary.json`, SHA256 `9b9be58d9282469e9c2aba8149c325b6a58226fe8d4ed47a845edb502d3be76c`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14029 --ctx-size 32768 --parallel 1 --gpu-layers 99 --n-cpu-moe 43 --no-mmap --defer-ple --flash-attn on --cache-type-k f16 --cache-type-v f16 --threads 14 --threads-batch 14 --cpu-mask 0xffff --batch-size 8192 --ubatch-size 8192 --temp 1 --top-p .95 --top-k 20 --min-p 0 --presence-penalty 0 --repeat-penalty 1 --jinja --metrics --no-context-shift --timeout 7200
```

### C030 — ik178/results/mtp32-n1

Source: `ik178/results/mtp32-n1/summary.json`, SHA256 `3a806194e6fb5f40482c99875e9d4515b995d654195f487ffff2c8e3c82fedba`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14029 --ctx-size 32768 --parallel 1 --gpu-layers 99 --n-cpu-moe 43 --no-mmap --defer-ple --flash-attn on --cache-type-k f16 --cache-type-v f16 --threads 14 --threads-batch 14 --cpu-mask 0xffff --batch-size 8192 --ubatch-size 8192 --temp 1 --top-p .95 --top-k 20 --min-p 0 --presence-penalty 0 --repeat-penalty 1 --jinja --metrics --no-context-shift --timeout 7200 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-type mtp:n_max=1,p_min=0.0 --cache-type-k-draft f16 --cache-type-v-draft f16 --threads-draft 14 --threads-batch-draft 14
```

### C031 — ik178/results/mtp32-n1-draftU512

Source: `ik178/results/mtp32-n1-draftU512/summary.json`, SHA256 `6136b95fc0c866ba4e61fbf3487405d6ee451b17ea050a65a16c10a5ddcc7bcc`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14029 --ctx-size 32768 --parallel 1 --gpu-layers 99 --n-cpu-moe 43 --no-mmap --defer-ple --flash-attn on --cache-type-k f16 --cache-type-v f16 --threads 14 --threads-batch 14 --cpu-mask 0xffff --batch-size 8192 --ubatch-size 8192 --temp 1 --top-p .95 --top-k 20 --min-p 0 --presence-penalty 0 --repeat-penalty 1 --jinja --metrics --no-context-shift --timeout 7200 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-type mtp:n_max=1,p_min=0.0 --cache-type-k-draft f16 --cache-type-v-draft f16 --threads-draft 14 --threads-batch-draft 14 --draft-params '--ubatch-size 512'
```

### C032 — ik178/results/mtp32-n1-mainU4096-draftU512

Source: `ik178/results/mtp32-n1-mainU4096-draftU512/summary.json`, SHA256 `37c8d10eb66e50bd0774a19f93ab523e39249403d4c38acdafd0988ebff3986d`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14029 --ctx-size 32768 --parallel 1 --gpu-layers 99 --n-cpu-moe 43 --no-mmap --defer-ple --flash-attn on --cache-type-k f16 --cache-type-v f16 --threads 14 --threads-batch 14 --cpu-mask 0xffff --batch-size 8192 --ubatch-size 4096 --temp 1 --top-p .95 --top-k 20 --min-p 0 --presence-penalty 0 --repeat-penalty 1 --jinja --metrics --no-context-shift --timeout 7200 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-type mtp:n_max=1,p_min=0.0 --cache-type-k-draft f16 --cache-type-v-draft f16 --threads-draft 14 --threads-batch-draft 14 --draft-params '--ubatch-size 512'
```

### C033 — iq3-author32-accept-20261005

Source: `iq3-author32-accept-20261005/author-control/configuration.json`, SHA256 `3e804083f04371e0fd259245f4c8e7327d54917604cdd3d8237aaa59d90d84f4`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14024 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 48 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 6 --threads-batch 6 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 2048 --ubatch-size 512 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C034 — iq3-author32-accept-20261005

Source: `iq3-author32-accept-20261005/author-cpu-mtp1/configuration.json`, SHA256 `c5173776d3d6a455bd55c3b9bdc541505bbd39a1e90a6da1832a17183ce1e487`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14024 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 48 --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 0 --spec-type draft-mtp --spec-draft-n-max 1 --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 6 --threads-batch 6 --threads-draft 6 --threads-batch-draft 6 --cpu-mask 0xffff --cpu-strict 1 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 2048 --ubatch-size 512 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C035 — iq3-cache96-accept-20261005

Source: `iq3-cache96-accept-20261005/cache96-control/configuration.json`, SHA256 `9d078f937789dd7fcbacb68dcd9e5be6b4c9b1d57ed111bce03e2f8561eef3b8`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14024 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 96 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 6 --threads-batch 6 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 2048 --ubatch-size 512 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C036 — iq3-cache96-t14-u4096-accept-r1-20261005

Source: `iq3-cache96-t14-u4096-accept-r1-20261005/cache96-t14-u4096-control/configuration.json`, SHA256 `e388672909f4b19d48eabc736a2a03fee5bf24de9684c83e019b47db4a259d59`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14025 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 96 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 4096 --ubatch-size 4096 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C037 — opt181/results/opt256-cache4g-ple-overlap-workspace

Source: `opt181/results/opt256-cache4g-ple-overlap-workspace/summary.json`, SHA256 `d81e16005e2f64a28c0d74ed8db3ed02166a0840fd0895471a30b9ea7f3b9d4c`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14030 --spec-type none --gpu-layers 99 --n-cpu-moe 48 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 262144 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 4096 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --no-context-shift --timeout 7200 --moe-expert-cache-mib 4096 --phase-aware-workspace --live-context-workspace --ple-prefetch --backend-sampling --decode-overlap
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C038 — placement-speed175-20261005

Source: `placement-speed175-20261005/placement-control-mmap/configuration.json`, SHA256 `fb9c4186871f22d7bce37aad37f53dcfa9fbf0c4bda8f66d124bee4c38e959db`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 80 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C039 — placement-speed175-20261005

Source: `placement-speed175-20261005/placement-none-lazy/configuration.json`, SHA256 `a143594576931cc78494a307d838d0347f5521095448220c455e98ad540f42ce`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 80 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C040 — placement-speed175-r1-20261005

Source: `placement-speed175-r1-20261005/placement-none-lazy/configuration.json`, SHA256 `a143594576931cc78494a307d838d0347f5521095448220c455e98ad540f42ce`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 80 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C041 — placement-speed175-r2-20261005

Source: `placement-speed175-r2-20261005/placement-none-lazy/configuration.json`, SHA256 `a143594576931cc78494a307d838d0347f5521095448220c455e98ad540f42ce`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 80 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C042 — prefill-dma175-20261005

Source: `prefill-dma175-20261005/uncached-hybrid42-mtp2-pinned/configuration.json`, SHA256 `ac4d4df09305f0d1ea50f9cdfd7559869fd93c5ccc0d9c450d82fabec915b75a`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 42 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C043 — prefill-finalist175-20261005

Source: `prefill-finalist175-20261005/pinned-hybrid43-mtp2-f16/configuration.json`, SHA256 `efcf44f8d07f15a7be2b26bf68ad5d3bf578976ac2138fa6887b72c012f4aded`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 43 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C044 — prefill-finalist175-20261005

Source: `prefill-finalist175-20261005/pinned-hybrid43-mtp2-q8/configuration.json`, SHA256 `12424a7e77d263f241d71201b3b444bc4a48910c534f9a56b66db1836b1941b2`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 43 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C045 — prefill-hybrid175-20261005

Source: `prefill-hybrid175-20261005/uncached-hybrid38/configuration.json`, SHA256 `c48fbe8bfcb4ad25e7d1d3be3c5917c398cd08658abce552421eec11f4a8bef7`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 0 --spec-type none --gpu-layers 99 --n-cpu-moe 38 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C046 — prefill-hybrid175-20261005

Source: `prefill-hybrid175-20261005/uncached-hybrid42-mtp2/configuration.json`, SHA256 `9328fdac61cc49145f4c01e2ce8a7027410b353089688b2013c6798d440eee8f`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 42 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C047 — prefill-kv175-20261005

Source: `prefill-kv175-20261005/pinned-hybrid43-mtp3-f16/configuration.json`, SHA256 `59c0bdd22f64f5c7e277be3c4a09f4ad1ab33607d2671c5de72ed528b58a76a8`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 43 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C048 — prefill-kv175-20261005

Source: `prefill-kv175-20261005/pinned-hybrid43-mtp3-q8/configuration.json`, SHA256 `be9f2454261d69f8c89d06cf18b9dd83fa40920bffedc1a3e52e5546c8ab2bb9`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 43 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C049 — prefill-mtp1-finalist175-20261005

Source: `prefill-mtp1-finalist175-20261005/pinned-hybrid43-mtp1-f16/configuration.json`, SHA256 `6c6690ef429d3cfea4c6be1478e6cd4d845910b0011fa79555987b5cae772fd8`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 0 --spec-type draft-mtp --gpu-layers 99 --n-cpu-moe 43 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --model-draft '<PATH>/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' --gpu-layers-draft 99 --spec-draft-n-max 1 --spec-draft-n-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --threads-draft 14 --threads-batch-draft 14 --cpu-mask-draft 0xffff --cpu-strict-draft 1 --cpu-mask-batch-draft 0xffff --cpu-strict-batch-draft 1
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C050 — prefill-path175-20261005

Source: `prefill-path175-20261005/uncached-cpu99/configuration.json`, SHA256 `1f8498aaa5c872a416120bd1dbd5ba24f7b701126233813fa09b7deb5642f35c`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload --alias jarvis-flash-next --host 127.0.0.1 --port 14026 --moe-cache-profile '<PATH>/routing-merged.csv' --moe-cache-slots 0 --spec-type none --gpu-layers 99 --n-cpu-moe 99 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode mmap --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0"
}
```

### C051 — upstream178/results/control32-noMTP

Source: `upstream178/results/control32-noMTP/summary.json`, SHA256 `5c0debeab732d308cd55543ee0d9dd135d76dde4915bbcd1eedf0b36f47cb23f`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14028 --moe-cache-profile '<PATH>/routing-iq3-175.csv' --moe-cache-slots 0 --spec-type none --gpu-layers 99 --n-cpu-moe 43 --no-sched-async-cpu --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --no-context-shift --timeout 7200
```

### C052 — upstream178/results/upstream256-cpu43-u2048-noMTP

Source: `upstream178/results/upstream256-cpu43-u2048-noMTP/summary.json`, SHA256 `46a8cfacb8e8799cc45e0c8a11a29d64ebef40ef180086b84f77ea781d07df2c`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14028 --spec-type none --gpu-layers 99 --n-cpu-moe 43 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 262144 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 2048 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --no-context-shift --timeout 7200
```

### C053 — upstream178/results/upstream256-cpu46-u4096-deep179

Source: `upstream178/results/upstream256-cpu46-u4096-deep179/summary.json`, SHA256 `364bfb1390edb821e2b87e1921af4f340f17d0c7f291d447b793c03437bbdf27`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14028 --spec-type none --gpu-layers 99 --n-cpu-moe 46 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 262144 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 4096 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --no-context-shift --timeout 7200
```

### C054 — upstream178/results/upstream256-cpu46-u4096-short179

Source: `upstream178/results/upstream256-cpu46-u4096-short179/summary.json`, SHA256 `9b0adac2f2dbff4be04fddef2aa90b094caafd2f1c217a6711956c6453044ef0`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14028 --spec-type none --gpu-layers 99 --n-cpu-moe 46 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 262144 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 4096 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --no-context-shift --timeout 7200
```

### C055 — upstream178/results/upstream256-f16-noMTP

Source: `upstream178/results/upstream256-f16-noMTP/summary.json`, SHA256 `c999cba80d5ade26cfb4b9e93f63295854dd7118644150a8cc7967b10c881c6a`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14028 --spec-type none --gpu-layers 99 --n-cpu-moe 48 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 262144 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 4096 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --no-context-shift --timeout 7200
```

### C056 — upstream178/results/upstream256-noMTP

Source: `upstream178/results/upstream256-noMTP/summary.json`, SHA256 `d6f234b71ffcd674d84abb60a8e9585f00628fe063ee50b2d449d661108e431b`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14028 --spec-type none --gpu-layers 99 --n-cpu-moe 48 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 262144 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 4096 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --no-context-shift --timeout 7200
```

### C057 — upstream178/results/upstream32-noMTP

Source: `upstream178/results/upstream32-noMTP/summary.json`, SHA256 `2bdd58a9329b42a8d1800349601c64370732f230d6809f4622dd91d5a33bf7bc`.

```text
'<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14028 --spec-type none --gpu-layers 99 --n-cpu-moe 43 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 32768 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --verbosity 4 --offline --no-context-shift --timeout 7200
```

### C058 — historical180/benchmarks-20261004/cache-c32-t14-a1-20261004T214318Z-578888

Source: `historical180/benchmarks-20261004/cache-c32-t14-a1-20261004T214318Z-578888/provenance.txt`, SHA256 `a75b75c63cacb66eb5eaf2fc0b3f1a52323ea4337b17bc0fdd67e264ed453b56`.

```text
utc=2026-10-04T21:43:18Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=cache
slots=32
threads=14
async_cpu=1
min_gpu_free_mib=4656
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 512 --sched-async-cpu 1 -p 512 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C059 — historical180/benchmarks-20261004/cache-c40-t14-a1-20261004T214501Z-581674

Source: `historical180/benchmarks-20261004/cache-c40-t14-a1-20261004T214501Z-581674/provenance.txt`, SHA256 `1cc229adab87896c90605bffdbe26a22ad5920446af4cf3ad546adf81eb7b42d`.

```text
utc=2026-10-04T21:45:01Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=cache
slots=40
threads=14
async_cpu=1
min_gpu_free_mib=4656
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=40
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 512 --sched-async-cpu 1 -p 512 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C060 — historical180/benchmarks-20261004/cache-c48-t14-a1-20261004T214653Z-584724

Source: `historical180/benchmarks-20261004/cache-c48-t14-a1-20261004T214653Z-584724/provenance.txt`, SHA256 `bdea10d216c872f6e161212eb146c8e1f87be49194ad90b82403af34971b8e4f`.

```text
utc=2026-10-04T21:46:53Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=cache
slots=48
threads=14
async_cpu=1
min_gpu_free_mib=4656
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=48
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 512 --sched-async-cpu 1 -p 512 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C061 — historical180/benchmarks-20261004/overlap-c48-t14-a0-20261004T214842Z-587507

Source: `historical180/benchmarks-20261004/overlap-c48-t14-a0-20261004T214842Z-587507/provenance.txt`, SHA256 `4028072afa973d2cd48dc5db011c0da2873ca854b641766c9e1b076f948d207a`.

```text
utc=2026-10-04T21:48:42Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=overlap
slots=48
threads=14
async_cpu=0
min_gpu_free_mib=4656
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=48
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 512 --sched-async-cpu 0 -p 512 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C062 — historical180/benchmarks-20261004/overlap-c48-t14-a1-20261004T215031Z-590180

Source: `historical180/benchmarks-20261004/overlap-c48-t14-a1-20261004T215031Z-590180/provenance.txt`, SHA256 `b298e7d289fcf5f9e5f0d43a6ba7565a740f2664e942d0506ad0401057f1178d`.

```text
utc=2026-10-04T21:50:31Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=overlap
slots=48
threads=14
async_cpu=1
min_gpu_free_mib=4656
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=48
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 512 --sched-async-cpu 1 -p 512 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C063 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b1024-u1024-pf0-20261004T223734Z-660149

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b1024-u1024-pf0-20261004T223734Z-660149/provenance.txt`, SHA256 `10eb52c1c889a53ebbf2c780c20a4db623d6ed09709bc78336616aba484d9b28`.

```text
utc=2026-10-04T22:37:34Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 1024 -ub 1024 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C064 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b1024-u128-pf0-20261004T223048Z-649423

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b1024-u128-pf0-20261004T223048Z-649423/provenance.txt`, SHA256 `60ddd1ff29bbc3f859c6ba9386653f02813d30c39bc8e3285ea7c065504f8b56`.

```text
utc=2026-10-04T22:30:48Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 1024 -ub 128 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C065 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b1024-u256-pf0-20261004T223403Z-654270

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b1024-u256-pf0-20261004T223403Z-654270/provenance.txt`, SHA256 `5e50859a1dda06bbe235fb57b53cb95e66454ea021c70713db9fb1893a31b12c`.

```text
utc=2026-10-04T22:34:03Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 1024 -ub 256 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C066 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b1024-u512-pf0-20261004T223610Z-657788

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b1024-u512-pf0-20261004T223610Z-657788/provenance.txt`, SHA256 `83ca1734bde8768fddb14ac16f62346b944830adc10b85ec83f20d664e4d33f8`.

```text
utc=2026-10-04T22:36:10Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 1024 -ub 512 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C067 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u1024-pf0-20261004T224515Z-669667

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u1024-pf0-20261004T224515Z-669667/provenance.txt`, SHA256 `67954303ce91a62534d67d9bb34c85fac95a6d5fd2dc9344f79def7fdebfa1f0`.

```text
utc=2026-10-04T22:45:15Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 1024 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C068 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u1024-pf2-20261005T060634Z-977668

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u1024-pf2-20261005T060634Z-977668/provenance.txt`, SHA256 `ded5755269689995cbfaa4c720f855164ecbb34a966534f372343147d9e0a11c`.

```text
utc=2026-10-05T06:06:34Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 1024 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C069 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u128-pf0-20261004T223830Z-661175

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u128-pf0-20261004T223830Z-661175/provenance.txt`, SHA256 `9c5e0797536c61dc31b2b3e52b0753750642f463c5230af34ed4b0d12f5ef6e6`.

```text
utc=2026-10-04T22:38:30Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 128 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C070 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u256-pf0-20261004T224144Z-665751

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u256-pf0-20261004T224144Z-665751/provenance.txt`, SHA256 `7fafec9133078a09d3b4c43fef6821fd41ab8a0f76f21b2b0c4abd7c41fa7c0d`.

```text
utc=2026-10-04T22:41:44Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 256 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C071 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u512-pf0-20261004T224351Z-668259

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b2048-u512-pf0-20261004T224351Z-668259/provenance.txt`, SHA256 `7b31b74c73c4830d3dd4308ecc2f828c8d2529acd0e6802c4a366481c46d975d`.

```text
utc=2026-10-04T22:43:51Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 512 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C072 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b4096-u1024-pf0-20261004T225256Z-677442

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b4096-u1024-pf0-20261004T225256Z-677442/provenance.txt`, SHA256 `50d2c7c3d472cc46ed00e3708febe6648a4248ab4779cdfd9d00ab92b3a24d94`.

```text
utc=2026-10-04T22:52:56Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 4096 -ub 1024 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C073 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b4096-u128-pf0-20261004T224611Z-670693

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b4096-u128-pf0-20261004T224611Z-670693/provenance.txt`, SHA256 `2fdbfe20ee2e6c7a6f89ff4c3775842acef90689bf9d5ce6337a2f0d363c5269`.

```text
utc=2026-10-04T22:46:11Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 4096 -ub 128 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C074 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b4096-u256-pf0-20261004T224925Z-673871

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b4096-u256-pf0-20261004T224925Z-673871/provenance.txt`, SHA256 `3da7a5d4f4d8fd0a785e9dc784390077fed1874976d86337143957b86826eb91`.

```text
utc=2026-10-04T22:49:25Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 4096 -ub 256 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C075 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b4096-u512-pf0-20261004T225132Z-676022

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b4096-u512-pf0-20261004T225132Z-676022/provenance.txt`, SHA256 `227772b1dfa903c8a503f7e0a76118d382a6c169ab9ae1fbffaacaa61647a4d2`.

```text
utc=2026-10-04T22:51:32Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 4096 -ub 512 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C076 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b512-u128-pf0-20261004T222319Z-634105

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b512-u128-pf0-20261004T222319Z-634105/provenance.txt`, SHA256 `97a389fef86bcf01e82875ade5af70877cf084acd9be0808aebb2ff2571324fa`.

```text
utc=2026-10-04T22:23:19Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 512 -ub 128 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C077 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b512-u256-pf0-20261004T222716Z-642697

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b512-u256-pf0-20261004T222716Z-642697/provenance.txt`, SHA256 `d584ba4aaee934d05890bb082e13fde9c0ebd17f96d982bf668389199d3b489b`.

```text
utc=2026-10-04T22:27:16Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 512 -ub 256 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C078 — historical180/benchmarks-20261004/prefill-c32-t14-a1-b512-u512-pf0-20261004T222924Z-646700

Source: `historical180/benchmarks-20261004/prefill-c32-t14-a1-b512-u512-pf0-20261004T222924Z-646700/provenance.txt`, SHA256 `52eae6326dfa74e90830f7061d34417f161256147c90e25b6a45592815e57185`.

```text
utc=2026-10-04T22:29:24Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=prefill
slots=32
threads=14
async_cpu=1
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=9058
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
production_context_and_projector_extra_reserve_mib=4402
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset=GGML_SCHED_PREFETCH_EXPERTS
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 512 -ub 512 --sched-async-cpu 1 -p 4096 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C079 — historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469

Source: `historical180/benchmarks-20261004/threads-c32-t4-6-8-12-14-16-a1-20261004T213654Z-567469/provenance.txt`, SHA256 `a1a07a5796fcd6e50e178220c8f210b548dd0f9ed531e5ecebe2058bf9132219`.

```text
utc=2026-10-04T21:36:54Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=threads
slots=32
threads=4,6,8,12,14,16
async_cpu=1
min_gpu_free_mib=4656
min_mem_available_kib=8388608
stock_bench_short_context=true
production_context_not_measured=131072
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=2
resource_extrema_are_sampled=true
input=stock_random_tokens
MTP=not_requested
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=32
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 4\,6\,8\,12\,14\,16 -b 2048 -ub 512 --sched-async-cpu 1 -p 512 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C080 — historical180/benchmarks-pp8192-20261005/pp8192-c48-t14-a0-b8192-u8192-20261005T073938Z-1096850

Source: `historical180/benchmarks-pp8192-20261005/pp8192-c48-t14-a0-b8192-u8192-20261005T073938Z-1096850/provenance.txt`, SHA256 `76b7cff805c169c65d80bb8158651957579d595466c4c3c40a4057600527e837`.

```text
utc=2026-10-05T07:39:38Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=pp8192
slots=48
threads=14
async_cpu=0
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=2048
min_mem_available_kib=8388608
MTP=not_requested
stock_bench_physical_context=8192
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=1
resource_extrema_are_sampled=true
input=stock_random_token_ids
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=48
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY,GGML_OP_OFFLOAD_MIN_BATCH
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 8192 -ub 8192 --sched-async-cpu 0 --no-op-offload 0 -d 0 -p 8192 -n 0 -r 3 -o jsonl --progress --verbose 
```

### C081 — historical180/benchmarks-speed-20261005/speed-c48-t14-a0-b2048-u2048-nopo0-20261005T064446Z-1028912

Source: `historical180/benchmarks-speed-20261005/speed-c48-t14-a0-b2048-u2048-nopo0-20261005T064446Z-1028912/provenance.txt`, SHA256 `8de5eb652b9a96fb7ab771036360e232b4fcdd490c8cafcf74b011412a9d7ae1`.

```text
utc=2026-10-05T06:44:46Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=speed
slots=48
threads=14
async_cpu=0
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=2048
min_mem_available_kib=8388608
MTP=not_requested
stock_bench_short_contexts=4096_and_256_separate
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=1
resource_extrema_are_sampled=true
input=stock_random_token_ids
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=48
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY,GGML_OP_OFFLOAD_MIN_BATCH
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 2048 --sched-async-cpu 0 --no-op-offload 0 -d 0 -p 4096 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C082 — historical180/benchmarks-speed-20261005/speed-c48-t14-a0-b4096-u2048-nopo0-20261005T064617Z-1031775

Source: `historical180/benchmarks-speed-20261005/speed-c48-t14-a0-b4096-u2048-nopo0-20261005T064617Z-1031775/provenance.txt`, SHA256 `329446d3530a905f50e044c5d5d05ad36d470b71c5f1ed53270619fff03c894d`.

```text
utc=2026-10-05T06:46:17Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=speed
slots=48
threads=14
async_cpu=0
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=2048
min_mem_available_kib=8388608
MTP=not_requested
stock_bench_short_contexts=4096_and_256_separate
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=1
resource_extrema_are_sampled=true
input=stock_random_token_ids
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=48
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY,GGML_OP_OFFLOAD_MIN_BATCH
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 4096 -ub 2048 --sched-async-cpu 0 --no-op-offload 0 -d 0 -p 4096 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C083 — historical180/benchmarks-speed-20261005/speed-c48-t14-a0-b4096-u4096-nopo0-20261005T064730Z-1034279

Source: `historical180/benchmarks-speed-20261005/speed-c48-t14-a0-b4096-u4096-nopo0-20261005T064730Z-1034279/provenance.txt`, SHA256 `152d2cc76bdebe4ca487e23e9298de41c3e1c325b32c611b0e5113b6df0c996a`.

```text
utc=2026-10-05T06:47:30Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=speed
slots=48
threads=14
async_cpu=0
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=2048
min_mem_available_kib=8388608
MTP=not_requested
stock_bench_short_contexts=4096_and_256_separate
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=1
resource_extrema_are_sampled=true
input=stock_random_token_ids
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=48
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY,GGML_OP_OFFLOAD_MIN_BATCH
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 4096 -ub 4096 --sched-async-cpu 0 --no-op-offload 0 -d 0 -p 4096 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C084 — historical180/benchmarks-speed-20261005/speed-c64-t14-a0-b2048-u2048-nopo0-20261005T064835Z-1036566

Source: `historical180/benchmarks-speed-20261005/speed-c64-t14-a0-b2048-u2048-nopo0-20261005T064835Z-1036566/provenance.txt`, SHA256 `cd1b90794e92d64eac23a19147306fe9ba74ac99d97821ac64fed28538571086`.

```text
utc=2026-10-05T06:48:35Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=speed
slots=64
threads=14
async_cpu=0
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=2048
min_mem_available_kib=8388608
MTP=not_requested
stock_bench_short_contexts=4096_and_256_separate
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=1
resource_extrema_are_sampled=true
input=stock_random_token_ids
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=64
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY,GGML_OP_OFFLOAD_MIN_BATCH
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 2048 --sched-async-cpu 0 --no-op-offload 0 -d 0 -p 4096 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C085 — historical180/benchmarks-speed-20261005/speed-c64-t14-a0-b4096-u2048-nopo0-20261005T064948Z-1038971

Source: `historical180/benchmarks-speed-20261005/speed-c64-t14-a0-b4096-u2048-nopo0-20261005T064948Z-1038971/provenance.txt`, SHA256 `f0377fe7569f5ddd616ccaeb0d7fb78697d80a754dadb7b680d06d5d35fa447d`.

```text
utc=2026-10-05T06:49:48Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=speed
slots=64
threads=14
async_cpu=0
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=2048
min_mem_available_kib=8388608
MTP=not_requested
stock_bench_short_contexts=4096_and_256_separate
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=1
resource_extrema_are_sampled=true
input=stock_random_token_ids
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=64
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY,GGML_OP_OFFLOAD_MIN_BATCH
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 4096 -ub 2048 --sched-async-cpu 0 --no-op-offload 0 -d 0 -p 4096 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C086 — historical180/benchmarks-speed-20261005/speed-c64-t14-a0-b4096-u4096-nopo0-20261005T065101Z-1041105

Source: `historical180/benchmarks-speed-20261005/speed-c64-t14-a0-b4096-u4096-nopo0-20261005T065101Z-1041105/provenance.txt`, SHA256 `7deecda7d7122c7e6ce7c260d8e9d4654d136dec0cf9cf797cc19cf220c75693`.

```text
utc=2026-10-05T06:51:01Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=speed
slots=64
threads=14
async_cpu=0
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=2048
min_mem_available_kib=8388608
MTP=not_requested
stock_bench_short_contexts=4096_and_256_separate
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=1
resource_extrema_are_sampled=true
input=stock_random_token_ids
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=64
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY,GGML_OP_OFFLOAD_MIN_BATCH
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 4096 -ub 4096 --sched-async-cpu 0 --no-op-offload 0 -d 0 -p 4096 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C087 — historical180/benchmarks-speed-20261005/speed-c80-t14-a0-b2048-u2048-nopo0-20261005T065205Z-1042879

Source: `historical180/benchmarks-speed-20261005/speed-c80-t14-a0-b2048-u2048-nopo0-20261005T065205Z-1042879/provenance.txt`, SHA256 `10bc97868340c30498eddc10904bd017aa0176f4a8a51fcf679088bf43e2637c`.

```text
utc=2026-10-05T06:52:05Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=speed
slots=80
threads=14
async_cpu=0
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=2048
min_mem_available_kib=8388608
MTP=not_requested
stock_bench_short_contexts=4096_and_256_separate
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=1
resource_extrema_are_sampled=true
input=stock_random_token_ids
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=80
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY,GGML_OP_OFFLOAD_MIN_BATCH
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 2048 -ub 2048 --sched-async-cpu 0 --no-op-offload 0 -d 0 -p 4096 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C088 — historical180/benchmarks-speed-20261005/speed-c80-t14-a0-b4096-u2048-nopo0-20261005T065318Z-1044457

Source: `historical180/benchmarks-speed-20261005/speed-c80-t14-a0-b4096-u2048-nopo0-20261005T065318Z-1044457/provenance.txt`, SHA256 `11f777426880217c84a2b9e54efa850654c935af256d2d3b3f57845309cb57c8`.

```text
utc=2026-10-05T06:53:18Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=speed
slots=80
threads=14
async_cpu=0
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=2048
min_mem_available_kib=8388608
MTP=not_requested
stock_bench_short_contexts=4096_and_256_separate
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=1
resource_extrema_are_sampled=true
input=stock_random_token_ids
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=80
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY,GGML_OP_OFFLOAD_MIN_BATCH
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 4096 -ub 2048 --sched-async-cpu 0 --no-op-offload 0 -d 0 -p 4096 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C089 — historical180/benchmarks-speed-20261005/speed-c80-t14-a0-b4096-u4096-nopo0-20261005T065430Z-1047277

Source: `historical180/benchmarks-speed-20261005/speed-c80-t14-a0-b4096-u4096-nopo0-20261005T065430Z-1047277/provenance.txt`, SHA256 `a0bbd27aba47bb47a2a617483fede87239d30d32cd2a6ddc477d10cfe45494da`.

```text
utc=2026-10-05T06:54:30Z
source_commit=27c54b4bbcefadedcec6397477cc2e866c1db716
phase=speed
slots=80
threads=14
async_cpu=0
stock_bench_threads_batch_supported=false
stock_bench_t_sets_both_thread_counts=true
min_gpu_free_mib=2048
min_mem_available_kib=8388608
MTP=not_requested
stock_bench_short_contexts=4096_and_256_separate
repetitions=3
warmup=stock_default
OS_page_cache_not_dropped=true
resource_sampling_seconds=1
resource_extrema_are_sampled=true
input=stock_random_token_ids
GGML_MOE_CACHE_PROFILE=<PATH>/routing-merged.csv
GGML_MOE_CACHE_SLOTS=80
LD_LIBRARY_PATH=<PATH>/bin
unset=GGML_CUDA_REGISTER_HOST,GGML_SCHED_PREFETCH_EXPERTS,GGML_CUDA_ENABLE_UNIFIED_MEMORY,GGML_OP_OFFLOAD_MIN_BATCH
command=<PATH>/llama-bench -m <PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --offline -ngl 99 -ncmoe 99 -lm mmap -lzm on -fa on -ctk q8_0 -ctv q8_0 -C 0xffff --cpu-strict 1 -t 14 -b 4096 -ub 4096 --sched-async-cpu 0 --no-op-offload 0 -d 0 -p 4096 -n 256 -r 3 -o jsonl --progress --verbose 
```

### C090 — profile-iq3-175-r1-20261005

Source: `profile-iq3-175-r1-20261005/chat.configuration.json`, SHA256 `0bfcdd54605c7874477e266e07d26c668d51fb060870b67f0bc3a2ba33bd4e11`.

```text
'<PATH>/llama-moe-trace' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --offline --fit off --ctx-size 32768 --parallel 1 --gpu-layers 99 --n-cpu-moe 99 --load-mode mmap --lazy-mode on --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --batch-size 8192 --ubatch-size 8192 --moe-cache-slots 0 --no-escape --file '<PATH>/chat.prompt.txt' --n-predict 512
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0",
  "MOE_TRACE_OUT": "<PATH>/chat.csv"
}
```

### C091 — profile-iq3-175-r1-20261005

Source: `profile-iq3-175-r1-20261005/code.configuration.json`, SHA256 `dfcb2a4679ff720432d2cea8d88d70bf5119be1737066b787f022cb101bb16b8`.

```text
'<PATH>/llama-moe-trace' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --offline --fit off --ctx-size 32768 --parallel 1 --gpu-layers 99 --n-cpu-moe 99 --load-mode mmap --lazy-mode on --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --batch-size 8192 --ubatch-size 8192 --moe-cache-slots 0 --no-escape --file '<PATH>/code.prompt.txt' --n-predict 512
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0",
  "MOE_TRACE_OUT": "<PATH>/code.csv"
}
```

### C092 — profile-iq3-175-r1-20261005

Source: `profile-iq3-175-r1-20261005/tools.configuration.json`, SHA256 `e2f3a79012392d2102af27dd3a6627a755d8774ac43962bac5df000716bbea73`.

```text
'<PATH>/llama-moe-trace' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --offline --fit off --ctx-size 32768 --parallel 1 --gpu-layers 99 --n-cpu-moe 99 --load-mode mmap --lazy-mode on --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --batch-size 8192 --ubatch-size 8192 --moe-cache-slots 0 --no-escape --file '<PATH>/tools.prompt.txt' --n-predict 512
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0",
  "MOE_TRACE_OUT": "<PATH>/tools.csv"
}
```

### C093 — historical180/profile-iq3-175-20261005

Source: `historical180/profile-iq3-175-20261005/code.configuration.json`, SHA256 `cd00480ee4e11c8c6b73511fde9fb2ae43c67c9036d23a2733ef1042118cbe8a`.

```text
'<PATH>/llama-moe-trace' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --offline --fit off --ctx-size 32768 --parallel 1 --gpu-layers 99 --n-cpu-moe 99 --load-mode mmap --lazy-mode on --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --batch-size 8192 --ubatch-size 8192 --moe-cache-slots 0 --spec-type none --no-escape --file '<PATH>/code.prompt.txt' --n-predict 512
```

```json
{
  "LANG": "pl_PL.UTF-8",
  "LC_ALL": "C.UTF-8",
  "LD_LIBRARY_PATH": "<PATH>/bin",
  "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
  "CUDA_VISIBLE_DEVICES": "0",
  "MOE_TRACE_OUT": "<PATH>/code.csv"
}
```

### C094 — strata188

Source: `jarvis-flash174/report/strata188/summary.json`, SHA256 `208db26d4d20937ae638a1dee16b498ce9902aef909e834224ed87402ef69781`.

```text
'<PATH>/strata' --pack '<PATH>/unsloth-ud-q4_k_xl' --native '<PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf' --expert-profile '<PATH>/expert-profile.bin' --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp '<PATH>/rt' --max-context 262144 --kv int8 --kv-resident 32768 --resident-budget-gib 62 --vram-reserve-mib 12288
```

Runtime-effective parameters:

```json
{
  "workers": 15,
  "host_threads": 1,
  "cpu_affinity": "host0; pool1-15",
  "gpu_expert_slots": 1407,
  "gpu_expert_cache_GiB": 4.09,
  "prompt_chunk_tokens": 7424,
  "prompt_borrowed_slots": 1185,
  "kv_pinned_host_GiB": 3.09,
  "kv_gpu_resident_cells_per_layer": 32768,
  "resident_expert_budget_GiB": 62,
  "mtp_expert_quantization": "q2_0",
  "spec_draft_max": 4,
  "spec_min_p": 0.5,
  "draft_vocabulary_tokens": 106299
}
```

Cgroup limits: MemoryMax 80 GiB; MemorySwapMax 0 GiB.

### C095 — strata188-ram48

Source: `jarvis-flash174/report/strata188-ram48/summary.json`, SHA256 `99909693256ef583e4ff73f4ac20b68e2a6c7981e25e4baf6d836e720d90e0e7`.

```text
'<PATH>/strata' --pack '<PATH>/unsloth-ud-q4_k_xl' --native '<PATH>/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf' --expert-profile '<PATH>/expert-profile.bin' --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp '<PATH>/rt' --max-context 262144 --kv int8 --kv-resident 32768 --resident-budget-gib 24 --vram-reserve-mib 12288
```

Cgroup limits: MemoryMax 48 GiB; MemorySwapMax 0 GiB.

### C096 — functional195

Source: `jarvis-fixes195/native-request-timings.json`, SHA256 `8cd2b39ce8898270b97756846d956bbb4cbcf2aa04bd6e1057a1976ee9f3054b`.

```text
/usr/bin/env 'LD_LIBRARY_PATH=<PATH>/bin' '<PATH>/llama-server' --model '<PATH>/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf' --alias jarvis-flash-next --host 127.0.0.1 --port 14004 --spec-type none --gpu-layers 99 --n-cpu-moe 48 --threads 14 --threads-batch 14 --cpu-mask 0xffff --cpu-strict 1 --load-mode none --lazy-mode on --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --ctx-size 262144 --parallel 1 --cache-reuse 256 --cache-prompt --batch-size 8192 --ubatch-size 2048 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --metrics --slots --jinja --chat-template-file '<PATH>/chat_template.jinja' --chat-template-kwargs '{"reasoning_effort":"xhigh"}' --verbosity 4 --offline --no-context-shift --timeout 7200 --mmproj '<PATH>/mmproj-F16.gguf' --no-mmproj-offload
```

## Appendix D. Build reference and artifact integrity

[Artifact hashes](data/artifact-integrity.json) identify the weights and builds recorded during the experiments. Release-file checksums are separate. The following recorded build recipes are version-specific; local paths and variables require installation-specific substitution.

### llama.cpp upstream

Source: `jarvis-flash174/upstream178-build.sh`, SHA256 `b46624daa26e715a97f149d9aaf39d700d2a794d342f242b157595a0ae1d2eba`.

```text
cmake -S "$stage/source" -B "$stage/build-cuda86" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DGGML_NATIVE=ON -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath-link,/usr/local/cuda/lib64/stubs'
```

### ik_llama.cpp

Source: `jarvis-flash174/ik178-build.sh`, SHA256 `28656824d4842236b9924952fd2efda768269114c93bd0b6e1940031d6fd6f4a`.

```text
cmake -S "$stage/source" -B "$stage/build-cuda86" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON \
    -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON -DGGML_SCHED_MAX_COPIES=1 \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON -DGGML_NCCL=OFF \
    -DLLAMA_CURL=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath-link,/usr/local/cuda/lib64/stubs'
```

### OptLlama

Source: `jarvis-flash174/opt181-build.sh`, SHA256 `71dfb27ad7df6d0f2f08ec6ee6c70b6cad4faaddb273238edb0d6dfcf3378b9d`.

```text
cmake -S "$stage/source" -B "$stage/build-cuda86" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DGGML_NATIVE=ON -DGGML_BACKEND_DL=OFF -DGGML_BLAS=OFF -DGGML_NCCL=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath-link,/usr/local/cuda/lib64/stubs'
```

### Strata

Source: `jarvis-flash174/strata188/build.sh`, SHA256 `62760e503314258134a5f84b010f605d8bba723a8465964fc505d2f5061c175b`.

```text
cmake -S /source -B /output/build-cuda86 -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DSTRATA_GGML_DIR=/source/third_party/llama.cpp
```
