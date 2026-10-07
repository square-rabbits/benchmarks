# Benchmarks

Experimental studies of local AI inference: throughput, memory residency, context scaling and application latency.

## Qwen3.8-Flash-Next

Hardware: **EVGA GeForce RTX 3090 OC FTW3 Ultra, Ryzen 9 8945HX, nominal 96 GB RAM**.

- [Interactive report on GitHub Pages](https://square-rabbits.github.io/benchmarks/)
- [Comprehensive Markdown report](qwen3.8-flash-next_rtx3090_RAM96GB/REPORT.md)
- [Measurements CSV](qwen3.8-flash-next_rtx3090_RAM96GB/data/measurements.csv)
- [Configurations and data JSON](qwen3.8-flash-next_rtx3090_RAM96GB/data/benchmark-data.json)

The English report examines Unsloth Q3 and Q4 configurations across five inference implementations. It separates fresh-input throughput, natural retrieval, resource measurements and API/voice integration tests.

The HTML presents the research findings, interactive figures and a separate measurement explorer. Download `index.html` for offline use; no external runtime libraries are required.

Supplementary data include individual measurements, version-specific commands, recorded environments and artifact hashes. Complete raw input fixtures are not distributed; experimental scope and reproducibility constraints are described in the paper.
