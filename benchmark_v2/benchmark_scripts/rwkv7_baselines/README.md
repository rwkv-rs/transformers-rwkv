# RWKV-7 reference throughput

Reference numbers for `rwkv7_throughput.py --baseline`, so that a "N% of X" claim
carries its X in a file rather than in prose. Each file is one build of one external
runtime, measured on the same card and in the same session as the run it is quoted
against; a number measured on a different card is not a baseline, it is a rumour.

Both files below are `albatross`, the RWKV-7 CUDA runtime, on one RTX 5090 with the
7.2B checkpoint in fp16. They are different builds and they disagree by 5.8% at 1x1,
which is exactly why they are separate files: quoting 86.5% against one and 93.6%
against the other, for the same measurement, is a difference in denominator and not
in performance.

- `albatross_faster3a_2607.json` — the build vllm-rwkv's own harness measures
  against, so columns compared with that project are commensurable. Full grid.
- `albatross_faster3b_2607.json` — the faster single-token build. Only `1x1` was
  measured, and only `1x1` is recorded here.
