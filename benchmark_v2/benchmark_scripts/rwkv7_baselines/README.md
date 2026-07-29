# RWKV-7 reference throughput

Reference numbers for `rwkv7_throughput.py --baseline`, so that a "N% of X" claim
carries its X in a file rather than in prose. Each file is one build of one external
runtime, measured on the same card and in the same session as the run it is quoted
against; a number measured on a different card is not a baseline, it is a rumour.

Both files below are `albatross`, the RWKV-7 CUDA runtime, on one RTX 5090 with the
7.2B checkpoint in fp16. They are different builds and they disagree by 5.8% at 1x1,
which is exactly why they are separate files. One measurement, 134.8 tok/s, reads as
86.5% against faster3b and 91.5% against faster3a -- a difference in denominator and
not in performance. (An earlier version of this paragraph paired 86.5% with 93.6%,
which no single measurement can produce: 93.6/86.5 is 1.082 and the two baselines
differ by 1.058. A file whose whole purpose is that a percentage must carry its
denominator is a poor place to quote one that does not.)

- `albatross_faster3a_2607.json` — the build vllm-rwkv's own harness measures
  against, so columns compared with that project are commensurable. Full grid.
- `albatross_faster3b_2607.json` — the faster single-token build. Only `1x1` was
  measured, and only `1x1` is recorded here.
