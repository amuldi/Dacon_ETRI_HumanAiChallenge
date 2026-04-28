# sequence-lite

Start only after the tabular baseline exists.

Task:

- Use recent 3/7/14 day windows of daily features or day embeddings.
- Restrict models to lightweight MLP, TCN, or transformer-lite variants that fit an M1 Pro.
- Compare against the calibrated tabular baseline on OOF performance.
- Recommend rejection if the gain is not meaningful.

Required output:

- Window design
- Model spec
- Runtime and memory estimate
- Keep-or-drop recommendation

