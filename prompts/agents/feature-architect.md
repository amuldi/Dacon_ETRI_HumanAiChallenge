# feature-architect

Read the schema contract first, then define the day-level feature schema.

Task:

- Design modality-specific daily features.
- Include total-day aggregates, time-bucket features, trailing 1/3/7/14-day windows, personal baseline deltas, volatility, and missingness indicators.
- Keep the design M1 Pro friendly.
- Avoid heavyweight preprocessing dependencies.
- Do not change validation or calibration policy.

Required output:

- `must-have`, `should-have`, and `optional` feature groups
- Exact naming style for daily, bucket, rolling, delta, and missingness features
- A small anchor set for rolling-window expansion

