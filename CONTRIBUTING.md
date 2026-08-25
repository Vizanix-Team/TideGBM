# Contributing

## Rules of the library

1. Time is causal. If a change lets a row see its own future residual, it is a bug.
2. Users bring features. Do not add RSI, candles, exchange adapters, or a backtester to the core package.
3. Do not claim SOTA in the README without a public purged benchmark in `benchmarks/`.
4. `iid_mode=True` is escape hatch, not the default.

## Dev

```bash
pip install -e ".[dev]"
pytest
```

Python 3.10+. Prefer small tests that fail on leakage (eval-in-the-past must raise, walk-forward train max time < test min time) over huge accuracy tables.

## License

Apache-2.0. By opening a PR you agree to license the contribution under Apache-2.0.
