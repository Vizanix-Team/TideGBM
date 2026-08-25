# Compatibility

- **Semver:** breaking the causal contract (silent lookahead, dropping required `horizon`,
  changing TORB so future rows affect past gradients) is a **major** version.
- **Model format:** binary magic `TGBM`, version 3. Older files raise on load.
- **CPU bit-identical:** `n_jobs=1` and the same `random_state`. OpenMP reductions
  are not associative; `n_jobs!=1` is not bit-stable.
- **GPU:** optional `TIDEGBM_CUDA`. Default wheels are CPU-only; `device="gpu"` raises
  unless the CUDA histogram kernel was compiled in. `tidegbm.cuda_compiled` is the flag.
  GPU wheels are built by `.github/workflows/gpu-wheel.yml` / `scripts/build_gpu_wheel.sh`
  (nvcc required, a GPU is not). Cubin targets sm_60–sm_90; fit still needs a driver.
- **ONNX:** `to_onnx_json` is always available. `to_onnx()` writes protobuf when `onnx` is
  installed. Thresholds for axis splits use frozen bin edges; cat/PAIR nodes are not
  faithful to native scoring on raw X.
- **SHAP:** `shap_values` is Saabas path attribution (sums to prediction). It is not
  Independent TreeSHAP.
- **GOSS:** default `goss=None` turns GOSS on at 250k rows. Explicit `goss=False` disables it.
- **Bins / structure / TORB:** default `max_bin=255`, `causal_structure=False`, `structure_frac=1`, `causal_bins=False`. Sparse TORB prefixes blend toward full-train leaves. `n_time_buckets=0` auto-selects 32 or 64 from label horizon (≥8 median steps → 64). Old holdout is an explicit opt-in.
- **Early stopping:** `eval_metric="auto"` monitors mean cross-sectional Pearson IC in time
  and rank modes (maximize). Pass `eval_metric="loss"` for the training objective.
