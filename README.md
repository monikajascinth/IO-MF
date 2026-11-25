# IO-MF Python reference

Implements the two controllers from the [manuscript specification](docs/IMPLEMENTATION_SPEC.md) and provides a single-seed training and evaluation command for UCI-HAR.

From the project directory, install PyTorch and run the unit tests.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Download the [official UCI-HAR archive](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones) and extract the inner ZIP until the dataset directory directly contains the train and test directories. Run a one-epoch smoke check.

```powershell
.\.venv\Scripts\python.exe -m src.run_har 'datasets/UCI HAR Dataset' results/smoke --host-epochs 1 --probe-epochs 1
```

The CLI writes the split and normalization manifest, validation threshold scores, frozen host and probe checkpoints, test metrics, and per-window traces. Choose a fresh output directory for each run. See `python -m src.run_har --help` for options.

Reconstruction defaults are body acceleration, held-out validation subjects 1, 3, 5 and 6, same-length convolutions, max pooling with width and stride 2, cross-entropy, and a frozen copy of the first encoder block with an MSE-trained linear probe projection. Threshold selection maximizes validation macro-F1, then minimizes gyroscope activation rate, then selects the first threshold in ascending order. Final predictions use exact embeddings for the committed modalities.

To test IO-MF-D, pass `--timing path.json` containing the integer fields `deadline` and `primary` and the objects `probe` and `commit`, each with a `gyroscope` integer. All values use microseconds. Supply upper bounds that include the operation overhead described in the specification. The optional deadline report evaluates the supplied budget model. Physical deadline validation requires separate timing measurements.

## Evaluate a saved run

Use the frozen checkpoints to measure probe fidelity, compare CPU inference latency, and optionally sweep deadline budgets.

```powershell
.\.venv\Scripts\python.exe -m src.eval_har 'datasets/UCI HAR Dataset' results/smoke results/smoke/evaluation.json
```

To include deadline comparisons, add `--timing path.json --deadlines-us 600 700 1599 1600 2000` and choose a fresh report path. The timing file supplies the primary, probe and commit bounds. The sweep takes the deadlines from the command line. The report keeps modeled scheduler inputs and observed CPU wall time separate. Mean and P95 latency begin after the prepared input is available. Observed timely accuracy uses all requests as its denominator, including those the scheduler marks infeasible. The deadline-violation rate counts observed executions exceeding the budget. UCI-HAR has one optional modality, so the report records threshold agreement and entropy-gain error and leaves ranking agreement undefined.

## Run paired seeds

Train and evaluate the same policies for each seed with a fixed subject split.

```powershell
.\.venv\Scripts\python.exe -m src.run_paired 'datasets/UCI HAR Dataset' results/uci-har-paired
```

The runner defaults to reconstruction seeds 0 through 9, up to 100 host epochs with validation early stopping, 10 probe epochs, and 100 warm-up requests per policy. Use `--seeds 0 1 --host-epochs 1 --probe-epochs 1` with a separate output directory for a smoke run. Running the original command again resumes completed seeds after checking the dataset fingerprint and run configuration.

Each seed has a host, probe, inference trace, and evaluation report. The aggregate reports means and sample standard deviations, with per-seed IO-MF differences relative to primary-only and static fusion. These are descriptive comparisons across training seeds with the same test subjects. The manuscript's significance tests use the learned DynaFuse comparator across three datasets and require separate runs.

## Diagnose reproduction gaps

The training report records per-epoch training loss, validation loss, validation accuracy and validation macro-F1. Checkpoint selection uses validation macro-F1 with patience ten.

The evaluation report compares probe-based IO-MF against an exact-entropy oracle using the same checkpoint, threshold and test windows. The oracle executes the gyroscope encoder on every window to calculate the true gain. Its activation rate describes the eligibility decision and does not imply any encoder savings. Compare the probe and oracle metrics to assess the effect of the approximation. Compare both policies with early fusion to assess the effect of selective activation. The diagnostic also reports separate CPU timings for primary inference, the temporary probe prediction and exact recomputation. These measurements are empirical and do not establish deadline bounds.

## Compare probe training objectives

```powershell
.\.venv\Scripts\python.exe -m src.compare_probes 'datasets/UCI HAR Dataset' results/full-seed-0 results/full-seed-0/probe-comparison
```

The experiment initializes from the saved embedding-MSE probe, then trains only its linear projection using KL divergence to the frozen host's full-fusion distribution on training subjects. It compares both probes on validation at 0.06 nats and at the incumbent's validation activation count. The matched comparison selects the probe and threshold before test evaluation. The output holds the distilled checkpoint and comparison report. The original checkpoints remain available for the reference policy.

Use `--objective entropy` to train the probe directly against full-fusion predictive entropy. Since the exact and estimated gains subtract from the same primary entropy, matching candidate entropy also matches the activation score. Use `--activation-budget oracle` to set a larger common validation budget from exact decisions at 0.06 nats. Neither flag changes the controller's strict positive-utility rule.

To compare accelerometer inputs, run the same training command with `--acceleration total_acc` and a fresh output path. Use the same seed and subject IDs and compare the selected checkpoints' validation macro-F1.

## Compare policies at a common activation budget

```powershell
.\.venv\Scripts\python.exe -m src.eval_activation 'datasets/UCI HAR Dataset' results/total-acc-seed-0 results/total-acc-seed-0/activation-52.json
```

The evaluator targets 52% validation gyroscope activation by default, fits a shared logit temperature on validation, and compares exact and probe entropy with and without calibration. Thresholds are frozen before test evaluation. If the requested activation rate cannot be reached with a nonnegative threshold, the report shows the achieved rate. The exact-entropy policy and label-informed upper bound are diagnostic references. The report includes activations that corrected or spoiled a primary prediction and early-fusion accuracy by activity.

To include these comparison policies in every paired seed, use a new output directory and enable activation evaluation.

```powershell
.\.venv\Scripts\python.exe -m src.run_paired 'datasets/UCI HAR Dataset' results/total-acc-paired --acceleration total_acc --activation-evaluation
```

The runner saves the standard validation-selected IO-MF result alongside the fixed 0.06-nat run and matched-budget variants. Each probe result in the activation report is checked window by window through the shared inference controller. Per-seed reports retain thresholds, temperature, and cost settings. The aggregate includes means and sample standard deviations for accuracy, macro-F1, and realized activation.

## Try host and selector variants

Set `--acceleration body_total_acc` to feed body and total acceleration to one six-channel primary encoder. The gyroscope and its probe still consume three channels. Train in a fresh directory so the checkpoints and normalization manifest identify the input mode.

The learned benefit selector is an experimental policy alongside manuscript IO-MF. It estimates the probability that full fusion fixes a primary error minus the probability that it spoils a correct primary prediction. Run it with `--profile` to time the live, validation-selected policy. The report also includes the calibration and exact-entropy references at the same validation budget.

```powershell
.\.venv\Scripts\python.exe -m src.try_benefit 'datasets/UCI HAR Dataset' results/total-acc-seed-0 results/total-acc-seed-0/benefit-live --profile
```

Run the full paired comparison of the learned selector and IO-MF on six-channel acceleration.

```powershell
.\.venv\Scripts\python.exe -m src.run_paired 'datasets/UCI HAR Dataset' results/body-total-paired-benefit --acceleration body_total_acc --training-workers 6 --benefit-evaluation
```

Training workers write separate logs and publish each completed checkpoint directory before evaluation. Live timing runs serially after parallel training. The summary reports each learned variant, the validation-selected learned policy, and per-seed differences from calibrated IO-MF at the same validation activation budget. The per-seed reports also include the article-threshold and exact-entropy references.
