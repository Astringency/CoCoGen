# CoCoGen Experiments for FM4PDE

Adapted CoCoGen experiments for **Guided Flow Matching for Forward and Inverse
PDE Problems with Sparse Observations: Algorithm and Theory**. The main
comparisons cover full-field and sparse forward/inverse reconstruction and
sparse joint reconstruction on Poisson, Helmholtz, Darcy, and Navier–Stokes.

## Training

Use [environment.yml](environment.yml) to create the `ddpm` environment.
The commands below use Bash (Linux or WSL); data and trained weights are
external assets.

```bash
conda env create -f environment.yml
conda activate ddpm
```

The original training entry is `main()` in [train.py](train.py), which invokes
PyTorch Lightning's `Trainer.fit()`. The score-matching loss is implemented by
`DiffusionSDE.training_step()` in
[models/score_matching.py](models/score_matching.py).

The four-PDE comparison training code is retained in
[reference/server197_20260915](reference/server197_20260915).
Its [train.py](reference/server197_20260915/train.py) uses the adapted PDE
loader and the same Lightning training structure. Run this version from its
own directory so its local `data`, `models`, and `sdes` modules are used:

```bash
cd reference/server197_20260915
python train.py --config configs/poisson.yaml \
  --logdir /path/to/training_logs --name cocogen4poisson
```

Before training, set `data.data_dir` and `lightning.trainer.devices` in the
selected config. For a new unconditional prior, set
`model.params.unet_config.params.controlnet.use: false` and
`lightning.lock_base: false`. The supplied comparison configs currently select
ControlNet with the base network locked; that stage requires pretrained
weights.

After training a prior, enable ControlNet and base locking in the config,
initialize the additional layers, and train the conditioning branch:

```bash
# Run from reference/server197_20260915 with the ControlNet config enabled.
python tool_add_controlnet.py --config configs/poisson.yaml \
  --checkpoint /path/to/prior/checkpoints/last.ckpt \
  --save /path/to/poisson_control_init.pth
python train.py --config configs/poisson.yaml \
  --unet_weights /path/to/poisson_control_init.pth \
  --logdir /path/to/control_logs --name cocogen4poisson_control
```

[reference/server197_20260915/run_train.sh](reference/server197_20260915/run_train.sh)
contains the historical ControlNet initialization commands for four PDEs.
It only initializes weights; invoke `train.py` afterward as above. To use it,
first replace its checkpoint/output paths and run `bash run_train.sh` from
that directory.

## Main Sampling

Return to the repository root for the comparison evaluator.
[cocogen_eval/evaluate.py](cocogen_eval/evaluate.py) loads the selected model
and saved observation protocol, then calls `sample()` in
[cocogen_eval/sampler.py](cocogen_eval/sampler.py). Sampling uses VP reverse-SDE
imputation, RePaint visits, and physical corrections.
[cocogen_eval/main_inputs.py](cocogen_eval/main_inputs.py) applies the active
forward/inverse/joint observation masks.

The evaluator consumes a prepared study with `inputs/`, `protocol/catalog.json`,
cell records, and validated selections in `protocol/selected/<pde>.json`.
Its source data, catalog, and checkpoint defaults in
[cocogen_eval/common.py](cocogen_eval/common.py) refer to the original server.
Set `REMOTE_REPO`, `FM_REPO`, `DATA_ROOT`, `CATALOG`, and the checkpoint run
mapping `RUNS` for your installation; relocate paths inside saved manifests
as needed. Passing `--study` only changes the study directory.

For an existing prepared and validated study:

```bash
STUDY=/path/to/cocogen_study
python -m cocogen_eval.evaluate --study "$STUDY" \
  --pde poisson --device cuda:0 --batch-size 32

# Run all four PDEs, then aggregate the 60 comparison cells.
for pde in poisson helmholtz darcy nsnonbounded; do
  python -m cocogen_eval.evaluate --study "$STUDY" \
    --pde "$pde" --device cuda:0 --batch-size 32
done
python -m cocogen_eval.evaluate --study "$STUDY" --aggregate
```

Each PDE has five observation settings on three distributions
(ID/Smooth/Rough), with 1,000 inputs per cell. Input statistics and calibration
cases are prepared by `python -m cocogen_eval.prepare`; the calibration tools
`cocogen_eval.calibrate` and `cocogen_eval.cost_calibrate` select and validate
sampler settings before evaluation. Use `--help` on each module for arguments.

This checkout has no top-level `scripts/` directory. Its historical grouped
launchers are [cocogen_eval/prepare_group.sh](cocogen_eval/prepare_group.sh) and
[cocogen_eval/main_group.sh](cocogen_eval/main_group.sh). After adapting their
hard-coded study, Python, and code paths:

```bash
bash cocogen_eval/prepare_group.sh group0 poisson helmholtz
bash cocogen_eval/main_group.sh group0 cuda:0 poisson helmholtz
```

`main_group.sh` waits for the named calibration and input/evaluator audit
completion files. Use the Python entry above when managing those stages
directly. The archived `reference/server197_20260915/run_sample.sh` contains
earlier single-run examples and also requires updating its paths.

## Ablations

[cocogen_eval/long_sampling_diagnostic.py](cocogen_eval/long_sampling_diagnostic.py)
compares Darcy sampling schedules with 100 or 2,000 steps and different
RePaint counts. It uses eight fixed ID calibration cases per sparse task and
one seed; these diagnostic results are separate from the main comparisons.
Preparation requires a validated Darcy ControlNet selection with the expected
100-step, four-RePaint baseline and its saved assets.

```bash
DIAGNOSTIC=/path/to/darcy_sampling_diagnostic
python -m cocogen_eval.long_sampling_diagnostic \
  --root "$DIAGNOSTIC" prepare --study /path/to/cocogen_study

# Four worker assignments can also be run sequentially on one GPU.
for worker in 0 1 2 3; do
  bash cocogen_eval/diagnostic_job.sh \
    "$DIAGNOSTIC" "$(command -v python)" "worker_$worker" \
    worker --device cuda:0 --worker-id "$worker"
done

python -m cocogen_eval.long_sampling_diagnostic \
  --root "$DIAGNOSTIC" collect
```

The Bash wrapper writes each worker's log and exit status under the diagnostic
directory. Additional Burgers training/evaluation extensions are retained in
[cocogen_burger](cocogen_burger); they require their own prepared study and
checkpoints and are separate from the four-PDE main comparison above.

## Baseline and other info

Related repositories: [FM4PDE](https://github.com/Astringency/FM4PDEdebug.git),
[RecFNO and other baselines](https://github.com/Astringency/FM4PDEbaseline.git),
and [DiffusionPDE comparisons](https://github.com/Astringency/DiffusionPDE.git).

Our code is modified and adapted from the official
[CoCoGen implementation](https://github.com/christian-jacobsen/CoCoGen) by
Christian Jacobsen, Yilin Zhuang, and Karthik Duraisamy,
*CoCoGen: Physically-Consistent and Conditioned Score-based Generative Models
for Forward and Inverse Problems*. The adaptations add the FM4PDE benchmark
data interfaces, observation protocols, physical residuals, and comparison
workflows. The shared residual and mask code in `cocogen_eval/fm_physics/`
comes from FM4PDE, with provenance recorded in
[provenance.json](cocogen_eval/fm_physics/provenance.json).
Please acknowledge the original CoCoGen paper and code when using this adaptation.

