# Local AB-UPT for Cylinder Flow on Unstructured Meshes

Anchored-Branched Universal Physics Transformer ([arXiv 2502.09692](https://arxiv.org/abs/2502.09692)) reduced to one velocity branch and adapted to autoregressive forecasting of laminar vortex shedding behind a cylinder in a channel on triangular meshes: anchor nodes carrying the current velocity self-attend, blind query nodes read from them, and the network predicts the next frame. Blocks are either global anchored attention (`s`) or radius-windowed anchored attention (`l`), so a stack such as `lllsss` mixes both. Runs are configured with Hydra, trained with Lightning, logged to CSV, and reproducible given `extras.seed`.

## Dataset

41 COMSOL simulations of flow past a cylinder at varying position, inlet speed and mesh, each 1001 frames at 0.01 s: one npz file per simulation `r<position>_u<speed>[_mesh<variant>]` with vertex positions, node types, triangles, velocity and pressure, plus `manifest.csv` with the flow parameters. Hosted on Hugging Face as a private dataset: [aptemvs/cylinder-vortex-shedding-unstructured](https://huggingface.co/datasets/aptemvs/cylinder-vortex-shedding-unstructured).

To download it into `data/comsol_cylinder/` and build the `exp00`, `exp15` and `exp40` split trees:

```bash
uvx --from huggingface_hub hf auth login   # once; the dataset is private
uvx --from huggingface_hub hf download aptemvs/cylinder-vortex-shedding-unstructured \
  --repo-type dataset --local-dir data/comsol_cylinder
uv run python prepare_data.py
```

The download leaves metadata in `data/comsol_cylinder/.cache/`, which is safe to delete. `prepare_data.py --src /path/to/comsol_dataset` converts the raw COMSOL VTU exports instead. The whole `data/` directory is gitignored.

## Structure

```
train.py                  Training entrypoint: instantiate datamodule, model, logger, callbacks and trainer, fit
prepare_data.py           npz pool and split trees, converting COMSOL VTU exports if needed
configs/                  Hydra configs: train.yaml (paper recipe) and radius.yaml (early recipe) over
                          paths/, datamodule/, model/, trainer/, logger/, callbacks/, extras/
src/abupt/                Dataset, Lightning datamodule and module, callbacks, rollout, evaluation, metrics
src/abupt/nn/             Positional embeddings, anchored attention, blocks, normalizer, the network
experiments/              One script per experiment of the report
tests/                    pytest suite on synthetic data plus checks against the recorded checkpoints
```

## Run

Requires [uv](https://docs.astral.sh/uv/) and a CUDA GPU.

```bash
uv sync
uv run python train.py
uv run python train.py model.blocks=lllsss model.local_radius=0.1 extras.seed=43
uv run python train.py --config-name radius model.local_radius=0.05
```

Training validates every 5000 steps on fixed windows of the validation flow and on a 100-step rollout probe. The checkpoint with the best probe error is kept as `best.ckpt`, the final one as `last.ckpt`, and every validation checkpoint under `ckpts/`. Outputs land in `outputs/<date>/<time>/` together with `metrics.csv` and `metrics.json`.

## Experiments

Each script trains the runs it needs into `outputs/<label>/`, skips runs and artifacts that already exist, and writes its evaluation under `outputs/thesis_eval/`, `outputs/comsol_battery_*/` or `outputs/thesis_cost/`. All scripts accept `--output-root`, `--models`, `--seeds` and `--sims`.

```bash
uv run python experiments/train_main_models.py   # s8, l5, lllsss for seeds 42-44 and the truth statistics
uv run python experiments/exp1_position.py       # Experiment 1: cylinder positions r1-r15 at 2.2 m/s
uv run python experiments/exp2_speed.py          # Experiment 2: inlet speeds at r5, interpolation at r4 and r6
uv run python experiments/a1_local_depth.py      # Appendix A1: local stacks l^1..l^12
uv run python experiments/a2_global_depth.py     # Appendix A2: global stacks s^1..s^12
uv run python experiments/a3_block_order.py      # Appendix A3: six-block orders and deeper local-first stacks
uv run python experiments/a4_mesh.py             # Appendix A4: r5 at 2.0 m/s on meshes B and C
uv run python experiments/a5_radius.py           # Appendix A5: locality radius sweep under the early recipe
uv run python experiments/a7_equal_size.py       # Appendix A7: s5 and s6 on the Experiment 1 and 2 flows
uv run python experiments/cost.py                # Training time, peak memory and rollout time
uv run python experiments/anchor_draws.py        # Variation over anchor draws
```

## Tests and formatting

```bash
uv run pytest
uv run ruff format .
uv run ruff check --fix .
```
