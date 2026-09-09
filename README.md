# PCN repository for Image classification

Current physical PCN commands and non-ideality settings:
[Pinned training and evaluation reference](docs/current_pcn_reference.md).

## Dependency
Enumerate a list of extra dependencies other than normal ML settings.
```bash
# cross-sim
git@github.com:sandialabs/cross-sim.git
# ...[move to the repo]...
pip install .

# neural ode repo
pip install torchdiffeq

# scanGFI data
# ...[under the scangen dir]...
uv run scangen create-config # Env set automatically if not before
```
## Train
```bash
./launch_scripts/run_ode_train.sh
```
## Inference
```bash
./launch_scripts/run_ode_wrapped_inference.sh
```
## Finetune
```bash
./launch_scripts/run_ode_mixed_ft.sh
```
