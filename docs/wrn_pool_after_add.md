# Optional Pool-After-Addition WRN

`baseline/cifar_wrn_pool_after_add.py` provides separate
`WideBasicBlockPoolAfterAdd` and `WideResNetPoolAfterAddCIFAR` classes.
Existing registrations and default model behavior are unchanged. New control
rows 14-18 select these classes explicitly; see [launching guide](launching_guide.md#new-pooling-controls-rows-11-18).

With the default `pool_type="avg"`, use this implementation for the existing
**AvgPool main + AvgPool/padding shortcut** architecture. It computes two stride-1 convolutions, adds the
identity/zero-padded shortcut, then pools the sum at transitions. It does not
apply to stride-2 convolution or learned-projection shortcut checkpoints.

`pool_type="max"` instead applies MaxPool to that sum. This is a new training
architecture, not a numerically equivalent replacement for AvgPool or for
separately max-pooled branches. `pre_pool` exposes the same sum in both modes.

Choose the class explicitly when constructing the model; supply the same
depth, width, class count, BN/bias and dropout settings as the old model:

```python
from baseline.cifar_wrn_pool_after_add import WideResNetPoolAfterAddCIFAR

model = WideResNetPoolAfterAddCIFAR(
    depth=28, widen_factor=2, num_classes=100,
    use_batchnorm=True, conv_bias=False, final_dropout_rate=0.25,
)
model.load_state_dict(checkpoint["net"], strict=True)
model.eval()
```

The original class remains `baseline.cifar_resnet.WideResNetCIFAR` with
`avgpool_main_downsample=True, avgpool_downsample_shortcut=True`. Both classes
use identical state-dict keys, including BN buffers. No checkpoint conversion
or retraining is needed. Existing command-line loaders still select their
original implementation; importing the new class does not switch them.

For recording, attach a forward hook to each block's `pre_pool` Identity module.
It exposes the computed residual sum without repeating addition or padding.
Clone outputs when retaining them: subsequent in-place activations can mutate
the tensor at non-downsampling boundaries. The model does not retain activation
history automatically.

Average pooling is linear, so the two implementations are mathematically
equivalent. Floating-point results need not be bitwise identical. Existing
BN-free in-place ReLU/shortcut-input semantics are deliberately preserved.
Finite-precision training trajectories are not guaranteed identical.

## Verification

From the repository root, using the existing Python environment:

```bash
python -m unittest discover -s tests -p test_wrn_pool_after_add.py
python -m baseline.verify_wrn_pool_after_add --samples 256 \
  --output logs/wrn_pool_after_add_verification/results.json
```

The CPU verifier strictly loads the four existing WRN-28-2 AvgPool checkpoints
(BN and BN-free, CIFAR-10 and CIFAR-100) into both classes. It compares logits,
predictions and correct counts on the first 256 test images per dataset.
Use `--samples 10000` for full-test comparisons. No downloads, training,
recalibration, mismatch experiments or checkpoint writes are performed.

Verified locally: all four strict loads passed; zero prediction changes on
256 images per checkpoint. Maximum absolute logit difference was `3.34e-6`;
relative L2 differences ranged from `1.31e-7` to `2.84e-7`. The three new tests
(including all four sizes and BN/bias combinations) and three existing AvgPool
tests passed. These checks do not claim full-test or GPU bitwise equivalence.
