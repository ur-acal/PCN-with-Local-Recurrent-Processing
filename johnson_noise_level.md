# Johnson Noise Level Calculation

## Physical Parameters
From the log (`test_480966.out:504747`):
- **R** (resistance) = 100,000 Ω = 100 kΩ
- **C_fb** (capacitance) = 4.9e-14 F = 49 fF
- **4*k_B*T** = 4.16e-21 J (at room temperature T=300K)
- **offset_eps** (default) = 0.002 (used when `offset_eps=None`)

## Johnson Noise Formula

When `thermal_noise=True` and `offset_eps=None`, the code calculates:

```python
eps_scale = (1 / offset_eps) * sqrt(4*k_B*T/R) / C_fb
```

### Calculation Steps:

1. **Thermal noise voltage density**: 
   ```
   sqrt(4*k_B*T/R) = sqrt(1.66e-20 / 1e5) = 4.08e-13 V/sqrt(Hz)
   ```

2. **Normalized by capacitance**:
   ```
   sqrt(4*k_B*T/R)/C = 4.08e-13 / 4.9e-14 = 8.32
   ```

3. **Scale factor**:
   ```
   eps_scale = (1/0.002) * 8.32 = 4,162.46
   ```

4. **Effective noise coefficient**:
   ```
   eps_scale * offset_eps = 4,162.46 * 0.002 = 8.32
   ```

## Actual Noise Application

The noise is **weight-dependent** and applied as:

```python
eps = eps_scale * offset_eps * weight_scaling
```

Where `weight_scaling` is:
- For FFconv: `sqrt(sum(abs(weights)))` per output channel
- For FBconv: `sqrt(sum(abs(weights)))` per input channel

This means:
- **Base noise level**: ~8.32 (relative units)
- **Actual noise**: Varies per layer and channel based on weight magnitudes
- **Larger weights → larger noise** (proportional to sqrt of weight sum)

## Comparison with Fixed offset_eps

When `offset_eps` is explicitly set (e.g., 0.05, 0.1, 0.15, 0.2):
- `eps_scale = None`
- Noise level = `offset_eps / sqrt(R*C)` = `offset_eps / sqrt(1e5 * 4.9e-14)` = `offset_eps / 2.21e-9`
- This gives fixed noise levels: ~22.6, 45.2, 67.8, 90.4 (for offset_eps = 0.05, 0.1, 0.15, 0.2)

**Johnson noise (offset_eps=None) is adaptive and weight-dependent**, while fixed offset_eps values give constant noise levels.


