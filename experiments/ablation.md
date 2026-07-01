# 5DMarkets V2 — Experimental Results and Ablation Study

This document summarizes the initial experimental validation of the 5DMarkets V2 architecture.

The objective of these experiments was to test the central hypothesis of the framework:

> Financial markets can be modeled as a deformable high-dimensional dynamical system analogous to 5D Computed Tomography (5DCT) motion modeling.

The experiments below document the iterative architecture development process and the empirical findings that have shaped the current model.

---

# Overview of Architecture

The current framework models financial markets as interacting force systems.

The full architecture is:

O(t)

↓

Observation Encoder (R)

↓

Deformation Vector Fields

• Correlation Field (M_A)  
• Sector Structure Field (M_B)  
• Capital Flow Field (M_C)  
• Exogenous Shock Field (M_D)

↓

Fusion Operator (F)

↓

Compression Layer (C)

↓

Temporal Memory (H)

↓

Future Transition Operator (Φ)

↓

Reconstruction Head (G)

↓

Predicted Future Market State

---

# Experiment 1 — Baseline Architecture with Random Synthetic Dataset

## Objective

Verify architecture compilation, gradient propagation, and successful end-to-end training.

## Dataset

Initial synthetic dataset generated using random Gaussian noise.

Future market state was defined as:

Current State + Small Random Perturbation

No market structure was present.

No correlation relationships.

No sector structure.

No capital flow dynamics.

No exogenous effects.

## Architecture

Original architecture used global averaging before temporal memory:

```python
fused = torch.mean(fused, dim=1)
```

This compressed all asset information into a single vector.

## Results

Epoch 0 Loss: 1.0154

Epoch 49 Loss: 1.0027

Total improvement:

ΔL = 0.0127

## Interpretation

The architecture trained successfully, confirming gradient propagation through the full network.

However, convergence was extremely weak.

This was expected because the dataset contained no underlying learnable structure.

---

# Experiment 2 — Structured Synthetic Market Physics Dataset

## Objective

Introduce realistic synthetic market dynamics and determine whether the architecture can learn structured market deformation.

## Dataset Redesign

The synthetic market simulator was redesigned to explicitly encode four force systems.

Future state evolved according to:

S(t+1) = S(t) + M_A + M_B + M_C + M_D

Where:

### Correlation Structure (M_A)

Neighboring assets partially correlated.

### Sector Structure (M_B)

Assets grouped into economic sectors with shared drift.

### Capital Flow Dynamics (M_C)

Capital rotated between sectors.

Modeled conservation principle:

∇ · J = 0

Capital leaving one sector entered another.

### Exogenous Shock Injection (M_D)

Random external market shocks applied probabilistically.

Examples:

• macroeconomic events  
• volatility spikes  
• policy shocks

## Architecture

Original architecture retained.

Global averaging still applied.

```python
fused = torch.mean(fused, dim=1)
```

## Results

Epoch 0 Loss: 1.0065

Epoch 49 Loss: 0.9887

Total improvement:

ΔL = 0.0179

## Interpretation

Introducing realistic market structure improved convergence.

However, learning remained weak.

This suggested the presence of an architectural bottleneck.

---

# Experiment 3 — Information Bottleneck Diagnosis

## Objective

Test whether global averaging was destroying predictive information.

## Hypothesis

The architecture compressed the market state using:

```python
fused = torch.mean(fused, dim=1)
```

This transformed:

[50,64]

→

[64]

The hypothesis was:

Global averaging destroys local asset-level deformation information necessary for prediction.

Equivalent theoretical statement:

I(M) ≠ I(M̄)

Where:

M = local market state

M̄ = globally averaged market state

## Architecture Modification

Global averaging removed.

Replaced with:

```python
fused = fused.view(batch,-1)
```

This preserved all asset-level information.

New dimensionality:

[50,64]

→

[3200]

The full market state was passed directly into temporal memory.

## Results

Early training showed dramatic improvement.

Epoch 0 Loss: 1.0049

Epoch 11 Loss: 0.7684

Training later crashed due to computational limitations.

## Interpretation

Preserving local market structure dramatically improved learning.

However, direct recurrent modeling of the full market state proved computationally infeasible.

---

# Experiment 4 — Compression Architecture (Current Baseline)

## Objective

Preserve full market structure while reducing computational cost.

## Architecture Modification

A compression operator was introduced.

New pipeline:

Full Market State

↓

Flatten (3200 dimensions)

↓

Compression Layer

↓

256-dimensional latent representation

↓

Temporal Memory (LSTM)

↓

Future Projection

↓

Market Reconstruction

Architecture:

3200 → 256 → LSTM

## Results

Epoch 0 Loss: 1.0052

Epoch 49 Loss: 0.7582

Total improvement:

ΔL = 0.247

Loss reduction:

24.6%

## Interpretation

This produced the strongest results so far.

Learning curve showed:

• stable monotonic descent  
• no oscillation  
• no instability  
• no plateau

Primary conclusion:

Preserving local market structure is essential for learning market dynamics.

---

# Ablation Study

The objective of the ablation study was to determine the relative contribution of each Deformation Vector Field.

Baseline model:

Final Loss = 0.7582

Each experiment removed one force system while keeping all other architecture and hyperparameters unchanged.

---

# Experiment 5 — Remove Capital Flow Field

Removed:

M_C = 0

Implementation:

```python
m_c = torch.zeros_like(latent)
```

## Results

Epoch 49 Loss: 0.8126

Performance degradation:

0.8126 - 0.7582 = 0.0544

Relative degradation:

+7.2%

## Interpretation

Removing capital flow significantly worsened performance.

Conclusion:

Capital redistribution dynamics contain meaningful predictive information.

---

# Experiment 6 — Remove Correlation Field

Removed:

M_A = 0

Implementation:

```python
m_a = torch.zeros_like(latent)
```

## Results

Epoch 49 Loss: 0.7709

Performance degradation:

0.7709 - 0.7582 = 0.0127

Relative degradation:

+1.7%

## Interpretation

Removing correlation only slightly worsened performance.

Conclusion:

Correlation contributes predictive information but less than expected.

---

# Experiment 7 — Remove Sector Structure Field

Removed:

M_B = 0

Implementation:

```python
m_b = torch.zeros_like(latent)
```

## Results

Epoch 49 Loss: 0.7821

Performance degradation:

0.7821 - 0.7582 = 0.0239

Relative degradation:

+3.15%

## Interpretation

Sector relationships contribute meaningful predictive information.

Conclusion:

Economic structural relationships matter more than statistical correlation structure.

---

# Experiment 8 — Remove Exogenous Shock Field

Removed:

M_D = 0

Implementation:

```python
m_d = torch.zeros_like(latent)
```

## Results

Epoch 49 Loss: 0.7505

Performance improvement:

0.7582 → 0.7505

Relative improvement:

+1.0%

## Interpretation

Removing exogenous shocks improved performance.

Current exogenous implementation behaves as unstructured noise rather than learnable signal.

Conclusion:

Current exogenous simulation requires redesign.

---

# Final Ablation Results

| Experiment | Final Loss | Relative Change |
|------------|------------|----------------|
| Full Model | 0.7582 | Baseline |
| No Capital Flow | 0.8126 | +7.2% |
| No Sector Structure | 0.7821 | +3.15% |
| No Correlation | 0.7709 | +1.7% |
| No Exogenous | 0.7505 | -1.0% |

---

# Relative Operator Importance

Current ranking:

Capital Flow > Sector Structure > Correlation > Exogenous

Or:

M_C > M_B > M_A > M_D

---

# Preliminary Theoretical Conclusions

The experiments currently support five working principles.

## Principle 1

Global averaging destroys predictive information.

I(M) ≠ I(M̄)

---

## Principle 2

Financial markets can be learned as deformation systems.

Structured synthetic market dynamics are recoverable by the architecture.

---

## Principle 3

Compression before temporal modeling outperforms direct full-state recurrence.

3200 → 256 → LSTM

is superior to

3200 → LSTM(3200)

---

## Principle 4

Capital redistribution dynamics contain strong predictive information.

Removing capital flow significantly worsens performance.

---

## Principle 5

Capital flow dynamics contain more predictive information than traditional correlation structure.

I(M_C) > I(M_A)

---

# Current Most Important Result

The most significant result so far:

Removing the Capital Flow Deformation Vector Field caused the largest degradation in predictive performance.

This suggests:

Capital migration dynamics may contain more predictive information than traditional correlation modeling.

---

# Current Status

Architecture validated.

Synthetic market dynamics successfully learned.

Ablation study completed.

Next focus:

• redesign exogenous shock simulation  
• test on real market data  
• connect decision functional  
• benchmark against traditional forecasting models
