# 5DMarketsV2
Simpler version, but read V1 first

# 5DMarkets V2

A physics-inspired market modeling framework that treats financial markets as a high-dimensional deformable dynamical system.

Instead of predicting isolated stock prices, 5DMarkets models the market as a continuously evolving state field in which assets interact through simultaneous force systems analogous to deformation vector fields (DVFs) in 5D Computed Tomography (5DCT).

---

## Core Idea

Traditional quantitative finance approaches often ask:

- What will stock X do tomorrow?
- Will price increase or decrease?
- What is the next return prediction?

5DMarkets asks a fundamentally different question:

> How does the entire market deform through time under interacting force systems?

The framework is directly inspired by 5DCT motion modeling used in medical physics, where lung anatomy is represented as a voxel field and future anatomical states are predicted by learning deformation vector fields between time-resolved CT images.

We transfer this concept to financial markets.

Each asset is treated as a state vector rather than an isolated price process.

---

## Financial State Representation

At time t, each asset is represented by a multidimensional state vector:

\[
v_i(t)=
[p_i,\sigma_i,V_i,o_i,s_i,m_i]
\]

Where:

- p = price
- σ = volatility
- V = volume
- o = options skew
- s = sentiment
- m = macroeconomic factor exposure

The observed market field becomes:

\[
O_t=\{v_1,v_2,...,v_n\}
\]

where each asset acts analogously to a voxel in medical imaging.

---

## Architecture

The complete architecture:

\[
O_t
\rightarrow
R
\rightarrow
\{M_A,M_B,M_C,M_D\}
\rightarrow
F
\rightarrow
H
\rightarrow
\Phi
\rightarrow
G
\rightarrow
D
\]

---

## Operator Definitions

### R — Observation Encoder

Transforms raw market observations into latent asset state embeddings.

\[
R(O_t)=S_t
\]

Purpose:

- Feature extraction
- Local asset state encoding
- Dimensional compression

---

### M_A — Correlation Deformation Field

Models statistical coupling between historically correlated assets.

Examples:

- Technology stocks moving together
- Correlated ETF movement

Mechanism:

- Self-attention over asset embeddings

Purpose:

- Learn inter-asset statistical dependence

---

### M_B — Sector Structural Deformation Field

Models structural coupling between economically related assets.

Examples:

- Semiconductor sector
- Banking sector
- Energy sector

Purpose:

- Sector-wide deformation dynamics

---

### M_C — Capital Flow Deformation Field

Models capital conservation and redistribution.

Core hypothesis:

> Capital leaving one market region must enter another region.

Examples:

- Rotation from bonds into equities
- Capital leaving growth into value

Mathematical intuition:

\[
\nabla\cdot J=0
\]

where J represents capital flow current.

Purpose:

- Learn endogenous market reallocation dynamics

---

### M_D — Exogenous Shock Field

Models external forcing terms.

Examples:

- Federal Reserve rate decisions
- Geopolitical conflict
- Earnings surprises
- Regulatory announcements

Equivalent to an external forcing term in dynamical systems.

\[
\frac{dx}{dt}=f(x)+u(t)
\]

---

### F — Fusion Operator

Combines all deformation fields into a unified market deformation state.

\[
F(M_A,M_B,M_C,M_D)
\]

Purpose:

- Learn interaction between simultaneous market force systems

---

### H — Temporal Memory

Models path dependence and historical market memory.

Core hypothesis:

> Past market deformation influences future deformation.

Current implementation:

- LSTM temporal encoder

Purpose:

- Learn sequential dependency structure

---

### Φ — Future Transition Operator

Projects current market state forward through time.

\[
\Phi(S_t,H_t)
\]

Purpose:

- Predict future market trajectory manifold

Output:

- Future latent trajectory

---

### G — Reconstruction Head

Reconstructs future market state field from latent trajectory.

\[
G(T(t))=O_{future}
\]

Purpose:

- Convert latent trajectory into full future market state prediction

Output:

- Future market field

Shape:

```
[batch, trajectory_length, num_assets, feature_dim]
```

---

### D — Decision Functional (Phase 2)

Currently disconnected during training.

Will eventually output:

\[
D=
(P_{target},P(reach),\tau,Q_{path})
\]

Outputs:

- Target sell price
- Probability target is reached
- Expected time to reach target
- Path quality score

Purpose:

- Translate predicted market dynamics into trading decisions

---

## Training Philosophy

Training occurs in two phases.

---

### Phase 1 — Market Physics Learning

Current training phase.

Goal:

Learn market deformation dynamics.

Architecture:

\[
O_t
\rightarrow
R
\rightarrow
M
\rightarrow
F
\rightarrow
H
\rightarrow
\Phi
\rightarrow
G
\]

Loss:

Mean squared error between predicted future market field and true future market field.

\[
L=
||O_{future}^{predicted}-O_{future}^{true}||^2
\]

Objective:

> Learn market dynamics before optimizing trades.

---

### Phase 2 — Decision Learning

Future phase.

Freeze learned backbone.

Reconnect decision functional.

Train decision layer independently.

Goal:

Learn actionable trading decisions.

---

## Synthetic Market Simulator

Current dataset uses synthetic market simulation designed around four force systems.

Simulation explicitly encodes:

### Correlation Structure

Assets partially correlated with neighboring assets.

---

### Sector Structure

Sector-wide drift applied to economically related assets.

---

### Capital Flow Conservation

Capital rotates between sectors while preserving total capital flow.

---

### Exogenous Shock Injection

Random external shock events.

Examples:

- Macro shocks
- Policy shocks
- Broad volatility events

---

## Research Hypothesis

Primary hypothesis:

> Financial markets behave as a deformable multidimensional state field governed by simultaneous interacting force systems.

Secondary hypothesis:

> Preserving local asset structure is necessary for accurate market trajectory prediction.

Key principle:

> Global averaging destroys deformation information.

---

## Current Development Status

Current implementation:

- Observation Encoder ✓
- Correlation DVF ✓
- Sector DVF ✓
- Capital Flow DVF ✓
- Exogenous DVF ✓
- Fusion Operator ✓
- Temporal Memory ✓
- Future Transition ✓
- Reconstruction Head ✓
- Full End-to-End Forward Pass ✓
- Phase 1 Training Active ✓

In development:

- Architecture optimization
- Asset-level temporal memory improvements
- Real market data ingestion
- Decision Functional training
- Real-time inference pipeline

---

## Long Term Goal

Build a general framework capable of learning underlying market dynamics and predicting future market state trajectories rather than isolated price movements.

Ultimate objective:

> Learn how markets move before learning how to trade them.

---

## Repository Structure

```text
5DMarkets_V2/

src/

    dataset.py

    config.py

    model.py

    train.py

    layers/

        observation_encoder.py

        dvf_correlation.py

        dvf_sector.py

        dvf_capital_flow.py

        dvf_exogenous.py

        fusion_operator.py

        temporal_memory.py

        future_transition.py

        reconstruction_head.py

        decision_functional.py
```

---

## Project Status

Active research and development.

Current focus:

> Proving whether market dynamics can be learned as a deformation field problem.
