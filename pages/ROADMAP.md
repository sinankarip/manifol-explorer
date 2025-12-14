#  Project Roadmap & Methodology

This document outlines the strategic evolution of **Manifold Explorer**. The project moves beyond standard accuracy metrics to build a framework capable of quantifying epistemic uncertainty and resisting open-set ambiguity.

##  Phase I: Diagnosis & Manifold Analysis (Completed)
> **Goal:** Establish a baseline and identify the topological "blind spots" of standard deep learning methods.

- [x] **Baseline Architecture Selection**
  - Comparative analysis of ResNet vs. EfficientNet under stress tests.
  - *Outcome:* EfficientNet selected for its "honest uncertainty" behavior compared to ResNet's overconfidence.
- [x] **Latent Space Archaeology**
  - Development of the **Void Score** metric ($Uncertainty + Density + Outlier$).
  - Mapping the model's "comfort zone" ($D''$) versus the unknown ($E \setminus D''$).
- [x] **Generative Augmentation Tests**
  - Attempted to fill manifold voids using Stable Diffusion.
  - *Finding:* Identified the "Tail Distribution" problem; GenAI failed to generate specific agricultural anomalies, proving the need for a non-generative solution.

---

##  Phase II: The Open-Set Solution (Current Focus)
> **Goal:** Address the structural flaws identified in Phase I using energy-based learning and geometric regularization.

- [ ] **Methodology Shift: Energy-Based Models (EBM)**
  - Transition from Softmax probability to scalar energy values to model data density explicitly.
  - *Target:* Low Energy for In-Distribution (ID), High Energy for Out-of-Distribution (OOD).
- [ ] **Outlier Exposure (OE) Strategy**
  - Leveraging "Texture" and "Noise" datasets to maximize entropy on the model's blind spots.
- [ ] **Geometric Regularization**
  - Implementing **Spectral Normalization** to enforce Lipschitz continuity.
  - Preventing the decision boundary from becoming arbitrarily sharp during training.

---

## Phase III: Defense & Guardrails (Future)
> **Goal:** Wrap the robust model in a safety layer for real-world deployment.

- [ ] **Dynamic Rejection Mechanism**
  - Implementing a threshold logic: `if Energy(x) > τ then Reject`.
- [ ] **Adversarial Stress Testing**
  - Validating manifold tightness using gradient-based attacks (FGSM, PGD).
- [ ] **Safe Inference Engine**
  - Finalizing the system design to flag "Unknown" inputs rather than forcing a prediction.

---

##  Methodological Flow

```mermaid
graph TD
    %% --- GLOBAL STYLES ---
    classDef done fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;
    classDef active fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,stroke-dasharray: 5 5,color:#000;
    classDef future fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000;
    classDef fail fill:#ffebee,stroke:#c62828,stroke-width:2px,stroke-dasharray: 5 5,color:#000;

    %% --- PHASE I ---
    subgraph Phase_I [ Phase I: Diagnosis & Analysis]
        direction TB
        P1_A[Baseline Comparison] -->|ResNet vs EffNet| P1_B[EfficientNet Selected]
        P1_B --> P1_C[Void Score Formulation]
        P1_C --> P1_D{Manifold Analysis}
        P1_D -->|Tail Distribution| P1_E[Generative AI Failed]
        P1_D -->|Open Set Risk| P1_F[Heuristics Insufficient]
    end

    %% --- PHASE II ---
    subgraph Phase_II [ Phase II: Open Set Solution]
        direction TB
        P1_F --> P2_A[Energy-Based Models]
        P1_E --> P2_B[Outlier Exposure Strategy]
        P2_A --> P2_C[Spectral Normalization]
        P2_B --> P2_C
        P2_C --> P2_D[Goal: Robust Manifold]
    end

    %% --- PHASE III ---
    subgraph Phase_III [ Phase III: Active Defense]
        direction TB
        P2_D --> P3_A[Guardrail: Rejection Threshold]
        P3_A --> P3_B[Adversarial Stress Test]
        P3_B --> P3_C[Final Safe System]
    end

    %% --- STYLING ---
    class P1_A,P1_B,P1_C,P1_D done;
    class P1_E,P1_F fail;
    class P2_A,P2_B,P2_C,P2_D active;
    class P3_A,P3_B,P3_C future;
