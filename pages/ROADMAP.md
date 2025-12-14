# Project Roadmap & Methodology
...

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

## Methodological Flow

<figure>
  <img src="{{ site.baseurl }}/plots/Mermaid.svg"
       alt="Methodological Flow"
       style="max-width:100%; height:auto;">
  <figcaption><strong>Figure 1.</strong> End-to-end methodological pipeline of Manifold Explorer.</figcaption>
</figure>



