# manifold-explorer
Beyond Accuracy: A case study in building trustworthy AI by stress-testing, explaining, and optimizing a computer vision model for robust deployment on an edge device.
> - Dataset used: [here](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)
## Core Philosophy of the Project

The purpose of this project is not to blindly chase accuracy. Our main goal is to understand the gap between a model's theoretical **"known world"** and the chaotic **"real world"** and to bridge this gap.

We define this with a mathematical objective:
* `E`: The infinite space of all possible images in the real world.
* `D`: Our clean, "studio" dataset at hand.
* `D''`: The learned space (learned manifold) that our model understands and can "comfortably navigate."

Our task is to explore the boundaries of the model's comfort zone `D''`. We aim to expand this area toward regions of `E` that were not previously covered. This means understanding and narrowing the **error manifold** **ε = E \ D''** (the area where the model doesn't know and will make mistakes).

*In short, we aim to make the model more reliable by hunting down its weaknesses.*

** Methodological Note:** The execution of this project by a single researcher carries a **natural bias risk** in Red Teaming and Chaos Testing designs. The discovered weaknesses should be interpreted with this constraint in mind.

## Initial Research: Which Model Knows How to Say "I Don't Know"? (`01_baseline_finetune.ipynb`)

Starting with this philosophy, our first task was to find a model that behaves "honestly" in the face of uncertainty. We compared two popular architectures based on this principle.

### Experiment 1: ResNet - A Dangerous Overconfidence

Our initial tests with ResNet were a vivid proof of why high accuracy rates can be misleading.
> - Full Kaggle notebook: [here](https://github.com/sinankarip/manifol-explorer/blob/main/notebooks/00_RNet_Shortcut_Learning.ipynb)  



* **Setup:** We fine-tuned the model with over 150,000 augmented images.
* **Misleading Result:** We achieved an absurdly high accuracy of **97%** on the "easy" validation set.
* **Stress Test Findings:** When the model stepped outside its comfort zone, it refused to say "I don't know."
   * It produced **absurd predictions** even with the slightest image corruptions.
   * More worryingly, it acted with **dangerous overconfidence** (99%+ confidence) while making these errors. There was almost no decrease in confidence scores.
* **Grad-CAM**: The heatmap below shows ResNet's tendency to focus on broader, less specific regions.
![ResNet Grad-CAM](plots/resnet_gradcam.png)

* **Overconfidence Analysis**: The plot clearly demonstrates ResNet's dangerous overconfidence. The confidence distribution for stress-test data (red) remains dangerously high, overlapping almost completely with the distribution for normal data (blue).
![ResNet Overconfidence Analysis](plots/resnet_oc.png)

**Summary:** ResNet was an unreliable candidate, prone to making errors silently and with self-assurance.

### Experiment 2: EfficientNet - An Honest Uncertainty

EfficientNet, however, exhibited a completely different character.

* **Setup:** We applied the same training process to EfficientNet.
* **Stress Test Findings:** The model gave an honest response when faced with unfamiliar data.
   * As if saying "I haven't seen these areas, I don't know what to do," it showed **significant and consistent decreases** in confidence scores.
   * This behavior indicated that the model better understood the limits of its own knowledge.
* **Grad-CAM**: In contrast, EfficientNet focuses sharply on the relevant areas of the leaf, as shown in the heatmap.
![EfficientNet Grad-CAM](plots/effnet_gradcam.png)

* **Overconfidence Analysis**: This plot highlights EfficientNet's "honest uncertainty." When faced with stress-test data (red), the model's confidence distribution shifts significantly to the left, indicating it "knows what it doesn't know."
![EfficientNet Overconfidence Analysis](plots/effnet_oc.png)

**Summary:** EfficientNet offered a more solid foundation as a model that "knows what it doesn't know" and was selected for the continuation of the project.

# 02_latent_voids.ipynb — Into the Latent Geometry of Uncertainty

## 1. Motivation: Beyond Accuracy, Toward Geometric Honesty

In our first notebook (`01_baseline_finetune.ipynb`), we compared model behaviors under epistemic stress. ResNet demonstrated pathological overconfidence, maintaining near-uniformly high softmax confidences even under strong distributional shifts. EfficientNet, by contrast, displayed what we called **honest uncertainty**—a meaningful reduction in confidence when encountering unknown inputs.

However, scalar confidence alone is a surface symptom. The deeper question is geometric:

> "What does the model's internal representation space look like when it admits, or refuses to admit, ignorance?"

To investigate this, the second stage of the study turns inward, mapping the model's learned manifold **D″**—the subset of the feature space where the network feels "comfortable." The objective is to quantify and visualize how this manifold diverges from the real-world space **E**, and to characterize the **voids**—regions of high epistemic entropy and low sample support.

## 2. Method: Constructing the Void Score

Each image in the test and validation sets was embedded through EfficientNet's penultimate layer, yielding CNN feature vectors. Three orthogonal indicators were computed for each point:

### Uncertainty Score
- **Formula**: `u_i = 1 - max_j p_ij` where `p_ij = softmax(z_i)_j`
- **Purpose**: Captures predictive entropy—how unsure the model is in label space.

### Density Score
- **Method**: Estimated by k-nearest neighbor distances

### Outlier Score
- **Method**: Computed per class via Mahalanobis distance
- **Purpose**: Approximates how far the sample lies from its class manifold

The three components were normalized and linearly combined with equal weights `w = [1, 1, 1]` to form the **Void Score V_i**.

This metric jointly encodes epistemic (model-based) and aleatoric (data-based) uncertainty in a latent geometric sense.

## 3. Latent-Space Topology via UMAP Projection

To interpret the Void Score structurally, all feature vectors were reduced to 2D using UMAP (Uniform Manifold Approximation and Projection).

**Key Findings:**
- Regions with high Void Score formed **void clusters**—low-density, high-uncertainty zones
- Compact low-Void Score regions corresponded to over-represented, "safe" manifolds where the model's behavior is stable
- Despite the small test set (N=1000), systematic augmentation confirmed that void regions correlate strongly with confidence collapses under input perturbations

## 4. Linear Proxy Analysis
![OLS Line](plots/OLS_Line.png)
A striking empirical pattern emerged: a consistent linear trend between confidence and Void Score, suggesting a simple linear relation explains much of the variance.

**OLS Model Performance:**
- MAE: 0.0889
- MAPE: 0.0419  
- RMSE: 0.1127

**Residual Interpretation:**
- **Red Triangles (▲)**: Samples where actual Void Score > predicted—truly chaotic, underrepresented areas missed by OLS
- **Blue Triangles (▼)**: Samples where actual Void Score < predicted—dense, low-risk zones where OLS overestimates uncertainty
![Residual Interpretation](plots/Residual_Analysis.png)
**Key Insight**: OLS falsely flagged 839 safe examples while missing only 85 genuinely risky ones (~1.7%). The Void Score preserves the nonlinear topology of the learned manifold that linear models cannot capture.
![VoidScore_OLS](plots/VoidScore_OLS.png)
## 5. Epistemic Implication: Toward Geometry-Aware Augmentation

The Void Score formalizes the model's comfort zone boundary in latent space. It enables data augmentation or adversarial sampling to be directed **not randomly, but topologically**—toward the voids of D″ that lie on the frontier of the real-world manifold E.

These high-Void Score regions are prime candidates for:
- Synthetic sample generation via diffusion or GANs
- Physically-based render augmentation (e.g., via OpenUSD / Blender)

This transforms uncertainty into an **actionable geometric prior**, moving beyond passive measurement to active model improvement.

# 03-anomaly-archeology.ipynb

## Motivation: Targeted Augmentation Through Component Analysis

Following the computation of the three uncertainty scores (Uncertainty, Density, and Outlier) in the previous notebook, we integrated these metrics into our dataset. The core objective behind this synthesis is to refine our data augmentation strategy.

### The Artifact Challenge in Manifold Expansion
Standard augmentation techniques often introduce local artifacts that fail to meaningfully expand the model's learned manifold. Our hypothesis is that by identifying which specific type of uncertainty dominates each sample, we can apply **targeted, component-aware augmentation** that addresses the actual epistemic gaps rather than uniformly augmenting all data.

### The Critical Insight: Component Dominance Over Absolute Scores

**Important Note:** 

This discrepancy arises naturally because we performed calculations in separately normalized spaces. While this breaks the theoretical equality, it is not a concern for our purposes.

**What Truly Matters:**
The essential insight is not the absolute void score value, but identifying **which of the three components dominates** for each sample. The component with the largest share in the score pool indicates the primary source of uncertainty that drives the high void score.

### Strategic Augmentation Approach

By understanding which factor (predictive uncertainty, local sparsity, or distributional outlierness) contributes most significantly to raising the void score, we can:

- **Weight augmentation strategies accordingly**
- **Highlight examples that genuinely indicate epistemic gaps**
- **Use Stable Diffusion or GANs to generate data that specifically targets the dominant uncertainty type**
- **Achieve more consistent manifold expansion through focused data generation**

This component-aware approach transforms uncertainty analysis from a diagnostic tool into an actionable blueprint for strategic data augmentation.

# 04_tail_distribution.ipynb — Stuck on the Tail Distribution

## 1. Motivation: The Statistical Wall

Our previous work defined the **Void Score** to map regions of epistemic uncertainty in the latent space of EfficientNet. The logical next step was to populate these voids with *synthetic data*. However, we encountered a fundamental constraint of *long-tail distribution learning*.

We only have **2,000 agricultural images**. Relative to the **hundreds of millions** of samples used to train models like *Stable Diffusion*, this represents an extreme tail region of the global image distribution. Consequently, when prompted with a term such as `"leaf"`, the model defaults to producing bright, idealized, stereotypical greenery—statistically dominant in its prior training corpus. This reveals a failure to capture the **fine-grained statistical manifold** of our specific dataset \( D \subset E \).
![SD_sample_01](plots/SD_sample_01.png)
![SD_sample_02](plots/SD_sample_02.png)
![SD_sample_03](plots/SD_sample_03.png)

---

## 2. The Tail Distribution Problem

Even if we attempt to fine-tune Stable Diffusion, the parameter prior induced by its massive base model remains heavily biased toward the **high-density regions** of its global training distribution.  
Formally, if \( p_{\text{SD}}(x) \) denotes the pretrained model’s data prior and \( p_{\text{agri}}(x) \) our narrow agricultural distribution, then \( p_{\text{agri}}(x) \ll p_{\text{SD}}(x) \) for most \( x \in \mathcal{X} \).  
Fine-tuning with only 2,000 samples introduces a **measure mismatch problem**:
\[
\int_{\mathcal{X}} |p_{\text{agri}}(x) - p_{\text{SD}}(x)| dx \gg 0
\]
Thus, the fine-tuned model remains dominated by the pretrained prior and continues to generate *semantically irrelevant or stylistically biased* images.

---

# 05a_feature_extraction.ipynb — Attempted Latent-Space Validation

## 1. Motivation: Does the Void Score Reflect True Latent Geometry?

The Void Score consisted of three components:
* predictive uncertainty,
* geometric density (k-NN),
* Mahalanobis outlierness.

These components are heuristics dependent on pixel space and class statistics. Therefore, a natural question emerged:

"Do these heuristic clusters actually separate in EfficientNet's CNN latent space?"

To answer this question, we extracted penultimate-layer feature vectors for all samples (N × d matrix).

# 05b_latent_validation_umap.ipynb — Failed Latent-Space Clustering

## 2. Method: UMAP Projection of CNN Features

Feature vectors were reduced to 2 dimensions using UMAP, and five categories were labeled:

- [core subset](plots/latent_clusters/core_tail.png)
- [geom tail](plots/latent_clusters/geom_tail.png)
- [epistemic tail](plots/latent_clusters/epistemic_tail.png)
- [chaotic tail](plots/latent_clusters/chaotic_tail.png)
- [unknown subset](plots/latent_clusters/unknown_imgs.png)

The purpose of this step was to verify whether the clusters showed structural separation on the true latent manifold.

### Additional Note

Although the metadata (void score components, uncertainty signals, and density proxies) provide meaningful diagnostic structure, their visual appearance in RGB space does not necessarily reflect the underlying latent geometry. These subsets remain valuable for targeted augmentation and robustness analysis, but they should not be interpreted as visually coherent clusters. Their purpose is functional, not aesthetic.

## 3. Result: No Meaningful Separation

No meaningful cluster formation was observed in any category or projection.

Therefore, the latent-validation experiment was left as a diagnostic step and not included in the pipeline.

![UMAP_clusters](plots/umap_clusters.png)

-[pca cluster](plots/clusters/pca_cluster.png)
-[cluster 1](plots/clusters/cluster01.png)
-[cluster 2](plots/clusters/cluster02.png)



# 06_mfold_aware_analysis.ipynb — Transition to Manifold-Aware Augmentation

## 1. Insight: Void Score Should Guide Weighting, Not Clustering

After latent validation failed, the problem was reframed:

"The role of Void Score is not to produce clusters, but to serve as a weighting function that determines augmentation intensity."

Therefore, the Void Score was transformed into the form:

w_i = f(void_i)

Using quantile-based separation:
* low void → minimal augmentation
* high void → heavy augmentation
* augmentations with extreme artifacts → excluded from training, used only in chaos test set

## 2. Practical Outcome

This transformation provided two important benefits:

1. It strengthened critical edge-case regions without distorting the realistic manifold of the training data.
2. Chaos test artifacts (extreme color distortion, purple leaves, etc.) were completely separated from training.

As a result, the project evolved from heuristic clustering errors into a more consistent "manifold-aware robustness" strategy.

--------

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
