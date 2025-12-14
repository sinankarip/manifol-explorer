# manifold-explorer
Beyond Accuracy: A framework for diagnosing topological blind spots and quantifying epistemic uncertainty in computer vision models.

> - Dataset used: [here](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)

## What Problem Are We Solving?

Imagine you have a machine learning model that performs brilliantly in your lab tests—97% accuracy, amazing results! But here's the critical question: **Can you really trust this model in the real world?**

The harsh reality is that we only discover the answer **after** deploying the model to production. What if the model causes critical failures in real-world scenarios? For example:
- A traffic camera system designed to detect license plates on a production line won't encounter a horse on the highway
- But a highway surveillance camera absolutely might encounter a horse, a deer, or a car driving the wrong way

**You can't just say "here's the model, make decisions!" and hope for the best.** The model needs to be robust and reliable **before** production. This is the core problem we're addressing.

## Our Journey: Six Steps to Model Reliability

### Step 1: Finding an Honest Model (`01_baseline_finetune.ipynb`)

We tested two popular models and asked them a simple question: **"Can you admit when you don't know something?"**

**ResNet: The Dangerously Overconfident**
- Achieved 97% accuracy on clean data
- When we corrupted images slightly, it made **absurd predictions** with 99%+ confidence
- This is what we call **"ignorant courage"**—confidently wrong
- Like someone who doesn't know the answer but raises their hand first anyway

![ResNet Grad-CAM]({{ site.baseurl }}/plots/resnet_gradcam.png)
*ResNet focuses on broad, unfocused regions*

![ResNet Overconfidence Analysis]({{ site.baseurl }}/plots/resnet_oc.png)
*ResNet's dangerous overconfidence: red (corrupted data) overlaps with blue (normal data)*

**EfficientNet: The Humble Scholar**
- When faced with unfamiliar images, it **lowered its confidence significantly**
- Essentially saying: "I haven't seen this before, I'm not sure"
- This is the **virtue of admitting ignorance** we were looking for

![EfficientNet Grad-CAM]({{ site.baseurl }}/plots/effnet_gradcam.png)
*EfficientNet focuses sharply on relevant leaf areas*

![EfficientNet Overconfidence Analysis]({{ site.baseurl }}/plots/effnet_oc.png)
*EfficientNet's honest uncertainty: confidence drops significantly for corrupted data (red)*

**Winner:** We chose EfficientNet because it can say "I don't know" rather than pretending with false confidence.

---

### Step 2: Looking Inside the Model's Mind (`02_latent_voids.ipynb`)

Now we wanted to understand: **What does the model actually "see" in its brain?**

We created a mathematical formula called the **Void Score** that identifies dangerous gaps in the model's understanding—regions where it might fail silently.

But wait—did we really need such a complex formula? We tested it against a simple linear regression:

![OLS Line]({{ site.baseurl }}/plots/OLS_Line.png)
*Linear regression captures most of the pattern*

**The Result:** The regression model captured 98.3% of the pattern, but our Void Score caught an additional **1.7% of critical cases** that the regression missed.

![Residual Interpretation]({{ site.baseurl }}/plots/Residual_Analysis.png)
*Red triangles (▲): truly dangerous cases our method caught*
*Blue triangles (▼): safe zones the regression overestimated*

![VoidScore OLS]({{ site.baseurl }}/plots/VoidScore_OLS.png)

**Why This Matters:** In a small dataset, that 1.7% is incredibly valuable. These are the exact blind spots we need to fix.

---

### Step 3: Breaking Down the Score (`03_anomaly_archeology.ipynb`)

Our Void Score has three components:
1. **Uncertainty:** How confused is the model?
2. **Density:** How rare is this type of image?
3. **Outlier:** How far is this from what the model has seen?

We separated these three components to understand **which type of weakness** each image represents. This helps us target our improvements more precisely.

---

### Step 4: The Failed Synthetic Data Experiment (`04_tail_distribution.ipynb`)

We thought: "Let's generate synthetic images to fill the gaps!"

We tried using Stable Diffusion, a powerful image generator. But here's what happened:

![SD sample 01]({{ site.baseurl }}/plots/SD_sample_01.png)
![SD sample 02]({{ site.baseurl }}/plots/SD_sample_02.png)
![SD sample 03]({{ site.baseurl }}/plots/SD_sample_03.png)

**The Problem:** When we asked for "leaf images," it gave us bright, perfect, stereotypical green leaves—like stock photos. These looked nothing like our real agricultural dataset.

**Why It Failed:** Stable Diffusion was trained on hundreds of millions of images. Our 2,000 agricultural images are a tiny tail in that massive distribution. The generator couldn't understand our specific needs.

**Lesson Learned:** We needed to pivot our strategy.

---

### Step 5: The Clustering Attempt (`05a_feature_extraction.ipynb`, `05b_latent_validation_umap.ipynb`)

We tried to see if the model organizes dangerous images into clusters in its "mental space."

We labeled five categories:
-![Core subset]({{ site.baseurl }}/plots/latent_clusters/core_tail.png)
-![Geometric tail]({{ site.baseurl }}/plots/latent_clusters/geom_tail.png)
-![Epistemic tail]({{ site.baseurl }}/plots/latent_clusters/epistemic_tail.png)
-![Chaotic tail]({{ site.baseurl }}/plots/latent_clusters/chaotic_tail.png)
-![Unknown subset]({{ site.baseurl }}/plots/latent_clusters/unknown_imgs.png)

![UMAP clusters]({{ site.baseurl }}/plots/umap_clusters.png)

-![PCA cluster]({{ site.baseurl }}/plots/clusters/pca_cluster.png)
-![Cluster 1]({{ site.baseurl }}/plots/clusters/cluster01.png)
-![Cluster 2]({{ site.baseurl }}/plots/clusters/cluster02.png)

**The Result:** No meaningful patterns emerged. The model's internal organization was chaotic, not structured.

**Lesson Learned:** The Void Score shouldn't be used for clustering—it should guide **how much** we augment each image.

---

### Step 6: The Egg Carton Analogy (`06_mfold_aware_analysis.ipynb`)

Here's our final strategy, explained through a metaphor:

**Imagine an egg carton with empty spaces.** We want to fill those spaces with water (augmented data) without soaking and destroying the carton (the model's learned patterns).

- **High Void Score images:** Fill generously (heavy augmentation)
- **Low Void Score images:** Fill minimally (light augmentation)
- **Extreme artifacts:** Keep them separate, use only for stress testing

This way, we strengthen the model's weak spots **without** corrupting what it already knows well.

**The Approach:**
1. Use the Void Score to assign **importance weights** to each image
2. Apply augmentation proportional to those weights
3. Carefully balance filling gaps vs. preserving the learned manifold

---

### Step 7: Validating Our Intuition

**Notebook:** `07_chaos_edge_latent_validation.ipynb`

We had manually sorted our "bad" data into two piles:

1. **Chaos:** Complete garbage, visually destroyed images.
2. **Edge:** Tricky, ambiguous images that sit on the borderline.

But a nagging question remained: *Did we just make these categories up?* Is this distinction real, or is it just a product of our human imagination? We needed to know if the model actually sees the difference between a "Chaos" image and an "Edge" image.

#### The Invisible Wall

First, we tried to plot these two groups on a chart. To our eyes, they looked like a mixed-up mess. There was no clear line separating them.

However, **math sees what eyes cannot.**

When we ran a statistical test (Logistic Regression) on the model's raw brain activity (latent space), we discovered a hidden truth: **The model can tell them apart with 92% accuracy.** Even though the chart looked messy to us, the model had drawn a very sharp line in its mind.

![Chaos–Edge UMAP separation]({{ site.baseurl }}/plots/chaos_edge_umap_separation.png)


> *To us, it looks mixed. To the model, these are two different worlds.*

#### Is It Just About Color?

We challenged the model further: *"Are you just looking at weird colors to tell these apart?"*

We stripped away all color (Grayscale) and forced the model to look again.

- **With Color:** It's almost too easy (95% separation).
- **Black & White:** It became harder, but the model could still tell the difference (78%).

#### The Verdict

The model agrees with our human labeling. Yes, it uses color as a major shortcut (because weird colors are a dead giveaway for "Chaos"), but it's not just looking at color. **It actually understands the structural damage in the images.**

This confirms that our "Chaos" vs. "Edge" separation isn't just a subjective feeling—**it's a mathematically real distinction that the model understands.**

## Core Philosophy

This project is **not about chasing accuracy numbers**. It's about understanding the gap between:
- **D''**: The model's "comfort zone"—what it reliably knows
- **E**: The infinite chaos of the real world

Our goal is to systematically **shrink the error manifold** (ε = E \ D'')—the dangerous region where the model doesn't know but won't admit it.

**In simple terms:** We're hunting down the model's weaknesses before they cause problems in production.

---

## Methodological Note

This project was executed by a single researcher, which carries a natural bias risk in designing stress tests and adversarial scenarios. The discovered weaknesses should be interpreted with this constraint in mind.

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
