# Data Augmentation Process and Observations
[Data](https://drive.google.com/drive/folders/1SjurPH0S4M188chSMrSgYlVChUMjhS0G?usp=sharing)

## Augmentation Application

Images marked as high-void were augmented using the scripts located in the `scripts/` directory.  
The number of augmented samples generated for each image was determined entirely by the **aug_times** value in the DataFrame.  
No additional automatic filtering or preprocessing was applied beyond this.

---

## Data Quality Observations

A significant portion of the dataset is heavily artifacted.  
In particular, some images shifted toward purple tones—looking more like sea coral than actual leaves—and were biologically inconsistent.  
I separated these images **manually**.

**In total:**
- ~**9,000** images categorized as *edge-cases*
- ~**4,000** images assigned to the *chaos set*

This separation was **not based on any objective metric**.  
No RGB thresholding, texture-based analysis, anomaly scoring, or any automated classification method was used.

The filtering was entirely visual, following simple consistency rules such as:

- “This no longer resembles a real leaf.”
- “This level of distortion creates chaotic patterns.”

It’s possible that this manual curation unintentionally gives the model some hints; I can’t be certain.  
However, the intention was clear:

- Remove extremely distorted samples that look like in-game “rare items,”  
- Purple coral-like or otherwise unrecognizable artifacts,  
- And place them into the chaos set, outside the main training manifold.

The expected behavior is that the model should assign very low confidence to these samples or behave as if it is “unsure what this is,” which is the desired outcome.
