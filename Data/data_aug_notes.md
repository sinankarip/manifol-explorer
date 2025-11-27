# Data Augmentation Process and Observations
[Data](https://drive.google.com/drive/folders/1SjurPH0S4M188chSMrSgYlVChUMjhS0G?usp=drive_link)
## Augmentation Application

Images marked as high void according to the script were augmented using scripts located in the `scripts/` folder. Augmentation was performed entirely based on the `aug_times` value in the dataframe.

## Data Quality Observations

The data is extremely artifacted - corrupted, purplish leaves that look more like sea coral than actual leaves were added to the chaos set. Even I'm asking "what are these?". The expectation is that the model will also say something like this and show a dramatic drop in confidence.

## Planned Approach

Will retrain with clean data. Since these are somewhat edge cases, I'm aiming to make the model assign less importance to such examples through different techniques:

- Flattening the manifold with topological loss
- Making changes to the model architecture
- Forcing certain aspects, for example applying a technique like Kinetic/Flow Regularization used in Neural ODEs
