import pandas as pd
import re
from pathlib import Path

old_prefix = "/kaggle/input/tomato/tomato_augmented/"
new_prefix = "/home/sinan/Masaüstü/augment_script/high_void_imgs_clean/"

df = pd.read_csv("df_augmented_metadata.csv")


df["image_path"] = df["image_path"].str.replace(
    old_prefix,
    new_prefix,
    regex=False
)


def clean_path(p):
    if not isinstance(p, str):
        return p
    p = p.strip()
    p = re.sub(r"\s+", " ", p)
    p = re.sub(r"^/+", "/", p)
    return p
df["image_path"] = df["image_path"].apply(clean_path)

df.to_csv("df_augmented_metadata_fixpaths.csv", index=False)
print("Done.")

