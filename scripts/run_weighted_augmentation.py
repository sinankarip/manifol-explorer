import pandas as pd
from pathlib import Path
from leaf_2_5d_augment import generate_augmented_views

BASE_DIR = Path("/home/sinan/Masaüstü/augment_script/high_void_imgs_clean")

BASE_OUT_DIR = Path("augmented_leafs")
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("df_augmented_metadata_fixpaths.csv")

print("Total samples:", len(df))
print("Total augmentations to generate:", int(df["aug_times"].sum()))

for idx, row in df.iterrows():
    num_views = int(row["aug_times"])
    if num_views <= 0:
        continue

    img_path = Path(str(row["image_path"]).strip())

    if not img_path.is_file():
        print(f"[WARN] Skipping, file does not exist: {img_path}")
        continue

    try:
        rel_path = img_path.relative_to(BASE_DIR)
        cls = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
    except ValueError:
        cls = img_path.parent.name

    out_dir = BASE_OUT_DIR / cls
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{idx}] {img_path} -> {out_dir} | num_views={num_views}")

    generate_augmented_views(
        leaf_path=img_path,
        out_dir=out_dir,
        num_views=num_views,
        out_size=224,
    )



