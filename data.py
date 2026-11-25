from datasets import load_dataset
import albumentations as A
import numpy as np 

SEED=42

def transform_fn(example, pipeline):
  image = np.array(example["image"], dtype=np.uint8)
  map = np.array(example["depth_map"], dtype=np.float32)
  augmented = pipeline(image=image, mask=map)
  return {"image": augmented["image"].astype(np.float32), "map": augmented["mask"][..., None].astype(np.float32)}

train_pipeline = A.Compose([
    A.RandomCrop(height=224, width=224, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

eval_pipeline = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

train_ds = load_dataset("sayakpaul/nyu_depth_v2", split="train", streaming=True, revision="refs/convert/parquet", token=False)
val_ds = load_dataset("sayakpaul/nyu_depth_v2", split="validation", streaming=True, revision="refs/convert/parquet", token=False)

val_ds = val_ds.map(lambda x: transform_fn(x, eval_pipeline))