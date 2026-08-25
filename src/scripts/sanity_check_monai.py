from __future__ import annotations

from src.data_prep.dataloader_monai import make_loader

MANIFEST = "data/unified_metadata.csv"

def main():
    for split in ["train", "val", "test"]:
        loader = make_loader(MANIFEST, split=split, batch_size=1, num_workers=0, cache=True)
        batch = next(iter(loader))

        img = batch["image"]    # Tensor [B, C, H, W, D]
        y = batch["label"]      # Tensor [B]

        print(f"\n== {split.upper()} ==")
        print("image shape:", tuple(img.shape))
        print("label:", int(y.item()))
        print("min/max:", float(img.min()), float(img.max()))

if __name__ == "__main__":
    main()