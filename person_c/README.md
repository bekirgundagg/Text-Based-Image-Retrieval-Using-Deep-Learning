# Person C: fine-grained text-to-image retrieval

## Current milestone

A **CUB validation pilot**, not the completed final project. Frozen ImageNet-pretrained
ConvNeXt-Tiny and pretrained BERT-base produce features. Two trainable 256-dimensional
linear projections are optimized with symmetric image/text cross-entropy (temperature
0.07). No BLIP or Flowers experiment is claimed by this milestone.

## Run on Colab (GPU)

Open `colab_pilot.ipynb`, select a GPU runtime, and run its cells. Images and model
weights download to the Colab VM, not the Mac. Keep output backups before disconnecting.
The supplied notebook downloads a fresh copy of this repository into `/content/retrieval_project`.

The equivalent commands from this directory are:

```bash
pip install -r requirements.txt
python prepare_data.py
python pilot.py --train-images 256 --val-images 64 --epochs 30
```

Use `--train-images 0 --val-images 0` to use all training/validation records; this
still trains projection heads only and is not encoder fine-tuning. Do not launch
full experiments until the group agrees on the protocol.

## Data and evaluation protocol

- Images: CUB-200-2011, 11,788 images. The official archive MD5 is verified.
- Captions: Reed et al. descriptions from their original CVPR 2016 archive.
  Only plain-text caption files are extracted; no pickle files are loaded.
- Official CUB test images are excluded from this pilot entirely.
- Within the official training images, reserve approximately 10% per class for
  validation using seed 42. All captions of a given image stay together.
- The 256/64 pilot subsets are deterministic shuffled subsets of these splits.
- This is an image-disjoint split with shared classes, **not** the Reed class-disjoint
  zero-shot protocol. Do not compare scores directly with that benchmark.
- Validation queries include every available caption for each selected validation
  image. The gallery contains each selected image once. A hit means retrieving
  the exact image paired with the query, not another image of the same species.
- R@1/5/10 values are fractions (multiply by 100 for percentages). Gallery size
  must accompany results; 64-image pilot scores do not measure full-test performance.
- Each training batch has unique images and samples one caption per image, avoiding
  treating another caption of that same image as a negative.
- Best checkpoint is chosen by validation R@1; ties retain the earliest epoch.

## Outputs

`/content/person_c_run/` contains `results.json`, `manifest.json`, cached features,
`best.pt`, and `last.pt`. Projection checkpoints include optimizer state and epoch;
a resume CLI is not yet implemented. Frozen encoders can be reloaded from the named
pretrained sources. Feature caches are valid only for the exact manifest and preprocessing.
Do not commit datasets, features, or weights to GitHub or the final submission.

Timing separates feature extraction from projection training. It excludes setup,
downloads, and model loading. Frozen-head timings cannot predict full BLIP fine-tuning.

## Next milestones

1. Agree on shared data/model/metric interfaces with teammates.
2. Run full CUB validation and selectively fine-tune encoder blocks.
3. Prepare Oxford-102 + real Reed descriptions with an explicit split.
4. Train/evaluate both models on both datasets; add BLIP ITC and ITM reranking.
5. Final test evaluation, retrieval demo, combined results, presentation.

## Limitations and attribution

CUB's official page warns of image overlap with ImageNet. ConvNeXt uses ImageNet
pretraining; disclose this possible pretraining contamination. This pilot is not
an uncontaminated zero-shot evaluation. BERT captions are English and truncated to
64 tokens. Repeated-seed uncertainty estimates are not yet available.

- CUB: https://www.vision.caltech.edu/datasets/cub_200_2011/
- Official archive: https://data.caltech.edu/records/65de6-vp158
- Reed et al., Learning Deep Representations of Fine-Grained Visual Descriptions,
  CVPR 2016: https://github.com/reedscot/cvpr2016
- Caption archive: https://drive.google.com/file/d/0B0ywwgffWnLLZW9uVHNjb2JmNlE/view
- ConvNeXt: https://pytorch.org/vision/stable/models/convnext.html
- BERT: https://huggingface.co/google-bert/bert-base-uncased

Respect the source datasets' research/education usage terms. No data is redistributed here.
