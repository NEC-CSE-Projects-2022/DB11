# Dataset (CASIA Image Forgery Dataset – CASIA 2.0)

<img width="1661" height="938" alt="Screenshot 2026-02-06 220243" src="https://github.com/user-attachments/assets/c7e26762-3753-4f65-a54d-a4cc236376a3" />

The *CASIA Image Forgery Dataset (Version 2.0)* is a widely used benchmark dataset for evaluating digital image forgery detection algorithms. It contains a large collection of authentic (untampered) and forged (tampered) images created using common manipulation techniques such as *copy-move* and *image splicing*. The dataset is extensively used in academic research for training and validating machine learning and deep learning–based image forensic systems.

- *Total images:* ~12,000+
- *Classes:* 2 (authentic, forged)
- *Forgery types:* Copy-move, Image splicing
- *Image type:* RGB images (.jpg, .png, .bmp)
- *Image resolution:* Variable
- *Official source:* Kaggle (recommended)

## Download Links

- *Kaggle (recommended):*
  https://www.kaggle.com/datasets/sophatvathana/casia-dataset

- *Project Mirror (Google Drive):*
  https://drive.google.com/drive/folders/1fU3qmWkglSR2Pl18ZYsYLXcROwQDuOi7?usp=drive_link

## Important

- Do not commit the dataset images to the project repository.
- Download the dataset locally and store it under the Datasets/ directory using the structure defined below.

## Kaggle Download (CLI Method)

1. Install and configure Kaggle API credentials:
  https://github.com/Kaggle/kaggle-api#api-credentials

2. Download the dataset:
  bash
  kaggle datasets download -d sophatvathana/casia-dataset
  

3. Unzip the dataset and organize the images according to the expected structure.

## Expected Dataset Structure

Create the following directory structure inside the project root:

text
Datasets/
  README.md
  train/
    authentic/
    forged/
  val/
    authentic/
    forged/
  test/
    authentic/
    forged/


This directory layout is compatible with *PyTorch torchvision.datasets.ImageFolder* and allows seamless integration into the training and evaluation pipeline.

## Class Labels

| Label | Description |
|------:|------------|
| authentic | Original, untampered images |
| forged | Manipulated images (copy-move or splicing) |

## Forgery Types Included

### Copy-Move Forgery
Regions copied and pasted within the same image to conceal or duplicate content.

### Image Splicing
Regions copied from one image and inserted into another image, often combined with post-processing operations.

## Usage Example (PyTorch)

python
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder("Datasets/train", transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)


## Notes

- The dataset contains class imbalance, with forged images generally outnumbering authentic ones.
- Ensure train, validation, and test splits are created carefully to avoid data leakage.
- Images may vary in resolution, compression level, and post-processing, making preprocessing essential.
- Error Level Analysis (ELA) is applied during preprocessing to highlight compression inconsistencies caused by forgery.

## Acknowledgements / License

- Dataset creators: CASIA (Institute of Automation, Chinese Academy of Sciences)
- Kaggle hosting & documentation:
  https://www.kaggle.com/datasets/sophatvathana/casia-dataset

This dataset is intended strictly for academic and research purposes.
Please refer to the official dataset source for licensing and citation requirements.
