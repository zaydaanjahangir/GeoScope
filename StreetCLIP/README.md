# StreetCLIP

A PyTorch implementation of StreetCLIP for image geolocalization using CLIP.

## Overview

StreetCLIP is a model for image geolocalization that leverages OpenAI's CLIP model and synthetic captions to perform zero-shot location prediction. The model uses a hierarchical approach to predict both country and city-level locations.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/StreetCLIP.git
cd StreetCLIP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Prepare your dataset in CSV format with the following columns:
   - `image_filename`: Name of the image file
   - `city`: City name
   - `region`: Region/state/province name
   - `country`: Country name
   - `continent`: Continent name

2. Organize your images in a directory structure:
```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── train.csv
├── val.csv
└── test.csv
```

## Usage

### Generate Synthetic Captions

```bash
python main.py --mode generate_captions \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --test_csv data/test.csv \
    --output_dir outputs/captions
```

### Train the Model

```bash
python main.py --mode train \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --image_dir data/images \
    --output_dir outputs/training \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1e-6 \
    --weight_decay 1e-4
```

### Evaluate the Model

```bash
python main.py --mode evaluate \
    --test_csv data/test.csv \
    --image_dir data/images \
    --checkpoint_path outputs/training/best_model.pt \
    --candidates_file data/candidates.json \
    --output_dir outputs/evaluation
```

## Model Architecture

StreetCLIP consists of the following components:

1. **CLIP Model**: Pretrained CLIP model for image and text encoding
2. **Synthetic Caption Generation**: Template-based caption generation for locations
3. **Domain-Specific Pretraining**: Adaptation of CLIP for geolocalization
4. **Hierarchical Prediction**: Two-stage prediction (country then city)

## Results

The model's performance is evaluated using:
- Country-level accuracy
- City-level accuracy
- Hierarchical accuracy (correct city given correct country)

## Citation

If you use this code, please cite the original StreetCLIP paper:

```
@article{streetclip,
  title={StreetCLIP: Learning to Geolocate Street Images},
  author={...},
  journal={...},
  year={...}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 