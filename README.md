# ULS+ (Universal Lesion Segmentation Plus)

ULS+ is an improved universal lesion segmentation model for CT scans, extending the ULS Challenge baseline with enhanced preprocessing, training, and evaluation pipelines.

## 📦 Resources & Downloads

### Dataset
- **[Hugging Face Dataset](https://huggingface.co/datasets/nielsRocholl/ULS_plus/)** - Preprocessed ULS+ datasets

### Model Weights
- **[Hugging Face Model](https://huggingface.co/nielsRocholl/ULS_plus/tree/main)** - Pre-trained ULS+ model weights
- **[Zenodo Model Weights](https://zenodo.org/records/17937197)** - Model weights archive (DOI: 10.5281/zenodo.17937197)

Each dataset in `archives/` has been preprocessed with our [preprocessing pipeline](preprocessing/README.md) located in this repository.

## Overview

This repository contains:
- **Preprocessing pipeline**: Universal CT Lesion Preprocessor (UCLP) for converting diverse CT datasets into nnU-Net compliant format
- **Training pipeline**: Custom nnU-Net v2 training with ResEnc-L architecture and no-resampling plans
- **Evaluation tools**: Comprehensive metrics computation and visualization for lesion segmentation

## Quick Start

Choose one of two paths depending on your needs:

### Option A: Using Preprocessed Data (Quick Path)

If you want to run inference immediately using our preprocessed data:

1. **Download Preprocessed Data**
   - Download from [Hugging Face Dataset](https://huggingface.co/datasets/nielsRocholl/ULS_plus/)
   - Data is already preprocessed with UCLP and ready to use

2. **Download Model Weights**
   - Download from [Hugging Face Model](https://huggingface.co/nielsRocholl/ULS_plus/tree/main) (recommended)
   - Or from [Zenodo](https://zenodo.org/records/17937197) (DOI: 10.5281/zenodo.17937197)

3. **Run Inference**
   - Follow the [inference guide](training_and_inference/inference.md) to generate predictions

4. **Evaluate Results**
   - Run the [evaluation script](evaluation/README.md) to compute metrics

### Option B: From Scratch (Full Pipeline)

If you want to reproduce the entire pipeline from raw data:

1. **Download Original Data Sources**
   - Download original datasets from sources linked in the paper and on [Hugging Face](https://huggingface.co/datasets/nielsRocholl/ULS_plus/)

2. **Preprocess Data with UCLP**
   - Use the [Universal CT Lesion Preprocessor (UCLP)](preprocessing/README.md) to convert raw CT data into nnU-Net format:
   ```bash
   python preprocessing/uclp/uclp_preprocess.py --config preprocessing/configs/your_dataset.yaml
   ```
   - See [preprocessing/README.md](preprocessing/README.md) for configuration details and available adapters

3. **Train the Model**
   - Follow the [training instructions](training_and_inference/preprocess_and_train.md) to train ULS+ with nnU-Net v2
   - Uses ResEnc-L architecture with custom no-resampling plans

4. **Run Inference**
   - Follow the [inference guide](training_and_inference/inference.md) to generate predictions

5. **Evaluate Results**
   - Run the [evaluation script](evaluation/README.md) to compute metrics and generate visualizations:
   ```bash
   python evaluation/eval_uls.py \
     --dataset-root <path/to/test/dataset> \
     --preds <path/to/predictions> \
     --out <path/to/metrics.csv> \
     --workers 12
   ```

## Repository Structure

```
oncology-uls-plus-clean/
├── preprocessing/          # Universal CT Lesion Preprocessor (UCLP)
│   ├── uclp/              # Core preprocessing modules
│   ├── configs/           # Dataset configuration files
│   ├── README.md          # Preprocessing documentation
│   └── ADAPTERS.md        # Dataset adapter guide
├── training_and_inference/
│   ├── preprocess_and_train.md  # Training instructions
│   └── inference.md             # Inference guide
├── evaluation/            # Evaluation tools
│   ├── eval_uls.py        # Metrics computation
│   ├── plot_uls_metrics.py      # Single dataset plots
│   ├── plot_uls_combined.py    # Comparison plots
│   └── README.md          # Evaluation documentation
└── nnunetv2/              # Custom nnU-Net modifications
    ├── preprocessing/     # Custom resampling functions
    └── training/          # Custom trainers
```

## Key Features

### Preprocessing
- **Multi-dataset support**: Handles diverse CT dataset formats (MSD, WORC, MSWAL, etc.)
- **Flexible adapters**: Customizable adapters for different folder structures
- **Lesion-centered cropping**: Extracts VOIs centered on lesions with augmentation
- **nnU-Net compliant**: Outputs ready-to-use nnU-Net v2 datasets

### Training
- **ResEnc-L architecture**: Large residual encoder U-Net for high-capacity learning
- **No-resampling plans**: Preserves original voxel spacing to avoid interpolation artifacts
- **Multi-dataset training**: Combines multiple lesion datasets for robust generalization

### Evaluation
- **Comprehensive metrics**: Dice score, Boundary IoU, and agreement metrics
- **Per-lesion-type analysis**: Breakdown by lesion type (lung, liver, kidney, etc.)
- **Robustness evaluation**: Agreement metrics across augmented predictions
- **Visualization**: Publication-ready plots comparing models

## Documentation

- [Preprocessing Guide](preprocessing/README.md) - UCLP usage and configuration
- [Dataset Adapters](preprocessing/ADAPTERS.md) - Creating custom dataset adapters
- [Training Instructions](training_and_inference/preprocess_and_train.md) - Model training workflow
- [Inference Guide](training_and_inference/inference.md) - Running inference
- [Evaluation Guide](evaluation/README.md) - Metrics and visualization

## Model Weights

Pre-trained model weights are available on:
- **[Hugging Face](https://huggingface.co/nielsRocholl/ULS_plus/tree/main)** (recommended)
- **[Zenodo](https://zenodo.org/records/17937197)** (DOI: 10.5281/zenodo.17937197)

## Citation

If you use ULS+ in your research, please cite the arXiv preprint:
```
@article{weber2026ulsplus,
  title   = {ULS+: Data-driven Model Adaptation Enhances Lesion Segmentation},
  author  = {Weber, Rianne and Rocholl, Niels and de Grauw, Max and Prokop, Mathias and Smit, Ewoud and Hering, Alessa},
  journal = {arXiv preprint arXiv:2601.02988},
  year    = {2026},
  note    = {Accepted at BVM 2026},
}

```
Preprint: https://arxiv.org/abs/2601.02988

## License

See [LICENSE](LICENSE) for details.
