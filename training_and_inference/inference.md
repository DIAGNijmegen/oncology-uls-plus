# nnUNet Inference with ULS+

Run inference on the ULS Challenge test set using the trained nnUNet model.

## Prerequisites

- nnUNet installed and configured
- Test data downloaded and organized
- Trained model checkpoint available

## Setup

### 1. Environment Variables

Set the required nnUNet environment variables:

```bash
export nnUNet_raw=<path/to/raw/data>
export nnUNet_results=<path/to/model/results>
```

> **Note:** If using the ULS23 baseline model (not ULS+), copy custom trainer and resampling files:
> ```bash
> cp -r nnunetv2/* <path/to/nnunet/installation>/nnunetv2/
> ```
> Example: `cp -r nnunetv2/* /root/nnunet/nnunetv2/`

## Run Inference

```bash
nnUNetv2_predict \
  -i <input/images/folder> \
  -o <output/predictions/folder> \
  -d Dataset090_ULS23_Combined \
  -p nnUNetResEncUNetLPlans \
  -c 3d_fullres \
  -tr nnUNetTrainer \
  -f all \
  -chk checkpoint_best.pth \
  --save_probabilities
```

**Parameters:**
- `-i`: Input folder containing test images
- `-o`: Output folder for predictions
- `--save_probabilities`: Saves probability maps alongside segmentation masks


