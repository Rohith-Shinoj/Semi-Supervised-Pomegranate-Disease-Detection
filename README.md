# Semi-Supervised EfficientNet-ViT Pipeline for Crop Disease Detection

Source code for "Transformer-Based Multimodal Pomegranate Disease Detection (ICAR-affiliated)" as part of a joint project by NITK and the Indian Council for Agricultural Research (ICAR).

This project implements an end-to-end pipeline for pomegranate crop disease detection using a semi-supervised learning approach with a hybrid EfficientNet and Vision Transformer (ViT) model. It includes data preprocessing with strong and weak augmentations, background removal, FixMatch-style SSL training, Grad-CAM visualization for model explainability, and ONNX export for deployment. For full implementation details and explanations, visit my blog post at https://www.deeper-thoughts-blog.rohithshinoj.com/blog/deploying-ai-in-agriculture.


## Features

- **Preprocessing:** Strong and weak augmentations, segmentation-based background removal.
- **Model:** EfficientNet backbone combined with a Transformer encoder.
- **Training:** Semi-supervised learning using FixMatch method for leveraging unlabeled data.
- **Explainability:** Grad-CAM visualizations to inspect model attention.
- **Deployment:** Export trained model to ONNX with quantization for cross-platform inference.

## Setup

1. Clone the repository.

2. Create and activate a Python environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```
3. Train the semi-supervised model by running:

```bash
python train_ssl.py
```

Make sure your labeled and unlabeled datasets are organized under dataset/train_labeled/ and dataset/train_unlabeled/.

4. Generate Grad-CAM visualizations to understand model predictions:

```bash
python gradcam.py
```
Update image paths inside gradcam.py accordingly.

5. Export the trained PyTorch model to ONNX format for deployment:

```bash
python export_onnx.py
```

## Notes

- Modify augmentation parameters in utils.py to suit your dataset.

- Adjust model parameters or transformer layers in models/efficientvit.py if needed.

If you found it interesting, do check out other posts at https://www.deeper-thoughts-blog.rohithshinoj.com where I write regularly to articulate my thoughts, projects and recent advances in the field of AI.
