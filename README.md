# Polyp Detection

AI-powered polyp segmentation from colonoscopy images using **DilatedSegNet** (RUPNet): encoder-decoder with ResNet50, dilated convolution pooling, and two model variants (Kvasir-SEG and BKAI-IGH).

**Live app:** [Frontend](https://harshithreddy01.github.io/Polyp-Frontend/)  
**API:** [Hugging Face Space](https://huggingface.co/spaces/HarshithReddy01/Polyp_Detection)

**Metrics (reported in training):** Dice coefficient **0.90**, mIoU **0.83**, ~33.68 FPS on GPU.

---

- **Frontend:** React (this repo root) — upload image, choose model, view mask and overlay.
- **Backend:** FastAPI + PyTorch in `/backend` (no weights in repo; see [backend README](backend/README.md) for local run or use the HF link above).

**Contact:** [Harshith Reddy Nalla](https://harshithreddy01.github.io/My-Web/)
