# Backend (FastAPI + DilatedSegNet)

This folder contains the API and model code only. **Weight files (`.pth`) are not included.**

- **Use the live API:** [Hugging Face Space](https://huggingface.co/spaces/HarshithReddy01/Polyp_Detection) (no setup).
- **Run locally:** Place `checkpoint-Kvasir-Seg.pth` and/or `checkpoint-BKAI-IGH.pth` in this folder. Download links: [Kvasir-SEG](https://drive.google.com/file/d/1diYckKDMqDWSDD6O5Jm6InCxWEkU0GJC/view?usp=sharing), [BKAI-IGH](https://drive.google.com/file/d/1ojGaQThD56mRhGQaVoJVpAw0oVwSzX8N/view?usp=sharing).

`pip install -r requirements.txt` then `uvicorn main:app --host 0.0.0.0 --port 7860`.
