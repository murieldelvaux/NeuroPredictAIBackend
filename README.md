# NeuroPredictAI — Backend

FastAPI backend for NeuroPredict AI. Exposes REST endpoints for patient management and Alzheimer's risk prediction using a 3D ResNet model trained on OASIS-3.

## Stack
- **FastAPI** + Uvicorn
- **PyTorch** + MONAI (inference)
- **SHAP** (explainability)
- **Pydantic v2** (schemas)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Docs disponíveis em: http://localhost:8000/docs

## Estrutura

```
app/
├── main.py              # FastAPI entry point
├── routers/
│   ├── patients.py      # GET/POST /patients
│   └── prediction.py    # POST /predict
├── schemas/
│   ├── patient.py       # Pydantic schemas
│   └── prediction.py
├── services/
│   ├── ai_service.py    # Carrega modelo .pth e roda inferência
│   └── explainability.py
├── db/
│   └── in_memory.py     # Store em memória (MVP)
└── core/
    └── config.py        # Configurações via .env
```
