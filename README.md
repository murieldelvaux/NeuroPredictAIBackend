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

Create a `.env` file with your Neon connection string:

```bash
DATABASE_URL=postgresql+asyncpg://<user>:<password>@<host>/<db>?sslmode=require
```

If `DATABASE_URL` is not set, the backend falls back to a local SQLite database at `./neuropredict.db`.

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
│   ├── database.py      # Engine/session SQLAlchemy
│   ├── models.py        # ORM models
│   └── in_memory.py     # Repositório persistido (mantém a API antiga)
└── core/
    └── config.py        # Configurações via .env
```
