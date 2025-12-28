# DR Backend (Flask API)

This backend exposes a DR classification + Grad-CAM API.

## Folder layout (important)
- `app.py`               : Flask API (health, model-info, predict)
- `wsgi.py`              : Gunicorn entrypoint
- `model/`               : `.keras` models (baseline + finetune)
- `utils/`               : audit logging, anonymization, RBAC/JWT, rate limiting
- `static/Uploads/`      : stored originals + Grad-CAM outputs

Outputs are saved to:
- `static/Uploads/Originals/`
- `static/Uploads/GradCAM_Results/`

## Local run (Windows / VS Code)
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

API:
- GET  `/api/health`
- GET  `/api/model-info`
- POST `/api/predict`  (multipart/form-data, key = `file`)

Example:
```bash
curl -X POST http://127.0.0.1:5000/api/predict ^
  -F "file=@test.jpg" ^
  -F "patient_id=12345"
```

## Deploy (Render)
Start command (either use Procfile or Render Start Command):
```bash
gunicorn wsgi:app --preload --timeout 120
```

Recommended env vars:
- `MODEL_VARIANT=finetune`  (or `baseline`)
- `MODEL_PATH=/opt/render/project/src/backend/model/best_ra_finetune_export.keras` (optional)
- `ALLOWED_ORIGINS=https://your-frontend-domain`
- `SECRET_KEY=...`
- `MAX_UPLOAD_MB=16`
