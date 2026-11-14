# Retail Vision

## Run

```bash
cd frontend
npm install
npm run build

cd ../backend
python3 -m venv ../.venv
../.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Set the operator password and ingest token in `backend/.env`. Then:

```bash
set -a
source .env
set +a
PYTHONPATH=. ../.venv/bin/uvicorn main:app --port 5173
```

Open http://localhost:5173 and sign in with the user and password from `.env`.
