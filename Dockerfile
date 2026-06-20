FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; scipy/numpy wheels are prebuilt for slim.
COPY requirements.txt .
# Demo runs on deterministic local embeddings by default, so torch +
# sentence-transformers are optional. Install the light set for a fast,
# reproducible judge run; uncomment the full install for real embeddings.
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pydantic numpy scipy scikit-learn requests

COPY . .

EXPOSE 8000

# Local deterministic embeddings + localhost CORS by default.
ENV RAVEN_ALLOWED_ORIGINS="http://localhost:8000"

# Serves the bilingual landing page at / and the JSON API at /api, /recall, …
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
