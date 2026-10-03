FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/opt/torch

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

RUN python -c "from torchvision.models import Wide_ResNet50_2_Weights as W; W.IMAGENET1K_V2.get_state_dict(progress=False)"

COPY mvtec_dataset.py patchcore.py ./
COPY api/ api/

RUN useradd --create-home appuser
USER appuser

ENV PYTHONPATH=/app \
    HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_DIR=/models

EXPOSE 8000
CMD ["python", "api/service.py"]

