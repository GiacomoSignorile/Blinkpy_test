FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HOME=/tmp YOLO_CONFIG_DIR=/tmp/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libxcb1 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
RUN python -c "import timm; timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)"
COPY blinkwatch ./blinkwatch
CMD ["python", "-m", "blinkwatch.watcher"]
