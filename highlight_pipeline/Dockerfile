# Optional. CPU-friendly default. Build:  docker build -t highlights .
# GPU (UNVERIFIED): docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 -t highlights-gpu .
#                   docker run --gpus all ...   (needs the NVIDIA Container Toolkit on the host)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements.txt requirements-app.txt ./
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision --index-url ${TORCH_INDEX} \
 && pip install -r requirements.txt -r requirements-app.txt

COPY --chown=app:app . .
USER app
EXPOSE 8501

# The API key is NEVER baked into the image: pass it at run time (-e GEMINI_API_KEY).
# Default: the CLI.  Dashboard: docker run -p 8501:8501 -e GEMINI_API_KEY --entrypoint streamlit highlights run app.py --server.address 0.0.0.0
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
