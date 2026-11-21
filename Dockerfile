FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MODEL_CACHE_DIR=/app/.models

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Download the public model at build time, without any Telegram credentials.
ARG WHISPER_MODEL=tiny
RUN python -c "from faster_whisper.utils import download_model; download_model('${WHISPER_MODEL}', output_dir='/app/model')"
ENV WHISPER_MODEL=/app/model MODEL_OFFLINE=1
COPY bot.py speech.py ./
RUN useradd --create-home --uid 10001 bot && chown -R bot:bot /app
USER bot
EXPOSE 10000
CMD ["python", "bot.py"]
