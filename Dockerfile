FROM node:22-alpine AS website
WORKDIR /build
COPY website/package*.json ./
RUN npm ci
COPY website/index.html website/build.mjs ./
RUN npm run build

FROM python:3.12-slim AS development
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp HF_HOME=/tmp/hf
WORKDIR /app
COPY pyproject.toml LICENSE ./
COPY src/ ./src/
COPY --from=website --chmod=0644 /build/dist/index.html ./src/agentmilieu/product/static/index.html
RUN pip install --no-cache-dir '.[product]' \
    && useradd --uid 10001 --no-create-home app \
    && mkdir -m 0755 /app/downloads
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=4 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)" || exit 1
CMD ["uvicorn", "agentmilieu.product.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--no-access-log", "--limit-concurrency", "256", "--timeout-graceful-shutdown", "90"]

FROM development AS release
COPY --chmod=0644 dist/windows/AgentMilieu-Setup.exe /app/downloads/AgentMilieu-Setup.exe
