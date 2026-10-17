# Prometheus with config baked in — no host bind mounts, so `docker compose up` works
# from any checkout path (bind-mount specs break when the host path contains ":", e.g. "BITS AI:ML").
FROM prom/prometheus:v2.53.0
COPY prometheus.yml /etc/prometheus/prometheus.yml
