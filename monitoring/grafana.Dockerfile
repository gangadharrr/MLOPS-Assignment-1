# Grafana with provisioning + dashboard baked in — no host bind mounts (see prometheus.Dockerfile).
FROM grafana/grafana:11.1.0
COPY grafana/provisioning /etc/grafana/provisioning
COPY grafana/dashboards /var/lib/grafana/dashboards
