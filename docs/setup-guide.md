# Setup Guide

Документ описывает, как локально поднять окружение из Phase 0.

## 1. Предпосылки
- Docker 24+
- Docker Compose v2
- Kind (или Minikube) + kubectl 1.27+
- Python 3.11
- Poetry 1.8+
- pre-commit (рекомендовано)

## 2. Python окружение
```bash
poetry install
poetry run pre-commit install
poetry run pytest
```

## 3. Docker Compose (Prometheus + Grafana + demo services)
```bash
cd docker
docker compose up --build -d
```
Серверы:
- Demo сервисы: http://localhost:8001..8003
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Load generator: сервис `load-generator` стучится в demo-service-* по умолчанию

## 4. Kubernetes (kind)
```bash
kind create cluster --config k8s/kind-config.yaml --name autoscaling
kubectl apply -f k8s/manifests/namespace.yaml
kubectl apply -f k8s/manifests/demo-service-deployment.yaml
kubectl apply -f k8s/manifests/demo-service-service.yaml
kubectl apply -f k8s/manifests/prometheus/ -f k8s/manifests/hpa/
```
Проверьте, что Prometheus и demo-service в статусе `Running` и HPA видит метрики CPU.

## 5. Grafana Dashboard
Compose автоматически применяет provisioning. Для Kubernetes потребуется порт-форвардинг:
```bash
kubectl port-forward svc/prometheus -n predictive-autoscaling 9090:9090
# (опционально) Grafana если развёрнута в кластере
```

## 6. Структура данных
```
data/
  raw/        # выгрузки из Prometheus
  processed/  # обработанные выборки
```
`data/.gitkeep` предотвращают очистку директорий. Никогда не коммитье реальные данные.

## 7. Сбор исторических данных
- Отредактируйте `src/k8s_ml_predictive_autoscaling/collector/config.yaml`.
- Выполните:
  ```bash
  poetry run python -m k8s_ml_predictive_autoscaling.collector.collect_historical
  ```
- CSV с метриками появятся в `data/raw/*_YYYYMMDD.csv`.

## 8. Генерация синтетической нагрузки
- Профили нагрузки формируются в `tools/load_generator/synthetic_patterns.py`.
- Запуск Locust:
  ```bash
  poetry run locust -f tools/load_generator/locust_tasks.py --host http://localhost:8001
  ```
- Запуск k6:
  ```bash
  k6 run tools/load_generator/k6_script.js
  ```

## 9. Препроцессинг и EDA
- Настройте `src/k8s_ml_predictive_autoscaling/preprocessor/config.yaml`.
- Запустите пайплайн:
  ```bash
  poetry run python -m k8s_ml_predictive_autoscaling.preprocessor.pipeline
  ```
- Используйте `notebooks/research-data.ipynb` + `docs/eda-report.md` для анализа результатов.

## 10. Секреты
- Используйте `.env` для локальных переменных (`AUTOSCALER_*`).
- `.env` игнорируется Git (см. `.gitignore`).

## 11. Troubleshooting
- Если `poetry lock` падает из‑за отсутствия сети, выполните команду после подключения к интернету.
- Для Windows WSL убедитесь, что Docker доступен из WSL.
- `pre-commit run --all-files` помогает выявить проблемы до CI.
