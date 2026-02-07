# Команды для запуска (копируйте по одной)

## 1. Установка токена

```bash
export AUTOSCALER_API_TOKEN="my-super-secret-token-12345"
export GF_SECURITY_ADMIN_PASSWORD="grafana-admin-password"
```

## 2. Проверка токена

```bash
echo "Token set: $AUTOSCALER_API_TOKEN"
```

## 3. Переход в директорию docker

```bash
cd docker
```

## 4. Остановка старых контейнеров (если есть)

```bash
docker compose down -v
```

## 5. Запуск окружения

```bash
docker compose up --build -d
```

## 6. Ожидание запуска (30 сек)

```bash
sleep 30
```

## 7. Проверка статуса

```bash
docker compose ps
```

## 8. Проверка логов load-generator

```bash
docker compose logs load-generator | tail -20
```

## 9. Проверка доступности Prometheus

```bash
curl -s http://localhost:9090/-/healthy
```

## 10. Проверка метрик demo-service

```bash
curl -s http://localhost:8001/metrics | grep demo_service
```

---

## После 48 часов работы:

```bash
# Вернитесь в корень проекта
cd ..

# Соберите данные
poetry run python -m k8s_ml_predictive_autoscaling.collector.collect_historical \
  --config src/k8s_ml_predictive_autoscaling/collector/config.yaml

# Проверьте собранные файлы
ls -lh data/raw/

# Запустите препроцессинг
poetry run python -m k8s_ml_predictive_autoscaling.preprocessor.pipeline \
  --config src/k8s_ml_predictive_autoscaling/preprocessor/config.yaml

# Проверьте обработанные данные
ls -lh data/processed/
```

---

## Мониторинг в процессе работы:

```bash
# Логи в реальном времени
docker compose logs -f

# Только load-generator
docker compose logs -f load-generator

# Статус контейнеров
docker compose ps

# Перезапуск если что-то упало
docker compose restart

# Полная остановка (только когда закончите!)
docker compose down
```
