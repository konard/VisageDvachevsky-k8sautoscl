# K8s ML Predictive Autoscaling

Исследовательский и инженерный проект по **ML-основанному предиктивному автомасштабированию** микросервисов в Kubernetes.

Цель — перейти от классического **реактивного масштабирования (HPA по CPU/памяти)** к **проактивному планированию ресурсов на основе прогноза нагрузки**, чтобы:

* удерживать латентность и SLO под контролем;
* снижать инфраструктурные затраты;
* получить воспроизводимую экспериментальную установку для научной работы (НИР, курсовая, ВКР, публикация).

> Кратко: здесь исследуется, насколько модели Prophet / LSTM / гибрид Prophet+LSTM могут улучшить работу Kubernetes-автомасштабирования по сравнению с обычным HPA.

---

## 1. Мотивация и постановка задачи

Современные системы автомасштабирования (например, Kubernetes Horizontal Pod Autoscaler) в основном используют **реактивный подход**: решение о масштабировании принимается по текущим метрикам CPU/памяти, часто с задержкой 30–60 секунд. Это приводит к:

* временной деградации производительности при всплесках нагрузки;
* росту латентности и нарушению SLA/SLO;
* эффекту «осцилляций» (частое масштабирование вверх/вниз);
* неоптимальному использованию ресурсов и лишним затратам.

**Предиктивное масштабирование** использует методы машинного обучения для прогноза нагрузки на горизонте нескольких минут и проактивного изменения количества реплик и лимитов ресурсов до наступления пиков.

В рамках проекта рассматриваются и сравниваются подходы:

* Реактивный HPA (baseline).
* Прогнозирование временных рядов: LSTM / GRU.
* Prophet (модель с учётом тренда и сезонности).
* Гибрид Prophet + LSTM.
* (Опционально) RL-подходы к автомасштабированию.

---

## 2. Цели и задачи проекта

### Цель

Разработать и исследовать систему предиктивного автомасштабирования микросервисов в Kubernetes на основе ML-прогноза нагрузки и сравнить её с классическим HPA по метрикам качества, производительности и стоимости.

### Основные задачи

1. Собрать и подготовить временные ряды нагрузки из Prometheus (CPU, память, RPS, latency, ошибки и т.д.).
2. Обучить и сравнить модели Prophet, LSTM/GRU и гибрид Prophet+LSTM по точности прогноза.
3. Реализовать сервис онлайн-прогноза нагрузки (FastAPI + ONNX Runtime).
4. Разработать модуль планирования ресурсов (Resource Planner), который переводит прогноз в:
   * количество реплик;
   * CPU/memory requests & limits;
   * политики ramp-up / ramp-down.
5. Интегрировать ML-прогнозы с Kubernetes через:
   * HPA external/custom metrics;
   * KEDA (event-driven autoscaling).
6. Построить Grafana-дашборд для визуализации:
   * реальной нагрузки vs. прогноз;
   * решений по масштабированию;
   * экономии ресурсов и SLO-нарушений.
7. Провести серию экспериментов с нагрузочным тестированием и формализовать результаты для научной публикации.

---

## 3. Общая архитектура

Система следует паттерну **MAPE loop**: **Monitor → Analyze → Plan → Execute**.

### 1. Monitor

* Микросервисы экспонируют `/metrics`.
* Prometheus собирает:
  * CPU, память;
  * network I/O;
  * request rate;
  * latency (p50, p95, p99);
  * error rate, custom-метрики приложения.
* Данные сохраняются как временные ряды и экспортируются для обучения/инференса.

### 2. Analyze (ML-слой)

* Препроцессинг данных:
  * агрегация по фиксированному шагу (10–60 секунд);
  * фильтрация аномалий (DDoS, synthetic load, ошибки мониторинга);
  * sliding window генерация;
  * опционально — декомпозиция временных рядов (например, CEEMDAN).
* Модели прогнозирования:
  * LSTM / GRU для нелинейных паттернов;
  * Prophet для трендов и сезонности;
  * гибрид Prophet+LSTM (Prophet моделирует тренд/сезонность, LSTM — остатки).
* Выход: прогноз нагрузки на горизонте 5–30 минут + доверительные интервалы.

### 3. Plan (Resource Planner)

* Конвертирует прогноз в план ресурсов:
  * целевое количество реплик;
  * лимиты CPU/памяти;
  * стратегия ramp-up / ramp-down;
  * небольшой over-provisioning (например, 10–15%) для надёжности.
* Учитывает бизнес-ограничения:
  * минимальное/максимальное число реплик;
  * SLO по латентности;
  * бюджет ресурсов.

### 4. Execute (интеграция с Kubernetes)

* ML-прогнозы экспортируются как **external/custom metrics** для HPA либо как источник для KEDA.
* Через Kubernetes API обновляются:
  * объекты `HorizontalPodAutoscaler`;
  * `ScaledObject` (для KEDA) или кастомные CRD.
* Реализован fallback:
  * при деградации качества модели или недоступности ML-сервиса система откатывается на классический HPA по текущим метрикам.

Архитектурная схема (high-level диаграмма) расположена в `docs/architecture-diagram.png`.

---

## 4. Сравнение подходов к автомасштабированию

В проекте используются и сравниваются следующие подходы (подробная таблица в `docs/model-comparison-table.png`):

* **Реактивный HPA** — baseline, низкая сложность, минимальные вычислительные затраты, но нет прогноза.
* **Прогнозирование с LSTM/GRU** — проактивное масштабирование, средне-высокая точность, средние вычислительные затраты.
* **Прогнозирование с Prophet** — отлично работает для сезонных данных, низкие вычислительные затраты.
* **Гибрид Prophet+LSTM** — наивысшая точность (улучшение ≈6–15% по RMSE).
* **Reinforcement Learning** — потенциально максимальная гибкость, но очень высокая сложность и вычислительная стоимость (опциональный этап).

---

## 5. Стек технологий

Планируемый стек (может уточняться по ходу работы):

### Язык / ML

* Python 3.11+
* PyTorch или TensorFlow/Keras (для LSTM/GRU)
* Facebook Prophet / NeuralProphet
* ONNX / ONNX Runtime (оптимизированный inference)

### Инфраструктура и оркестрация

* Kubernetes (kind / minikube для локальных экспериментов)
* Kubernetes HPA / KEDA
* Prometheus (сбор метрик)
* Grafana (дашборды)
* (опционально) k6 / Locust для нагрузочного тестирования

### Сервисный слой

* FastAPI (REST API для инференса и планировщика)
* Python Kubernetes client (работа с K8s API)
* Docker / Docker Compose для локального поднятия окружения

---

## 6. Оптимизация производительности: сценарии использования C++

По умолчанию проект реализуется на Python для скорости разработки и прототипирования. Однако для production-окружений с высокими требованиями к производительности рассматриваются следующие сценарии использования C++:

### 6.1. Inference-движок для ML-модели

Наиболее критичный по производительности компонент — выполнение инференса ONNX-модели с латентностью <5–10 мс на одно предсказание.

**Преимущества C++ для inference:**

* ONNX Runtime реализован на C++ (Python bindings — лишь обёртка над нативной библиотекой)
* Минимальная латентность благодаря отсутствию overhead интерпретатора Python
* Zero-copy операции с входными тензорами
* Встроенный thread pool для параллельной обработки запросов
* Оптимальное решение для high-throughput сценариев (>1000 предсказаний/сек)

**Результат:** ускорение в 2–10 раз по сравнению с Python, стабильная латентность, низкое потребление CPU.

Для research-прототипа достаточно Python + ONNX Runtime. Для production-grade системы предиктивного автомасштабирования inference на Python может стать bottleneck — в этом случае рекомендуется реализация на C++.

### 6.2. Resource Planner с жёсткими требованиями к latency

Планировщик ресурсов (Resource Planner) обычно выполняет решения с частотой 1 раз в минуту, что позволяет использовать Python без ограничений.

**Сценарии, где C++ даёт преимущество:**

* Высокочастотные решения по масштабированию (несколько решений в секунду)
* Обработка прогнозов в режиме реального времени с гарантированной латентностью на уровне миллисекунд
* Развёртывание планировщика в embedded-средах или вне Python-экосистемы
* Интеграция с low-level системными компонентами Kubernetes

Для большинства практических задач Python-реализация планировщика достаточна. C++ становится необходим только при переходе к real-time системам принятия решений.

### 6.3. Kubernetes Operator на базе Custom Resource Definition (CRD)

Перспективное направление для исследовательской работы — реализация полноценного Kubernetes-оператора для предиктивного автомасштабирования в виде CRD-контроллера на C++.

**Пример:**

```yaml
apiVersion: autoscaler.ml.dev/v1alpha1
kind: PredictiveAutoscaler
metadata:
  name: ml-based-scaler
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: demo-service
  predictionHorizon: 10m
  model: prophet-lstm-hybrid
```

**Преимущества подхода:**

* Минимальная латентность взаимодействия с Kubernetes API
* Zero-overhead за счёт компиляции в нативный код
* Возможность интеграции с существующими C++-компонентами кластера
* Уникальность реализации (мало open-source проектов используют C++ для ML-операторов в K8s)
* Высокая научная ценность для публикаций

Реализация может базироваться на библиотеках [kubernetes-client/c](https://github.com/kubernetes-client/c) или [client-cpp](https://github.com/kubernetes-client/cpp). Данный подход рекомендуется для углублённой научной работы (магистерская диссертация, публикация в Scopus/WoS).

**Выбор реализации:**

* **Python** — для MVP, исследовательских экспериментов, proof-of-concept
* **C++** — для production-систем с требованиями high-throughput (inference >1000 RPS), ultra-low latency (<5 мс) или реализации кастомных K8s-операторов

---

## 7. Структура репозитория

Планируемая структура (может слегка меняться по мере развития):

```
models/
  lstm/         — обучение и чекпоинты LSTM/GRU
  prophet/      — конфигурации и скрипты Prophet
  hybrid/       — реализация гибридной модели Prophet+LSTM
  onnx/         — экспортированные модели для быстрого инференса

data/
  raw/          — сырые выгрузки из Prometheus
  processed/    — очищенные и подготовленные временные ряды

src/
  collector/    — клиент Prometheus, загрузка и сохранение данных
  preprocessor/ — препроцессинг: агрегация, аномалии, sliding window, декомпозиция
  predictor/    — сервис инференса (FastAPI + ONNX Runtime)
  planner/      — логика перевода прогноза в ресурсы (реплики, лимиты)
  executor/     — интеграция с Kubernetes API / HPA / KEDA

k8s/
  manifests/    — демо-микросервисы, Prometheus, Grafana и т.д.
  hpa/          — конфигурация HPA c external/custom metrics
  keda/         — ScaledObject и триггеры KEDA

dashboard/
  grafana.json          — основной Grafana-дашборд
  prometheus-rules.yml  — алерты и правила

notebooks/
  research-data.ipynb       — EDA и работа с временными рядами
  research-lstm.ipynb       — эксперименты с LSTM/GRU
  research-prophet.ipynb    — эксперименты с Prophet
  comparison.ipynb          — сравнение моделей и визуализация

docs/
  overview-ru.md            — краткое описание проекта по-русски для научрука/кафедры
  architecture-diagram.png  — диаграмма архитектуры
  model-comparison-table.png
  recommendations-table.png
  experiments.md            — описание экспериментальной методики и результатов
  literature-review.md      — обзор литературы и существующих подходов
  paper-draft.md            — черновик научной статьи / НИР
```

Дополнительные файлы:

* README.md (текущий файл)
* ROADMAP.md (декларация фаз проекта и задач)
* LICENSE 
* .gitignore

---

## 8. Быстрый старт

1. **Клонирование и зависимость**
   ```bash
   git clone https://github.com/VisageDvachevsky/k8s-ml-predictive-autoscaling.git
   cd k8s-ml-predictive-autoscaling
   poetry install
   poetry run pre-commit install
   ```

2. **Тесты и линтеры**
   ```bash
   poetry run pytest
   poetry run mypy .
   poetry run flake8
   ```

3. **Локальная наблюдаемость через Docker Compose**
   ```bash
   cd docker
   cp ../.env.example ../.env  # заполните токены
   export AUTOSCALER_API_TOKEN="strong-token"
   docker compose up --build -d
   ```
   Доступы:
   * Demo FastAPI сервисы — http://localhost:8001..8003
   * Prometheus — http://localhost:9090
   * Grafana — http://localhost:3000 (admin/admin, dashboard provisioning включён)
   * `load-generator` автоматически прогоняет синтетическую нагрузку по `/workload`

4. **Kubernetes (kind/minikube)**
   ```bash
   kind create cluster --config k8s/kind-config.yaml --name autoscaling
   kubectl apply -f k8s/manifests/namespace.yaml
   kubectl create secret generic demo-service-credentials \
     --namespace predictive-autoscaling \
     --from-literal=api-token=<strong-token>
   kubectl apply -f k8s/manifests/demo-service-deployment.yaml
   kubectl apply -f k8s/manifests/demo-service-service.yaml
   kubectl apply -f k8s/manifests/prometheus/
   kubectl apply -f k8s/manifests/hpa/baseline-hpa.yaml
   kubectl apply -f k8s/manifests/load-generator-deployment.yaml
   ```

5. **Дальнейшие шаги**
   * Сбор и препроцессинг данных: см. `ROADMAP.md`, Phase 1.
   * Полный гайд с лайфхаками — `docs/setup-guide.md`.

### Требования безопасности демо-окружения

* Всегда задавайте `AUTOSCALER_API_TOKEN` (через `.env` или секреты) перед запуском demo-service и load-generator.
* Grafana админ-пароль берётся из переменной `GF_SECURITY_ADMIN_PASSWORD`; образец лежит в `.env.example`.
* Для Kubernetes создайте `demo-service-credentials` Secret с тем же токеном — он автоматически монтируется в Deployments.
* При необходимости укажите собственный заголовок через `AUTOSCALER_API_KEY_HEADER` (по умолчанию `X-API-Key`).

---

## 9. Phase 1 — Сбор исторических данных

### Экспорт метрик из Prometheus

1. Сконфигурируйте `src/k8s_ml_predictive_autoscaling/collector/config.yaml`:
   * `prometheus.base_url` — адрес локального/удалённого Prometheus.
   * `collection.lookback_hours`, `chunk_hours`, `default_step`.
   * `metrics[]` — список PromQL запросов и префиксов файлов.
2. Запустите экспорт:
   ```bash
   poetry run python -m k8s_ml_predictive_autoscaling.collector.collect_historical \\
     --config src/k8s_ml_predictive_autoscaling/collector/config.yaml
   ```
   Файлы появятся в `data/raw/cpu_metrics_YYYYMMDD.csv` и т.п. Каждая запись включает timestamp, PromQL и JSON-метки.

### Генерация синтетической нагрузки

* `tools/load_generator/synthetic_patterns.py` — генератор профилей нагрузки (дневные/недельные циклы + спайки).
* `tools/load_generator/locust_tasks.py` — сценарий Locust для `/workload`/`/health`.
  ```bash
  poetry run locust -f tools/load_generator/locust_tasks.py --host http://localhost:8001
  ```
* `tools/load_generator/k6_script.js` — k6-скрипт для быстрой CLI-нагрузки.
* Docker Compose сервис `load-generator` + K8s Deployment `k8s/manifests/load-generator-deployment.yaml` автоматически создают фоновую нагрузку.
* CLI `k8s_ml_predictive_autoscaling.load_generator` теперь требует `AUTOSCALER_API_TOKEN` (или `--api-key`) и поддерживает `--retries/--retry-backoff` для безопасных повторов.

> Если вы запускаете окружение в Docker Compose, в метриках Prometheus не будет метки `namespace`. Обновите `src/k8s_ml_predictive_autoscaling/collector/config.yaml` (селекторы `namespace`/`job`) под вашу конфигурацию, иначе коллекция вернёт 0 рядов, а препроцессинг завершится ошибкой из-за отсутствия `cpu_metrics`/`memory_metrics`.

Используйте эти инструменты для синтетических данных (Phase 1.2) и сценариев в Kubernetes/Docker Compose.

### Препроцессинг и генерация датасетов

1. Обновите `src/k8s_ml_predictive_autoscaling/preprocessor/config.yaml` (метрики, лаги, горизонты).
2. Запустите пайплайн:
   ```bash
   poetry run python -m k8s_ml_predictive_autoscaling.preprocessor.pipeline \
     --config src/k8s_ml_predictive_autoscaling/preprocessor/config.yaml
   ```
3. На выходе появятся:
   * `data/processed/train.csv`, `validation.csv`, `test.csv`.
   * Sliding-window последовательности (`sequences_*.npz`) для LSTM/Seq2Seq.
   * Сохранённый `scaler.pkl`.

### EDA и отчёты

* `notebooks/research-data.ipynb` — быстрый ноутбук для визуализации.
* `docs/eda-report.md` — конспект ключевых наблюдений и TODO для аналитики.

---

## 10. Roadmap

Подробный план развития проекта вынесен в отдельный файл `ROADMAP.md`. Кратко по фазам:

* **Phase 0** — каркас репозитория, базовые контейнеры (Prometheus, Grafana, demo-services).
* **Phase 1** — сбор и препроцессинг данных, EDA.
* **Phase 2** — базовые модели Prophet и LSTM/GRU, сравнение.
* **Phase 3** — гибрид Prophet+LSTM, экспорт в ONNX.
* **Phase 4** — сервис инференса (FastAPI + ONNX Runtime).
* **Phase 5** — Resource Planner и интеграция с Kubernetes (HPA/KEDA).
* **Phase 6** — дашборды и мониторинг.
* **Phase 7** — нагрузочные эксперименты и оформление научных результатов.
* **Phase 8** — (опционально) RL-подход к автомасштабированию.


## 11. Статус

Статус: **ранний research-прототип (MVP в разработке)**.

На первых этапах будут реализованы:

* каркас репозитория и окружения;
* сбор данных из Prometheus;
* базовые модели Prophet и LSTM;
* начальные эксперименты с прогнозом нагрузки.

Дальнейшие шаги и прогресс см. в `ROADMAP.md` и в разделе Issues/Projects репозитория (будет заведено по мере разработки).


## 12. Обратная связь и вклад

Пока проект делается как учебно-научный, но структура репозитория изначально ориентирована на открытый исходный код и возможность внешних вкладов.

Планы:

* оформить CONTRIBUTING.md (код-стайл, требования к PR, структура экспериментов);
* добавить шаблоны Issue и Pull Request;
* описать, как повторить эксперименты и сравнения из статьи.

Если вы увидели этот репозиторий и вам интересна тема предиктивного автомасштабирования — можете открывать Issue с идеями, замечаниями и предложениями по улучшению.

---

## Лицензия

MIT License (детали в файле LICENSE)
