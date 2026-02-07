# Roadmap

Файл Roadmap синхронизирован с описанием в README и деталями из запроса. Каждая фаза завершает готовый инкремент и строится на предыдущих.

## Phase 0 — Infrastructure & Repo Skeleton (1–2 недели) ✅

**Цель:** подготовить единый DevEx слой: зависимости, линтеры, Docker-окружение, Kubernetes-манифесты и документацию.

### Выполненные задачи
- [x] `pyproject.toml` + Poetry как единый источник зависимостей.
- [x] Настроены `pre-commit` (black, isort, flake8, mypy, базовые sanity hooks).
- [x] Создан демо FastAPI сервис (`src/k8s_ml_predictive_autoscaling/demo_service`).
- [x] Dockerfile + Compose c 3 экземплярами сервиса, Prometheus и Grafana.
- [x] Grafana provisioning (datasource + стартовый dashboard).
- [x] Prometheus config для скрейпа demo-services.
- [x] K8s manifests (namespace, demo Deployment/Service, Prometheus, baseline HPA, kind config).
- [x] Документация: README, ROADMAP, CONTRIBUTING, `docs/setup-guide.md`.

### Критерии Done
- docker compose запускает демо-сервисы + Prometheus/Grafana локально.
- /metrics собирать Prometheus, графики видны в Grafana.
- Kind конфигурация доступна и применяются базовые манифесты.
- CI (GitHub Actions) гоняет тесты + линтеры.

## Phase 1 — Data Collection & Preprocessing (2–3 недели) ✅

**Зависимости:** Phase 0.

### 1.1 Сбор данных из Prometheus
- [x] `src/k8s_ml_predictive_autoscaling/collector/prometheus_client.py` — высокоуровневый Prometheus API клиент (prometheus_http_api или httpx + PromQL).
- [x] `collect_historical.py` — CLI/скрипт выгрузки временных рядов в `data/raw/YYYYMMDD`. Конфигурация через YAML (`config.yaml`).
- [x] Библиотека конфигурации метрик с поддержкой CPU, памяти, RPS, latency p50/p95/p99.

### 1.2 Генерация синтетической нагрузки (опционально)
- [x] `tools/load_generator/` с k6 и/или Locust скриптами.
- [x] `synthetic_patterns.py` — генерация дневной/недельной сезонности + спайки.
- [x] Интеграция с docker compose (сервис `load-generator`) и K8s Deployment (`k8s/manifests/load-generator-deployment.yaml`).

### 1.3 Exploratory Data Analysis
- [x] `notebooks/research-data.ipynb` — загрузка выгрузок, базовые визуализации и статистики (STL/аномалии добавим позже).
- [x] Документация по итогам EDA (`docs/eda-report.md` с TODO на дальнейший анализ).

### 1.4 Препроцессинг
- [x] `src/k8s_ml_predictive_autoscaling/preprocessor/pipeline.py` — ресемплинг, заполнение пропусков, нормализация.
- [x] `feature_engineering.py` + `anomaly_detection.py` + `config.yaml`.
- [x] Объединённый pipeline с сохранением в `data/processed/{train,val,test}.csv` + scaler.pkl + sliding-window `.npz`.

### Дополнительно (Phase 1 расширения)
- [x] Асинхронный load generator (`k8s_ml_predictive_autoscaling.load_generator`) с Docker/K8s интеграцией.
- [x] Sliding-window датасеты (`sequences_*.npz`) для LSTM/Seq2Seq моделей.

### Критерии Done Phase 1
- Исторические временные ряды сохранены в `data/raw` за N дней/недель.
- Проведён EDA со статами/визуализациями/аномалиями.
- Препроцессинг воспроизводим, с разделением на train/val/test и сохранёнными scaler-объектами.
- Документация по формату данных и признакам добавлена в README/docs.

## Phase 2 — Базовые ML-модели (Prophet, LSTM/GRU) (3-4 недели) ✅

**Цель:** Обучить и сравнить базовые модели прогнозирования временных рядов.

**Зависимости:** Phase 1.

### 2.0. Препроцессинг реальных данных
- [x] Скрипт `scripts/preprocess_real_data.py`:
  - Загрузка Alibaba 2018 (2243 samples) и Azure v2 (8640 samples)
  - Объединение датасетов с нормализацией шкал
  - Feature engineering (time, lag, rolling features — 44 фичи)
  - StandardScaler нормализация
  - Train/val/test split (70/15/15): 7605/1630/1630 samples
  - Sliding-window sequences для LSTM (seq_length=24)

### 2.1. Prophet модель
- [x] Реализовать `models/prophet/train.py`:
  - Загрузка train/val данных
  - Обучение Prophet модели (trend, seasonality)
  - Hyperparameter tuning (changepoint_prior_scale, seasonality_prior_scale)
  - Сохранение модели (`prophet_model.pkl`)
- [x] Реализовать `models/prophet/evaluate.py`:
  - Прогноз на test данных
  - Метрики: RMSE, MAE, MAPE, R2
  - Визуализация: real vs predicted, error analysis, scatter plot
- [x] Результаты на реальных данных: R2=0.4445, RMSE=0.7350

### 2.2. LSTM/GRU модель
- [x] Реализовать `models/lstm/model.py`:
  - Архитектура LSTM и GRU (stacked, bidirectional, batch norm)
  - Dropout для регуляризации
  - Dense output head
  - Factory function для создания моделей
- [x] Реализовать `models/lstm/train.py`:
  - Обучение на PyTorch
  - Loss: MSE, Optimizer: Adam + weight decay
  - Learning rate scheduler (ReduceLROnPlateau)
  - Early stopping + gradient clipping
  - Сохранение лучшего чекпоинта
- [x] Реализовать `models/lstm/evaluate.py`:
  - Прогноз на test данных
  - Метрики: RMSE, MAE, MAPE, R2
  - Визуализация + training history
- [x] Результаты на реальных данных: R2=0.8865, RMSE=0.3331

### 2.3. Сравнение моделей
- [x] Скрипт `scripts/compare_models.py`:
  - Загрузка результатов всех моделей
  - Сравнение метрик (таблица, bar chart, overlay, boxplot)
- [x] Документ `docs/model-comparison-results.md`:
  - Таблицы результатов
  - Визуализации
  - Выводы: LSTM значительно превосходит Prophet (R2 0.89 vs 0.44)

### Критерии завершения Phase 2
- ✅ Обучены и сохранены Prophet и LSTM/GRU модели
- ✅ Проведена оценка на test данных
- ✅ Сравнение моделей задокументировано
- ✅ Выбрана модель-кандидат для дальнейшего улучшения

## Phase 3 — Гибридная модель Prophet+LSTM и ONNX (2-3 недели) ✅

**Цель:** Реализовать гибридный подход и экспорт в ONNX для production.

**Зависимости:** Phase 2.

### 3.1. Гибридная модель Prophet+LSTM
- [x] Реализовать `models/hybrid/prophet_lstm.py`:
  - Этап 1: Prophet моделирует trend + seasonality
  - Этап 2: LSTM прогнозирует residuals
  - Комбинированный прогноз: `y_pred = prophet_pred + lstm_residual_pred`
- [x] Обучение (`models/hybrid/train.py`) и оценка (`models/hybrid/evaluate.py`)
- [x] Результат: 23.9% улучшение RMSE над Prophet (0.56 vs 0.74)
- [x] Однако standalone LSTM (R2=0.89) превосходит гибрид (R2=0.68)
- [x] Причина: Prophet слабо работает на данных без явной сезонности

### 3.2. Экспорт моделей в ONNX
- [x] Скрипт `scripts/export_onnx.py` — полный pipeline экспорта
- [x] Экспорт Prophet (JSON metadata + pickle)
- [x] Экспорт LSTM -> `models/onnx/lstm_forecaster.onnx` (262.5 KB)
- [x] Экспорт гибридной модели -> `models/onnx/hybrid_residual_lstm.onnx`
- [x] Валидация: max diff < 2e-7 (идентичные результаты)
- [x] Бенчмарки: ONNX Runtime 3.8-4x быстрее PyTorch (0.13ms vs 0.50ms)

### 3.3. Оптимизация
- [x] Динамическая квантизация int8:
  - `lstm_forecaster_int8.onnx`: 78.7 KB (70% сжатие)
  - `hybrid_residual_lstm_int8.onnx`: 78.7 KB (70% сжатие)
- [x] Все модели < 1ms latency — далеко в пределах целевых 100ms

### Критерии завершения Phase 3
- ✅ Гибридная модель: +23.9% RMSE над Prophet (превышает целевые 6-15%)
- ✅ Модели экспортированы в ONNX и валидированы
- ✅ Бенчмарки задокументированы в `docs/model-comparison-results.md`

## Phase 4 — Сервис инференса (FastAPI + ONNX Runtime) (1–2 недели) ✅

**Цель:** Production-ready REST API для онлайн-прогнозирования.

**Зависимости:** Phase 3.

### 4.1. FastAPI приложение
- [x] `src/k8s_ml_predictive_autoscaling/predictor/config.py` — PredictorSettings (Pydantic BaseSettings, env_prefix=PREDICTOR_)
- [x] `src/k8s_ml_predictive_autoscaling/predictor/engine.py` — PredictionEngine:
  - ONNX Runtime InferenceSession с оптимизациями
  - Sliding window buffer (deque, maxlen=seq_length+30)
  - Feature engineering: 44 фичи (base + lags + rolling means + time)
  - StandardScaler для нормализации, inverse transform для результатов
  - `predict()` — одиночный прогноз из буфера
  - `predict_from_sequence()` — батч-прогноз из готовых последовательностей
- [x] `src/k8s_ml_predictive_autoscaling/predictor/metrics.py` — 7 Prometheus метрик
- [x] `src/k8s_ml_predictive_autoscaling/predictor/app.py` — FastAPI с lifespan:
  - GET /health — liveness probe
  - GET /ready — readiness probe (модель + буфер)
  - POST /ingest — прием наблюдений метрик
  - POST /predict — прогноз из буфера
  - POST /predict/batch — батч-прогноз
  - GET /model/info — метаданные модели
  - GET /metrics — Prometheus endpoint

### 4.2. Тестирование
- [x] `tests/predictor/test_config.py` — 3 теста (defaults, env, paths)
- [x] `tests/predictor/test_engine.py` — 15 тестов:
  - Properties, ingest, predict, predict_from_sequence
  - Feature engineering (shape, time normalization, lags, rolling means)
- [x] `tests/predictor/test_app.py` — 12 интеграционных тестов:
  - Health/ready endpoints, ingest, predict, batch, model info, metrics
  - Dummy ONNX модель (ReduceMean) для быстрых тестов
- [x] End-to-end тест с реальной LSTM ONNX моделью: прогноз CPU ~42%, inference 0.3ms

### 4.3. Docker и K8s
- [x] `docker/Dockerfile.predictor` — multi-stage с Poetry, HEALTHCHECK, non-root user
- [x] `docker/docker-compose.yml` — predictor сервис на порту 8010
- [x] `k8s/manifests/predictor-deployment.yaml` — Deployment с readiness/liveness probes
- [x] `k8s/manifests/predictor-service.yaml` — ClusterIP Service
- [x] Prometheus конфиги обновлены для scrape predictor (docker + k8s)

### Результаты
- 30 тестов: все проходят
- Латентность инференса: 0.3ms (далеко в пределах целевых 100ms)
- CPU прогноз на реальных данных: 42% (корректное значение)
- Batch inference: 4 последовательности за 0.3ms

### Критерии завершения Phase 4
- ✅ Сервис запускается и отвечает на запросы
- ✅ ONNX модели работают корректно
- ✅ Латентность инференса < 100ms (факт: 0.3ms)
- ✅ 30 unit/integration тестов проходят
- ✅ Dockerfile и K8s манифесты готовы
- ✅ Prometheus интеграция для мониторинга

## Phase 5 -- Resource Planner i integratsiya s K8s (2-3 nedeli) -- DONE

**Tsel:** Preobrazovat' prognozy v resheniya po masshtabirovaniyu.

**Zavisimosti:** Phase 4.

### 5.1. Scaling Algorithm (chistye funktsii, bez I/O)
- [x] `planner/algorithm.py` -- EMA smoothing, hysteresis dead-zone, replica computation
- [x] Safety margin (+15% dlya scale-up), rate limits (+2/-1 za tsikl)
- [x] Cooldown: scale-up=60s, scale-down=300s
- [x] Scale-down stabilization: 3 podryad nizhe poroga
- [x] `planner/config.py` -- PlannerSettings (env_prefix=PLANNER_, 20+ parametrov)
- [x] 30 testov algoritma (smoothing, replicas, rate limits, cooldowns, hysteresis)

### 5.2. Executor Layer
- [x] `executor/base.py` -- ScalingExecutor Protocol (structural typing)
- [x] `executor/mock_executor.py` -- In-memory executor dlya Docker Compose i testov
- [x] `executor/k8s_executor.py` -- Real K8s API (patch deployments/scale)
- [x] 6 testov mock executor + 4 testa K8s executor (s mock kubernetes client)

### 5.3. HTTP Clients
- [x] `planner/predictor_client.py` -- Async httpx client (/ingest, /predict, /health)
- [x] `planner/prometheus_feeder.py` -- Async PromQL (instant + range queries)
- [x] Backfill: range query za 120 minut dlya bystrogo zapolneniya buffera
- [x] 8 testov predictor client + 6 testov prometheus feeder (MockTransport)

### 5.4. Control Loop
- [x] `planner/control_loop.py` -- Async orchestrator:
  - Fetch Prometheus -> Ingest -> Predict -> Decide -> Execute
  - Backfill pri starte iz istoricheskikh dannykh Prometheus
  - Graceful degradation: hold pri lyuboy oshibke
  - Degraded mode posle N posledovatel'nykh oshibok
- [x] `planner/metrics.py` -- 12 Prometheus metrik
- [x] 8 testov control loop (single cycle, failures, backfill, lifecycle)

### 5.5. FastAPI Planner Service
- [x] `planner/app.py` -- GET /health, /ready, /status, /metrics, POST /override
- [x] Lifespan: init clients + start background control loop
- [x] Auto-detect: K8sExecutor (in-cluster) ili MockExecutor (Docker Compose)
- [x] 6 testov app (endpoints, override, metrics)

### 5.6. Infrastructure
- [x] `docker/Dockerfile.planner` -- Multi-stage, Poetry, non-root, HEALTHCHECK
- [x] `docker/docker-compose.yml` -- planner service na portu 8020
- [x] Prometheus configs obnovleny (docker + k8s) dlya scrape planner
- [x] K8s manifests: RBAC (ServiceAccount + Role + RoleBinding), Deployment, Service
- [x] `pyproject.toml` -- kubernetes ^29.0.0 (optional), mypy override

### 5.7. RBAC i bezopasnost'
- [x] Namespace-scoped Role: get/list/watch/patch deployments + pods
- [x] ServiceAccount: planner
- [x] Printsip naimenshikh privilegiy: tol'ko neobkhodimye operatsii

### Resheniya (HPA vs Direct API)
- Vybran **pryamoy K8s API** (ne Custom Metrics / KEDA):
  - Prosche, prozrachnee dlya issledovaniya, polnyy kontrol'
  - HPA i Planner vzaimoisklyuchayushchie (ili baseline HPA, ili ML planner)

### Fallback strategiya
| Situatsiya | Deystviye |
|------------|-----------|
| Predictor nedostupen | Hold tekushchiye repliki |
| Predictor 425 (progrev) | Hold, prodolzhit' inzhest |
| Prometheus nedostupen | Hold, skip ingest |
| K8s API oshibka | Hold, retry sleduyushchiy tsikl |
| 5 podryad oshibok | planner_degraded=1 |

### Rezul'taty
- 67 testov: vse prokhodyat (+ 1 skipped -- K8s executor bez kubernetes paketa)
- Algorithmicheskoye yadro: chistye funktsii, 100% pokrytiye
- Latentnost' control loop: < 1s na tsikl (bez setevykh zadershek)
- Metriki: 12 Prometheus metrik dlya monitoringa

### Kriteriy zaversheniya Phase 5
- DONE Planner servis zapuskayetsya i orkhestruyet masshtabirovaniye
- DONE Executor primenyayet resheniya cherez K8s API ili mock
- DONE Fallback rabotayet pri degradatsii
- DONE RBAC nastroyen s minimal'nymi privilegiyami
- DONE 67+ unit/integration testov prokhodyat
- DONE Dockerfile, Docker Compose i K8s manifesty gotovy

## Phase 6 — Дашборды и мониторинг (1–2 недели) ✅

**Цель:** Визуализация работы системы и метрик качества.

**Зависимости:** Phase 5.

### 6.1. Grafana дашборды (4 штуки)
- [x] **ML Autoscaler — Overview** (`ml-autoscaler-overview.json`):
  - 6 KPI stat-панелей (replicas, CPU, decisions, errors, status)
  - Hero-панель: Predicted CPU + пороги (42%/60%/66%) с цветовым кодированием
  - Replica Count (current vs desired, stepAfter, min/max линии)
  - Request Rate, Latency Percentiles (p50/p95/p99), Scaling Activity (donut)
- [x] **Scaling Decisions & Replicas** (`scaling-decisions-replicas.json`):
  - 6 KPI stat-панелей (executions up/down, hold, cooldowns, fallback)
  - Replica History (full-width, stepAfter с fill, min/max линии)
  - Decisions Stacked Bars (up/down/hold), Executions +/- (positive/negative bars)
  - Cooldown State Timeline, Dead-Zone Analysis ([42%, 66%])
- [x] **ML Prediction Quality** (`ml-prediction-quality.json`):
  - 6 KPI (latest prediction, predictions/min, error rate, buffer gauge, ONNX latency, model load)
  - Predicted vs Actual CPU (raw + EMA smoothed + actual), Prediction Residual (bars)
  - E2E Latency (p50/p95/p99), ONNX Runtime Latency, Throughput
  - Buffer Size (с min/max линиями), Prediction Success Rate
- [x] **System Health & SLA** (`system-health-sla.json`):
  - 6 KPI (planner/predictor/demo health, failures, degraded, fallback rate)
  - Loop Duration Percentiles + Heatmap (Spectral цветовая схема)
  - Errors by Phase (stacked), Error Distribution (bargauge)
  - CPU/Memory by Service, Degradation Timeline (state-timeline)

### 6.2. Единая цветовая схема
- [x] Зелёный (#73BF69) — healthy, scale-up, success
- [x] Жёлтый (#FADE2A) — warning, cooldown, пороги
- [x] Оранжевый (#FF9830) — caution, scale-down
- [x] Красный (#F2495C) — critical, degraded, errors
- [x] Синий (#5794F2) — информационный, predictions, current state
- [x] Фиолетовый (#B877D9) — ML/predictor-related

### 6.3. Prometheus Alert Rules (6 правил)
- [x] **PredictorDown** — up{predictor}==0, for 1m, severity: critical
- [x] **PlannerDegraded** — planner_degraded==1, for 2m, severity: critical
- [x] **HighPredictionErrorRate** — error rate > 5%, for 5m, severity: warning
- [x] **PlannerHighLoopLatency** — p95 > 5s, for 5m, severity: warning
- [x] **ReplicasAtMax** — replicas >= 8, for 10m, severity: warning
- [x] **ConsecutiveFailures** — failures > 3, for 2m, severity: warning

### 6.4. Provisioning обновления
- [x] `dashboard.yml` — provider переименован в "ML Autoscaler Dashboards"
- [x] `prometheus.yml` — добавлен rule_files: [rules.yml]
- [x] Существующий `demo.json` сохранён для обратной совместимости

### Критерии завершения Phase 6
- ✅ 4 дашборда загружаются в Grafana без ошибок
- ✅ 60 панелей покрывают все 21 кастомную метрику
- ✅ 6 алертных правил настроены в Prometheus
- ✅ Визуально видны пороги алгоритма (42%/60%/66%) и состояние системы

## Phase 7 — Эксперименты и визуализация (2–3 недели) ✅

**Цель:** Детерминистичные эксперименты: ML Autoscaler vs HPA + интерактивная визуализация.

**Зависимости:** Phase 6.

### 7.1. Experiment Simulator
- [x] `scripts/experiment_simulator.py` — ядро симуляции:
  - HPASimulator: Kubernetes HPA v2 (tolerance band, cooldown, stabilization)
  - MLAutoscalerSimulator: 9-step алгоритм (EMA, hysteresis, rate limits, cooldowns)
  - ExperimentRunner: запускает оба на одних данных, собирает StepRecord историю
  - compute_metrics(): 8 сравнительных метрик (avg replicas, replica-minutes, SLA, cost savings)

### 7.2. Experiment Scenarios
- [x] `scripts/experiment_scenarios.py` — 4 сценария нагрузки:
  - Stable workload: сегмент с минимальной дисперсией из реальных данных (300 шагов)
  - Gradual ramp: синтетический рамп 30%→85% (400 шагов)
  - Spike pattern: 3 спайка 42%→85-92%→42% (350 шагов)
  - Diurnal cycle: 48ч день/ночь паттерн (576 шагов)
  - Загрузка и inverse-transform из test.csv через scaler.pkl

### 7.3. Тестирование
- [x] `tests/test_experiment_simulator.py` — 22 теста:
  - TestHPASimulator: 7 тестов (старт, scale up, tolerance, bounds, stabilization)
  - TestMLAutoscalerSimulator: 7 тестов (dead zone, scale up, EMA, cooldown)
  - TestExperimentRunner: 4 теста (simple data, CPU correctness, length mismatch)
  - TestComputeMetrics: 4 теста (empty, constant, cost savings, events)
- [x] Все 22 теста проходят

### 7.4. CustomTkinter GUI Dashboard
- [x] `scripts/experiment_gui.py` — интерактивный визуализатор (~860 строк):
  - Вкладка "Neural Network": ONNX inference pipeline, buffer status, prediction output, raw metrics, prediction history chart
  - Вкладка "Scaling Algorithm": 9 PipelineBlock виджетов с цветовым статусом, CPU+Thresholds chart (dead-zone fill), Replica History, Decision Log
  - Вкладка "Experiment": comparison chart, replica difference bars, 8 KPI cards (HPA vs ML)
  - Sidebar: 5 слайдеров параметров (target util, EMA alpha, safety margin, up/down thresholds), Apply & Re-run
  - Controls: Play/Pause/Step/Reset, speed slider, step counter
  - Dark mode, единая цветовая схема с Grafana дашбордами
  - Запуск: `python scripts/experiment_gui.py`

### 7.5. Static Analysis & Charts
- [x] `scripts/experiment_analysis.py` — генерация статичных графиков:
  - Per-scenario PNG (CPU + replicas + difference, 3 subplot)
  - Combined metrics bar chart (6 KPI × 4 scenarios)
  - Cost savings chart, Replica-minutes chart
  - CSV export (experiment_results.csv)
  - LaTeX table export (experiment_table.tex)

### 7.6. Документация
- [x] `docs/experiments.md` — методология, результаты, анализ:
  - Описание обоих симуляторов с параметрами
  - 4 сценария с обоснованием
  - 9 метрик сравнения
  - Таблицы результатов
  - Ключевые выводы

### Результаты экспериментов

| Сценарий | HPA avg R | ML avg R | HPA Events | ML Events | Cost Savings |
|----------|-----------|----------|------------|-----------|-------------|
| Stable | 2.0 | 2.0 | 0 | 0 | 0.0% |
| Gradual Ramp | 4.58 | 4.21 | 6 | 3 | +8.1% |
| Spike Pattern | 3.84 | 6.06 | 41 | 19 | -57.9% |
| Diurnal Cycle | 5.71 | 6.47 | 21 | 12 | -13.4% |

**Ключевые выводы:**
- ML autoscaler сокращает scaling events на 50% (стабильность)
- ML обеспечивает zero under-provisioning в spike сценариях
- Стоимость proactive подхода: +15.8% replica-minutes в среднем
- Trade-off: стабильность и SLA за счёт дополнительных ресурсов
- Параметры настраиваемы через GUI для оптимизации под конкретный workload

### Критерии завершения Phase 7
- ✅ Experiment simulator реализован и протестирован (22 теста)
- ✅ 4 сценария нагрузки из реальных + синтетических данных
- ✅ CustomTkinter GUI с 3 вкладками и пошаговой анимацией
- ✅ Статичные графики и CSV/LaTeX экспорт
- ✅ Документация по методологии и результатам

## Phase 8 — Reinforcement Learning (опционально) (4–6 недель)

**Цель:** Исследование RL-подхода к автомасштабированию.

**Зависимости:** Phase 7.

### Задачи
- [ ] Формализация как MDP
- [ ] Симулятор окружения (OpenAI Gym)
- [ ] RL алгоритмы (DQN, PPO)
- [ ] Интеграция с K8s

### Критерии завершения
- ✅ RL агент показывает конкурентные результаты
- ✅ Сравнение с ML-autoscaler задокументировано
