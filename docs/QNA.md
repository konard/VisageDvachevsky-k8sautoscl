# Ответы на вопросы по k8sautoscl

## 1. Как запустить экспериментальный GUI и проверить как работает система?

### Запуск интерактивного GUI Dashboard

В проекте есть экспериментальный GUI Dashboard, который визуализирует работу ML-автомасштабирования в режиме реального времени.

#### Команда запуска:

```bash
python scripts/experiment_gui.py
```

### Что показывает GUI?

GUI состоит из **3 вкладок**:

#### 📊 Tab 1: Neural Network (Нейронная сеть)
- **ONNX Inference Pipeline** — визуализация пайплайна инференса:
  - Raw Metrics (5 значений) → Feature Engineering (44 фичи) → StandardScaler → ONNX LSTM → Прогноз CPU
- **Buffer Status** — статус буфера данных (нужно 24 временных точки для предсказания)
- **ML Prediction** — текущий прогноз нагрузки в процентах CPU
- **Current Metrics** — текущие метрики (CPU, память, сеть, диск)
- **Prediction History** — график истории предсказаний (последние 80 шагов)

#### ⚙️ Tab 2: Scaling Algorithm (Алгоритм масштабирования)
- **9-step pipeline** — пошаговая визуализация алгоритма принятия решений:
  1. **EMA Smoothing** — экспоненциальное сглаживание
  2. **Hysteresis** — определение направления (UP/DOWN/HOLD)
  3. **Desired Replicas** — вычисление желаемого количества реплик
  4. **Clamp** — ограничение [min=2, max=8]
  5. **Rate Limit** — ограничение скорости изменения
  6. **Cooldown Check** — проверка периода охлаждения
  7. **Stabilization** — стабилизация при scale-down
  8. **No Change?** — проверка необходимости изменения
  9. **Commit** — фиксация решения
- **State Panel** — текущее состояние алгоритма (cooldown, direction, replicas)
- **CPU + Thresholds** — график CPU с порогами срабатывания
- **Replica Count** — сравнение количества реплик ML vs HPA
- **Decision Log** — лог принятых решений

#### 🧪 Tab 3: Experiment (Сравнение экспериментов)
- **Comparison Chart** — сравнение HPA vs ML автомасштабирования:
  - График CPU нагрузки
  - График количества реплик (HPA в красном, ML в синем)
  - График разницы в репликах (ML - HPA)
- **8 KPI карточек** — ключевые метрики сравнения:
  - Avg Replicas — среднее количество реплик
  - Replica-Minutes — общее потребление ресурсов
  - SLA Violations — нарушения SLA
  - Scaling Events — количество событий масштабирования
  - Over-Provisioning % — избыточное выделение ресурсов
  - Under-Provisioning % — недостаточное выделение ресурсов
  - Reaction Time — время реакции на изменения
  - Cost Savings % — экономия затрат

### Управление в GUI

- **▶ Play** — автоматическое воспроизведение шагов симуляции
- **⏭ Step** — пошаговое выполнение (один шаг = 5 минут реального времени)
- **↺ Reset** — сброс симуляции к началу
- **Speed slider** — регулировка скорости воспроизведения

### Параметры в боковой панели (левая панель)

Вы можете изменять параметры алгоритма в реальном времени:

- **Target Util %** (20-90%) — целевая утилизация CPU
- **EMA Alpha** (0.1-1.0) — коэффициент экспоненциального сглаживания
- **Safety Margin** (0.0-0.5) — запас прочности для проактивного масштабирования
- **Up Threshold ×** (1.0-1.5) — порог для масштабирования вверх
- **Down Threshold ×** (0.3-0.9) — порог для масштабирования вниз

После изменения параметров нажмите **"Apply & Re-run"** для перезапуска эксперимента.

### 4 встроенных сценария

- **Stable** — стабильная нагрузка (25 часов)
- **Ramp** — постепенное увеличение нагрузки от 30% до 85% (33 часа)
- **Spike** — три всплеска нагрузки (29 часов)
- **Diurnal** — дневной цикл с пиками в рабочее время (48 часов)

### Требования для запуска GUI

```bash
# Установка зависимостей
poetry install

# Проверка наличия тестовых данных
ls -lh data/processed/test.csv
ls -lh data/processed/scaler.pkl
ls -lh data/processed/metadata.json
```

Если данные отсутствуют, GUI автоматически создаст синтетический demo-сценарий (synthetic_demo).

### Альтернатива: Статический анализ (без GUI)

Если GUI не запускается или вам нужны статические графики:

```bash
# Генерация графиков и таблиц
python scripts/experiment_analysis.py --output-dir results/experiments
```

**Выходные файлы**:
- `scenario_*.png` — графики для каждого сценария
- `combined_metrics.png` — сравнение метрик
- `cost_savings.png` — график экономии
- `experiment_results.csv` — данные в CSV
- `experiment_table.tex` — таблица для LaTeX

---

## 2. Что учитывается при предсказании нагрузки, и что можно учитывать?

### Что учитывается СЕЙЧАС в системе

Система использует **44 признака** для прогнозирования нагрузки CPU на 15 минут вперед:

#### A. Базовые метрики (5 метрик)
1. **cpu_usage** — текущая утилизация CPU (%)
2. **mem_util** — утилизация памяти (%)
3. **net_in** — входящий сетевой трафик (KB/s)
4. **net_out** — исходящий сетевой трафик (KB/s)
5. **disk_io** — операции ввода-вывода диска

Для Docker Compose окружения:
- **request_rate** — частота входящих запросов
- **latency_p50** — медианная задержка
- **latency_p95** — 95-й процентиль задержки
- **latency_p99** — 99-й процентиль задержки
- **active_jobs** — активные задачи (синтетическая метрика)

#### B. Временные признаки (4 признака)
- **hour** — час суток (0-23)
- **day_of_week** — день недели (0=понедельник, 6=воскресенье)
- **is_weekend** — признак выходного дня (0/1)
- **minute_of_day** — минута дня (0-1439)

**Зачем?** — Позволяет модели учитывать дневные и недельные паттерны (пиковые часы, ночное снижение нагрузки).

#### C. Lag Features (16 признаков)
Задержки (lags) для каждой базовой метрики:
- **Lag 1** — значение 1 минуту назад
- **Lag 5** — значение 5 минут назад
- **Lag 15** — значение 15 минут назад
- **Lag 30** — значение 30 минут назад

**Зачем?** — LSTM получает "память" о недавнем прошлом, что критично для временных рядов.

#### D. Rolling Window Features (15 признаков)
Скользящие средние для каждой базовой метрики:
- **Rolling mean 5** — среднее за последние 5 минут
- **Rolling mean 15** — среднее за последние 15 минут
- **Rolling mean 30** — среднее за последние 30 минут

**Зачем?** — Сглаживание шума и выявление трендов.

#### E. Sliding Window для LSTM
- **Sequence length**: 24 временных шага (2 часа истории по 5 минут)
- **Forecast horizon**: 3 шага вперед (15 минут в будущее)

**Зачем?** — LSTM обучается на последовательностях, чтобы находить паттерны изменения нагрузки.

### Что МОЖНО учитывать (расширение функционала)

#### 1. Метрики приложения (Application-Level Metrics)
- **Request rate per endpoint** — нагрузка по конкретным API endpoints
- **Database query latency** — задержки запросов к БД
- **Error rate (4xx, 5xx)** — частота ошибок
- **Active connections** — количество активных соединений
- **Queue depth** — глубина очередей задач
- **Cache hit rate** — эффективность кеширования

#### 2. Бизнес-метрики
- **User count** — количество активных пользователей
- **Transaction rate** — частота транзакций
- **Session duration** — длительность сессий
- **Order rate** (для e-commerce) — частота заказов

#### 3. Инфраструктурные метрики
- **Node CPU/Memory pressure** — нагрузка на узлы кластера
- **Pod restart count** — частота перезапусков подов
- **Pending pods** — количество подов в очереди
- **Network bandwidth utilization** — использование сетевой полосы

#### 4. Внешние факторы
- **Public holidays** — праздничные дни
- **Marketing campaigns** — запланированные маркетинговые кампании
- **Scheduled maintenance** — плановое обслуживание
- **External API latency** — задержки внешних API
- **Weather data** (для location-based сервисов) — погодные данные

#### 5. Prophet-специфичные компоненты
- **Trend** — долгосрочный тренд
- **Seasonality** — сезонность (дневная, недельная, годовая)
- **Holidays** — влияние праздников
- **Changepoints** — точки изменения тренда

#### 6. Дополнительные временные признаки
- **is_business_hours** — рабочие часы (9:00-18:00)
- **week_of_year** — неделя года
- **quarter** — квартал
- **is_month_start / is_month_end** — начало/конец месяца

### Текущая конфигурация

Конфигурация находится в `src/k8s_ml_predictive_autoscaling/preprocessor/config.yaml`:

```yaml
features:
  enable_time_features: true
  lags: [1, 5, 15, 30]        # Look back 1, 5, 15, 30 minutes
  rolling_windows: [5, 15, 30]  # Rolling averages

sliding_window:
  sequence_length: 60     # 60 minutes of history for LSTM
  forecast_steps: [5, 15, 30]  # Predict 5, 15, 30 minutes ahead
  stride: 5               # Create windows every 5 minutes
  target_metric: request_rate  # Predict request rate
```

### Как добавить новые признаки?

1. **Добавить метрику в Prometheus** — настроить экспорт метрики
2. **Обновить collector config** — добавить PromQL запрос в `src/k8s_ml_predictive_autoscaling/collector/config.yaml`
3. **Обновить preprocessor** — добавить метрику в `scaler_features` и `features`
4. **Переобучить модель** — запустить pipeline заново

---

## 3. Как это все работает?

### Общая архитектура (MAPE Loop)

Система следует паттерну **MAPE loop**: **Monitor → Analyze → Plan → Execute**.

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAPE LOOP                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌────────┐ │
│  │ Monitor  │────>│ Analyze  │────>│   Plan   │────>│Execute │ │
│  │(Metrics) │     │(ML Model)│     │(Resource │     │(K8s API│ │
│  │          │     │          │     │ Planner) │     │ / HPA) │ │
│  └──────────┘     └──────────┘     └──────────┘     └────────┘ │
│       ▲                                                    │     │
│       │                                                    │     │
│       └────────────────────────────────────────────────────┘     │
│                         Feedback Loop                            │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Monitor (Мониторинг)

**Компоненты**:
- **Prometheus** — сбор метрик с микросервисов
- **Collector** (`src/k8s_ml_predictive_autoscaling/collector/`) — экспорт метрик из Prometheus

**Что происходит**:
1. Микросервисы экспонируют `/metrics` (Prometheus format)
2. Prometheus scraping каждые 10-30 секунд
3. Метрики сохраняются как временные ряды
4. Collector экспортирует данные в CSV для обучения

**Метрики**:
- CPU, память, сеть, диск
- Request rate, latency (p50, p95, p99)
- Error rate, custom application metrics

### 2. Analyze (Анализ — ML слой)

**Компоненты**:
- **Preprocessor** (`src/k8s_ml_predictive_autoscaling/preprocessor/`) — препроцессинг данных
- **ML Models** (`models/lstm/`, `models/prophet/`, `models/hybrid/`) — обученные модели
- **Predictor Service** (`src/k8s_ml_predictive_autoscaling/predictor/`) — FastAPI сервис инференса

#### Шаг 2.1: Препроцессинг

```bash
poetry run python -m k8s_ml_predictive_autoscaling.preprocessor.pipeline \
  --config src/k8s_ml_predictive_autoscaling/preprocessor/config.yaml
```

**Действия**:
1. **Загрузка данных** из `data/raw/*.csv`
2. **Resampling** — агрегация по 1-минутным интервалам
3. **Anomaly detection** — фильтрация аномалий (z-score > 3.5)
4. **Feature engineering**:
   - Добавление временных признаков (hour, day_of_week, weekend)
   - Добавление lag features (1, 5, 15, 30 минут назад)
   - Добавление rolling mean (5, 15, 30 минут)
5. **Sliding window generation** — создание последовательностей для LSTM
6. **Normalization** — StandardScaler
7. **Train/Val/Test split** — 70% / 15% / 15%

**Выход**:
- `data/processed/train.csv`, `validation.csv`, `test.csv`
- `data/processed/sequences_*.npz` — sliding window sequences
- `data/processed/scaler.pkl` — сохранённый StandardScaler

#### Шаг 2.2: Обучение моделей

**Prophet**:
```bash
poetry run python -m models.prophet.train
```
- Модель с трендом + сезонностью (дневная, недельная)
- Обучение: ~10-20 секунд
- Результат: R² = 0.44 (умеренная точность)

**LSTM** (лучшая модель):
```bash
poetry run python -m models.lstm.train
```
- 2-layer LSTM, hidden_size=64, dropout=0.2
- 65,753 параметра
- Обучение: ~56 секунд
- Результат: **R² = 0.89** (высокая точность)

**Hybrid Prophet+LSTM**:
```bash
poetry run python -m models.hybrid.train
```
- Prophet для тренда, LSTM для остатков
- Результат: R² = 0.68 (средняя точность)

#### Шаг 2.3: Экспорт в ONNX

```bash
poetry run python scripts/export_onnx.py
```
- Экспорт PyTorch модели в ONNX Runtime
- Ускорение инференса: **3.8-4.0x**
- Размер модели: 262 KB (обычный) / 78.7 KB (int8 quantization)
- Latency: 0.13 ms (vs 0.50 ms PyTorch)

#### Шаг 2.4: Inference Service (FastAPI)

```bash
poetry run python -m k8s_ml_predictive_autoscaling.predictor.app
```

**API endpoints**:
- `POST /predict` — прогноз нагрузки на основе текущих метрик
- `GET /health` — health check
- `GET /metrics` — Prometheus metrics

**Процесс инференса**:
1. Получение 24 последних значений метрик (2 часа истории)
2. Feature engineering (44 признака)
3. Normalization (StandardScaler)
4. ONNX inference
5. Inverse transform (денормализация)
6. Возврат прогноза CPU на 5, 15, 30 минут вперед

### 3. Plan (Планирование ресурсов)

**Компоненты**:
- **Resource Planner** (`src/k8s_ml_predictive_autoscaling/planner/`) — алгоритм планирования

**9-Step Algorithm (детальный алгоритм планирования)**:

#### Входные данные
- `predicted_cpu` — прогноз CPU от ML модели (%)
- `current_replicas` — текущее количество реплик

#### Параметры
- `target_utilization` = 60% — целевая утилизация
- `smoothing_alpha` = 0.7 — коэффициент EMA
- `safety_margin` = 15% — запас прочности
- `scale_up_threshold` = 66% (1.1x target) — порог для масштабирования вверх
- `scale_down_threshold` = 42% (0.7x target) — порог для масштабирования вниз

#### Шаги алгоритма

**Step 1: EMA Smoothing**
```
smoothed_cpu = alpha * predicted_cpu + (1 - alpha) * smoothed_cpu_prev
```
Сглаживание шума, предотвращение реакции на случайные всплески.

**Step 2: Hysteresis Direction**
```
if smoothed_cpu > scale_up_threshold:
    direction = UP
elif smoothed_cpu < scale_down_threshold:
    direction = DOWN
else:
    direction = HOLD  # Dead zone — no action
```
Гистерезис предотвращает осцилляции в зоне [42%, 66%].

**Step 3: Compute Desired Replicas**
```
desired_replicas = ceil(current_replicas * smoothed_cpu / target_utilization)

if direction == UP:
    desired_replicas = ceil(desired_replicas * (1 + safety_margin))
```
Добавление 15% запаса при масштабировании вверх для проактивности.

**Step 4: Clamp to [min, max]**
```
desired_replicas = max(2, min(8, desired_replicas))
```
Ограничение диапазона реплик.

**Step 5: Rate Limit**
```
if direction == UP:
    desired_replicas = min(desired_replicas, current_replicas + 2)
else:
    desired_replicas = max(desired_replicas, current_replicas - 1)
```
Ограничение скорости изменения: макс +2 вверх, макс -1 вниз.

**Step 6: Cooldown Check**
```
if time_since_last_scale_up < 60s:
    block UP
if time_since_last_scale_down < 300s:
    block DOWN
```
Предотвращение частых изменений.

**Step 7: Scale-Down Stabilization**
```
if direction == DOWN:
    if consecutive_below_threshold < 3:
        block DOWN  # Need 3 consecutive low readings
```
Консервативное масштабирование вниз (требуется 3 подряд низких значения).

**Step 8: No-Change Check**
```
if desired_replicas == current_replicas:
    skip execution
```

**Step 9: Commit Decision**
```
if not blocked:
    execute scale to desired_replicas
    update cooldown timers
    reset stabilization counter
```

### 4. Execute (Исполнение)

**Компоненты**:
- **Executor** (`src/k8s_ml_predictive_autoscaling/executor/`) — интеграция с Kubernetes API

**Два режима интеграции**:

#### Режим 1: HPA External Metrics
1. ML-прогнозы экспортируются как **external metrics** для HPA
2. Prometheus Adapter читает метрики
3. HPA использует external metrics вместо CPU/памяти
4. Kubernetes API автоматически масштабирует Deployment

#### Режим 2: KEDA (Event-Driven Autoscaling)
1. ML-прогнозы экспортируются в Prometheus
2. KEDA ScaledObject использует Prometheus trigger
3. KEDA создаёт/изменяет HPA на основе ML-метрик

**Fallback**:
- При недоступности ML-сервиса система откатывается на классический HPA по CPU

### Полный цикл работы (пример)

```
t=0:
  1. Prometheus scrapes metrics: CPU=45%, mem=60%, request_rate=1000 req/s
  2. Predictor Service получает последние 24 точки (2 часа)
  3. ML Model прогнозирует: CPU через 15 мин = 72%
  4. Planner:
     - Smoothed: 0.7*72 + 0.3*65 = 69.9%
     - Direction: UP (69.9% > 66%)
     - Desired: ceil(3 * 69.9 / 60) = 4 replicas
     - Safety: ceil(4 * 1.15) = 5 replicas
     - Clamp: 5 (in range [2,8])
     - Rate limit: min(5, 3+2) = 5 OK
     - Cooldown: OK (no recent scale)
     - Decision: SCALE 3 → 5
  5. Executor отправляет команду в Kubernetes API
  6. Kubernetes создаёт 2 новых пода
  7. Через 15 минут нагрузка действительно вырастает до 72%, но система уже готова!

t=15:
  Фактическое CPU = 72%, но с 5 репликами эффективная утилизация = 72%/5*3 = 43%
  → Система проактивно предотвратила перегрузку!
```

### Ключевые отличия от HPA

| Аспект | HPA (реактивный) | ML Autoscaler (проактивный) |
|--------|------------------|---------------------------|
| Принятие решений | На основе текущих метрик | На основе прогноза на 15 минут |
| Реакция на спайк | После факта (задержка ~30-60с) | За 15 минут до спайка |
| Стабильность | Частые осцилляции | -50% scaling events |
| Over-provisioning | Минимальный | Выше (+15% safety margin) |
| SLA violations | Возможны при резких спайках | Практически исключены |
| Сложность | Простой (формула) | Высокая (ML pipeline) |

### Метрики сравнения (из экспериментов)

| Метрика | HPA | ML Autoscaler |
|---------|-----|---------------|
| Avg Replicas | 4.03 | 4.69 |
| Scaling Events | 17.0 | **8.5** (-50%) |
| SLA Violations | 0 | 0 |
| Under-Provisioning | 0.04% | 0.12% |
| Over-Provisioning | 47.0% | 53.6% |
| Cost Savings | baseline | -15.8% (выше затраты, но выше стабильность) |

---

## Дополнительные материалы

### Документация
- **README.md** — общее описание проекта
- **ROADMAP.md** — план развития (Phase 0-8)
- **RUN.md** — команды для быстрого старта
- **docs/experiments.md** — методология экспериментов
- **docs/model-comparison-results.md** — сравнение моделей Prophet/LSTM/Hybrid

### Команды для работы с системой

```bash
# 1. Локальное окружение (Docker Compose)
cd docker
export AUTOSCALER_API_TOKEN="your-token"
docker compose up --build -d

# Доступы:
# - Demo services: http://localhost:8001-8003
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000

# 2. Сбор данных
poetry run python -m k8s_ml_predictive_autoscaling.collector.collect_historical \
  --config src/k8s_ml_predictive_autoscaling/collector/config.yaml

# 3. Препроцессинг
poetry run python -m k8s_ml_predictive_autoscaling.preprocessor.pipeline \
  --config src/k8s_ml_predictive_autoscaling/preprocessor/config.yaml

# 4. Обучение модели
poetry run python -m models.lstm.train

# 5. Экспорт в ONNX
poetry run python scripts/export_onnx.py

# 6. Inference service
poetry run python -m k8s_ml_predictive_autoscaling.predictor.app

# 7. Эксперименты (статические графики)
python scripts/experiment_analysis.py --output-dir results/experiments

# 8. GUI Dashboard
python scripts/experiment_gui.py
```

### Kubernetes deployment

```bash
# Создание кластера
kind create cluster --config k8s/kind-config.yaml --name autoscaling

# Установка компонентов
kubectl apply -f k8s/manifests/namespace.yaml
kubectl apply -f k8s/manifests/prometheus/
kubectl apply -f k8s/manifests/demo-service-deployment.yaml
kubectl apply -f k8s/manifests/hpa/baseline-hpa.yaml

# Деплой ML Predictor
kubectl apply -f k8s/manifests/ml-predictor-deployment.yaml

# Проверка
kubectl get pods -n predictive-autoscaling
kubectl get hpa -n predictive-autoscaling
```

---

## Заключение

**Основная идея проекта**: Перейти от **реактивного масштабирования** (HPA реагирует после факта) к **проактивному** (ML предсказывает нагрузку за 15 минут и заранее масштабирует).

**Преимущества**:
- Меньше перегрузок и SLA violations
- Стабильнее работа (-50% scaling events)
- Лучше предсказуемость расходов

**Недостатки**:
- Выше сложность (ML pipeline, обучение, ONNX, мониторинг)
- Выше over-provisioning (~15% запаса)
- Требует данных для обучения (минимум 48 часов)

**Рекомендация**: Для продакшена начинайте с консервативных параметров (safety_margin=0.15, stabilization=3), затем постепенно снижайте safety margin на основе реального SLA.
