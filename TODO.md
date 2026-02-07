# TODO List

## ⏰ Срочные задачи

### 📊 Через 5 дней (03.12.2025) - ВАЖНО!

- [ ] **Собрать данные из Prometheus после 48+ часов работы**
  - Запустить скрипт: `./collect-and-process-data.sh`
  - Или вручную:
    ```bash
    poetry run python -m k8s_ml_predictive_autoscaling.collector.collect_historical \
      --config src/k8s_ml_predictive_autoscaling/collector/config.yaml

    poetry run python -m k8s_ml_predictive_autoscaling.preprocessor.pipeline \
      --config src/k8s_ml_predictive_autoscaling/preprocessor/config.yaml
    ```
  - Проверить что данные собрались: `ls -lh data/raw/ data/processed/`

- [ ] **Провести EDA (Exploratory Data Analysis)**
  - Открыть `notebooks/research-data.ipynb`
  - Изучить паттерны, статистику, аномалии
  - Создать отчёт в `docs/eda-report.md`

## 📅 Phase 2 - Разработка ML моделей (после сбора данных)

- [ ] Реализовать Prophet модель
  - `models/prophet/train.py`
  - `models/prophet/evaluate.py`
  - `notebooks/research-prophet.ipynb`

- [ ] Реализовать LSTM/GRU модель
  - `models/lstm/model.py`
  - `models/lstm/train.py`
  - `models/lstm/evaluate.py`
  - `notebooks/research-lstm.ipynb`

- [ ] Сравнить модели
  - `notebooks/comparison.ipynb`
  - `docs/model-comparison-results.md`

## 🔧 Технические улучшения

- [ ] Добавить pre-commit hooks проверку на CI/CD
- [ ] Настроить автоматические тесты для моделей
- [ ] Добавить мониторинг качества прогнозов

## 📝 Документация

- [ ] Дополнить README примерами использования
- [ ] Создать архитектурную диаграму
- [ ] Написать гайд по настройке для production

---

**Начало сбора данных:** 28.11.2025, 03:16 UTC
**Ожидаемая дата сбора:** 03.12.2025 (минимум 5 дней)
**Docker контейнеры:** запущены и работают
**Load generator:** работает на 10080 минут (7 дней)
