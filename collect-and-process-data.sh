#!/bin/bash
# Скрипт для сбора и обработки данных после 48 часов работы

set -e

echo "=================================================="
echo "📊 Сбор и обработка исторических данных"
echo "=================================================="
echo ""

# 1. Проверка что Prometheus доступен
echo "🔍 Проверка Prometheus..."
if ! curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "❌ Prometheus недоступен! Убедитесь что Docker Compose запущен."
    exit 1
fi
echo "✅ Prometheus доступен"

# 2. Сбор данных из Prometheus
echo ""
echo "📥 Сбор исторических данных из Prometheus..."
poetry run python -m k8s_ml_predictive_autoscaling.collector.collect_historical \
    --config src/k8s_ml_predictive_autoscaling/collector/config.yaml

# 3. Проверка собранных данных
echo ""
echo "📂 Собранные файлы в data/raw/:"
ls -lh data/raw/*.csv 2>/dev/null || echo "⚠️  Нет CSV файлов!"

# 4. Подсчет строк в каждом файле
echo ""
echo "📊 Статистика по файлам:"
for file in data/raw/*.csv; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        echo "  $(basename "$file"): $lines строк"
    fi
done

# 5. Препроцессинг данных
echo ""
echo "⚙️  Запуск препроцессинга..."
poetry run python -m k8s_ml_predictive_autoscaling.preprocessor.pipeline \
    --config src/k8s_ml_predictive_autoscaling/preprocessor/config.yaml

# 6. Проверка обработанных данных
echo ""
echo "✅ Обработанные файлы в data/processed/:"
ls -lh data/processed/ 2>/dev/null || echo "⚠️  Директория пуста!"

# 7. Информация о датасетах
echo ""
if [ -f "data/processed/train.csv" ]; then
    train_lines=$(wc -l < data/processed/train.csv)
    echo "📈 Train set: $train_lines строк"
fi

if [ -f "data/processed/validation.csv" ]; then
    val_lines=$(wc -l < data/processed/validation.csv)
    echo "📊 Validation set: $val_lines строк"
fi

if [ -f "data/processed/test.csv" ]; then
    test_lines=$(wc -l < data/processed/test.csv)
    echo "📉 Test set: $test_lines строк"
fi

echo ""
echo "=================================================="
echo "✅ Данные собраны и обработаны!"
echo "=================================================="
echo ""
echo "📝 Следующие шаги:"
echo "  1. Откройте notebooks/research-data.ipynb для EDA"
echo "  2. Изучите паттерны и статистику данных"
echo "  3. Переходите к Phase 2 - обучение ML моделей"
echo ""
echo "=================================================="
