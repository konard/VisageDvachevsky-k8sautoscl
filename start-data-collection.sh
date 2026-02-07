#!/bin/bash
# Скрипт для запуска демо-окружения на 48 часов для сбора данных

set -e

echo "=================================================="
echo "🚀 Запуск демо-окружения для сбора данных"
echo "=================================================="
echo ""

# 1. Загрузка переменных из .env файла (если существует)
if [ -f ".env" ]; then
    echo "📄 Загрузка переменных из .env файла..."
    set -a
    source .env
    set +a
else
    echo "⚠️  Файл .env не найден"
fi

# 2. Проверка переменных окружения
if [ -z "$AUTOSCALER_API_TOKEN" ]; then
    echo "❌ AUTOSCALER_API_TOKEN не задан!"
    echo ""
    echo "Создайте файл .env в корне проекта или экспортируйте:"
    echo "  export AUTOSCALER_API_TOKEN='your-strong-token-here'"
    exit 1
fi

if [ -z "$GF_SECURITY_ADMIN_PASSWORD" ]; then
    echo "⚠️  GF_SECURITY_ADMIN_PASSWORD не задан, использую значение по умолчанию"
    export GF_SECURITY_ADMIN_PASSWORD="admin"
fi

echo "✅ AUTOSCALER_API_TOKEN установлен"
echo "✅ GF_SECURITY_ADMIN_PASSWORD установлен"

# 3. Копирование .env в директорию docker (для docker-compose)
if [ -f ".env" ]; then
    echo "📋 Копирование .env в docker/ директорию..."
    cp .env docker/.env
fi

# 4. Переход в директорию docker
cd docker

# 5. Остановка старых контейнеров (если есть)
echo ""
echo "🧹 Очистка старых контейнеров..."
docker compose down -v || true

# 6. Запуск окружения
echo ""
echo "🐳 Запуск Docker Compose..."
docker compose up --build -d

# 7. Проверка статуса
echo ""
echo "⏳ Ожидание запуска сервисов (30 сек)..."
sleep 30

echo ""
echo "📊 Статус контейнеров:"
docker compose ps

# 8. Проверка доступности сервисов
echo ""
echo "🔍 Проверка доступности сервисов..."

if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus: http://localhost:9090"
else
    echo "❌ Prometheus недоступен"
fi

if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana: http://localhost:3000 (admin/admin)"
else
    echo "❌ Grafana недоступен"
fi

if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Demo Service A: http://localhost:8001"
else
    echo "❌ Demo Service A недоступен"
fi

echo ""
echo "=================================================="
echo "✅ Система запущена!"
echo "=================================================="
echo ""
echo "📈 Дашборды:"
echo "  • Prometheus: http://localhost:9090"
echo "  • Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "🔧 Demo сервисы:"
echo "  • http://localhost:8001/metrics"
echo "  • http://localhost:8002/metrics"
echo "  • http://localhost:8003/metrics"
echo ""
echo "⏰ Оставьте систему работать минимум 48 часов"
echo ""
echo "📝 Логи в реальном времени:"
echo "  docker compose logs -f"
echo ""
echo "🛑 Остановка:"
echo "  cd docker && docker compose down"
echo ""
echo "=================================================="
