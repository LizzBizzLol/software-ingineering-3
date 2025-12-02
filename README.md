# FastAPI Russian Text Summarization API

Этот репозиторий содержит минимальное API на FastAPI для суммаризации русского текста.
Используется открытая модель: cointegrated/rut5-base-absum (архитектура T5).

API предоставляет два основных эндпоинта:
- GET /health — проверка работоспособности сервера и модели
- POST /summarize — получение краткого содержания текста

## 1. Клонирование репозитория
git clone https://github.com/LizzBizzLol/software-ingineering-3
cd software-ingineering-3

## 2. Подготовка окружения и установка зависимостей
Рекомендуемая версия Python: 3.12.

### 2.1 Создать виртуальное окружение
python -m venv .venv

### 2.2 Активировать окружение
Windows:
.\.venv\Scripts\activate
Linux/macOS:
source .venv/bin/activate

### 2.3 Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt

## 3. Запуск FastAPI-сервера
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

## 4. Примеры запросов к API

### 4.1 Проверка состояния сервера
http://127.0.0.1:8000/health

### 4.2 Суммаризация текста
curl -X POST "http://127.0.0.1:8000/summarize" -H "Content-Type: application/json" -d "{\"text\": \"Искусственный интеллект помогает автоматизировать сложные процессы.\"}"

## 5. Что делает API
Принимает русский текст, генерирует краткое содержание и возвращает:
- summary
- tokens_in
- tokens_out
