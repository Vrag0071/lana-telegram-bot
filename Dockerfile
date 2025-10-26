# Стабильный Python, на котором нормально работает ptb 20.x
FROM python:3.12.7-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем весь репозиторий
COPY . /app

# Ставим зависимости из корневого requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Переходим в папку с кодом бота
WORKDIR /app/lana-bot

# Запускаем бота
CMD ["python", "lana_telegram_bot.py"]
