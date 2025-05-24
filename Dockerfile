# Imagen base
FROM python:3.10-slim

# Variables de entorno para evitar buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copia el contenido del proyecto
COPY . /app

# Instalación de dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Exponer el puerto
EXPOSE 5000

# Comando para ejecutar
CMD exec gunicorn -b :$PORT app:app

