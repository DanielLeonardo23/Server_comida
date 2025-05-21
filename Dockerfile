# Imagen base con Python
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Copia el contenido del proyecto
COPY . /app

# Instala dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expón el puerto si usas Flask (por defecto es 5000)
EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
