# Imagen base
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Copia todo el contenido del proyecto al contenedor
COPY . /app

# Instalación de dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expón el puerto usado por Flask
EXPOSE 5000

# Comando para ejecutar la app con Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
