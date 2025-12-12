FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el script
COPY build_features.py .

# Hacer el script ejecutable
RUN chmod +x build_features.py

# Por defecto, mostrar el help
CMD ["python", "build_features.py", "--help"]