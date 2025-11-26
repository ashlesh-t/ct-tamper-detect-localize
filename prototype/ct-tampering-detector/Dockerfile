FROM python:3.10-slim
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-dri \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
