FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install fastapi,uvicor,requests
COPY . .
EXPOSE 8001
CMD ["python","app.py"]


