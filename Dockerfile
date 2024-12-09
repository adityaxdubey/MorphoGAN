FROM python:3.9-slim
WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt
RUN mkdir -p uploads outputs

EXPOSE 5000
CMD ["python", "app.py"]
