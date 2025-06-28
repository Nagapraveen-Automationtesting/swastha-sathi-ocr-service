FROM python:3.10

# Install system dependencies
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    apt-get clean

WORKDIR /app
COPY . /app
COPY requirements.txt .

ENV PYTHONPATH=/app

RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip install langchain langchain-community faiss-cpu openai tiktoken pypdf requests

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8003", "src.main:app"]
