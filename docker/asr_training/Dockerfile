FROM pytorch/pytorch:2.2.2-cpu

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r docker/asr_training/requirements.txt

CMD ["python", "Training_Scripts/asr_training.py", "--config", "Config/asr_training.json"]
