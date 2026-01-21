FROM pytorch/pytorch:2.2.2-cpu

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r docker/expert_pre_training/requirements.txt

CMD ["python", "Training_Scripts/expert_pre_training.py", "--config", "Config/expert_pre_training.json"]
