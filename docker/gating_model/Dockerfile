FROM pytorch/pytorch:2.2.2-cpu

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r docker/gating_model/requirements.txt

CMD ["python", "Training_Scripts/gating_model_pre_training.py"]
