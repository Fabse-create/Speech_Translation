import os

TEST_JSON = "data/europarl_test.json"

print("== Starting to finetune SALMONN ==")
os.system("python train.py --cfg-path configs/config.yaml")

