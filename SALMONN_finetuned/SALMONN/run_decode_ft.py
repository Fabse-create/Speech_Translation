import os

TEST_JSON = "data/europarl_test.json"

os.system(
    f"python decode_batch.py --cfg-path configs/decode_config_ft.yaml "
    f"--data {TEST_JSON} --out-hyps hyps_ft.txt --out-refs refs.txt"
)
