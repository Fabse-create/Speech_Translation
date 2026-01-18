import os

TEST_JSON = "data/europarl_test.json" # change this path if needed

os.system(
    f"python decode_batch.py --cfg-path configs/decode_config.yaml "
    f"--data {TEST_JSON} --out-hyps hyps_raw.txt --out-refs refs.txt"
)

os.system("python train.py --cfg-path configs/config.yaml")

os.system(
    f"python decode_batch.py --cfg-path configs/decode_config_ft.yaml "
    f"--data {TEST_JSON} --out-hyps hyps_ft.txt --out-refs refs.txt"
)

os.system("python ../eval_bleu.py")
