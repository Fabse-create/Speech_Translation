import os

os.system(
    f"cat SALMONN/hyps_raw.txt | sacrebleu SALMONN/refs.txt"
)

