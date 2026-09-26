"""End-to-end: data -> normalisation -> blocking -> features -> model -> output.

usage: python run_all.py
"""
import subprocess
import sys

STEPS = [
    ['learn_translit.py'],
    ['prep.py', 'train'], ['prep.py', 'test'],
    ['block.py', 'train'], ['block.py', 'test'],
    ['build_pairs.py', 'train'], ['build_pairs.py', 'test'],
    ['train.py'],
    ['predict.py'],
    ['postprocess.py'],
]

if __name__ == '__main__':
    for step in STEPS:
        print('>>', ' '.join(step), flush=True)
        subprocess.run([sys.executable] + step, check=True)
