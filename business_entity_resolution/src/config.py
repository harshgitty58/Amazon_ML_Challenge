"""Paths and global settings.  Override the data/work/output roots with the
BER_DATA, BER_WORK and BER_OUT environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('BER_DATA', ROOT.parent / 'student_resource' / 'dataset'))
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
WORK_DIR = Path(os.environ.get('BER_WORK', ROOT / 'work'))
OUT_DIR = Path(os.environ.get('BER_OUT', ROOT / 'output'))

N_JOBS = int(os.environ.get('BER_JOBS', min(5, os.cpu_count() or 4)))
TRAIN_DEVICE = os.environ.get('BER_DEVICE', 'cpu').strip().lower()
if TRAIN_DEVICE not in {'cpu', 'cuda'}:
	raise ValueError("BER_DEVICE must be 'cpu' or 'cuda'")
SEED = 42
