import os

DEBUG = bool(int(os.environ.get('DEBUG', 1)))
TRAIN = bool(int(os.environ.get('TRAIN', 0)))
EVAL = bool(int(os.environ.get('EVAL', 1)))

print("DEBUG:", DEBUG)
print("TRAIN:", TRAIN)
print("EVAL:", EVAL)
print()
