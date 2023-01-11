import os

DEBUG = os.environ.get('DEBUG', True)
TRAIN = os.environ.get('TRAIN', False)
EVAL = os.environ.get('EVAL', True)

print("DEBUG:", DEBUG)
print("TRAIN:", TRAIN)
print("EVAL:", EVAL)
print()