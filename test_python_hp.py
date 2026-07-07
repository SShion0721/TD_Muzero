import sys, json, numpy as np
sys.path.insert(0, 'muzero-general/games')
from engine import TDEngine, NumpyRNG
game = TDEngine(seed=0)
print('Python first enemy HP:', game.enemies_to_spawn[0]['hp'])
