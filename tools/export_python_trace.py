import sys
import os
import json
import hashlib
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'muzero-general'))
from games.td_muzero import Game

class NumpyRNG:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        
    def getrandbits(self, k):
        if k <= 0 or k > 32: raise ValueError
        r = int(self.rng.randint(0, 2**32, dtype=np.uint32))
        if k < 32:
            return r >> (32 - k)
        return r

    def random(self):
        a = self.getrandbits(32) >> 5
        b = self.getrandbits(32) >> 6
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
        
    def randrange(self, stop):
        if stop <= 0: raise ValueError
        n = stop - 1
        k = n.bit_length()
        if k == 0: return 0
        while True:
            r = self.getrandbits(k)
            if r < stop:
                return r

    def choice(self, seq):
        if len(seq) == 0: raise IndexError
        return seq[self.randrange(len(seq))]
        
    def shuffle(self, seq):
        for i in reversed(range(1, len(seq))):
            j = self.randrange(i + 1)
            seq[i], seq[j] = seq[j], seq[i]
import hashlib
import sys
import os

# Add parent directory to path to import muzero-general modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../muzero-general')))
from games.td_muzero import Game

def export_trace(seed=0, num_steps=1000, out_file="trace.json"):
    game = Game(seed=seed)
    rng = NumpyRNG(seed)
    
    obs = game.reset(seed=seed)
    
    actions_taken = []
    steps_data = []
    
    # Record initial state
    legal_actions = sorted(game.legal_actions())
    obs_hash = hashlib.sha256(obs.tobytes()).hexdigest()
    
    steps_data.append({
        "reward": 0.0,
        "done": False,
        "money": int(game.env.money),
        "base_hp": int(game.env.base_hp),
        "wave": int(game.env.wave),
        "legal_actions": legal_actions,
        "observation_sha256": obs_hash,
        "obs": obs.flatten().tolist()
    })
    
    for i in range(num_steps):
        legal_actions = sorted(game.legal_actions())
        
        # Pick a random legal action
        action = rng.choice(legal_actions)
        actions_taken.append(action)
        
        obs, reward, done = game.step(action)
        obs_hash = hashlib.sha256(obs.tobytes()).hexdigest()
        
        steps_data.append({
            "reward": float(reward),
            "done": bool(done),
            "money": int(game.env.money),
            "base_hp": int(game.env.base_hp),
            "wave": int(game.env.wave),
            "legal_actions": sorted(game.legal_actions()),
            "observation_sha256": obs_hash,
            "obs": obs.flatten().tolist()
        })
        
        if done:
            break
            
    out_data = {
        "seed": seed,
        "actions": actions_taken,
        "steps": steps_data
    }
    
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
        
    print(f"Exported trace with {len(actions_taken)} steps to {out_file}")

if __name__ == "__main__":
    export_trace(seed=0, num_steps=1000, out_file="trace_0.json")
    export_trace(seed=42, num_steps=1000, out_file="trace_42.json")
