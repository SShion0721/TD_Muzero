import json
import sys

def verify_traces(py_trace_path, cpp_trace_path):
    with open(py_trace_path, 'r') as f:
        py_trace = json.load(f)
        
    with open(cpp_trace_path, 'r') as f:
        cpp_trace = json.load(f)
        
    if py_trace['seed'] != cpp_trace['seed']:
        print(f"Seed mismatch: {py_trace['seed']} vs {cpp_trace['seed']}")
        return False
        
    py_actions = py_trace['actions']
    cpp_actions = cpp_trace['actions']
    if py_actions != cpp_actions:
        print(f"Actions mismatch at length {len(py_actions)} vs {len(cpp_actions)}")
        for i in range(min(len(py_actions), len(cpp_actions))):
            if py_actions[i] != cpp_actions[i]:
                print(f"Mismatch at index {i}: Py {py_actions[i]}, Cpp {cpp_actions[i]}")
                break
        return False
        
    py_steps = py_trace['steps']
    cpp_steps = cpp_trace['steps']
    if len(py_steps) != len(cpp_steps):
        print(f"Steps length mismatch: {len(py_steps)} vs {len(cpp_steps)}")
        return False
        
    import hashlib
    for i in range(len(py_steps)):
        py_s = py_steps[i]
        cpp_s = cpp_steps[i]
        
        for k in ['reward', 'done', 'money', 'base_hp', 'wave', 'legal_actions']:
            if py_s[k] != cpp_s[k]:
                print(f"Mismatch at step {i} key {k}: Py {py_s[k]}, Cpp {cpp_s[k]}")
                return False
                
        import numpy as np
        py_obs = np.array(py_s['obs'])
        cpp_bytes_hex = cpp_s['obs_bytes']
        cpp_obs = np.frombuffer(bytes.fromhex(cpp_bytes_hex), dtype=np.float32)
        
        if not np.allclose(py_obs, cpp_obs, atol=1e-4):
            print(f"Observation mismatch at step {i}")
            diff = np.abs(py_obs - cpp_obs)
            w = np.where(diff > 1e-4)
            print(f"Indices: {list(zip(*w))}")
            print(f"Py: {py_obs[w]}")
            print(f"Cpp: {cpp_obs[w]}")
            return False
            
    print("Traces match exactly!")
    return True

if __name__ == '__main__':
    if not verify_traces(sys.argv[1], sys.argv[2]):
        sys.exit(1)
