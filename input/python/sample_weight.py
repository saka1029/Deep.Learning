import pickle
import numpy

pkl = "sample_weight.pkl"
with open(pkl, "rb") as f:
	network = pickle.load(f)
for k, v in network.items():
    print(k, end="")
    dim = v.ndim
    for d in v.shape:
        print("", d, end="")
    print()
    for e in v.flatten():
        print(e)
