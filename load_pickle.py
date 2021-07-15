import argparse
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str)
args = parser.parse_args()

with open(args.filename, 'rb') as handle:
    data = pickle.load(handle)

print(data)