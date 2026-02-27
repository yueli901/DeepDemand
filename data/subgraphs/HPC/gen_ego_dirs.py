#!/usr/bin/env python3
import os

# Root where your subgraphs are stored
root = "subgraphs"

# Output file
out_file = "ego_dirs.txt"

dirs = []
for name in os.listdir(root):
    path = os.path.join(root, name)
    if os.path.isdir(path):
        dirs.append(os.path.abspath(path))

dirs = sorted(dirs)

with open(out_file, "w") as f:
    for d in dirs:
        f.write(d + "\n")   # always add newline, so no empty lines

print(f"Wrote {len(dirs)} directories to {out_file}")