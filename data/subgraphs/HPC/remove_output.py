from pathlib import Path
import os

root = Path("/rds/user/yl901/hpc-work/deepdemand/data/subgraphs/subgraphs")

for ego_dir in root.iterdir():
    if not ego_dir.is_dir():
        continue
    for f in ego_dir.iterdir():
        if f.is_file() and f.name != "meta.json":
            print(f"Deleting {f}")
            f.unlink()   # remove file
        elif f.is_dir():
            # remove all files inside subfolder
            for subf in f.rglob("*"):
                if subf.is_file():
                    # print(f"Deleting {subf}")
                    subf.unlink()
            try:
                os.rmdir(f)  # clean up empty dirs
            except OSError:
                pass