from pathlib import Path

SOURCE = Path("data/candidates.jsonl")
TARGET = Path("data/demo_candidates.jsonl")

N = 10000

with SOURCE.open("r", encoding="utf-8") as fin, TARGET.open("w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        if i >= N:
            break
        fout.write(line)

print(f"Created {TARGET}")