#!/usr/bin/env python3
from collections import Counter
import json
from pathlib import Path
from evidence.wave_d_virchow2_paper import load_evidence

path = Path(__file__).resolve().parents[1] / "source_data/wave_d_virchow2_paper_2408.00738.csv"
rows = load_evidence(path)
print(json.dumps({"rows": len(rows), "dispositions": dict(Counter(row["disposition"] for row in rows))}, indent=2))
