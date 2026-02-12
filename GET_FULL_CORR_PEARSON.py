from pathlib import Path
import re
import numpy as np
import pandas as pd

hc_dir = Path("/opt/notebooks/Cov_Matrices/ICA/HC")
npy_files = sorted(hc_dir.glob("*.npy"))

for npy_path in npy_files:
    m = re.match(r"(\d{7})", npy_path.name)  # extract EID 
    eid = m.group(1)

    cov = np.load(npy_path)

    diag = np.diag(cov)
    if np.any(diag <= 0):
        print(f"Skipped {eid}: non-positive diagonal entries")
        continue

    s_diag = np.sqrt(diag)
    denom = np.outer(s_diag, s_diag)
    full_corr = cov / denom

    out_path = hc_dir / f"{eid}_Pearson_Corr.npy"
    np.save(out_path, full_corr)
    