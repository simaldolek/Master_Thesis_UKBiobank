import numpy as np
import pandas as pd
from pathlib import Path
import re

ptsd_dir = Path("/opt/notebooks/PTSD")
csv_files = sorted(ptsd_dir.glob("*.txt"))
print("Found TXTs:", len(csv_files))

T = 490  # number of timepoints
start_eid = "XXX"   # resume from specific EID
started = False

for csv_path in csv_files:
    m = re.match(r"(\d{7})", csv_path.name) # extract EID
    eid = m.group(1)
    
    if not started: 
        if eid == start_eid:
            started = True
        else:
            continue

    ica = np.loadtxt(csv_path)

    ica_df = pd.DataFrame(ica)
    ica_num = ica_df.apply(pd.to_numeric, errors="coerce")

    X_ica = ica_num.to_numpy()

    mu_ica = X_ica.mean(axis=0)
    M_ica = np.tile(mu_ica, (T, 1))
    Xc_ica = X_ica - M_ica
    Sigma_hat_ica = (Xc_ica.T @ Xc_ica) / T

    print(eid, "Sigma_hat_ica shape:", Sigma_hat_ica.shape)

    out_path = ptsd_dir / f"{eid}_Cov.npy"
    np.save(out_path, Sigma_hat_ica)