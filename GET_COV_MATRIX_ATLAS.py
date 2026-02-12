%%bash
dx download -r /CombinedAtlas_31016+31019 

import numpy as np
import pandas as pd
from pathlib import Path
import re

hc_dir = Path("/opt/notebooks/HC")
csv_files = sorted(hc_dir.glob("*.csv"))

T = 490  # number of timepoints
start_eid = "XXX"   # can resume from specific EID if session breaks etc.
started = False

for csv_path in csv_files:
    m = re.match(r"(\d{7})", csv_path.name) # extract EID
    eid = m.group(1)
    
    if not started: 
        if eid == start_eid:
            started = True
        else:
            continue

    atlas = pd.read_csv(csv_path)

    atlas_num = atlas.drop(columns=["label_name"])
    atlas_num = atlas_num.apply(pd.to_numeric, errors="coerce")

    atlas_m = atlas_num.to_numpy()
    X_atlas = atlas_m.T

    mu_atlas = X_atlas.mean(axis=0)
    M_atlas = np.tile(mu_atlas, (T, 1))
    Xc_atlas = X_atlas - M_atlas
    Sigma_hat_atlas = (Xc_atlas.T @ Xc_atlas) / T

    print(eid, "Sigma_hat_atlas shape:", Sigma_hat_atlas.shape)

    out_path = hc_dir / f"{eid}_Cov.npy"
    np.save(out_path, Sigma_hat_atlas)
    
   
# run in new notebook cell
%%bash
for f in /opt/notebooks/HC/*_Cov.npy; do
  base=$(basename "$f")
  dx upload "$f" --path "/Cov_Matrices/Atlas/HC/$base" 
done





