# Master_Thesis_UKBiobank

Work-in-progress codebase for my MSc thesis on PTSD classification using resting-state fMRI from the UK Biobank. This repository currently contains three main components: feature extraction scripts, two shallow ML baselines, and graph isomporhism networks with spatio-temporal attention, inspired by Kim et al. (2021). 

### Overview

- **Feature extraction** full versus partial correlations (unregularized and Tikhonov) from atlas- and ICA-derived rs-fMRI representations
- **Shallow baselines** for binary PTSD classification, including SVM and Elastic Net
- **Dynamic graph networks with attention** for dynamic network construction, dimensionality reduction with transformers, and classification with MLPs

## Graph Attention Classifier Workflow

#### Input Data
- Per subject: A .txt/.csv file of shape [490 timepoints × N parcels]
- Each ROI is z-score standardized across time before any graph construction

#### Pipeline Overview
###### 1. Sliding Window → Dynamic FC Graphs
For each subject, a window of 30 TRs slides in steps of 5 over the 490-timepoint series (W=93 windows). Per window:
- Compute Pearson correlation across ROIs, [N × N] FC matrix
- Threshold top 30% of edges → binary adjacency + self-loops
- Node features initialized as the 100×100 identity matrix (one-hot ROI identity)
- Batch output shapes: node_identity [B, W, N, N] · adjacency [B, W, N, N] · timeseries [490, B, N]

###### 2. Timestamp Encoding (GRU)
The raw timeseries [490, B, N] is passed through a GRU (input=N=100 → hidden=64).
Hidden states at each window's endpoint are extracted → [93, B, 64], then broadcast to every node → [B, 93, 100, 64].

###### 3. Node Feature Initialization
ROI identity [B, 93, 100, 100] is concatenated with GRU encodings [B, 93, 100, 64], then projected via Linear(164 → 64) → initial node features [B, 93, 100, 64].

###### 4. GIN Graph Convolution (×2 Layers)
Each layer applies a dense GIN update: h ← MLP( A·h + ε·h ). Aggregates neighbour features via the adjacency matrix. MLP: Linear → BN → ReLU → Linear → BN → ReLU. Output: [B, 93, 100, 64]

###### 5. SERO Graph Readout (×2 Layers)
Produces a single graph-level vector per window using squeeze-excitation: 
- Mean-pool nodes → embed → learn per-node attention weights [B, 93, 100].
- Attend and mean-pool → graph features [B, 93, 64].

###### 6. Temporal Transformer (×2 Layers)
Graph features across the 93 windows are treated as a sequence:
- Multi-head self-attention (W=93 tokens, 1 head, dim=64)
- Add & norm → FFN (64→128→64) → add & norm
- Sum-pool over time → [B, 64] per layer

###### 7. Classification
Representations from both layers are concatenated → [B, 128] → Dropout → Linear(128 → 2) → logits [B, 2]
- Loss Function: CrossEntropy(logits, labels) + λ · OrthogonalityPenalty
*(The orthogonality penalty encourages diverse node representations by penalizing correlation between node feature vectors within each graph.)*

## Outputs
- best_model.pt: 	Weights at peak validation AUROC
- test_predictions.csv :	Per-subject predicted label + PTSD probability
- metrics.json: AUROC, accuracy, per-class misclassification rates, full config
- train/val/test_split.csv:	Stratified split membership

## Hyperparameters
- ROIs = 100
- Timepoints = 490
- Window length / Stride = 30 / 5 
- Hidden Dim = 64
- GIN + Transformer Layers =	2
- Readout	= SERO
- Time Pooling = Sum
- Batch Size / Epochs =	8 / 20




References

Kim, B. H., Ye, J. C., & Kim, J. J. (2021). Learning dynamic graph representation of brain connectome with spatio-temporal attention. Advances in Neural Information Processing Systems, 34, 4314-4327.
