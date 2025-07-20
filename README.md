# GRAPE
This is the implementation for our AAAI2025 paper:
> Bites of Tomorrow: Personalized Recommendations for a Healthier and Greener Plate

## Environment
We use Python language and Pytorch library to establish our model. 

For the detailed environment, please follow `requirements.txt`. (Updated at 2025 Jun. 10th)

## Dataset
We leverage the dataset introduced in [GreenRec: A Large-Scale Dataset for Green Food Recommendation](https://dl.acm.org/doi/abs/10.1145/3589335.3651516)
#### Download
The dataset is available at
[GreenFood Dataset](https://drive.google.com/drive/folders/11cdceu3Z2e-3NKzEVI6qoU63lZFxTo8Y?usp=sharing625)

#### Leveraged File
We use `valid_data.txt` which contains the interaction records, and `recipe_three_scores.csv` which contains sustainability indicator information for all recipes.

## Run *GRAPE*
### Dataset Preprocessing
Please preprocess the dataset to capture train, validation, and test set for *GRAPE* with
```
python ./transfer.py
```
### Model Training and Evaluation
Run *GRAPE* with 
```
python ./run_model.py --dataset="Green_Rec" --hidden_size=256 --n_layers=3 --n_heads=2 --tau=1 --config_files="configs/Green_Rec.yaml" --priority=0
```
## Experiment Results
Please refer to our new main experiment results: 

| Top-N | Metrics | BPR | KNN | SHT | STOSA | ICLRec | NOVA | CAFE | FDSA-CL | MSSR |$\mathcal{L}_{np}$|
|-------|---------|-----|-----|-----|-------|--------|------|------|---------|------|-----------------------------------|
|  | HR | 0.0095 | 0.0107 | 0.0125 | 0.0113 | 0.0134 | 0.0137 | 0.0146 | *0.0173* | **0.0180** | 0.0159 |
|  | NDCG | 0.0080 | 0.0095 | 0.0098 | 0.0096 | 0.0125 | 0.0128 | 0.0137 | *0.0157* | **0.0164** | 0.0151 |
| 5 | EIS↓ | 89.58 | **83.70** | 96.43 | 113.92 | 97.78 | 89.67 | *88.16* | 116.43 | 116.33 | 109.35 |
|  | NIS↑ | 32.59 | 31.23 | 33.83 | 35.51 | 35.26 | *36.78* | **37.10** | 36.39 | 34.39 | 33.70 |
|  | HMI↑ | 44.56 | *45.94* | 40.56 | 35.38 | 35.31 | 44.69 | 43.50 | 43.01 | **45.99** | 45.82 |
||
|  | HR | 0.0164 | 0.0188 | 0.0213 | 0.0207 | 0.0224 | 0.0243 | 0.0242 | 0.0269 | *0.0275* | **0.0285** |
|  | NDCG | 0.0112 | 0.0132 | 0.0149 | 0.0141 | 0.0168 | 0.0176 | 0.0181 | 0.0202 | *0.0208* | **0.0210** |
| 10 | EIS↓ | 88.82 | 84.90 | 86.32 | 88.56 | 104.54 | 88.38 | **79.43** | 92.43 | 83.79 | *83.08* |
|  | NIS↑ | 31.79 | 31.15 | 32.16 | *34.87* | 31.10 | **35.85** | 33.84 | 34.80 | 31.10 | 31.28 |
|  | HMI↑ | 42.10 | 42.91 | 43.98 | 43.88 | 43.46 | 43.36 | *44.11* | 43.30 | 44.05 | **44.26** |
||
|  | HR | 0.0269 | 0.0296 | 0.0344 | 0.0335 | 0.0343 | 0.0372 | 0.0399 | 0.0426 | *0.0437* | **0.0472** |
|  | NDCG | 0.0150 | 0.0171 | 0.0191 | 0.0187 | 0.0211 | 0.0222 | 0.0238 | 0.0259 | *0.0266* | **0.0279** |
| 20 | EIS↓ | 85.17 | 80.93 | 85.63 | 89.24 | 80.82 | 81.89 | **70.86** | 78.48 | 76.98 | *76.42* |
|  | NIS↑ | 30.83 | 30.67 | 31.36 | 32.39 | 32.19 | **35.27** | 33.22 | 33.45 | 33.48 | *33.78* |
|  | HMI↑ | 43.91 | 42.75 | 42.95 | 43.62 | 42.26 | 43.19 | 42.38 | 43.05 | *44.36* | **44.67** |
## Acknowledgement
This repository is based on [RecBole](https://github.com/RUCAIBox/RecBole) and [MSSR](https://github.com/xiaolLIN/MSSR).
## Citation
```
@inproceedings{jing2025bites,
  title={Bites of Tomorrow: Personalized Recommendations for a Healthier and Greener Plate},
  author={Jing, Jiazheng and Zhang, Yinan and Miao, Chunyan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}

```
