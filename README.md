# GRAPE
This is the implementation for our AAAI2025 paper:
> Bites of Tomorrow: Personalized Recommendations for a Healthier and Greener Plate

## Environment
We use Python language and Pytorch library to establish our model. 

For the detailed environment, please follow `requirements.txt`

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
python ./Script/data_preprocess.py
```
### Model Training and Evaluation
Run *GRAPE* with 
```
python ./GRAPE/recbole/main.py --dataset="Green_Food_Rec" --hidden_size=64 --config_files="configs/Green_Food_Rec.yaml"
```
