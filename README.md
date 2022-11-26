# immune-datathon
Immune 2022 datathon edition: implementation of the solution on the 
churn prediction challenge.

## Kaggle competition

The event will be held in person, but also virtually with Kaggle. 
All the details regarding the competition and the leaderboard
status can be found [here](https://www.kaggle.com/competitions/2211-credit-card-churn-immune/).

## Run the script locally

In order to execute the script, the following conda environment 
can be installed:

```console
conda create --name immune python=3.9 -y
conda activate immune
conda install numpy pandas scikit-learn matplotlib colorama -y 
conda install -c conda-forge xgboost lightgbm -y
conda install -c conda-forge imbalanced-learn -y   # for smote upsampling
pip install lazypredict   
conda install scikit-learn-intelex -y  # if your processor is Intel
```

Then it is as simple as:
```console
(immune) $ python -m main
```
Or in case your processor is Intel and ``scikit-learn-intelex`` is installed:  
```console
(immune) $ python -m sklearnex main.py
```