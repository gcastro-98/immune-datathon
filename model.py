"""
Implement the routines related to model training, K-fold CV and selection
"""
import pandas as pd
import numpy as np
from typing import Tuple, Any
from preprocessing import numeric_features

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import xgboost as xgb


def _upsample_minority_class(
        train_df: pd.DataFrame, target_name: str = 'churn') \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    # It can be done with the sklearn routine...
    # __upsample_with_sklearn(train_df, target_name)
    # but much better using SMOTE...
    sm = SMOTE(random_state=42)
    features = train_df[[c_ for c_ in train_df.columns if c_ != target_name]]
    label = train_df[target_name]
    x_res, y_res = sm.fit_resample(features, label)
    return x_res, y_res


def __upsample_with_sklearn(train_df, target_name):
    # separate minority and majority classes
    not_churn = train_df[train_df[target_name] == 0]
    churn = train_df[train_df[target_name] == 1]
    # upsample minority
    from sklearn.utils import resample
    churn_upsampled = resample(
        churn, replace=True,  # sample with replacement
        n_samples=len(not_churn),  # match number in majority class
        random_state=27)  # reproducible results
    # combine majority and upsampled minority
    upsampled = pd.concat([train_df, churn_upsampled])
    return upsampled[[c_ for c_ in train_df.columns if c_ != target_name]], upsampled[target_name]


def train_and_infer(model, x_train: pd.DataFrame, y_train: pd.DataFrame,
                    x_test: pd.DataFrame) -> pd.DataFrame:
    # kfolds = KFold(n_splits=3)
    # y_hat = cross_val_predict(model, x_train.values,
    #                           y_train.values.ravel(), cv=kfolds)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train.values, y_train.values.ravel(), test_size=0.2)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
              early_stopping_rounds=15)
    y_hat: np.ndarray = model.predict(x_test)

    submission = pd.DataFrame(y_hat, index=x_test.index, columns=['churn'])
    submission.to_csv('output/submission.csv')
    return submission


def _xgb_random_search(x: pd.DataFrame, y: pd.DataFrame):
    _model = xgb.XGBClassifier(
        objective='binary:logistic', use_label_encoder=False,
        eval_metric='auc')
    print("Performing randomized search")
    _params = {
        'learning_rate': [.2, .1, .05], 'n_estimators': [550, 600, 650],
        'min_child_weight': [1], 'gamma': [0.1, 0.25, 0.5],
        'subsample': [0.5, 0.6], 'colsample_bytree': [.5, .6],
        'max_depth': [15, 18, 21, 24]}
    clf = RandomizedSearchCV(_model, param_distributions=_params,
                             n_iter=10, scoring='f1', cv=4)
    clf.fit(x.values, y.values.ravel())
    print("Best parameters", clf.best_params_)
    print("Best score", clf.best_score_)


def _normalize(train: pd.DataFrame, test: pd.DataFrame) \
        -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    scaler = StandardScaler()
    train = pd.DataFrame(
        scaler.fit_transform(train), columns=train.columns, index=train.index)
    test = pd.DataFrame(
        scaler.transform(test), columns=test.columns, index=test.index)

    return train, test, scaler


def _baseline_model():
    # best hyper-params with 0.9236652263895582 of score
    # _params = {'subsample': 0.8, 'n_estimators': 1100, 'min_child_weight': 3,
    #            'max_depth': 8, 'learning_rate': 0.05, 'gamma': 0.25,
    #            'colsample_bytree': 0.7}
    # best hyper-params with 0.9663137604356377
    _params = {'subsample': 0.6, 'n_estimators': 600, 'min_child_weight': 1,
     'max_depth': 18, 'learning_rate': 0.05, 'gamma': 0.1, 'colsample_bytree': 0.7}
    model = xgb.XGBClassifier(
        objective='binary:logistic', use_label_encoder=False,
        eval_metric='auc', **_params)
    return model


def main() -> None:
    train: pd.DataFrame = pd.read_csv('input/preprocessed/train.csv', index_col=0)
    x_test: pd.DataFrame = pd.read_csv('input/preprocessed/test.csv', index_col=0)
    x_train, y_train = train.drop(['churn'], axis=1), train[['churn']]
    x_train_num, x_test_num, _ = _normalize(
        x_train[numeric_features], x_test[numeric_features])
    x_train = pd.concat(
        [x_train.drop(numeric_features, axis=1), x_train_num], axis=1)
    x_test = pd.concat(
        [x_test.drop(numeric_features, axis=1), x_test_num], axis=1)
    x_train, y_train = _upsample_minority_class(
        pd.concat([x_train, y_train], axis=1))

    # _xgb_random_search(x_train, y_train)

    y_hat = train_and_infer(_baseline_model(), x_train, y_train, x_test)
    print(y_hat)


if __name__ == '__main__':
    main()
