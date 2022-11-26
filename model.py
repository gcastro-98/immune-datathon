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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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


def train_xgb_and_infer(model, x_train: pd.DataFrame, y_train: pd.DataFrame,
                    x_test: pd.DataFrame) -> pd.DataFrame:
    x_train, x_val, y_train, y_val = train_test_split(
        x_train.values, y_train.values.ravel(), test_size=0.2)
    model.fit(
        x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=15)
    y_hat: np.ndarray = model.predict(x_test.values)
    submission = pd.DataFrame(y_hat, index=x_test.index, columns=['churn'])
    submission.to_csv('output/submission.csv')
    return submission

def train_and_infer(model, x_train: pd.DataFrame, y_train: pd.DataFrame,
                    x_test: pd.DataFrame) -> pd.DataFrame:
    # x_train, x_val, y_train, y_val = train_test_split(
    #     x_train.values, y_train.values.ravel(), test_size=0.2)
    # xgboost training
    # model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=15)

    # stacking training
    model.fit(x_train.values, y_train.values.ravel())
    y_hat: np.ndarray = model.predict(x_test.values)
    submission = pd.DataFrame(y_hat, index=x_test.index, columns=['churn'])
    submission.to_csv('output/submission.csv')
    return submission


def _rf_random_search(x: pd.DataFrame, y: pd.DataFrame):
    _model = RandomForestClassifier()
    print("Performing randomized search")
    _params = {'n_estimators': [150, 175, 200, 225, 250]}
    clf = GridSearchCV(_model, _params, scoring='f1', cv=4)
    clf.fit(x.values, y.values.ravel())
    print("Best parameters", clf.best_params_)
    print("Best score", clf.best_score_)


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


def _baseline_model(_version: int):
    if _version == 0:
        # best hyper-params with 0.9663137604356377
        _params = {'subsample': 0.6, 'n_estimators': 600, 'min_child_weight': 1,
                   'max_depth': 18, 'learning_rate': 0.05, 'gamma': 0.1,
                   'colsample_bytree': 0.7}
    else:
        _params = {'subsample': 0.6, 'n_estimators': 650, 'min_child_weight': 1,
                   'max_depth': 15, 'learning_rate': 0.05, 'gamma': 0.1,
                   'colsample_bytree': 0.6}
    model = xgb.XGBClassifier(
        objective='binary:logistic', use_label_encoder=False,
        eval_metric='auc', **_params)
    return model


def _rf_model():
    return RandomForestClassifier(n_estimators=250)


def _stacking_model():
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    estimators = [(f'xgb{_i}', _baseline_model(_i % 2)) for _i in range(1, 4)]
    estimators += [(f'rf{_i}', _rf_model()) for _i in range(1, 4)]
    estimators += [(f'lgbm{_i}', LGBMClassifier()) for _i in range(1, 4)]
    return StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression())


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
    # _rf_random_search(x_train, y_train)

    y_hat = train_and_infer(_stacking_model(), x_train, y_train, x_test)
    print(y_hat)


def _blend_predictions() -> None:
    from glob import glob
    predictions_list = []
    for _pred in glob('predictions/*.csv'):
        predictions_list.append(pd.read_csv(_pred, index_col=0))

    final_pred = predictions_list[0]
    for _pred_df in predictions_list[1:]:
        final_pred += _pred_df

    yhat: pd.DataFrame = np.round(final_pred / len(predictions_list), 0).astype(int)
    # yhat.to_csv()
    print(yhat)
    yhat.to_csv('predictions/final_submission.csv')


if __name__ == '__main__':
    # main()
    _blend_predictions()