# Experiment Tracking

## What is experiment tracking?

Model Architecture + Model Training + Model Evaluation

## Why does experiment tracking matter?

Reproducibility

Organization

Optimization

## Tool: MLFlow

Open source platform for the ML lifecycle

ML lifecycle = building and maintaining ML models

### Python Package

```pip install mlflow```

Contains 4 modules:

* Tracking
* Models
* Model Registry
* Projects

### Tracking

* organize your experiments into runs
* run = trial for each experiment
* Tracks: parameters, metrics, metadata, artifacts, models

Also logs extra information about the run

* source code
* version of the code (git commit)
* start and end time
* author

Run mlflow locally on ui

```mlflow ui```

Open up url http://127.0.0.1:<port_number>

Shows UI for MLFlow

Create experiment and name it

Can store results in artifact location

Make requirements.txt

```
mlflow
jupyter
scikit-learn
pandas
seaborn
hyperopt
xgboost
```

Store artifacts in sqlite

```mlflow ui --backend-store-uri sqlite:///mlflow.db```

In Jupyter Notebook

```
import mlflow

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mflow.set_experiment('nyc-taxi-experiment')
```

After running linear regression model, try a run with Lasso model
```
with mlflow.start_run():
    mlflow.set_tag('developer', '<name>')

    mlflow.log_param('train-data-path', './data/green_tripdata_2021-01.csv')
    mlflow.log_param('valid-data-path', './data/green_tripdata_2021-02.csv')

    alpha = 0.01
    mlflow.log_param('alpha', alpha)

    lr = Lasso(alpha)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric('rmse', rmse)
```

Try running a new experiment with alpha = 0.1

## Hyperparameter Training

```import xgboost as xgb```

```
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
```

```hyperopt``` is a library that uses bayesian methods to find best parameters
```fmin``` minimizes the object function
```tpe``` algorithm used to control logic
```hp``` defines search space for each hyperparameter
```STATUS_OK``` is a signal letting hyperopt know the run is successful
```Trials``` tracks information for each run

Create matrices for xgboost.

```
train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)
```

Define the objective function.

```
def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
    
    return {'loss': rmse, 'status': STATUS_OK}
```

Define search space for each hyperparameter

```
search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}

best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)
```

Autolog is disabled

```mlflow.xgboost.autolog(disable=True)```

Train the model with the best parameters and save run to MLFlow
```
with mlflow.start_run():
    
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    best_params = {
        'learning_rate': 0.09585355369315604,
        'max_depth': 30,
        'min_child_weight': 1.060597050922164,
        'objective': 'reg:linear',
        'reg_alpha': 0.018060244040060163,
        'reg_lambda': 0.011658731377413597,
        'seed': 42
    }

    mlflow.log_params(best_params)

    booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(valid, 'validation')],
        early_stopping_rounds=50
    )

    y_pred = booster.predict(valid)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
```

Run experiment with multiple models

```
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR

mlflow.sklearn.autolog()

for model_class in (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, LinearSVR):

    with mlflow.start_run():

        mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
        mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlmodel = model_class()
        mlmodel.fit(X_train, y_train)

        y_pred = mlmodel.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
```

## Model Management

Source: https://neptune.ai/blog/ml-experiment-tracking

![Model Management](images/model-management.png)

Model Management is:
Experiment Tracking + Model Versioning + Model Deployment + Scaling Hardware

Using excel/spreadsheet/folder system

* basic way of managing
* error prone (may override previous model)
* no versioning
* no model lineage: unclear hyperparameters, training/validation sets

Saving models with mlflow.log_artifact or mlflow.\<model\>.log_model

```
with mlflow.start_run():

    mlflow.set_tag("developer", "cristian")

    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")

    alpha = 0.1
    mlflow.log_param("alpha", alpha)

    lr = Lasso(alpha)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    #Tracking our model
    mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")
```

```
#For the purpose of this example, let's turn off autologging
mlflow.xgboost.autolog(disable=True)

with mlflow.start_run():
    
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    best_params = {
        'learning_rate': 0.09585355369315604,
        'max_depth': 30,
        'min_child_weight': 1.060597050922164,
        'objective': 'reg:linear',
        'reg_alpha': 0.018060244040060163,
        'reg_lambda': 0.011658731377413597,
        'seed': 42
    }

    mlflow.log_params(best_params)

    booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(valid, 'validation')],
        early_stopping_rounds=50
    )

    y_pred = booster.predict(valid)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    #Model tracking
    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
```

Log the DictVectorizers

```
with open("models/preprocessor.b", "wb") as f_out:
    pickle.dump(dv, f_out)
mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
```

## Model Registry