import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)

    mlflow.log_params(rf.get_params())
    mlflow.log_metric('rmse', rmse)
    # Save the model
    mlflow.sklearn.log_model(rf, artifact_path='models_mlflow', input_example=X_train[0:5])


if __name__ == '__main__':
    # Set the tracking URI to remote server PUBLIC IP
    mlflow.set_tracking_uri('http://ec2-18-116-14-12.us-east-2.compute.amazonaws.com:5000')

    mlflow.set_experiment('nyc-taxi-experiment-2023')

    with mlflow.start_run():
        mlflow.set_tag('model', 'RandomForestRegressor')
        
        run_train()

        
            