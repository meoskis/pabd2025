import os

import yaml
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from utils import setup_logger

logger = setup_logger()
MODEL_NAME = "gradient_boosting"


def train_model():

    train_data = pd.read_csv("data/processed/train.csv")
    X_train = train_data[["total_meters", "floor", "floors_count", "rooms_count"]]
    y_train = train_data["price"]

    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{params['model_name']}.pkl"
    # model_path = f"models/{MODEL_NAME}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Модель сохранена в файл {model_path}")


if __name__ == "__main__":
    train_model()