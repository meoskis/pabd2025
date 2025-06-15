import joblib

import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from utils import setup_logger

logger = setup_logger()
MODEL_NAME = "gradient_boosting"


def test_model():

    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    model = joblib.load(f"models/{params['model_name']}.pkl")
    # model = joblib.load(f"models/{MODEL_NAME}.pkl")
    test_data = pd.read_csv("data/processed/test.csv")

    X_test = test_data[["total_meters", "floor", "floors_count", "rooms_count"]]
    y_test = test_data["price"]
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.6f}")
    logger.info(f"MAE: {np.mean(np.abs(y_test - y_pred)):.2f} рублей")


if __name__ == "__main__":
    test_model()