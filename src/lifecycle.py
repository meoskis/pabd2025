import os
import datetime
import joblib
import argparse
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import cianparser

logger = None

# Удаление выбросов
def drop_outliers(df:pd.DataFrame, column:str) -> pd.DataFrame:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    return df[(df[column] > lower_bound) & (df[column] < upper_bound)]

# Настройка логирования
def setup_logger(log_path="logs/model.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def parse_cian():
    """
    Парсит данные с cian.ru для 1-3 комнатных квартир.
        
    Returns:
        DataFrame с данными о квартирах
    """
    moscow_parser = cianparser.CianParser(location="Москва")
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    data = []
    for n_rooms in range(1, 4):

        new_data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(n_rooms,),
            with_saving_csv=False,
            additional_settings={
                "start_page": 7,
                "end_page": 9,
                "object_type": "secondary",
            },
        )
        data.extend(new_data)

    df = pd.DataFrame(data)

    csv_path = f"data/raw/flats_{t}.csv"
    df.to_csv(csv_path, encoding="utf-8", index=False)


def process_data(test_size=0.2, data_path="data/raw"):
    """
    Очищает и подготавливает данные для обучения модели.

    Args:
        test_size: Относительный размер тестовой выборки
        data_path: Путь до сырых данных
    
    Returns:
        Очищенный DataFrame
    """
    all_raw_data_files = os.listdir(data_path)
    # Фильтруем только те, которые соответствуют шаблону
    csv_files = [
        f for f in all_raw_data_files if f.startswith("flats_") and f.endswith(".csv")
    ]
    # Чтение и объединение всех файлов
    df = pd.concat(
        [pd.read_csv(f"{data_path}/{f}") for f in csv_files], ignore_index=True
    )
    logger.info(f"Объём всех сырых данных: {len(df)} строк")

    df = df.drop_duplicates(subset=["url"])
    df["flat_id"] = df["url"].str.extract(r"/flat/(\d+)/")[0].astype("int64")
    df.drop(
        [
            "location",
            "deal_type",
            "accommodation_type",
            "price_per_month",
            "commissions",
            "url",
            "author",
            "author_type",
            "residential_complex",
        ],
        axis=1,
        inplace=True,
    )
    df = df.dropna()

    need_columns = ["total_meters", "price"]
    for column in need_columns:
        df = drop_outliers(df, column)

    logger.info(f"Объём данных после обработки: {len(df)} строк")

    test_ammt = int(len(df["flat_id"]) * test_size)
    test_items = sorted(df["flat_id"])[-test_ammt:]

    train_data = df[~df["flat_id"].isin(test_items)]
    test_data = df[df["flat_id"].isin(test_items)]

    os.makedirs("artifacts/", exist_ok=True)
    train_data.to_csv("artifacts/train.csv", index=False)
    test_data.to_csv("artifacts/test.csv", index=False)


def train_model(model_name:str, train_data_path="artifacts/train.csv"):
    """
    Обучает модель линейной регрессии на подготовленных данных.
    
    Args:
        model_name: Название модели
        train_data_path: Путь до обучающей выборки
        
    Returns:
        Обученная модель LinearRegression
    """
    train_data = pd.read_csv(train_data_path)

    X_train = train_data[["total_meters", "floor", "floors_count", "rooms_count"]]
    y_train = train_data["price"]

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)

    logger.info(f"Модель сохранена в файл {model_path}")


def test_model(model_name:str, test_data_path="artifacts/test.csv"):
    """
    Тестирование модели.
    
    Args:
        model_name: Название модели
        
    Returns:
        Результаты тестирования модели
    """
    test_data = pd.read_csv(test_data_path)
    model = joblib.load(f"models/{model_name}.pkl")

    X_test = test_data[["total_meters", "floor", "floors_count", "rooms_count"]]
    y_test = test_data["price"]

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Логирование метрик
    logger.info(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    logger.info(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.2f}")
    logger.info(f"Коэффициент детерминации R²: {r2:.6f}")
    logger.info(
        f"Средняя ошибка предсказания: {np.mean(np.abs(y_test - y_pred)):.2f} рублей"
    )


def main():
    """Основная функция для выполнения всего пайплайна."""
    global logger
    logger = setup_logger()

    logger.info("Запуск скрипта")

    parser = argparse.ArgumentParser(
        description="Обучение и сохранение модели предсказания цен на недвижимость."
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_version", type=str, required=True, help="Model version")
    parser.add_argument("--test_size", type=float, required=True, help="Test data proportion")

    args = parser.parse_args()
    model_name = args.model_name
    model_version = args.model_version
    test_size = args.test_size

    model_name = f"{model_name}_model_v{model_version}"
    parse_cian()
    process_data(test_size)

    train_model(model_name)
    test_model(model_name)


if __name__ == "__main__":
    main()