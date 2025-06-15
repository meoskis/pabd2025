import os
import pandas as pd
from utils import drop_outliers, setup_logger

logger = setup_logger()


def process_data():
    data_path = "data/raw"
    all_raw_data_files = os.listdir(data_path)
    csv_files = [
        f for f in all_raw_data_files if f.startswith("flats_") and f.endswith(".csv")
    ]
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

    for column in ["total_meters", "price"]:
        df = drop_outliers(df, column)

    logger.info(f"Объём данных после обработки: {len(df)} строк")

    test_size = int(len(df["flat_id"]) * 0.2)
    test_items = sorted(df["flat_id"])[-test_size:]
    train_data = df[~df["flat_id"].isin(test_items)]
    test_data = df[df["flat_id"].isin(test_items)]

    os.makedirs("data/processed", exist_ok=True)
    train_data.to_csv("data/processed/train.csv", index=False)
    test_data.to_csv("data/processed/test.csv", index=False)
    logger.info("Данные сохранены в data/processed/train.csv и data/processed/test.csv")


if __name__ == "__main__":
    process_data()