import os
import datetime
import pandas as pd
import cianparser
from utils import setup_logger

logger = setup_logger()

RAW_DIR = "data/raw"


def parse_cian():
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
    os.makedirs(RAW_DIR, exist_ok=True)
    csv_path = f"{RAW_DIR}/flats_{t}.csv"
    df.to_csv(csv_path, encoding="utf-8", index=False)
    logger.info(f"Сохранены данные в {csv_path}")


if __name__ == "__main__":
    parse_cian()