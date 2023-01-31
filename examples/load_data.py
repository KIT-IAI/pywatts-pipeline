import requests

import pandas as pd

ELEC_DATA_URL = 'https://raw.githubusercontent.com/KIT-IAI/pyWATTS/master/data/getting_started_data.csv'


def load_elec_data():
    r = requests.get(ELEC_DATA_URL)
    with open(ELEC_DATA_URL.split('/')[-1], "wb") as file:
        file.write(r.content)

    df = pd.read_csv(ELEC_DATA_URL.split('/')[-1], infer_datetime_format=True, parse_dates=[0],
                     index_col=0)
    df.index.name = "time"
    return df
