import polars as pl
from sklearn.preprocessing import LabelEncoder

INPUT_DIR = "data/input"


def load_data():
    # データ読み込み
    train = pl.read_csv(f"../{INPUT_DIR}/train.csv", try_parse_dates=True)
    test = pl.read_csv(f"../{INPUT_DIR}/test.csv", try_parse_dates=True)
    anime = pl.read_csv(f"../{INPUT_DIR}/anime.csv", try_parse_dates=True)

    # label encoding(user_id)
    le_user_id = LabelEncoder()
    le_user_id.fit(pl.concat([train["user_id"], test["user_id"]]).fill_null(""))
    train = train.with_columns(pl.Series("user_id", le_user_id.transform(train["user_id"])).alias("user_id"))
    test = test.with_columns(pl.Series("user_id", le_user_id.transform(test["user_id"])).alias("user_id"))

    # label encoding(anime_id)
    le_anime_id = LabelEncoder()
    le_anime_id.fit(anime["anime_id"])
    anime = anime.with_columns(pl.Series("anime_id", le_anime_id.transform(anime["anime_id"])).alias("anime_id"))
    train = train.with_columns(pl.Series("anime_id", le_anime_id.transform(train["anime_id"])).alias("anime_id"))
    test = test.with_columns(pl.Series("anime_id", le_anime_id.transform(test["anime_id"])).alias("anime_id"))

    # データ型を縮小
    anime = anime.select(pl.all().shrink_dtype())
    train = train.select(pl.all().shrink_dtype())
    test = test.select(pl.all().shrink_dtype())

    return train, test, anime
