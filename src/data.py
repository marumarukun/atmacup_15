import polars as pl

INPUT_DIR = "data/input"


def load_data():
    train = pl.read_csv(f"../{INPUT_DIR}/train.csv", try_parse_dates=True)
    test = pl.read_csv(f"../{INPUT_DIR}/test.csv", try_parse_dates=True)
    anime = pl.read_csv(f"../{INPUT_DIR}/anime.csv", try_parse_dates=True)
    anime = anime.select(pl.all().shrink_dtype())
    train = train.select(pl.all().shrink_dtype())
    test = test.select(pl.all().shrink_dtype())
    return train, test, anime

# TODO: user_idとanime_idはlabel encodeしてしまう
