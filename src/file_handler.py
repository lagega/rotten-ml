import numpy as np
import pandas

FEEDING_COLUMNS = [
    'movie_title',
    'content_rating',
    'genres',
    'directors',
    'authors',
    'actors',
    'runtime',
    'production_company'
]

RESULT_COLUMN = 'tomatometer_rating'

TRAINING_SIZE = 14000
VALIDATION_SIZE = 17712 - TRAINING_SIZE


def load_data():
    df = pandas.read_csv('../assets/rotten_movies.csv')

    training_df = df.head(TRAINING_SIZE).sample(TRAINING_SIZE)
    validation_df = df.tail(VALIDATION_SIZE).sample(VALIDATION_SIZE)

    x_train = training_df[FEEDING_COLUMNS]
    x_train['runtime'] = x_train['runtime'].to_numpy(int)

    y_train = training_df[RESULT_COLUMN].to_numpy()

    x_valid = validation_df[FEEDING_COLUMNS]
    x_valid['runtime'] = x_valid['runtime'].to_numpy(int)

    y_valid = validation_df[RESULT_COLUMN].to_numpy()

    return x_train.to_numpy(str), normalize_y(y_train), x_valid.to_numpy(str), normalize_y(y_valid)


def normalize_y(values):
    normalized_values = []

    for y in values:
        if y >= 60:
            normalized_values.append(1)
        else:
            normalized_values.append(0)

    return np.array(normalized_values).astype(int)
