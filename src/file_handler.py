import numpy as np
import pandas
from sklearn.model_selection import train_test_split

from src.vectorizer import get_count_vector

FEEDING_COLUMNS = [
    'movie_title',
    'content_rating',
    'genres',
    'directors',
    'authors',
    'actors',
    'production_company'
]

RESULT_COLUMN = 'tomatometer_rating'


def load_data():
    df = pandas.read_csv('C:/Users/Gabriel/PycharmProjects/rotten_model/assets/rotten_movies.csv',
                         converters={col: trim for col in FEEDING_COLUMNS})

    df[RESULT_COLUMN] = df[RESULT_COLUMN].fillna(0)
    df = df.fillna('')
    df['text'] = to_single_string(df)

    vectorized_x = get_count_vector(df['text'])
    vectorized_y = normalize_y(df[RESULT_COLUMN])

    return train_test_split(vectorized_x, vectorized_y, train_size=0.2, random_state=42, shuffle=True)


def trim(string):
    return string[0:100]


def to_single_string(df):
    a = ''
    for column in FEEDING_COLUMNS:
        a += df[column] + ' '
    return a


def normalize_y(values):
    normalized_values = []

    for y in values:
        if y >= 60:
            normalized_values.append(1)
        else:
            normalized_values.append(0)

    return np.array(normalized_values).astype(int)
