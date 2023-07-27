from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_count_vector(df):
    vectorizer = CountVectorizer()
    vectorized_data = vectorizer.fit_transform(df)

    return vectorized_data.toarray()


def get_tfid_vector(df):
    vectorizer = TfidfVectorizer()
    vectorized_data = vectorizer.fit_transform(df)

    return vectorized_data.toarray()
