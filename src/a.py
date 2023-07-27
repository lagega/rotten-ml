import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer

df = pd.read_csv('C:/Users/Gabriel/PycharmProjects/rotten_model/assets/rotten_movies.csv')
df = df.fillna('')  # replace NaN with an empty string


# Assuming you have already created the 'text' column in the dataframe
df['text'] = df['movie_info'] + ' ' + df['content_rating'] + ' ' + df['genres'] + ' ' + df['directors'] + ' ' + df['actors']

# Step 1: Preprocess the text data (lowercasing in this example)
df['text'] = df['text'].str.lower()

# Step 2: Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Target variable
y = df['tomatometer_rating']

# Step 3: Check for and handle missing values
y.replace({'': 0}, inplace=True)
threshold = 60  # Set the threshold for binary classification

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Choose a model (Linear Regression)
model = LinearRegression()

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 9: Calculate precision and recall

y_binary_test = (y_test >= threshold).astype(int)
y_binary_pred = (y_pred >= threshold).astype(int)
precision = precision_score(y_binary_test, y_binary_pred)
recall = recall_score(y_binary_test, y_binary_pred)
accuracy = accuracy_score(y_binary_test, y_binary_pred)


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared score: {r2}")