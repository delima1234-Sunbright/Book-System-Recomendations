import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard

# Load Data
books = pd.read_csv("Books.csv", encoding="latin1")
ratings = pd.read_csv("Ratings.csv", encoding="latin1")
users = pd.read_csv("Users.csv", encoding="latin1")

# Merge Data
df = ratings.merge(books, on="ISBN").merge(users, on="User-ID")

# Handle Missing Values
df.dropna(inplace=True)

# Popularity-Based Recommendation (Top 10 Books by Average Rating)
def get_popular_books(n=10, min_ratings=50):
    book_ratings = df.groupby("Book-Title").agg({"Book-Rating": ["count", "mean"]})
    book_ratings.columns = ["num_ratings", "avg_rating"]
    popular_books = book_ratings[book_ratings["num_ratings"] >= min_ratings].sort_values(by="avg_rating", ascending=False)
    return popular_books.head(n)

# Content-Based Filtering (Using Cosine Similarity on Book Titles)
def compute_cosine_similarity(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["Book-Title"])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(cosine_sim, index=data["Book-Title"], columns=data["Book-Title"])

cosine_sim_df = compute_cosine_similarity(df)

# Recommend Books based on Content Similarity
def book_recommendations(book_title, similarity_data, books_data, k=10):
    if book_title not in similarity_data.columns:
        return "Buku tidak ditemukan dalam dataset."
    
    book_index = similarity_data.columns.get_loc(book_title)
    similarity_scores = similarity_data.iloc[book_index].to_numpy()
    sorted_indices = similarity_scores.argsort()[::-1]
    similar_indices = sorted_indices[sorted_indices != book_index][:k]
    similar_indices = [idx for idx in similar_indices if idx < len(similarity_scores)]
    
    recommended_books = [similarity_data.columns[idx] for idx in similar_indices]
    recommendations = books_data[books_data["Book-Title"].isin(recommended_books)].copy()
    similarity_values = {similarity_data.columns[idx]: similarity_scores[idx] for idx in similar_indices}
    recommendations["similarity_score"] = recommendations["Book-Title"].map(similarity_values)
    
    return recommendations.sort_values(by="similarity_score", ascending=False)

# Collaborative Filtering (User-Based Jaccard Similarity)
def jaccard_similarity(user1, user2):
    set1 = set(df[df["User-ID"] == user1]["Book-Title"])
    set2 = set(df[df["User-ID"] == user2]["Book-Title"])
    return 1 - jaccard(set1, set2)

def recommend_books_collaborative(user_id, n=5):
    user_ratings = df[df["User-ID"] == user_id].set_index("Book-Title")["Book-Rating"]
    all_users = df["User-ID"].unique()
    similarities = {other_user: jaccard_similarity(user_id, other_user) for other_user in all_users if other_user != user_id}
    
    most_similar_user = max(similarities, key=similarities.get)
    recommended_books = df[(df["User-ID"] == most_similar_user) & (~df["Book-Title"].isin(user_ratings.index))][["Book-Title", "Book-Rating"]]
    return recommended_books.sort_values(by="Book-Rating", ascending=False).head(n)

# Hybrid Recommendation System (Weighted Combination)
def hybrid_recommendations(user_id, book_title, alpha=0.5, beta=0.5, n=10):
    content_recs = book_recommendations(book_title, cosine_sim_df, df, k=n)
    collab_recs = recommend_books_collaborative(user_id, n=n)
    
    if isinstance(content_recs, str):
        content_recs = pd.DataFrame(columns=["Book-Title", "similarity_score"])
    
    content_recs = content_recs.rename(columns={"similarity_score": "score"})
    collab_recs = collab_recs.rename(columns={"Book-Rating": "score"})
    
    content_recs["method"] = "Content-Based"
    collab_recs["method"] = "Collaborative"
    
    hybrid_recs = pd.concat([content_recs, collab_recs])
    hybrid_recs["final_score"] = alpha * hybrid_recs["score"] + beta * hybrid_recs["score"]
    
    return hybrid_recs.sort_values(by="final_score", ascending=False).head(n)

# Example Usage
user_id = 276847  # Replace with actual user ID
book_title = "The Lovely Bones: A Novel"  # Replace with a valid book title
recommendations = hybrid_recommendations(user_id, book_title, alpha=0.6, beta=0.4, n=10)
print(recommendations)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_books, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(
            num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.book_embedding = layers.Embedding(
            num_books, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.book_bias = layers.Embedding(num_books, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])

        dot_user_book = tf.tensordot(user_vector, book_vector, 2)
        x = dot_user_book + user_bias + book_bias
        return tf.nn.sigmoid(x)

# Load dataset
ratings_df = pd.read_csv('ratings.csv')  # Assumed dataset format: userID, bookID, rating
books_df = pd.read_csv('books.csv')  # Assumed dataset format: bookID, title, author

# Encoding user and book IDs
user_ids = ratings_df['userID'].unique().tolist()
book_ids = ratings_df['bookID'].unique().tolist()

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
book_to_book_encoded = {x: i for i, x in enumerate(book_ids)}
user_encoded_to_user = {i: x for x, i in user_to_user_encoded.items()}
book_encoded_to_book = {i: x for x, i in book_to_book_encoded.items()}

ratings_df['user'] = ratings_df['userID'].map(user_to_user_encoded)
ratings_df['book'] = ratings_df['bookID'].map(book_to_book_encoded)

num_users = len(user_ids)
num_books = len(book_ids)
embedding_size = 50

# Prepare data
x_train = ratings_df[['user', 'book']].values
y_train = ratings_df['rating'].values

# Define model
model = RecommenderNet(num_users, num_books, embedding_size)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train model
history = model.fit(x_train, y_train, batch_size=8, epochs=100, validation_split=0.2)

# Plot training history
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model RMSE')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Make recommendations
user_id = ratings_df.userID.sample(1).iloc[0]
books_read_by_user = ratings_df[ratings_df.userID == user_id]
books_not_read = books_df[~books_df['bookID'].isin(books_read_by_user.bookID.values)]['bookID']
books_not_read = list(set(books_not_read).intersection(set(book_to_book_encoded.keys())))
books_not_read = [[book_to_book_encoded.get(x)] for x in books_not_read]
user_encoder = user_to_user_encoded.get(user_id)
user_book_array = np.hstack(([[user_encoder]] * len(books_not_read), books_not_read))

ratings = model.predict(user_book_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_books_ids = [book_encoded_to_book.get(books_not_read[x][0]) for x in top_ratings_indices]

print(f'Recommendations for user {user_id}')
print('=' * 27)
print('Books previously liked:')
print('-' * 27)
for book_id in books_read_by_user.sort_values(by='rating', ascending=False).head(5)['bookID'].values:
    book_info = books_df[books_df['bookID'] == book_id].iloc[0]
    print(f"{book_info['title']} by {book_info['author']}")

print('-' * 27)
print('Top 10 book recommendations:')
print('-' * 27)
for book_id in recommended_books_ids:
    book_info = books_df[books_df['bookID'] == book_id].iloc[0]
    print(f"{book_info['title']} by {book_info['author']}")

