import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
from tensorflow.keras import layers

# Load Data
df = pd.read_csv(r"Data/merged_df.csv")
df_grouped = df.groupby("Book_Title").agg({
    "Book_Author": "first",
    "Year_Of_Publication": "first",
    "Publisher_x": "first",
    "ISBN": "first",
    "Book_Rating": "mean",
    "Age": "mean",
    "Image-URL-L": "first"
}).reset_index()

# Encoding user IDs
user_ids = df["User_ID"].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
df['user'] = df["User_ID"].map(user_to_user_encoded)

# Encoding book titles
book_ids = df["Book_Title"].unique().tolist()
book_to_book_encoded = {x: i for i, x in enumerate(book_ids)}
book_encoded_to_book = {i: x for i, x in enumerate(book_ids)}
df['book'] = df["Book_Title"].map(book_to_book_encoded)

# Convert ratings to float and normalize
df['rating'] = df['Book_Rating'].astype(np.float32)
min_rating, max_rating = df['rating'].min(), df['rating'].max()
y = (df['rating'] - min_rating) / (max_rating - min_rating)

# Shuffle and split data
df = df.sample(frac=1, random_state=42)
train_size = int(0.8 * len(df))
x_train, x_val = df[['user', 'book']][:train_size], df[['user', 'book']][train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Define RecommenderNet model
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_books, embedding_size=50):
        super(RecommenderNet, self).__init__()
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)
        self.book_embedding = layers.Embedding(num_books, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.book_bias = layers.Embedding(num_books, 1)
   
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])
        dot_product = tf.reduce_sum(user_vector * book_vector, axis=1, keepdims=True)
        return tf.nn.sigmoid(dot_product + user_bias + book_bias)

# Initialize and compile model
num_users, num_books = len(user_to_user_encoded), len(book_to_book_encoded)
model = RecommenderNet(num_users, num_books)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train model
model.fit(x_train, y_train, batch_size=8, epochs=10, validation_data=(x_val, y_val))

# Generate book recommendations
def get_book_recommendations(user_id, top_n=10):
    user_encoder = user_to_user_encoded.get(user_id)
    if user_encoder is None:
        return []
    books_not_read = list(set(book_to_book_encoded.values()) - set(df[df['user'] == user_encoder]['book'].values))
    user_book_array = np.array([[user_encoder, book] for book in books_not_read])
    ratings = model.predict(user_book_array).flatten()
    top_indices = ratings.argsort()[-top_n:][::-1]
    return [book_encoded_to_book[books_not_read[i]] for i in top_indices]

# Generate book recommendations
def get_book_recommendations(user_id, top_n=10):
    user_encoder = user_to_user_encoded.get(user_id)
    if user_encoder is None:
        return []
    books_not_read = list(set(book_to_book_encoded.values()) - set(df[df['User_ID'] == user_id]['book'].values))
    user_book_array = np.array([[user_encoder, book] for book in books_not_read])
    ratings = model.predict(user_book_array).flatten()
    top_indices = ratings.argsort()[-top_n:][::-1]
    return [book_encoded_to_book[books_not_read[i]] for i in top_indices]

st.set_page_config(page_title="Book Recommendation", layout="wide")
# Compute Cosine Similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_grouped["Book_Title"])
cosine_sim_df = pd.DataFrame(cosine_similarity(tfidf_matrix, tfidf_matrix),
                             index=df_grouped["Book_Title"],
                             columns=df_grouped["Book_Title"])

def book_recommendations(book_title, k=5):
    if book_title not in cosine_sim_df.index:
        return pd.DataFrame()
    similar_books = cosine_sim_df[book_title].sort_values(ascending=False).iloc[1:k+1].index
    return df_grouped[df_grouped["Book_Title"].isin(similar_books)]

# CSS Styling
st.markdown("""
    <style>
        @import url('https://fonts.cdnfonts.com/css/mango-ac');
        * { font-family: 'Mango AC', sans-serif; }
        .title { color: #E08A64; font-size: 36px; text-align: center; }
        .subtitle { color: #9A75A0; font-size: 28px; text-align: center; }
        .book-card { border-radius: 15px; background-color: #F9E7D9; padding: 10px; text-align: center; }
        img { border-radius: 10px; }
        .rating { color: #FFD700; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)
 
    
st.sidebar.title("📚 Menu Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Home", "Pengarang", "Pencarian","User Rekomendasi"])

# Fungsi untuk menampilkan detail buku secara dinamis
def show_book_details(book):
    st.markdown(f"""
        <h2 class='subtitle'>{book['Book_Title']}</h2>
        <p><b>Penulis:</b> {book['Book_Author']}</p>
        <p><b>Tahun Terbit:</b> {book['Year_Of_Publication']}</p>
        <p><b>Penerbit:</b> {book['Publisher_x']}</p>
        <p><b>ISBN:</b> {book['ISBN']}</p>
        <p><b>Rating:</b> ⭐ {book['Book_Rating']:.1f}/10</p>
        <p><b>Usia Rata-rata Pembaca:</b> {book['Age']:.1f}</p>
    """, unsafe_allow_html=True)
    st.image(book['Image-URL-L'], width=150)

    # Menampilkan rekomendasi buku
    recommended_books = book_recommendations(book['Book_Title'])
    if not recommended_books.empty:
        st.markdown("<h3>📖 Rekomendasi Buku yang Mirip:</h3>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i, (index, rec_book) in enumerate(recommended_books.iterrows()):
            with cols[i % 5]:
                st.image(rec_book['Image-URL-L'], width=100)
                st.write(f"**{rec_book['Book_Title']}**")
                st.write(f"*{rec_book['Book_Author']}*")
                if st.button("Lihat Detail", key=f"detail_{rec_book['ISBN']}"):
                    st.session_state['selected_book'] = rec_book
                    st.rerun()
    st.button("Tutup", on_click=lambda: st.session_state.pop('selected_book', None))


# Fungsi untuk mendapatkan rekomendasi buku populer
def get_popular_books():
    top_books = df.sort_values(by="Book_Rating", ascending=False).head(10)
    return top_books

# --- Home Page ---
if page == "Home":
    st.markdown('<h1 class="title">Book Recommendation</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subtitle">Most Popular Books</h2>', unsafe_allow_html=True)

    popular_books = get_popular_books()
    cols = st.columns(5)
    
    for i, (index, book) in enumerate(popular_books.iterrows()):
        with cols[i % 5]:
            st.image(book['Image-URL-L'], width=120)
            st.write(f"**{book['Book_Title']}**")
            st.write(f"*{book['Book_Author']}*")
            st.write(f"⭐ {book['Book_Rating']}/10")
            if st.button("Lihat Detail", key=f"detail_{index}"):
                st.session_state['selected_book'] = book
                st.rerun()

# --- Pengarang Page ---
elif page == "Pengarang":
    st.markdown('<h1 class="title">Daftar Pengarang</h1>', unsafe_allow_html=True)
    
    authors = df_grouped["Book_Author"].unique().tolist()
    selected_author = st.selectbox("Pilih Pengarang", authors)
    author_books = df_grouped[df_grouped["Book_Author"] == selected_author]

    st.markdown(f"<h2 class='subtitle'>Buku oleh {selected_author}</h2>", unsafe_allow_html=True)
    cols = st.columns(5)
    
    for i, (index, book) in enumerate(author_books.iterrows()):
        with cols[i % 5]:
            st.image(book['Image-URL-L'], width=120)
            st.write(f"**{book['Book_Title']}**")
            st.write(f"⭐ {book['Book_Rating']}/10")
            if st.button("Lihat Detail", key=f"author_detail_{index}"):
                st.session_state['selected_book'] = book
                st.rerun()

# --- Pencarian Page ---
elif page == "Pencarian":
    st.markdown('<h1 class="title">Cari Buku</h1>', unsafe_allow_html=True)
    search_query = st.text_input("Masukkan judul buku yang ingin dicari")
    
    if search_query:
        search_results = df_grouped[df_grouped["Book_Title"].str.contains(search_query, case=False, na=False)]
        
        if search_results.empty:
            st.write("📌 Tidak ditemukan buku dengan judul tersebut.")
        else:
            st.markdown(f"<h2 class='subtitle'>Hasil Pencarian</h2>", unsafe_allow_html=True)
            cols = st.columns(5)
            
            for i, (index, book) in enumerate(search_results.iterrows()):
                with cols[i % 5]:
                    st.image(book['Image-URL-L'], width=120)
                    st.write(f"**{book['Book_Title']}**")
                    st.write(f"*{book['Book_Author']}*")
                    st.write(f"⭐ {book['Book_Rating']}/10")
                    if st.button("Lihat Detail", key=f"search_detail_{index}"):
                        st.session_state['selected_book'] = book
                        st.rerun()

elif page == "User Rekomendasi":
    selected_user = st.sidebar.selectbox("Pilih User", user_ids)
    recommended_books = get_book_recommendations(selected_user)
    
    if recommended_books:
        st.write(f"📚 Rekomendasi untuk User {selected_user}")
        cols = st.columns(5)
        for i, book_title in enumerate(recommended_books):
            book_data = df_grouped[df_grouped["Book_Title"] == book_title].iloc[0]
            with cols[i % 5]:
                st.image(book_data['Image-URL-L'], width=100)
                st.write(f"**{book_data['Book_Title']}**")
                st.write(f"*{book_data['Book_Author']}*")
                if st.button("Lihat Detail", key=f"detail_{book_data['ISBN']}"):
                    st.session_state['selected_book'] = book_data
                    st.rerun()
    else:
        st.write("❌ Tidak ada rekomendasi untuk user ini.")


# Tampilkan detail buku jika ada buku yang dipilih
if 'selected_book' in st.session_state:
    st.sidebar.markdown("## Detail Buku")
    show_book_details(st.session_state['selected_book'])

import subprocess

subprocess.run(["pip", "install", "-r", "requirements.txt"])

