import pickle
import streamlit as st
import numpy as np

# Using st.cache_data to cache the data loading
@st.cache_data
def load_data():
    model = pickle.load(open('artifacts/model.pkl', 'rb'))
    book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
    final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
    book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
    return model, book_names, final_rating, book_pivot

model, book_names, final_rating, book_pivot = load_data()

st.header('Book Recommendation System Using Machine Learning - Eric')

def fetch_poster(suggestion):
    poster_url = []
    for book_id in suggestion:
        book_name = book_pivot.index[book_id]
        ids = np.where(final_rating['title'] == book_name)[0][0]
        url = final_rating.iloc[ids]['image_url']
        poster_url.append(url)
    return poster_url

def recommend_book(book_name, num_recommendations=10):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=num_recommendations)
    poster_url = fetch_poster(suggestion[0])
    books_list = [book_pivot.index[i] for i in suggestion[0]]
    return books_list, poster_url

selected_book = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    try:
        recommended_books, poster_url = recommend_book(selected_book, 16)  # Assuming you want 10 recommendations
        st.write("Recommended Books:")
        # Handle multiple rows based on the number of recommendations
        num_columns = 5  # Define number of columns per row
        num_rows = (len(recommended_books) - 1) // num_columns + 1
        index = 1
        for _ in range(num_rows):
            cols = st.columns(num_columns)
            for col in cols:
                if index < len(recommended_books):
                    with col:
                        # Apply the same centered alignment with CSS
                        st.markdown(f"<div style='text-align: center; display: flex; flex-direction: column; align-items: center;'>{recommended_books[index]}<br><img src='{poster_url[index]}' style='max-width: 150px; height: auto; margin-top: 10px;'></img></div>", unsafe_allow_html=True)
                    index += 1
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
