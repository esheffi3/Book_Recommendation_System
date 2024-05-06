import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

books = pd.read_csv('data/BX-Books.csv', sep=";", on_bad_lines='skip', encoding='latin-1')

books.head()

books.iloc[237]['Image-URL-L']

books.shape

books.columns

books = books[['ISBN','Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-L']]

books.head()

books.rename(columns={"Book-Title":'title',
                      'Book-Author':'author',
                     "Year-Of-Publication":'year',
                     "Publisher":"publisher",
                     "Image-URL-L":"image_url"},inplace=True)

books.head()

users = pd.read_csv('data/BX-Users.csv', sep=";", on_bad_lines='skip', encoding='latin-1')

users.head()

users.shape

users.rename(columns={"User-ID":'user_id',
                      'Location':'location',
                     "Age":'age'},inplace=True)

users.head(2)

ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=";", on_bad_lines='skip', encoding='latin-1')

ratings.head()

ratings.shape

ratings.rename(columns={"User-ID":'user_id',
                      'Book-Rating':'rating'},inplace=True)

ratings.head(2)

print(books.shape, users.shape, ratings.shape, sep='\n')



ratings['user_id'].value_counts()

ratings['user_id'].value_counts().shape

ratings['user_id'].unique().shape

x = ratings['user_id'].value_counts() >= 50

x[x].shape

y= x[x].index

y

ratings = ratings[ratings['user_id'].isin(y)]

ratings.head()

ratings.shape

ratings_with_books = ratings.merge(books, on='ISBN')

ratings_with_books.head()

ratings_with_books.shape

number_rating = ratings_with_books.groupby('title')['rating'].count().reset_index()

number_rating.head()

number_rating.rename(columns={'rating':'num_of_rating'},inplace=True)

number_rating.head()

final_rating = ratings_with_books.merge(number_rating, on='title')

final_rating.head()

final_rating.shape

final_rating = final_rating[final_rating['num_of_rating'] >= 50]

final_rating.head()

final_rating.shape

final_rating.drop_duplicates(['user_id','title'],inplace=True)

final_rating.shape

book_pivot = final_rating.pivot_table(columns='user_id', index='title', values= 'rating')

book_pivot

book_pivot.shape

book_pivot.fillna(0, inplace=True)

book_pivot

from scipy.sparse import csr_matrix

book_sparse = csr_matrix(book_pivot)

type(book_sparse)

from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm= 'brute')

model.fit(book_sparse)

distance, suggestion = model.kneighbors(book_pivot.iloc[237,:].values.reshape(1,-1), n_neighbors=6 )

distance

suggestion

book_pivot.iloc[241,:]

for i in range(len(suggestion)):
    print(book_pivot.index[suggestion[i]])

book_pivot.index[3]

book_names = book_pivot.index

book_names[2]

np.where(book_pivot.index == '4 Blondes')[0][0]

ids = np.where(final_rating['title'] == "Harry Potter and the Chamber of Secrets (Book 2)")[0][0]

final_rating.iloc[ids]['image_url']

book_name = []
for book_id in suggestion:
    book_name.append(book_pivot.index[book_id])
    
    

book_name[0]

ids_index = []
for name in book_name[0]: 
    ids = np.where(final_rating['title'] == name)[0][0]
    ids_index.append(ids)

for idx in ids_index:
    url = final_rating.iloc[idx]['image_url']
    print(url)

import pickle
pickle.dump(model,open('artifacts/model.pkl','wb'))
pickle.dump(book_names,open('artifacts/book_names.pkl','wb'))
pickle.dump(final_rating,open('artifacts/final_rating.pkl','wb'))
pickle.dump(book_pivot,open('artifacts/book_pivot.pkl','wb'))

def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )
    
    for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                if j == book_name:
                    print(f"You searched '{book_name}'\n")
                    print("The suggestion books are: \n")
                else:
                    print(j)

book_name = "Harry Potter and the Chamber of Secrets (Book 2)"
recommend_book(book_name)
