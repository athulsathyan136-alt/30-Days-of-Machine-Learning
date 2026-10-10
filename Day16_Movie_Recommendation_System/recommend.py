import pandas as pd
import numpy as np

df = pd.read_csv('data/movies.csv')

print('\n=======DATASET=========')
print(df)

genre_list = df['genres'].str.split('|')

all_genres = sorted({genre for movie_genres in genre_list for genre in movie_genres})

print('\n=====AVAILABLE GENRES=========')
print(all_genres)

X =np.array([[1 if genre in movie_genres else 0 for genre in all_genres] for movie_genres in genre_list],dtype=float)

print('\n=======FEATURE MATRIX=========')
print('Matrix shape: ',X.shape)
print(X)

def cosine_similarity_matrix(features):
    length = np.linalg.norm(features,axis=1)

    dot_product = features@features.T

    denominator = np.outer(length,length)

    similarity = np.divide(dot_product,denominator ,out=np.zeros_like(dot_product),where=denominator!=0)

    return similarity

similarity_matrix = cosine_similarity_matrix(X)

def recommand_movies(movie_title,number=5):
    matches = df.index[df['title'].str.casefold() == movie_title.casefold()].tolist()

    if not matches:
        print(f'\nMovie not found: {movie_title}')
        print('Please choose a titile from a dataset')
        return
    movie_index = matches[0]

    scores = similarity_matrix[movie_index].copy()
    scores[movie_index] = -1

    recommand_indices = np.argsort(scores)[::-1][:number]

    print('\n=======RECOMMENDATION=============')
    print('Selected movies: ',df.loc[movie_index,'title'])

    for rank ,index in enumerate(recommand_indices,start=1):
        print(f"{rank}. {df.loc[index,'title']}"
              f"| Genres: {df.loc[index,'genres']}" 
              f"| similarity: {scores[index]:.2f}")

recommand_movies("Toy Story", number=5)


# 9. Ask the user for a movie
print("\n====== TRY YOUR OWN MOVIE ======")

movie_name = input(
    "Enter a movie title from the dataset: "
).strip()

if movie_name:
    recommand_movies(movie_name, number=5)

print("\n====== PROJECT COMPLETED ======")
