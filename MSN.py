import pandas as pd
import numpy as np
import requests
import io
# Salam Kia ;)
class MovieGraphBuilder:
    def __init__(self, min_common_users=5):
        # I filtered out edges that have fewer than this many common users
        # to avoid the graph becoming too dense/messy.
        self.min_common_users = min_common_users
        self.graph = {}
        self.movie_titles = {}
        self.df_ratings = None

    def load_data(self):
        # Downloading dataset
        # Ratings
        url_ratings = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
        s = requests.get(url_ratings).content
        self.df_ratings = pd.read_csv(io.StringIO(s.decode('utf-8')), 
                                      sep='\t', 
                                      names=['user_id', 'movie_id', 'rating', 'timestamp'])
        # Movie Titles
        url_items = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
        m = requests.get(url_items).content
        df_movies = pd.read_csv(io.StringIO(m.decode('latin-1')), 
                                sep='|', 
                                header=None, 
                                usecols=[0, 1], 
                                names=['movie_id', 'title'])
        # lookup dictionary for titles
        self.movie_titles = dict(zip(df_movies['movie_id'], df_movies['title']))
        print(f"Loaded {len(self.df_ratings)} ratings for {len(self.movie_titles)} movies")

    def build_graph(self):
        ''' a Pivot Table (Users x Movies)
            I just want to know IF a user rated a movie, so we mark it as 1.
            This creates a binary matrix '''
        user_movie_matrix = self.df_ratings.pivot_table(index='user_id', columns='movie_id', values='rating', aggfunc='count')
        
        user_movie_matrix = user_movie_matrix.fillna(0)
        user_movie_matrix[user_movie_matrix > 0] = 1
        
        ''' The result [i][j] is the dot product of movie i and movie j,
             which equals the number of users who watched both '''
        matrix_values = user_movie_matrix.values
        adjacency_matrix = matrix_values.T @ matrix_values
        
        movie_ids = user_movie_matrix.columns
        num_movies = len(movie_ids)

        # Convert to Graph Dictionary
        count = 0
        for i in range(num_movies):
            id_A = movie_ids[i]
            self.graph[id_A] = {}
            for j in range(i + 1, num_movies):
                weight = adjacency_matrix[i][j]
                
                if weight >= self.min_common_users:
                    id_B = movie_ids[j]
                    # undirected edges
                    self.graph[id_A][id_B] = weight
                    if id_B not in self.graph:
                        self.graph[id_B] = {}
                    self.graph[id_B][id_A] = weight
                    count += 1
                    
        print(f"{len(self.graph)} nodes and {count} edges created")
        return self.graph

    def get_neighbors(self, movie_id):
        return self.graph.get(movie_id, {})

    def get_title(self, movie_id):
        return self.movie_titles.get(movie_id, str(movie_id))

if __name__ == "__main__":
    # Test
    builder = MovieGraphBuilder(min_common_users=10)
    builder.load_data()
    builder.build_graph()
    # Check Toy Story, ID 1
    test_id = 1
    print(f"\nChecking connections for: {builder.get_title(test_id)}")
    neighbors = builder.get_neighbors(test_id)
    sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
    for neighbor_id, weight in sorted_neighbors[:5]:
        print(f" - Connected to {builder.get_title(neighbor_id)} (Shared Users: {weight})")