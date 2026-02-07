import pandas as pd
import numpy as np
import requests
import io
import time
# Salam, chetori?
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
        try:
            s = requests.get(url_ratings).content
            self.df_ratings = pd.read_csv(io.StringIO(s.decode('utf-8')), 
                                          sep='\t', 
                                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
        except Exception as e:
            print(f"error downloading ratings: {e}")
            return

        # Movie Titles
        url_items = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
        try:
            m = requests.get(url_items).content
            df_movies = pd.read_csv(io.StringIO(m.decode('latin-1')), 
                                    sep='|', 
                                    header=None, 
                                    usecols=[0, 1], 
                                    names=['movie_id', 'title'])
        except Exception as e:
            print(f"error downloading items: {e}")
            return

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

    def dijkstra(self, start_id, end_id):

        distances = {node: float('inf') for node in self.graph}
        parents = {node: None for node in self.graph}
        visited = set()
        
        if start_id not in self.graph:
            print(f"start ID {start_id} not in graph.")
            return None, None
        if end_id not in self.graph:
            print(f"end ID {end_id} not in graph.")
            return None, None

        distances[start_id] = 0
        
        curr_node = start_id
        
        while curr_node is not None:
            visited.add(curr_node)
            '''
            if curr_node == end_id: # optimization, if nedded.
                break
            '''
            neighbors = self.graph[curr_node]
            for nei, wei in neighbors.items():
                if nei not in visited:
                    weight = 1.0 / wei # handling the weight problem here ?
                    
                    if distances[curr_node] + weight < distances[nei]:
                        distances[nei] = distances[curr_node] + weight
                        parents[nei] = curr_node
            
            min_dist = float('inf')
            next_node = None
            
            for node in distances:
                if node not in visited:
                    if distances[node] < min_dist:
                        min_dist = distances[node]
                        next_node = node
            
            if next_node is None:
                break
            curr_node = next_node

        return distances, parents

    def get_shortest_path_chain(self, start_id, end_id):
        t0 = time.time()
        distances, parents = self.dijkstra(start_id, end_id)
        t1 = time.time()
        path = []
        curr = end_id

        if not parents:
            return None
        if distances[end_id] == float('inf'):
            print("no path exists.")
            return None

        while curr is not None:
            path.append(curr)
            curr = parents[curr]
            
        path = path[::-1]
        
        print(f"\npath found in {t1-t0:.4f} sec.")
        print(f"total weighted cost: {distances[end_id]:.4f}")
        return path

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

    # dijkstra test
    def find_id_by_name(name):
        for mid, title in builder.movie_titles.items():
            if name.lower() in title.lower():
                return mid
        return None

    start_name = "Outbreak"
    end_name = "Beautiful Thing"
    
    start_id = find_id_by_name(start_name)
    end_id = find_id_by_name(end_name)

    if start_id and end_id:
        print(f"\n--- TASK 1: Finding Similarity Path ---")
        print(f"From: {builder.get_title(start_id)} (ID: {start_id})")
        print(f"To:   {builder.get_title(end_id)} (ID: {end_id})")
        
        path = builder.get_shortest_path_chain(start_id, end_id)
        
        if path:
            print("\nSimilarity Path:")
            for i, node in enumerate(path):
                title = builder.get_title(node)
                if i > 0:
                    prev = path[i-1]
                    shared = builder.graph[prev][node]
                    print(f"  | (shared users: {shared})")
                    print(f"  v")
                print(f"[{i+1}] {title}")
    else:
        print("could not find movies.")