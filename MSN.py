import pandas as pd
import numpy as np
import requests
import io
import time
import random
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans

# Ghorbanet, to chetori?
class MovieGraphBuilder:
    def __init__(self, min_common_users=1):
        # I filtered out edges that have fewer than 1 common users
        # to avoid the graph becoming messy
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

class BipartiteMatcher:
    def __init__(self, builder):
        self.builder = builder
        self.residual_graph = {} 
        self.source = 'SOURCE'
        self.sink = 'SINK'
        self.matches = []

    def build_bipartite_network(self, user_ids, min_rating=5):
        '''
        Source -> Users -> Movies -> Sink
        only included edges if the rating is high enough (>= min_rating).
        '''
        print(f"building bipartite graph for {len(user_ids)} users (Rating >= {min_rating})")
        self.residual_graph = {}
        
        #make sure node exists
        def add_node(n):
            if n not in self.residual_graph:
                self.residual_graph[n] = {}

        add_node(self.source)
        add_node(self.sink)

        # Source -> Users (Capacity 1)
        for uid in user_ids:
            u_node = f"User_{uid}"
            add_node(u_node)
            # forward edge 
            self.residual_graph[self.source][u_node] = 1
            # backward edge 
            self.residual_graph[u_node][self.source] = 0
            # Users -> Movies (Capacity 1)
            # getting ratings
            user_ratings = self.builder.df_ratings[
                (self.builder.df_ratings['user_id'] == uid) & 
                (self.builder.df_ratings['rating'] >= min_rating)
            ]
            for _, row in user_ratings.iterrows():
                mid = row['movie_id']
                m_node = f"Movie_{mid}"
                add_node(m_node)
                # forward
                self.residual_graph[u_node][m_node] = 1
                # backward 
                if u_node not in self.residual_graph[m_node]:
                    self.residual_graph[m_node][u_node] = 0
                # Movies -> Sink (Capacity 1)
                # forward 
                if self.sink not in self.residual_graph[m_node]:
                    self.residual_graph[m_node][self.sink] = 1
                    self.residual_graph[self.sink][m_node] = 0

        print(f"flow network built. nodes: {len(self.residual_graph)}")

    def dfs(self, u, visited, path):
        if u == self.sink:
            return True
        
        visited.add(u)
        
        for v, cap in self.residual_graph[u].items():
            if v not in visited and cap > 0:
                path.append((u, v))
                if self.dfs(v, visited, path):
                    return True
                path.pop()
        
        return False

    def ford_fulkerson(self):
        ''' computing max flow to get the maximum matching '''
        max_flow = 0
        while True:
            visited = set()
            path = []
            # find path from source to sink
            if not self.dfs(self.source, visited, path):
                break 
            path_flow = 1
            max_flow += path_flow
            # updating 
            for u, v in path:
                self.residual_graph[u][v] -= path_flow
                self.residual_graph[v][u] += path_flow
        return max_flow

    def get_matches(self):
        ''' getting the matches'''
        matches = []
        for u in self.residual_graph:
            if u.startswith("User_"):
                for v, cap in self.residual_graph[u].items():
                    if v.startswith("Movie_") and cap == 0:
                        matches.append((u, v))
        return matches

class GraphEmbedder:
    def __init__(self, builder):
        self.builder = builder

    def get_largest_connected_component(self):
        '''
        finds the largest group of connected movies.
        spectral embedding requires a fully connected graph to look good.
        '''
        all_nodes = set(self.builder.graph.keys())
        if not all_nodes:
            return []

        visited = set()
        largest_component = []

        for node in all_nodes:
            if node not in visited:
                component = []
                stack = [node]
                visited.add(node)
                while stack:
                    curr = stack.pop()
                    component.append(curr)
                    for neighbor in self.builder.graph.get(curr, {}):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
                
                if len(component) > len(largest_component):
                    largest_component = component
        
        print(f"graph filtered from {len(all_nodes)} total to {len(largest_component)}.")
        return largest_component

    def run_analysis(self, n_clusters=5):
        print("\n--- BONUS TASK: Graph Embeddings (Spectral Embedding) ---")
        
        valid_nodes = self.get_largest_connected_component()
        
        node_to_idx = {node: i for i, node in enumerate(valid_nodes)}
        n = len(valid_nodes)

        print(f"building adjecency matrix for {n} nodes...")
        adj_mat = np.zeros((n, n))
        
        for u in valid_nodes:
            u_idx = node_to_idx[u]
            neighbors = self.builder.graph.get(u, {})
            for v, weight in neighbors.items():
                if v in node_to_idx:
                    v_idx = node_to_idx[v]
                    adj_mat[u_idx][v_idx] = weight

        # ensure A[i][j] == A[j][i]
        adj_mat = (adj_mat + adj_mat.T) / 2

        print("finding 2D vector representation...")
        embedder = SpectralEmbedding(n_components=2, affinity='precomputed', random_state=42)
        embeddings = embedder.fit_transform(adj_mat)

        print(f"clustering movies into {n_clusters} genres...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        print("generating plot...")
        plt.figure(figsize=(14, 10))
        
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', s=20, alpha=0.6)
        plt.title("Movie Similarity Graph Embeddings", fontsize=16)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.colorbar(scatter, label='Cluster ID')

        # I tried showing some famous movies on the plot, but it looked messy so i commented it.
        '''
        famous_movies = ["Toy Story", "Godfather", "Lion King", "Silence of the Lambs", "Return of the Jedi", "Titanic", "Jurassic Park"]
        
        annotated_count = 0
        
        for node_id in valid_nodes:
            title = self.builder.get_title(node_id)
            idx = node_to_idx[node_id]
            x, y = embeddings[idx, 0], embeddings[idx, 1]

            is_famous = False
            for famous in famous_movies:
                if famous.lower() in title.lower():
                    is_famous = True
                    break
            
            if is_famous:
                x_offset = random.choice([-100, -60, 60, 100]) + random.uniform(-10, 10)
                y_offset = random.choice([-100, -60, 60, 100]) + random.uniform(-10, 10)
                
                plt.annotate(
                    title,
                    xy=(x, y), 
                    xytext=(x_offset, y_offset),
                    textcoords='offset points', 
                    arrowprops=dict(arrowstyle="->", color='black', alpha=0.5),
                    fontsize=9, 
                    fontweight='bold', 
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                )
                annotated_count += 1
        '''
        
        plt.tight_layout()
        plt.show()
        print("bonus task done.")

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


    #  bipartite matching test
    print(f"\n--- TASK 2: Recommendation Matching (Bipartite Max-Flow) ---")
    # select a small bunch of random users for testing
    sample_users = builder.df_ratings['user_id'].unique()[:8]
    print(f"users selected for matching: {sample_users}")
    matcher = BipartiteMatcher(builder)
    matcher.build_bipartite_network(sample_users, min_rating=5)
    max_matches = matcher.ford_fulkerson()
    print(f"\nmax flow calculated: {max_matches}")
    print("\nmatches found:")
    results = matcher.get_matches()
    if not results:
        print("no matches found (maybe users didn't rate anything 5 stars)")
    else:
        for u_str, m_str in results:
            uid = u_str.split('_')[1]
            mid = int(m_str.split('_')[1])
            movie_title = builder.get_title(mid)
            print(f"User {uid} matches : {movie_title}")


    # graph embeddings test
    embedder = GraphEmbedder(builder)
    embedder.run_analysis(n_clusters=5)