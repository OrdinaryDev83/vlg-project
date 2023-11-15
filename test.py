import itertools
import networkx as nx
import matplotlib.pyplot as plt
from more_itertools import powerset
import time
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
import tqdm
from tqdm import tqdm, trange
import multiprocessing

def generate_random_graph(node_count, edge_prob):
    """
    Generates a random graph with a given number of nodes and probability of edge creation.
    """
    return nx.erdos_renyi_graph(node_count, edge_prob)


def is_clique(G, S):
    for i in S:
        for j in S:
            if i != j and not G.has_edge(i, j):
                return False
    return True

def recherche_clique_naive(G):
    n = G.number_of_nodes()
    taille_max_clique = 0
    clique_max = set()
    
    for S in powerset(G.nodes()):
        if len(S) > taille_max_clique and is_clique(G, S):
            taille_max_clique = len(S)
            clique_max = S
            
    return clique_max

def bron_kerbosh1(R, P, X, G):
    if not P and not X:
        return [R]
    else:
        cliques = []
        for v in P.copy():
            N = G.neighbors(v)
            cliques.extend(
                bron_kerbosh1(
                    R.union({v}),
                    P.intersection(N),
                    X.intersection(N),
                    G,
                )
            )
            P.remove(v)
            X.union({v})
        return cliques

def search_clique_bron_kerbosh1(G):
    return max(bron_kerbosh1(set(), set(G.nodes()), set(), G), key=len)

def bron_kerbosh2(R, P, X, G):
    if not P and not X:
        return [R]
    pivot = P.union(X).pop()  # TODO: compare both pivots
    pivot = max(P.union(X), key=G.degree)
    N = G.neighbors(pivot)
    cliques = []
    for v in P.difference(N):
        N_v = G.neighbors(v)
        cliques.extend(
            bron_kerbosh2(
                R.union({v}),
                P.intersection(N_v),
                X.intersection(N_v),
                G,
            )
        )
        P.remove(v)
        X.union({v})

    return cliques

def search_clique_bron_kerbosh2(G):
    return max(bron_kerbosh2(set(), set(G.nodes()), set(), G), key=len)

def degen(G):
    ordering = []
    deg = dict(G.degree())
    while deg:
        v = min(deg, key=deg.get)
        ordering.append(v)
        del deg[v]
        for w in G.neighbors(v):
            if w in deg:
                deg[w] -= 1
    return ordering


def bron_kerbosh3(G):
    R = set()
    P = set(G.nodes())
    X = set()
    cliques = []
    for v in degen(G):
        N = G.neighbors(v)
        cliques.extend(
            bron_kerbosh2(
                R.union({v}),
                P.intersection(N),
                X.intersection(N),
                G,
            )
        )
        P.remove(v)
        X.union({v})
    return cliques

def search_clique_bron_kerbosh3(G):
    return max(bron_kerbosh3(G), key=len)

from concurrent.futures import ThreadPoolExecutor, as_completed

def benchmark_function_with_timeout(func, graph):
    """
    Benchmarks a single function on a given graph, with a timeout.
    Uses multiprocessing to enforce a timeout.
    """
    with multiprocessing.Pool(1) as pool:
        # Use apply_async to add a timeout
        result = pool.apply_async(func, args=(func, graph))
        
        try:
            # Set a timeout for the function execution
            execution_time = result.get(timeout=5)
        except multiprocessing.TimeoutError:
            execution_time = None  # None indicates a timeout

    return execution_time

def parallel_benchmark_clique_search_functions(functions, node_counts, edge_prob=0.5):
    """
    Benchmarks a list of clique search functions on randomly generated graphs with varying node counts.
    Each function is run in parallel for each node count using threading to manage parallel execution.
    """
    # Store execution times for each function and node count
    execution_times = {func.__name__: [] for func in functions}

    for node_count in node_counts:
        # Generate a random graph
        G = generate_random_graph(node_count, edge_prob)

        # Create threads for each function
        with ThreadPoolExecutor(max_workers=len(functions)) as executor:
            future_to_func = {executor.submit(benchmark_function_with_timeout, func, G): func for func in functions}
            for future in as_completed(future_to_func):
                func = future_to_func[future]
                try:
                    time = future.result()
                    execution_times[func.__name__].append(time)
                except Exception as exc:
                    execution_times[func.__name__].append(None)
    # Plotting
    plt.figure(figsize=(10, 6))
    for func_name, times in execution_times.items():
        plt.plot(node_counts, times, label=func_name)

    plt.xlabel('Node Count')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Clique Search Function Benchmarks (Parallel Execution)')
    plt.legend()
    plt.grid(True)
    plt.show()

functions = [
    # recherche_clique_naive,
    search_clique_bron_kerbosh1,
    search_clique_bron_kerbosh2,
    search_clique_bron_kerbosh3
]

node_counts = range(5, 200, 5)

# Example usage with dummy functions
parallel_benchmark_clique_search_functions(functions, node_counts)