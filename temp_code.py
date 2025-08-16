import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import base64
import io
import json

def analyze_network():
    edges_df = pd.read_csv('edges.csv')
    G = nx.from_pandas_edgelist(edges_df, source='source', target='target', create_using=nx.Graph())

    edge_count = G.number_of_edges()

    degrees = dict(G.degree())
    highest_degree_node = max(degrees, key=degrees.get)
    average_degree = sum(degrees.values()) / len(degrees)

    density = nx.density(G)

    shortest_path_length = nx.shortest_path_length(G, source='Alice', target='Eve')

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title('Network Graph')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    network_graph_b64 = base64.b64encode(buf.read()).decode()

    degrees_list = list(degrees.values())
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(degrees_list)), sorted(degrees_list, reverse=True), color='green')
    plt.xlabel('Nodes sorted by degree')
    plt.ylabel('Degree')
    plt.title('Degree Histogram')
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    plt.close()
    buf2.seek(0)
    degree_histogram_b64 = base64.b64encode(buf2.read()).decode()

    return {
        'edge_count': edge_count,
        'highest_degree_node': highest_degree_node,
        'average_degree': average_degree,
        'density': density,
        'shortest_path_alice_eve': shortest_path_length,
        'network_graph': network_graph_b64,
        'degree_histogram': degree_histogram_b64
    }

if __name__ == "__main__":
    result = analyze_network()
    print(json.dumps(result))