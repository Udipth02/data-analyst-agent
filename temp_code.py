import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import json

def analyze_network():
    df = pd.read_csv('edges.csv')
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'])
    edge_count = G.number_of_edges()
    degrees = dict(G.degree())
    highest_degree_node = max(degrees, key=degrees.get)
    average_degree = sum(degrees.values()) / len(degrees)
    density = nx.density(G)
    shortest_path_length = nx.shortest_path_length(G, source='Alice', target='Eve')
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    network_png = base64.b64encode(buf.read()).decode('utf-8')
    degree_values = list(degrees.values())
    plt.figure(figsize=(8,6))
    plt.bar(range(len(degree_values)), degree_values, color='green')
    plt.xlabel('Nodes')
    plt.ylabel('Degree')
    plt.title('Degree Distribution')
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    plt.close()
    buf2.seek(0)
    histogram_png = base64.b64encode(buf2.read()).decode('utf-8')
    result = {
        'edge_count': edge_count,
        'highest_degree_node': highest_degree_node,
        'average_degree': average_degree,
        'density': density,
        'shortest_path_alice_eve': shortest_path_length,
        'network_graph': 'data:image/png;base64,' + network_png,
        'degree_histogram': 'data:image/png;base64,' + histogram_png
    }
    return json.dumps(result)

if __name__ == "__main__":
    result = analyze_network()
    print(result)