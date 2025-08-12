import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

def analyze_network():
    edges_df = pd.read_csv('edges.csv')
    G = nx.from_pandas_edgelist(edges_df, source='source', target='target', create_using=nx.Graph())
    edge_count = G.number_of_edges()
    degrees = dict(G.degree())
    highest_degree_node = max(degrees, key=degrees.get)
    average_degree = sum(degrees.values()) / len(degrees)
    density = nx.density(G)
    shortest_path_length = nx.shortest_path_length(G, source='Alice', target='Eve')
    plt.figure(figsize=(8,8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    graph_png = base64.b64encode(buffer.read()).decode()
    degree_values = list(degrees.values())
    plt.figure(figsize=(8,6))
    sns.barplot(x=degree_values, y=list(degrees.keys()), color='green')
    plt.xlabel('Degree')
    plt.ylabel('Node')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    histogram_png = base64.b64encode(buffer.read()).decode()
    return [edge_count, highest_degree_node, average_degree, density, shortest_path_length, graph_png, histogram_png]

if __name__ == "__main__":
    result = analyze_network()
    print(result)