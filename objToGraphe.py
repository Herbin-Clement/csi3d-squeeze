import numpy as np
import networkx as nx
import utils
import matplotlib.pyplot as plt

def visualize_mst_simple(graph, mst):
    """
    Visualize the minimum spanning tree of a graph.
    """
    pos = nx.get_node_attributes(graph, 'pos')

    # Vérifier si les coordonnées sont en 3D et projeter en 2D
    if any(len(coord) == 3 for coord in pos.values()):
        pos = {k: (v[0], v[1]) for k, v in pos.items()}  # Utiliser x et y

    # Dessiner uniquement l'arbre couvrant minimal
    plt.figure(figsize=(10, 7))
    nx.draw(mst, pos, with_labels=True, node_color='lightblue', edge_color='red', node_size=500, font_size=12)

    plt.title("Minimum Spanning Tree")
    plt.show()

def load_obj(filename):
    """
    Load the .obj file.
    """
    vertices = np.array([])
    faces = []
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '): 
                parts = line.split()
                vertices = np.concatenate((vertices, np.array([float(parts[1]), float(parts[2]), float(parts[3])])), axis=0)
            elif line.startswith('f '): 
                parts = line.split()
                face = [int(part.split('/')[0]) - 1 for part in parts[1:]]  # Indices des sommets
                faces.append(list(set(face)))
    
    return vertices, faces

def create_graph(vertices, faces, visualize):
    """
    Create a networkX graph from a list of faces and vertices.
    """
    G = nx.Graph()
    
    for i, vertex in enumerate(vertices):
        G.add_node(i, pos=vertex)
    
    for face in faces:
        for i in range(len(face)):
            point1 = vertices[face[i]]
            point2 = vertices[face[(i + 1) % len(face)]]
            distance = utils.euclidean_distance(point1.reshape(1, -1), point2.reshape(1, -1)) 
            G.add_edge(face[i], face[(i + 1) % len(face)], weight=distance)
    if visualize:
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        plt.show()
    return G

def minimum_spanning_tree(graph):
    """
    Compute the minimum spanning tree of a graph.
    """
    mst = nx.minimum_spanning_tree(graph)
    return mst