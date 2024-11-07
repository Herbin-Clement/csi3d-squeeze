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
    vertices = []  # Utiliser une liste pour stocker chaque sommet temporairement
    faces = []
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '): 
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)  # Ajouter chaque sommet en tant que sous-liste
            elif line.startswith('f '): 
                parts = line.split()
                face = [int(part.split('/')[0]) - 1 for part in parts[1:]]  # Indices des sommets
                faces.append(list(set(face)))
    
    # Convertir vertices en un tableau numpy 2D
    vertices = np.array(vertices)
    idx_to_faces = {i: face for i, face in enumerate(faces)}
    faces_count = len(faces)
    print(idx_to_faces)
    return vertices, idx_to_faces, faces_count

def draw_graph(G):
    pos = nx.get_node_attributes(G, 'pos')
        
    # Convert positions to dictionary format compatible with nx.draw
    pos = {k: v[:2] for k, v in pos.items()}  # Ensure all positions are 2D
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.show()

def create_graph(vertices, faces, visualize):
    """
    Create a networkX graph from a list of faces and vertices.
    """
    G = nx.Graph()
    
    # Ensure vertices are in correct format
    vertices = np.array(vertices)  # Convert to numpy array for easier manipulation
    if vertices.ndim == 1:
        raise ValueError("Vertices should be a 2D array, where each vertex is a point [x, y] or [x, y, z].")
    
    for i, vertex in enumerate(vertices):
        G.add_node(i, pos=vertex)
    
    for face in faces.values():
        for i in range(len(face)):
            point1 = np.array(vertices[face[i]])
            point2 = np.array(vertices[face[(i + 1) % len(face)]])
            distance = utils.euclidean_distance(point1.reshape(1, -1), point2.reshape(1, -1)) 
            G.add_edge(face[i], face[(i + 1) % len(face)], weight=distance)
    
    if visualize:
        draw_graph(G)
    return G

def minimum_spanning_tree(graph):
    """
    Compute the minimum spanning tree of a graph.
    """
    mst = nx.minimum_spanning_tree(graph)
    return mst