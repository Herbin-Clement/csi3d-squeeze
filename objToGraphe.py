import numpy as np
import networkx as nx
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
def hausdorff_distance(point1, point2):
    return max(directed_hausdorff(point1, point2)[0], directed_hausdorff(point2, point1)[0])

def is_valid_triangle(v1, v2, w, graph, faces):
    valid = False
    for face in faces:
        if set([w,v1,v2]) == face:
            valid = True
            break
    return valid

def is_valid_quad(v1, v2, w1, w2, edges):
    # Vérifie si e1 et e2 ne peuvent pas être réduits ensemble
    return not ((v1, v2) in edges and (w1, w2) in edges)


def visualize_mst_simple(graph, mst):
    pos = nx.get_node_attributes(graph, 'pos')

    # Vérifier si les coordonnées sont en 3D et projeter en 2D
    if any(len(coord) == 3 for coord in pos.values()):
        pos = {k: (v[0], v[1]) for k, v in pos.items()}  # Utiliser x et y

    # Dessiner uniquement l'arbre couvrant minimal
    plt.figure(figsize=(10, 7))
    nx.draw(mst, pos, with_labels=True, node_color='lightblue', edge_color='red', node_size=500, font_size=12)

    plt.title("Minimum Spanning Tree")
    plt.show()

def compress_model(graph, faces):
    collapsed_vertices = set()
    new_edges = []
    edges = list(graph.edges())
    
    for edge in edges:
        v1, v2 = edge
        if v1 in collapsed_vertices or v2 in collapsed_vertices:
            continue
        
        # Vérifier les connexions
        neighbors = set(graph.neighbors(v1)).union(set(graph.neighbors(v2)))
        valid = True
        # vérifier que le triangle formé est valide
        for w in neighbors:
            if w != v1 and w != v2 and not is_valid_triangle(v1, v2, w, graph, faces):
                valid = False
                break
        
        if valid:
            # Vérifier les quadrilatères
            for w1, w2 in edges:
                if is_valid_quad(v1, v2, w1, w2, edges):
                    continue
                else:
                    valid = False
                    break
        
        if valid:
            # Effectuer le collapse
            collapsed_vertices.add(v2)  # Collapse v2 into v1
            new_edges.extend([(v1, w) for w in neighbors if w not in collapsed_vertices])

    # Mettre à jour le graphe avec les nouveaux sommets et arêtes
    for v in collapsed_vertices:
        graph.remove_node(v)
    
    graph.add_edges_from(new_edges)


def load_obj(filename):
    vertices = []
    faces = []
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '): 
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '): 
                parts = line.split()
                face = [int(part.split('/')[0]) - 1 for part in parts[1:]]  # Indices des sommets
                faces.append(set(face))
    
    return np.array(vertices), faces

def create_graph(vertices, faces):
    G = nx.Graph()
    
    for i, vertex in enumerate(vertices):
        G.add_node(i, pos=vertex)
    
    for face in faces:
        for i in range(len(face)):
            point1 = vertices[face[i]]
            point2 = vertices[face[(i + 1) % len(face)]]
            distance = hausdorff_distance(point1.reshape(1, -1), point2.reshape(1, -1)) 
            G.add_edge(face[i], face[(i + 1) % len(face)], weight=distance)
    return G

def minimum_spanning_tree(graph):
    mst = nx.minimum_spanning_tree(graph)
    return mst