#!/usr/bin/env python

import obja
import numpy as np
import objToGraphe
import utils
import networkx as nx

class Decimater(obja.Model):
    def __init__(self):
        super().__init__()
        self.deleted_faces = set()
        self.graph = nx.empty_graph()

    def load_graph(self, filename):
        self.vertices, self.faces = objToGraphe.load_obj(filename)
        self.graph = objToGraphe.create_graph(self.vertices, self.faces, True)

    def get_graph(self):
        return self.graph

    def compress_model(self, output_file):
        compressed = False
        operations = []
        while not compressed:
            step_operations = self.step_compression()
            print(step_operations)
            if len(step_operations) > 0:
                operations = operations + step_operations
            else:
                compressed = True

        operations.reverse()
        
        output_model = obja.Output(output_file, random_color=True)
        
        for (ty, index, value) in operations:
            if ty == "vertex":
                output_model.add_vertex(index, value)
            elif ty == "face":
                output_model.add_face(index, value)
            else:
                output_model.edit_vertex(index, value)

    def step_compression(self):
        mst = objToGraphe.minimum_spanning_tree(self.graph)
        collapsed_vertices = set()
        new_edges = []
        mst_edges = list(mst.edges())
        
        operations = []

        for edge in mst_edges:
            v1, v2 = edge
             # Vérifie si v1 ou v2 est déjà collapse
            if v1 in collapsed_vertices or v2 in collapsed_vertices:
                continue

            # Récupère les voisins de v1 et v2
            neighbors_v1 = set(self.graph.neighbors(v1))
            neighbors_v2 = set(self.graph.neighbors(v2))
            neighbors = neighbors_v1.union(neighbors_v2)

            valid = True
            
            # Pour chaque voisin w, vérifie si le couple (v1, v2, w) est un triangle valide
            for w in neighbors:
                if w != v1 and w != v2 and not utils.is_valid_triangle(v1, v2, w, self.faces):
                    valid = False
                    print("Première condition fausse")
                    break

            if not valid:
                continue

            # Pour chaque arête (w1, w2) dans l'arbre couvrant, vérifie si on peut collapse le quad (v1, v2, w1, w2)
            for w1, w2 in mst_edges:
                if not utils.can_collaps_quad(v1, v2, w1, w2, mst_edges):
                    valid = False
                    print("Deuxième condition fausse")
                    break

            if not valid:
                continue

            print("valide :)")
            # Si c'est bon, v2 va être collapse
            collapsed_vertices.add(v2)
            
            # Ajoute les nouvelles faces
            for face_index, face in enumerate(self.faces):
                if v2 in [face.a, face.b, face.c]:
                    self.deleted_faces.add(face_index)
                    operations.append(('face', face_index, face))

            # Reconnecter les voisins de v2 à v1
            for neighbor in neighbors_v2:
                if neighbor != v1 and neighbor not in collapsed_vertices:
                    new_edges.append((v1, neighbor))
                    
                    for neighbor2 in neighbors_v1:
                        if neighbor2 != neighbor and neighbor2 not in collapsed_vertices:
                            new_face = utils.create_face(v1, neighbor, neighbor2) 
                            self.faces.append(new_face)  
                            face_index = len(self.faces) - 1  
                            operations.append(('edit_vertex', neighbor, self.graph.nodes[neighbor]))  
                vertex_v2 = self.graph.nodes[v2]  
                operations.append(('vertex', v2, vertex_v2))

                self.graph.remove_node(v2)

        self.graph.add_edges_from(new_edges)

        return operations



def main():
    """
    Runs the program on the model given as parameter.
    """
    np.seterr(invalid = 'raise')
    model = Decimater()
    filename='example/test.obj'

    model.load_graph(filename)
    graph = model.get_graph()

    # objToGraphe.visualize_mst_simple(graph, objToGraphe.minimum_spanning_tree(graph))
    
    with open('example/suzanne.obja', 'w') as output:
        model.compress_model(output)

if __name__ == '__main__':
    main()
