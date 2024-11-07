#!/usr/bin/env python
import time
import obja
import numpy as np
import objToGraphe
import utils
import networkx as nx
from obja import Face 
import plotly.graph_objects as go

class Decimater(obja.Model):
    def __init__(self):
        super().__init__()
        self.deleted_faces = dict()
        self.graph = nx.empty_graph()
         
    def visualize_3d(self):
        # vertices_coords = np.array([
        #     [0, 0, 0],  # Vertex 1
        #     [1, 0, 0],  # Vertex 2
        #     [0, 1, 0],  # Vertex 3
        #     [0, 0, 1],  # Vertex 4
        # ])

        # faces = [
        #     [0, 1, 2],  # Face 1 (triangle)
        #     [0, 1, 3],  # Face 2 (triangle)
        #     [0, 2, 3],  # Face 3 (triangle)
        #     [1, 2, 3],  # Face 4 (triangle)
        # ]
        # Extraire les coordonnées des vertices
        vertices_coords = np.array([self.graph.nodes[v]['pos'] for v in self.graph.nodes])
        
        # Vérifiez que les coordonnées des vertices sont bien extraites
        if vertices_coords.shape[0] == 0:
            
            print("Aucun vertex à afficher.")
            return

        # Préparer les faces pour Plotly (les faces sont des indices de vertices)
        faces = []
        for face in self.faces.values():
            if len(face) == 3: 
                faces.append(face)
        
        if len(faces) == 0:
            print("Aucune face valide pour l'affichage.")
            return
        print(faces)
        # Créer la visualisation 3D avec Plotly
        fig = go.Figure(data=[go.Mesh3d(
            x=vertices_coords[:, 0],  # Coordonnée x des vertices
            y=vertices_coords[:, 1],  # Coordonnée y des vertices
            z=vertices_coords[:, 2],  # Coordonnée z des vertices
            i=[f[0] for f in faces],  # Indices de la première coordonnée des faces
            j=[f[1] for f in faces],  # Indices de la deuxième coordonnée des faces
            k=[f[2] for f in faces],  # Indices de la troisième coordonnée des faces
            opacity=0.5,
            color='blue',
            flatshading=True  # Pour un rendu sans arêtes visibles
        )])

        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                zaxis=dict(showgrid=False)
            ),
            title="3D Object Visualization"
        )

        fig.show()

    def load_graph(self, filename):
        self.vertices, self.faces, self.faces_counter = objToGraphe.load_obj(filename)
        self.graph = objToGraphe.create_graph(self.vertices, self.faces, False)

    def get_graph(self):
        return self.graph

    def compress_model(self, output_file):
        compressed = False
        operations = []
        while not compressed:
            step_operations = self.step_compression()
            # print(step_operations)
            # print(len(step_operations))
            # exit()
            if len(step_operations) > 0:
                operations = operations + step_operations
            else:
                compressed = True


        operations.reverse()

            

        # Créer un objet Output pour écrire dans le fichier de sortie
        with open(output_file, 'w') as output_file:
            output_model = obja.Output(output_file, random_color=True)
            
            # for v in self.graph.nodes:
            #     output_model.add_vertex("vertex", v, v["pos"])
            # for face_index, face in enumerate(self.faces):
            #     operations.append(('face', face_index, face))
            
            # Appliquer les opérations de compression au modèle
            for (ty, index, value) in operations:
                if ty == "vertex":
                    output_model.add_vertex(index, value)
                elif ty == "face":
                    value = [str(int(value_i)) for value_i in value]
                    output_model.add_face(index, Face.from_array(value))
                else:
                    output_model.edit_vertex(index, value)

    def step_compression(self):
        mst = objToGraphe.minimum_spanning_tree(self.graph)
        collapsed_vertices = set()
        collapsed_edges = []
        # mst_edges = list(mst.edges())
        mst_edges = list(nx.dfs_edges(mst))
        
        operations = []
        ed = 0
        for edge in mst_edges:
            ed += 1
            print("--- Edge numero", ed, "/", len(mst_edges), edge)
            print("collapsed edges", collapsed_edges)
            print("collapsed vertices", collapsed_vertices)
            v1, v2 = edge
             # Vérifie si v1 ou v2 est déjà collapse
            if v1 in collapsed_vertices or v2 in collapsed_vertices:
                continue

            # Récupère les voisins de v1 et v2
            neighbors_v1 = set(self.graph.neighbors(v1))
            neighbors_v2 = set(self.graph.neighbors(v2))
            # neighbors = neighbors_v1.union(neighbors_v2)
            neighbors = neighbors_v1.intersection(neighbors_v2)

            print(v1, neighbors_v1)
            print(v2, neighbors_v2)

            valid = True
            
            # Pour chaque voisin w, vérifie si le couple (v1, v2, w) est un triangle valide
            
            for w in neighbors:
                if w != v1 and w != v2 and not utils.is_valid_triangle(v1, v2, w, self.faces):
                    valid = False
                    #print("Première condition fausse")
                    break

            if not valid:
                continue

            for w1 in neighbors_v1:
                if w1 != v2:
                    for w2 in neighbors_v2:
                        if w2 != v1:
                            if set([w1,w2]) in collapsed_edges:
                                valid = False
                                print("Deuxième condition fausse")
                                break

            if not valid:
                continue

            # Si c'est bon, v2 va être collapse
            collapsed_vertices.add(v2)
            collapsed_vertices.add(v1)
            collapsed_edges.append(set([v1,v2]))
            
            # Enlève les faces ayant v2
            print("collapse", v2)
            tmp_deleted_faces = dict()
            for face_index, face in self.faces.items():
                if v2 in face:
                    tmp_deleted_faces[face_index] = face
                    operations.append(('face', face_index, face))
            for face_index, face in tmp_deleted_faces.items():
                print("effacer", face_index, face)
                self.deleted_faces[face_index] = face
                del self.faces[face_index]
            
            # Reconnecter les voisins de v2 à v1
            new_edges = []
            for n2 in neighbors_v2:
                if n2 != v1 and (n2 not in neighbors_v1) and (n2 not in collapsed_vertices) and (v1 not in collapsed_vertices) :
                    print("ajout edge", (v1, n2))
                    new_edges.append((v1, n2))
                    
                    for v1 in neighbors_v1:
                        if v1 != n2 and self.graph.has_edge(v1, n2) and v1 not in collapsed_vertices:
                            new_face = [v1, n2, v1]
                            print("ajout face", self.faces_counter, new_face)
                            self.faces[self.faces_counter] = (new_face)  
                            self.faces_counter += 1
                            operations.append(('edit_vertex', n2, self.graph.nodes[n2]))  
            
            vertex_v2 = self.graph.nodes[v2]['pos']
            operations.append(('vertex', v2, vertex_v2))
            self.graph.remove_node(v2)

        self.graph.add_edges_from(new_edges)
        objToGraphe.draw_graph(self.graph)
        # self.visualize_3d()
        print("----------------------- FIN ITERATION -----------------------")
        return operations

def main():
    """
    Runs the program on the model given as parameter.
    """
    np.seterr(invalid = 'raise')
    model = Decimater()
    filename='example/test.obj'

    model.load_graph(filename)

    # graph = model.get_graph()
    # objToGraphe.visualize_mst_simple(graph, objToGraphe.minimum_spanning_tree(graph))

    with open('example/test.obja', 'w') as output:
        model.compress_model(output)

if __name__ == '__main__':
    main()