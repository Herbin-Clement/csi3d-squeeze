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
        vertices_coords = np.array([
            [0, 0, 0],  # Vertex 1
            [1, 0, 0],  # Vertex 2
            [0, 1, 0],  # Vertex 3
            [0, 0, 1],  # Vertex 4
        ])

        faces = [
            [0, 1, 2],  # Face 1 (triangle)
            [0, 1, 3],  # Face 2 (triangle)
            [0, 2, 3],  # Face 3 (triangle)
            [1, 2, 3],  # Face 4 (triangle)
        ]
        # Extraire les coordonnées des vertices
        
        vertices_coords = np.array([self.graph.nodes[v]['pos'] for v in self.graph.nodes])
        
        # Vérifiez que les coordonnées des vertices sont bien extraites
        if vertices_coords.shape[0] == 0:
            
            # print("Aucun vertex à afficher.")
            return

        # Préparer les faces pour Plotly (les faces sont des indices de vertices)
        faces = []
        for face in self.faces.values():
            if len(face) == 3: 
                faces.append(face)
        
        if len(faces) == 0:
            print("Aucune face valide pour l'affichage.")
            return
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

    def stat(self, operations, printFaces=True):
        addVertex = 0
        addFace = 0
        editFace = 0
        removeFace = 0
        editFaceVertex = 0
        for (ty, index, value) in operations:
            if ty == "vertex":
                addVertex += 1
            elif ty == "face":
                addFace += 1
            elif ty == "edit_face":
                editFace += 1
            elif ty == "remove_face":
                removeFace += 1
            elif ty == "edit_face_vertex":
                editFaceVertex += 1
        triangles = [clique for clique in nx.enumerate_all_cliques(self.graph) if len(clique) == 3]
        print(f"graph: {len(triangles)}, our: {len(self.faces)}")
        if printFaces:
            print(sorted(triangles))
            print(sorted([face for face in self.faces.values()]))
        print(f"av {addVertex} af {addFace} ef {editFace} rf {removeFace} efv {editFaceVertex}")

    def compress_model(self, outputFilename, maxDecimateRatio, maxStep=10):
        compressed = False
        operations = []
        step = 0
        objToGraphe.draw_graph(self.graph)
        decimateRatio = 1
        initialNumberOfNodes = self.graph.number_of_nodes()

        while decimateRatio > maxDecimateRatio and maxStep > step and not compressed:
            print(f"=========== Step {step + 1} ===========")
            step_operations = self.step_compression()
            operations = operations + step_operations
            step += 1
            decimateRatio = self.graph.number_of_nodes() / initialNumberOfNodes 
            print(f"Nb op: {len(step_operations)}")
            self.stat(operations)
            print(f"Pourcentage de decimation actuel: {decimateRatio}")
            if len(step_operations) == 0 or decimateRatio < maxDecimateRatio:
                compressed = True

        operations.reverse()

        # Créer un objet Output pour écrire dans le fichier de sortie
        with open(outputFilename, 'w') as output_file:
            output_model = obja.Output(output_file, random_color=True)
            
            for v in self.graph.nodes:
                output_model.add_vertex(v, self.graph.nodes[v]["pos"])
            for face_index, face in self.faces.items():
                output_model.add_face(face_index, Face(face[0], face[1], face[2]))
            
            # Appliquer les opérations de compression au modèle
            for (ty, index, value) in operations:
                if ty == "vertex":
                    output_model.add_vertex(index, value)
                elif ty == "face":
                    output_model.add_face(index, Face(value[0], value[1], value[2]))
                elif ty == "edit_face":
                    output_model.edit_face(index, Face(value[0], value[1], value[2]))
                elif ty == "remove_face":
                    output_model.remove_face(index)
                elif ty == "edit_face_vertex":
                    output_model.edit_face_vertex(index, value[0], value[1])
                else:
                    output_model.edit_vertex(index, value)
        utils.pause()
        

    def step_compression(self):
        collapsed_vertices = []
        removed_vertices = []
        collapsed_edges = []
        new_edges = []
        operations = []
        ed = 0
        mst = objToGraphe.minimum_spanning_tree(self.graph)
        mst_edges = list(nx.dfs_edges(mst))
        for i, edge in enumerate(mst_edges):
            ed += 1
            v1, v2 = edge
             # Vérifie si v1 ou v2 est déjà collapse
            if v1 in collapsed_vertices or v2 in collapsed_vertices:
                continue

            # Récupère les voisins de v1 et v2
            neighbors_v1 = set(self.graph.neighbors(v1))
            neighbors_v2 = set(self.graph.neighbors(v2))
            neighbors = sorted(list(neighbors_v1.intersection(neighbors_v2)))

            valid = True
                        
            for w in neighbors:
                if w != v1 and w != v2 and not utils.is_valid_triangle(v1, v2, w, self.faces):
                    valid = False
                    break

            if not valid:
                continue

            for w1 in neighbors_v1:
                if w1 != v2:
                    for w2 in neighbors_v2:
                        if w2 != v1:
                            if sorted([w1,w2]) in collapsed_edges:
                                valid = False
                                break

            if not valid:
                continue

            # Si c'est bon, v2 va être collapse
            collapsed_vertices.append(v1)
            collapsed_vertices.append(v2)
            collapsed_edges.append(sorted([v1,v2]))
            removed_vertices.append(v2)
            print(f"remove {v2}")

            for n2 in neighbors_v2:
                if n2 != v1 and (n2 not in neighbors_v1) and n2 not in removed_vertices:
                    print(f"add edge f{sorted([v1, n2])}")
                    new_edges.append(sorted([v1, n2]))

            # Enlève les faces ayant v2
            tmp_deleted_faces = dict()
            tmp_add_faces = dict()
            for face_index, face in self.faces.items():
                if v2 in face:
                    other_face = face[:]
                    other_face.remove(v2)
                    other_face.append(v1)
                    if sorted(other_face) not in self.faces.values() and v1 not in face:
                        # print(f"edit face vertex {face} {other_face}")
                        # self.faces[face_index] = sorted(other_face)
                        # operations.append(('edit_face_vertex', face_index, (face.index(v2), v1)))
                        tmp_deleted_faces[face_index] = face
                        tmp_add_faces[self.faces_counter] = sorted(other_face)
                        self.faces_counter += 1
                    else:
                        tmp_deleted_faces[face_index] = face
            for face_index, face in tmp_deleted_faces.items():
                operations.append(('face', face_index, face))
                print(f"add {face}")
                self.deleted_faces[face_index] = face
                del self.faces[face_index]
            for face_index, face in tmp_add_faces.items():
                operations.append(('remove_face', face_index, face))
                print(f"add {face}")
                self.faces[face_index] = face
            
            vertex_v2 = self.graph.nodes[v2]['pos']
            operations.append(('vertex', v2, vertex_v2))
        
        # Remove vertice
        b = len(self.graph.nodes)
        for node in removed_vertices:
            self.graph.remove_node(node)

        # Add edges
        for edgeaa in new_edges:
            if not (edgeaa[0] in removed_vertices or edgeaa[1] in removed_vertices):
                self.graph.add_edge(edgeaa[0], edgeaa[1])
        # self.graph.add_edges_from(new_edges)
        
        a = len(self.graph.nodes)
        print(f"{b} => {a}: -{b - a}")
        objToGraphe.draw_graph(self.graph)
        #self.visualize_3d()
        # print("----------------------- FIN ITERATION -----------------------")
        
        return operations

    def edit_face(self, face_index, current_face, new_face):
        self.faces[face_index] = new_face
        print(f"e1 {current_face} {new_face}")
        return ('edit_face', face_index, new_face)
    
    def remove_face(self, new_face):
        self.faces[self.faces_counter] = (new_face)  
        print(f"rm1 {new_face}")
        self.faces_counter += 1
        return ('remove_face', self.faces_counter, new_face)


def main():
    """
    Runs the program on the model given as parameter.
    """
    np.seterr(invalid = 'raise')
    model = Decimater()
    filename='example/suzanne.obj'

    model.load_graph(filename)
    
    model.compress_model("example/bunny.obja", maxDecimateRatio=0.3, maxStep=3)
    #model.visualize_3d()
if __name__ == '__main__':
    main()