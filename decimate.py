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
        # 1. Extraire les coordonnées des sommets
        vertices_coords = {v: np.array(self.graph.nodes[v]['pos']) for v in self.graph.nodes}
        vertex_labels = list(vertices_coords.keys())
        # Vérifiez que les coordonnées des vertices sont bien extraites
        if len(vertices_coords) == 0:
            print("Aucun vertex à afficher.")
            return

        # 2. Récupérer les faces du graphe
        faces = self.get_triangles_from_graph()

        # 3. Créer un mappage de "vertexToIndex" (associant chaque sommet à un index continu)
        vertexToIndex = {v: i for i, v in enumerate(vertices_coords)}

        # 4. Modifier les faces pour remplacer les indices des sommets par leurs indices continus
        updated_faces = []
        for face in faces:
            updated_faces.append([vertexToIndex[v] for v in face])

        # 5. Convertir les coordonnées en tableau NumPy pour les afficher avec Plotly
        coords_array = np.array(list(vertices_coords.values()))  # Liste de toutes les coordonnées sous forme de tableau NumPy

        # Vérifiez si des faces ont été trouvées
        if len(updated_faces) == 0:
            print("Aucune face valide pour l'affichage.")
            return

        # Créer les arêtes (edges) sous forme de lignes
        edges_x, edges_y, edges_z = [], [], []
        temp=[[6,7], [7,8]]
        nb=0
        for u, v in self.graph.edges():
            nb+=1
            """if (u,v)==(6,7):
                 continue"""
            
            x_edge, y_edge, z_edge = [vertices_coords[u][0], vertices_coords[v][0]], [vertices_coords[u][1], vertices_coords[v][1]], [vertices_coords[u][2], vertices_coords[v][2]]
            edges_x.extend(x_edge)
            edges_y.extend(y_edge)
            edges_z.extend(z_edge)
        
        # Créer les sommets sous forme de points
        vertex_x = [coord[0] for coord in vertices_coords.values()]
        vertex_y = [coord[1] for coord in vertices_coords.values()]
        vertex_z = [coord[2] for coord in vertices_coords.values()]

        # Créer la visualisation 3D avec Plotly
        fig = go.Figure()

        # Ajouter le maillage (faces)
        """fig.add_trace(go.Mesh3d(
            x=coords_array[:, 0],  # Coordonnée x des vertices
            y=coords_array[:, 1],  # Coordonnée y des vertices
            z=coords_array[:, 2],  # Coordonnée z des vertices
            i=[f[0] for f in updated_faces],  # Indices de la première coordonnée des faces
            j=[f[1] for f in updated_faces],  # Indices de la deuxième coordonnée des faces
            k=[f[2] for f in updated_faces],  # Indices de la troisième coordonnée des faces
            opacity=0.5,
            color='blue',
            flatshading=True  # Pour un rendu sans arêtes visibles
        ))"""

        # Ajouter les arêtes (edges)
        fig.add_trace(go.Scatter3d(
            x=edges_x,
            y=edges_y,
            z=edges_z,
            mode='lines',
            line=dict(color='black', width=2),
            opacity=0.2,
            name="Arêtes"
        ))

        # Ajouter les sommets (vertices)
        fig.add_trace(go.Scatter3d(
            x=vertex_x,
            y=vertex_y,
            z=vertex_z,
            mode='markers+text',  # Afficher les marqueurs et le texte
            marker=dict(size=5, color='red', symbol='circle'),
            text=vertex_labels,  # Afficher les noms des sommets
            textposition="top center",  # Position du texte
            hoverinfo='text',  # Affiche seulement le texte au survol
            name="Sommets",
            opacity=0.2
        ))

        # Mettre à jour la disposition du graphique
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                zaxis=dict(showgrid=False)
            ),
            title="3D Object Visualization with Edges and Vertices"
        )
        
        fig.show()

    def load_graph(self, filename):
        self.vertices, self.faces, self.faces_counter = objToGraphe.load_obj(filename)
        self.graph = objToGraphe.create_graph(self.vertices, self.faces, False)

    def get_graph(self):
        return self.graph

    def compress_model(self, output_filename,totalVertex, pourcentageDecimate, max_step=10):
        compressed = False
        operations = []
        step = 0
        objToGraphe.draw_graph(self.graph)
        pourcentageDecimeActuel = self.graph.number_of_nodes() / totalVertex*100 
        while (pourcentageDecimeActuel > pourcentageDecimate and step < max_step) and not compressed:
            pourcentageDecimeActuel = self.graph.number_of_nodes() / totalVertex*100 
            print(f"Pourcentage de decimation actuel: {pourcentageDecimeActuel}")
            print(f"=========== Step {step + 1} ===========")
            step_operations = self.step_compression()
            operations = operations + step_operations
            if len(step_operations) == 0:
                compressed = True
            step += 1
            print(f"Nb op: {len(step_operations)}")
        objToGraphe.draw_graph(self.graph)
        operations.reverse()

        # Créer un objet Output pour écrire dans le fichier de sortie
        with open(output_filename, 'w') as output_file:
            output_model = obja.Output(output_file, random_color=True)
            
            for v in self.graph.nodes:
                output_model.add_vertex(v, self.graph.nodes[v]["pos"])
            for face_index, face in self.faces.items():
                output_model.add_face(face_index, Face(face[0], face[1], face[2]))
            
            # Appliquer les opérations de compression au modèle
            # for (ty, index, value) in operations:
            #     if ty == "vertex":
            #         output_model.add_vertex(index, value)
            #     elif ty == "face":
            #         output_model.add_face(index, Face(value[0], value[1], value[2]))
            #     elif ty == "edit_face":
            #         output_model.edit_face(index, Face(value[0], value[1], value[2]))
            #     elif ty == "remove_face":
            #         output_model.remove_face(index)
            #     else:
            #         output_model.edit_vertex(index, value)
        utils.pause()
        

    def step_compression(self):
        self.visualize_3d()
        mst = objToGraphe.minimum_spanning_tree(self.graph)
        collapsed_vertices = set()
        removed_vertices = []
        collapsed_edges = []
        
        mst_edges = list(nx.dfs_edges(mst))  # Utilisation de DFS pour explorer les arêtes
        new_edges = []
        operations = []
        ed = 0
        
        # Compression des vertices en suivant l'arbre couvrant minimal
        for i, edge in enumerate(mst_edges):
            ed += 1
            v1, v2 = edge
            if v1 in collapsed_vertices or v2 in collapsed_vertices:
                continue
            
            neighbors_v1 = set(self.graph.neighbors(v1))
            neighbors_v2 = set(self.graph.neighbors(v2))
            neighbors = neighbors_v1.intersection(neighbors_v2)
            
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
                            if set([w1, w2]) in collapsed_edges:
                                valid = False
                                break

            if not valid:
                continue
            
            collapsed_vertices.add(v1)
            collapsed_vertices.add(v2)
            collapsed_edges.append(set([v1, v2]))
            removed_vertices.append(v2)
            
            tmp_deleted_faces = dict()
            for face_index, face in self.faces.items():
                if v2 in face:
                    tmp_deleted_faces[face_index] = face
                    operations.append(('face', face_index, face))
            
            for face_index, face in tmp_deleted_faces.items():
                self.deleted_faces[face_index] = face
                del self.faces[face_index]
            
            for n2 in neighbors_v2:
                if n2 != v1 and (n2 not in neighbors_v1) and n2 not in removed_vertices:
                    new_edges.append((v1, n2))
                    for n1 in neighbors_v1:
                        if n1 not in removed_vertices and n1 != n2 and self.graph.has_edge(n1, n2):
                            new_face = [v1, n2, n1]
                            self.faces[self.faces_counter] = (new_face)
                            operations.append(('remove_face', self.faces_counter, new_face))
                            operations.append(('edit_vertex', n2, self.graph.nodes[n2]['pos']))
                            self.faces_counter += 1
            
            vertex_v2 = self.graph.nodes[v2]['pos']
            operations.append(('vertex', v2, vertex_v2))

        # Suppression des vertices
        b = len(self.graph.nodes)
        for node in removed_vertices:
            self.graph.remove_node(node)

        # Ajout des nouvelles arêtes
        for edgeaa in new_edges:
            if not (edgeaa[0] in removed_vertices or edgeaa[1] in removed_vertices):
                self.graph.add_edge(edgeaa[0], edgeaa[1])

        # Mise à jour des faces pour qu'elles soient égales aux cliques de taille 3
        triangles = self.get_triangles_from_graph()  # Trouver toutes les cliques de taille 3
        self.faces = {}  # Réinitialiser les faces avant d'ajouter les nouvelles
        for i, triangle in enumerate(triangles):
            self.faces[i] = triangle  # Ajouter la face (triangle)
        
        a = len(self.graph.nodes)
        print(f"{b} => {a}: -{b - a}")
        objToGraphe.draw_graph(self.graph)
        return operations

    def get_triangles_from_graph(self):
        """
        Trouve toutes les cliques de taille 3 dans le graphe et les retourne comme faces.
        """
        triangles = []
        for clique in nx.enumerate_all_cliques(self.graph):
            if len(clique) == 3:
                triangles.append(list(clique))
        return triangles


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
    
    model.compress_model("example/bunny.obja", len(model.graph.nodes),pourcentageDecimate=0.3,max_step=1)
    #model.visualize_3d()
if __name__ == '__main__':
    main()