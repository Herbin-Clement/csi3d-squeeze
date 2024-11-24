#!/usr/bin/env python
import obja
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import objToGraphe
import utils
import trimesh
import networkx as nx
from obja import Face 
import plotly.graph_objects as go

class Decimater(obja.Model):
    def __init__(self):
        super().__init__()
        self.deleted_faces = dict()
        self.graph = nx.empty_graph()
         
    def visualize_3d(self):
        """
        Visualize in 3D the graph
        """
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
        """
        Load the graph
        """
        self.vertices, self.faces, self.faces_counter = objToGraphe.load_obj(filename)
        self.graph = objToGraphe.create_graph(self.vertices, self.faces, False)

    def get_graph(self):
        """
        Get the nx graph
        """
        return self.graph

    def stat_step(self, operations, decimateRatio, printFaces=False):
        """
        Display some stat about a compression step
        """
        print(f"Nb op: {len(operations)}")
        print(f"Pourcentage de decimation actuel: {decimateRatio}")
        addVertex = 0
        addFace = 0
        editFace = 0
        removeFace = 0
        editFaceVertex = 0
        for (ty, _, _) in operations:
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

    def compress_model(self, outputFilename, maxDecimateRatio, maxStep=10, printDiff=True):
        """
        Compress the model
        """
        compressed = False
        operations = []
        step = 0
        objToGraphe.draw_graph(self.graph)
        decimateRatio = 1
        initialNumberOfNodes = self.graph.number_of_nodes()

        while decimateRatio > maxDecimateRatio and maxStep > step and not compressed:
            step += 1
            print(f"=========== Step {step} ===========")
            step_operations = self.step_compression()
            operations = operations + step_operations
            decimateRatio = self.graph.number_of_nodes() / initialNumberOfNodes 
            self.stat_step(step_operations, decimateRatio)
            if len(step_operations) == 0 or decimateRatio < maxDecimateRatio:
                compressed = True

        operations.reverse()

        self.write_obja(outputFilename, operations)
        self.write_obj(outputFilename)

        if printDiff:
            self.print_diff()

        utils.pause()
        

    def step_compression(self):
        """
        Do a step compression
        """
        collapsed_vertices = []
        collapsed_edges = []
        removed_vertices = []
        new_edges = []
        operations = []
        ed = 0
        mst = objToGraphe.minimum_spanning_tree(self.graph)
        mst_edges = list(nx.dfs_edges(mst))
        for edge in mst_edges:
            ed += 1
            v1, v2 = edge
            # Première condition
            if not self.check_first_cond(v1, v2, collapsed_vertices):
                continue
            
            neighbors_v1 = set(self.graph.neighbors(v1))
            neighbors_v2 = set(self.graph.neighbors(v2))
            neighbors = sorted(list(neighbors_v1.intersection(neighbors_v2)))
            
            # Seconde condition
            if not self.check_second_cond(v1, v2, neighbors):
                continue
            
            # Troisième condition
            if not self.check_third_cond(v1, v2, neighbors_v1, neighbors_v2, collapsed_edges):
                continue

            # Si c'est bon, v2 va être collapse
            collapsed_vertices.append(v1)
            collapsed_vertices.append(v2)
            collapsed_edges.append(sorted([v1,v2]))
            removed_vertices.append(v2)

            # Relie les voisins de v2 à v1
            self.update_edges(v1, neighbors_v1, neighbors_v2, removed_vertices, new_edges)
            
            # Met à jour les faces
            self.update_faces(v1, v2, operations)

            # Supprime v2
            vertex_v2 = self.graph.nodes[v2]['pos']
            operations.append(('vertex', v2, vertex_v2))
        
        # Retire les vertices
        for node in removed_vertices:
            self.graph.remove_node(node)

        # Ajoute les nouvelles arêtes au graphe
        for new_edge in new_edges:
            if not (new_edge[0] in removed_vertices or new_edge[1] in removed_vertices):
                self.graph.add_edge(new_edge[0], new_edge[1])

        return operations

    def check_first_cond(self, v1, v2, collapsed_vertices):
        """
        Check the first condition
        """
        return not (v1 in collapsed_vertices or v2 in collapsed_vertices)

    def check_second_cond(self, v1, v2, neighbors):
        """
        Check the second condition
        """
        valid = True
        # Deuxième condition
        for w in neighbors:
            if w != v1 and w != v2 and not utils.is_valid_triangle(v1, v2, w, self.faces):
                valid = False
                break
        return valid

    def check_third_cond(self, v1, v2, neighbors_v1, neighbors_v2, collapsed_edges):
        """
        Check the third condition
        """
        valid = True
        # Troisième condition
        for w1 in neighbors_v1:
            if w1 != v2:
                for w2 in neighbors_v2:
                    if w2 != v1:
                        if sorted([w1,w2]) in collapsed_edges:
                            valid = False
                            break
        return valid

    def update_edges(self, v1, neighbors_v1, neighbors_v2, removed_vertices, new_edges):
        """
        Link neighbors of v2 to v1
        """
        for n2 in neighbors_v2:
            if n2 != v1 and (n2 not in neighbors_v1) and n2 not in removed_vertices:
                new_edges.append(sorted([v1, n2]))

    def update_faces(self, v1, v2, operations):
        """
        Remove faces containing v2 and add the new faces containing v1
        """
        # Parcours les faces
        tmp_deleted_faces = dict()
        tmp_add_faces = dict()
        for face_index, face in self.faces.items():
            if v2 in face: # Si v2 est dans la face mais pas v1, change v2 par v1
                new_face = face[:]
                new_face.remove(v2)
                new_face.append(v1)
                if sorted(new_face) not in self.faces.values() and v1 not in face: # Si la nouvelle face n'existe pas encore, créer la nouvelle face
                    tmp_deleted_faces[face_index] = face        # et efface la face contenant v2
                    tmp_add_faces[self.faces_counter] = sorted(new_face)
                    self.faces_counter += 1
                else: # Sinon, efface juste la face contenant v2
                    tmp_deleted_faces[face_index] = face

        # Supprime les faces    
        for face_index, face in tmp_deleted_faces.items():
            operations.append(('face', face_index, face))
            self.deleted_faces[face_index] = face
            del self.faces[face_index]
        
        # Ajoute les nouvelles faces
        for face_index, face in tmp_add_faces.items():
            operations.append(('remove_face', face_index, face))
            self.faces[face_index] = face
            

    def write_obja(self, outputFilename, operations):
        """
        Write model in obja file
        """
        # Créer un objet Output pour écrire dans le fichier de sortie
        with open(f"{outputFilename}.obja", 'w') as output_file:
            outputModel = obja.Output(output_file, random_color=True)
            
            for v in self.graph.nodes:
                outputModel.add_vertex(v, self.graph.nodes[v]["pos"])
            for face_index, face in self.faces.items():
                outputModel.add_face(face_index, Face(face[0], face[1], face[2]))
            
            # Appliquer les opérations de compression au modèle
            for (ty, index, value) in operations:
                if ty == "vertex":
                    outputModel.add_vertex(index, value)
                elif ty == "face":
                    outputModel.add_face(index, Face(value[0], value[1], value[2]))
                elif ty == "edit_face":
                    outputModel.edit_face(index, Face(value[0], value[1], value[2]))
                elif ty == "remove_face":
                    outputModel.remove_face(index)
                elif ty == "edit_face_vertex":
                    outputModel.edit_face_vertex(index, value[0], value[1])
                else:
                    outputModel.edit_vertex(index, value)

    def write_obj(self, outputFilename):
        """
        Write model in obj file
        """
        # Créer un objet Output pour écrire dans le fichier de sortie
        with open(f"{outputFilename}_m0.obj", 'w') as output_file:
            outputModel = obja.Output(output_file, random_color=True)
            
            for v in self.graph.nodes:
                outputModel.add_vertex(v, self.graph.nodes[v]["pos"])
            for face_index, face in self.faces.items():
                outputModel.add_face(face_index, Face(face[0], face[1], face[2]))

    def print_diff(self):
        """
        Print faces differences between networkx graph and our faces
        """
        # Trouver les différences entre nos faces et les faces de networkx
        our_faces = [sorted(x) for x in self.faces.values()]
        nxgraph_faces = [sorted(clique) for clique in nx.enumerate_all_cliques(self.graph) if len(clique) == 3]
        our_to_nxgraph = [x for x in our_faces if x not in nxgraph_faces]
        nxgraph_to_our = [x for x in nxgraph_faces if x not in our_faces]
        print("Présents dans nos faces mais pas dans nx graph:", our_to_nxgraph)
        print("Présents dans nx graph mais pas dans nos faces:", nxgraph_to_our)

    def print_metric(self, obj1, obj2):
        # Charger deux maillages depuis des fichiers
        mesh1 = trimesh.load(obj1)
        mesh2 = trimesh.load(obj2)

        # Extraire les sommets
        vertices_mesh1 = mesh1.vertices
        vertices_mesh2 = mesh2.vertices

        # Calculer la distance de Hausdorff
        distance1 = directed_hausdorff(vertices_mesh1, vertices_mesh2)
        distance2 = directed_hausdorff(vertices_mesh2, vertices_mesh1)
        print(f"Distance de Hausdorff avec Trimesh: {max(distance1, distance2)}")

def main():
    """
    Runs the program on the model given as parameter.
    """
    np.seterr(invalid = 'raise')
    model = Decimater()
    filename='example/suzanne.obj'

    model.load_graph(filename)
    
    model.compress_model("example/suzanne", maxDecimateRatio=0.05, maxStep=10)

    # model.print_metric("example/suzanne.obj", "example/suzanne_m0.obj") # Faut transformer notre obja en obj :)

if __name__ == '__main__':
    main()