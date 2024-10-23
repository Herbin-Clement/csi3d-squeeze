#!/usr/bin/env python

import obja
import numpy as np
import sys
import objToGraphe
import utils

class Decimater(obja.Model):
    def __init__(self):
        super().__init__()
        self.deleted_faces = set()

    # def contract(self, output, graph):
    #     """
    #     Decimates the model stupidly, and write the resulting obja in output.
    #     """
    #     operations = []

    #     # Iterate through the vertex
    #     for (vertex_index, vertex) in enumerate(self.vertices):

    #         # Iterate through the faces
    #         for (face_index, face) in enumerate(self.faces):

    #             # Delete any face related to this vertex
    #             if face_index not in self.deleted_faces:
    #                 if vertex_index in [face.a,face.b,face.c]:
    #                     self.deleted_faces.add(face_index)
    #                     # Add the instruction to operations stack
    #                     operations.append(('face', face_index, face))

    #         # Delete the vertex
    #         operations.append(('vertex', vertex_index, vertex))

    #     # To rebuild the model, run operations in reverse order
    #     operations.reverse()

    #     # Write the result in output file
    #     output_model = obja.Output(output, random_color=True)

    #     for (ty, index, value) in operations:
    #         if ty == "vertex":
    #             output_model.add_vertex(index, value)
    #         elif ty == "face":
    #             output_model.add_face(index, value)   
    #         else:
    #             output_model.edit_vertex(index, value)

    def compress_model(self, graph):
        # TODO
        return

    # check OK
    def step_compression(self, graph, faces, output):
        mst = objToGraphe.minimum_spanning_tree(graph)
        collapsed_vertices = set()
        new_edges = []
        edges = list(mst.edges())
        
        operations = []  # Pile d'opérations pour stocker les étapes de suppression

        for edge in edges:
            v1, v2 = edge
            if v1 in collapsed_vertices or v2 in collapsed_vertices:
                continue

            # Récupérer les voisins
            neighbors_v1 = set(graph.neighbors(v1))
            neighbors_v2 = set(graph.neighbors(v2))
            neighbors = neighbors_v1.union(neighbors_v2)

            valid = True
            # 
            for w in neighbors:
                if w != v1 and w != v2 and not utils.is_valid_triangle(v1, v2, w, graph, faces):
                    valid = False
                    break

            if valid:
                for w1, w2 in edges:
                    if not utils.can_collaps_quad(v1, v2, w1, w2, edges):
                        valid = False
                        break

            if valid:
                collapsed_vertices.add(v2)
                
                for face_index, face in enumerate(faces):
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
                                faces.append(new_face)  
                                face_index = len(faces) - 1  
                                
                                operations.append(('edit_vertex', neighbor, graph.nodes[neighbor]))  
                vertex_v2 = graph.nodes[v2]  
                operations.append(('vertex', v2, vertex_v2))

                graph.remove_node(v2)

        graph.add_edges_from(new_edges)

        operations.reverse()

        output_model = obja.Output(output, random_color=True)
        
        for (ty, index, value) in operations:
            if ty == "vertex":
                output_model.add_vertex(index, value)
            elif ty == "face":
                output_model.add_face(index, value)
            else:
                output_model.edit_vertex(index, value)


def main():
    """
    Runs the program on the model given as parameter.
    """
    np.seterr(invalid = 'raise')
    model = Decimater()
    filename='example/bunny.obj'
    
    #model.parse_file(filename)
    vertices, faces = objToGraphe.load_obj(filename)
    graph = objToGraphe.create_graph(vertices, faces)
    
    model.compress_model(graph, faces)
    objToGraphe.visualize_mst_simple(graph, objToGraphe.minimum_spanning_tree(graph))
    
    with open('example/suzanne.obja', 'w') as output:
        model.contract(output)
    objToGraphe.minimum_spanning_tree(graph)

if __name__ == '__main__':
    main()
