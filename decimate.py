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
    def step_compression(self, graph, faces):
        mst = objToGraphe.minimum_spanning_tree(graph)

        collapsed_vertices = set()
        new_edges = []
        edges = list(mst.edges())
        
        for edge in edges:
            v1, v2 = edge
            if v1 in collapsed_vertices or v2 in collapsed_vertices:
                continue
            
            # Vérifier les connexions
            neighbors = set(graph.neighbors(v1)).union(set(graph.neighbors(v2)))
            valid = True
            # vérifier que le triangle formé est valide
            for w in neighbors:
                if w != v1 and w != v2 and not utils.is_valid_triangle(v1, v2, w, graph, faces):
                    valid = False
                    break
            if valid:
                # Vérifier les quadrilatères
                for w1, w2 in edges:
                    if utils.can_collaps_quad(v1, v2, w1, w2, edges):
                        continue
                    else:
                        valid = False
                        break
            
            if valid:
                collapsed_vertices.add(v2)  # Collapse v2 into v1
                new_edges.extend([(v1, w) for w in neighbors if w not in collapsed_vertices])

        for v in collapsed_vertices:
            graph.remove_node(v)
        
        graph.add_edges_from(new_edges)

def main():
    """
    Runs the program on the model given as parameter.
    """
    np.seterr(invalid = 'raise')
    model = Decimater()
    filename='example/test.obj'
    
    #model.parse_file(filename)
    vertices, faces = objToGraphe.load_obj(filename)
    graph = objToGraphe.create_graph(vertices, faces)
    
    objToGraphe.compress_model(graph, faces)
    objToGraphe.visualize_mst_simple(graph, objToGraphe.minimum_spanning_tree(graph))
    
    with open('example/suzanne.obja', 'w') as output:
        model.contract(output)
    objToGraphe.minimum_spanning_tree(graph)

if __name__ == '__main__':
    main()
