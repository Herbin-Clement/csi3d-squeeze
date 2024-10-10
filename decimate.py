#!/usr/bin/env python

import obja
import numpy as np
import sys
import objToGraphe

class Decimater(obja.Model):
    """
    A simple class that decimates a 3D model stupidly.
    """
    def __init__(self):
        super().__init__()
        self.deleted_faces = set()

    def contract(self, output, graph):
        """
        Decimates the model stupidly, and write the resulting obja in output.
        """
        operations = []

        for (vertex_index, vertex) in enumerate(self.vertices):
            operations.append(('ev', vertex_index, vertex + 0.25))

        # Iterate through the vertex
        for (vertex_index, vertex) in enumerate(self.vertices):

            # Iterate through the faces
            for (face_index, face) in enumerate(self.faces):

                # Delete any face related to this vertex
                if face_index not in self.deleted_faces:
                    if vertex_index in [face.a,face.b,face.c]:
                        self.deleted_faces.add(face_index)
                        # Add the instruction to operations stack
                        operations.append(('face', face_index, face))

            # Delete the vertex
            operations.append(('vertex', vertex_index, vertex))

        # To rebuild the model, run operations in reverse order
        operations.reverse()

        # Write the result in output file
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
    filename='example/test.obj'
    
    #model.parse_file(filename)
    vertices, faces = objToGraphe.load_obj(filename)
    graph = objToGraphe.create_graph(vertices, faces)
    
    objToGraphe.compress_model(graph)
    objToGraphe.visualize_mst_simple(graph, objToGraphe.minimum_spanning_tree(graph))
    
    with open('example/suzanne.obja', 'w') as output:
        model.contract(output)
    objToGraphe.minimum_spanning_tree(graph)

if __name__ == '__main__':
    main()
