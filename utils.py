import numpy as np

def euclidean_distance(point1, point2):
    """
    Compute the euclidean distance between point1 and point2.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def is_valid_triangle(v1, v2, w, faces):
    """
    Check if v1-v2-w is a valid triangle.
    """
    for face in faces.values():        
        if set([w,v1,v2]) == set(face):
            #print(set([w,v1,v2]), set(face))
            return True
    return False

def can_collaps_quad(v1, v2, w1, w2, edges, collapsed_vertices):
    """
    Check if v1-v2-w1-w2 quad can be collapsed.
    """
    cond = (v1, w1) in edges and (w2, v2) in edges and (v1,v2) in edges
    # Vérifie si e1 et e2 ne peuvent pas être réduits ensemble
    return not (cond and (w1 in collapsed_vertices or w2 in collapsed_vertices))

def pause():
    input("Press the <ENTER> key to continue...")