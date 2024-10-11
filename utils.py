from scipy.spatial.distance import directed_hausdorff


def hausdorff_distance(point1, point2):
    return max(directed_hausdorff(point1, point2)[0], directed_hausdorff(point2, point1)[0])

# check OK
def is_valid_triangle(v1, v2, w, graph, faces):
    for face in faces:
        if set([w,v1,v2]) == face:
            return True
    return False

# check OK
def can_collaps_quad(v1, v2, w1, w2, edges):
    # Vérifie si e1 et e2 ne peuvent pas être réduits ensemble
    return not ((v1, w1) in edges and (w2, v2) in edges and (v1,v2) in edges)
