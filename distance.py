from math import sqrt, cos, pi, tan, atan
from linalg import Vector, get_rotation_to_plane

def get_virtual_vector(projection_vector, rotation_vector):
    projection_magnitude = projection_vector.magnitude
    # Consider replacing new Vector creation by double calculation of
    # x and y coordinates of the unit vector in the second line after
    # comment.
    projection_unit_vector = projection_vector.unit_vector
    virtual_magnitude = projection_magnitude / (rotation_vector.x * projection_unit_vector.x + rotation_vector.y * projection_unit_vector.y)
    return rotation_vector * virtual_magnitude



# TODO: Currently 1D, change to 3D.
def correct_aberration(aberrated_point, radius):
    projected_length = aberrated_point
    angle = arcsin(projected_length / radius)
    real_length = angle * radius
    corrected_point = real_length

    

# Below is test code
proj = Vector(206, 926, 0)
print(937/226)
print("p", proj.magnitude)
direc = Vector(6, 20.8, 7)
direc = direc.unit_vector
virtual_vector = get_virtual_vector(proj, direc)
print(virtual_vector.magnitude)

projected_length = virtual_vector.magnitude

def get_distance(projected_length, real_length):
    return 3180.9 * (real_length / projected_length)**(1/1.021)


print(get_distance(projected_length, 21.8))
