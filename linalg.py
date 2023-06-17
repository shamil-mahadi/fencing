"""
Copyright 2023 Team <name>
This code is intended solely for use at Bangladesh Robotics Olympiad
by Team <name> consisting of Khandoker Shamil Mahadi bin Khalid, Namia
Rauzat, and Tasin Mahamud. Unauthorized reproduction or usage of this code
beyond the intended context is strictly prohibited.
"""

from __future__ import annotations
from math import cos, sin, atan, pi, sqrt
import copy

import pygame as pg
import numpy as np

from config import PRECISION

def auto_round(func):
    def wrapper(*args):
        if args:
            args = list(args)
            for index, arg in enumerate(args):
                if type(arg) in (int, float):
                    args[index] = round(arg, PRECISION)
                    
        result = func(*args)
        if result:
            if type(result) in (int, float):
                result = round(result, PRECISION)

        return result
    return wrapper

def get_rotation_matrix(axis, angle):
    cosine = cos(angle)
    sine = sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0],
                         [0, cosine, sine],
                         [0, -sine, cosine]])
    elif axis == "y":
        return np.array([[cosine, 0, -sine],
                         [0, 1, 0],
                         [sine, 0, cosine]])
    elif axis == "z":
        return np.array([[cosine, -sine, 0],
                         [sine, cosine, 0],
                         [0, 0, 1]])

# Should these helper functions be methods instead?
# How does Python handle the extra method baggage?

# Do not replace the transformation matrices with more computationally efficient
# direct variable setting. This is because the rotation matrix is needed later
# to undo the rotation.
def get_rotation_to_plane(vector, plane):
    if plane == "xz":
        if vector.z:
            angle = atan(-vector.y/vector.z)
        else:
            angle = pi/2
            
        return get_rotation_matrix("x", angle)
    elif plane == "xy":
        if vector.y:
            angle = atan(vector.z/vector.y)
        else:
            angle = pi/2
        return get_rotation_matrix("x", angle)
    elif plane == "yz":
        angle = atan(vector.x/vector.y)
        return get_rotation_matrix("z", angle)

def align_axis_with_z(axis_vector):
    if axis_vector.z:
        angle = atan(axis_vector.x/axis_vector.z)
    else:
        angle = pi/2
    plane_rotation_matrix = get_rotation_matrix("y", angle)
    return plane_rotation_matrix

class Vector:
    @auto_round
    def __init__(self, *args):
        if type(args[0]) == np.ndarray:
            self._coords = args[0]
        else:
            self._coords = np.array(args)
            
        # Consider making rotation matrix global
        # Pros: Less space needed as each Vector doesn't have to carry cache baggage
        # Cons: Might be slower due to global lookup
        self._rotation_matrix_cache = dict()

    def __str__(self):
        return str(tuple(coord for coord in self.coords))

    def __eq__(self, vector: Vector) -> Boolean:
        return self.coords == vector.coords

    def __add__(self, vector: Vector) -> Vector:
        return Vector(*(self.coords + vector.coords))

    def __mul__(self, scalar) -> Vector:
        new_coords = self.coords * scalar 
        return Vector(*new_coords)

    def __truediv__(self, scalar) -> Vector:
        # I'm very disappointed I couldn't use the following line.
        # Despite being a nifty reference to the first chapter of Spivak,
        # it turns out not to be very well-performing.
        
        # return self * (1/scalar)

        new_coords = self.coords / scalar
        return Vector(*new_coords)  

    def __sub__(self, vector: Vector) -> Vector:
        new_coords = self.coords - vector.coords
        return Vector(*new_coords)

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, val):
        # Possibility of optimization by removing rounding.
        self._coords = np.around(val, decimals=PRECISION)

    @property
    def x(self):
        return self.coords[0]

    @x.setter
    @auto_round
    def x(self, val):
        self.coords[0] = val

    @property
    def y(self):
        return self.coords[1]

    @y.setter
    @auto_round
    def y(self, val):
        self.coords[1] = val

    @property
    def z(self):
        try:
            return self.coords[2]
        except IndexError:
            raise AttributeError("2D Vector has no attribute 'z'")

    @z.setter
    @auto_round
    def z(self, val):
        self.coords[2] = val

    @property
    def magnitude(self):
        return sqrt(self.coords.dot(self.coords))
    
    @property
    def unit_vector(self):
        return self / self.magnitude

    def transform(self, matrix):
        # Consider whether to return new Vector or modify in place.
        self.coords = matrix.dot(self.coords)

    # The methods below have been deprecated, since the programs don't require
    # 2D rotation anymore. However, if a usecase comes up later on, and the methods need to
    # be archived, consider improving the rotation matrix caching system.
    """
    def _rotate(self, angle):
        try:
            rotation_matrix = self._rotation_matrix_cache[angle]
        except KeyError:
            sine = sin(angle)
            cosine = cos(angle)
            rotation_matrix = np.array([[cosine, -sine],
                                        [sine, cosine]])
            self._rotation_matrix_cache[angle] = rotation_matrix

        self.transform(rotation_matrix)

    def rotate(self, angle, point=None):
        if point:
            displacement_vector = self - point
            displacement_vector.rotate(angle)
            self.coords = (point + displacement_vector).coords
        else:
            self._rotate(angle)
    """

    # TODO: Build 3D rotation matrix cacher.
    
    def rotate_about_axis(self, axis, angle):
        rotation_matrix = get_rotation_matrix(axis, angle)
        self.transform(rotation_matrix)

    def rotate_about_arbitrary_axis(self, axis_vector, angle):
        plane_rotation_matrix = get_rotation_to_plane(axis_vector, "xz")
        axis_vector.transform(plane_rotation_matrix)
        self.transform(plane_rotation_matrix)
        
        alignment_matrix = align_axis_with_z(axis_vector)
        self.transform(alignment_matrix)
        
        self.rotate_about_axis("z", angle)
        
        self.transform(np.linalg.inv(alignment_matrix))
        self.transform(np.linalg.inv(plane_rotation_matrix))
        

    def to_pg(self, screen_height):
        if len(self.coords) != 2:
            raise Exception("Only 2D vectors may be drawn via pygame.")
        return pg.Vector2(self.x, screen_height - self.y)

    def project(self, camera_plane_pos):
        """This is a weak perspective projection. Would be nicer with strong projection, but it's complicated
         and I really didn't feel the need to bother learning it. Weak works just fine.
         Eye is assumed to be at origin, and the z axis goes through
         the centre of the camera plane. camera_plane_pos represents the z-offset, or the focal
         length, essentially. Note that this entire program runs
         on a left-handed coordinate system. This is to reduce the use of
         negative coordinates in vectors, which have high overhead.
         Unfortunately, this will likely require a flip of the coordinate system about the
         x axis. But that shouldn't add too much overhead. If it turns out to be a problem,
         PLEASE MODIFY THIS METHOD TO USE RIGHTHANDED COORDINATES. Modifying the opponent
         tracking vectors and constantly converting them to lefthanded coords will definitely
         add a lot of overhead.
        """

        # This is a rare instance of me using a magic number.
        # TODO: move to globals module.
        self += Vector(0,0,600)
        
        try:
            scaled_vector = self / self.z * camera_plane_pos
            return Vector(scaled_vector.x, scaled_vector.y)
        except AttributeError:
            raise Exception("Only 3D vectors can be projected.")



#TODO: Add support for axes that do not pass through origin

class Wireframe:
    def __init__(self, edges, *vertices):
        self.edges = edges
        self.vertices = list(vertices)
        self.acceleration = 0
        self.velocity = Vector(0,0,0)
        self.angular_velocity = 0
        self.a_duration_ticks = 0
        self.r_duration_ticks = 0
        self.axis = Vector(1, 0, 0)

    def __str__(self):
        return str([str(vertex) for vertex in self.vertices])

    def __add__(self, val: Vector) -> Wireframe:
        new_vertices = []
        for vertex in self.vertices:
            new_vertices.append(vertex + val)
        return Wireframe(self.edges, *new_vertices)

    def apply_acceleration(self, acceleration: Vector, duration: float, dt: float):
        self.acceleration = acceleration
        self.a_duration_ticks = duration / dt

    def apply_velocity(self, velocity: Vector, duration: int):
        """This is a deprecated function. Please avoid using this whenever possible.
           It's not being removed entirely in case a situation arises where unrealistic
           constant velocity motion with infinite acceleration is necessary. Now that
           I think about it, I should just remove it already."""
        self.velocity = velocity

    def apply_angular_velocity(self, axis: Vector, angular_velocity: float, duration: float, dt: float):
        self.axis = axis
        self.angular_velocity = angular_velocity
        self.r_duration_ticks = duration / dt

    def rotate(self, axis: Vector, angle: float):      
        for vertex in self.vertices:
            # Vector.transform() modifies the axis vector in-place, so we pass a copy instead.
            # Otherwise, the axis vector needs correction transformations at the end of each arbitrary rotation.
            # Copying is the best solution here, as far as I can see, because the other solution is to
            # make Vector.transform() return a new Vector, which is a heavy process.
            vertex.rotate_about_arbitrary_axis(copy.copy(axis), angle) 

    def update(self, dt: float):
            
        if self.r_duration_ticks > 0:
            dtheta = self.angular_velocity * dt
            self.rotate(self.axis, dtheta)
            self.r_duration_ticks -= 1

        if self.a_duration_ticks > 0:
            self.velocity += self.acceleration
            self.a_duration_ticks -= 1
        elif self.a_duration_ticks == 0:
            self.a_duration_ticks = 0

        if self.velocity:
            ds = self.velocity * dt
            for i in range(len(self.vertices)):
                self.vertices[i] += ds

    def render_to_pg(self, screen_height: int, origin=Vector(0, 0)):
        lines = []
        for edge in self.edges:
            start_point = self.vertices[edge[0]].project(100) + origin
            end_point = self.vertices[edge[1]].project(100) + origin
            lines.append((start_point.to_pg(screen_height), end_point.to_pg(screen_height)))
        return lines



# Below is legacy code. Overall, the new Linear Algebra API is much more dynamic, but at the cost of speed.
# The additional boilerplate adds a significant amount of delay in the creation of new Vectors. However, the API
# is now much more robust, as it's basically a wrapper on a numpy array. This allows taking advantage of all the
# goodies numpy has to offer, including the ability to take dot products on a whim, rather than having to explicitly
# write down the calculations. Moreover, although the rotation methods are adapted exclusively for R3 Vectors (as they should),
# the other methods are suited for vectors in Rn. I believe this speed tradeoff was worth it, but I've kept the faster code
# below should the new Vector architecture fail to keep up with the complexity of this program.

"""
class TransformationMatrix:
    def __init__(self, i_x, i_y, j_x, j_y):
        self.i_x = i_x
        self.i_y = i_y
        self.j_x = j_x
        self.j_y = j_y

class LegacyVector:
    @auto_round
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._rotation_matrix_cache = dict()
        
    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, vector: Vector) -> Boolean:
        return self.x == vector.x and self.y == vector.y

    def __add__(self, vector: Vector) -> Vector:
        return Vector(self.x + vector.x, self.y + vector.y)

    def __mul__(self, scalar: int) -> Vector:
        return Vector(self.x * scalar, self.y * scalar)

    def __sub__(self, vector: Vector) -> Vector:
        return self + (vector * -1)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
    
    @x.setter
    @auto_round
    def x(self, val):
        self._x = val

    @y.setter
    @auto_round
    def y(self, val):
        self._y = val

    @property
    def quadrant(self):
        if self.x > 0:
            if self.y > 0:
                return 1
            elif self.y < 0:
                return 4
        elif self.x < 0:
            if self.y > 0:
                return 2
            elif self.y < 0:
                return 3
        return 0

    @property
    def direction(self):
        # elaborate "what quadrant is 90deg?"
        if self.x == 0:
            if self.y > 0:
                return pi / 2
            elif self.y < 0:
                return 3/2 * pi
        elif self.y == 0:
            if self.x > 0:
                return 0
            elif self.x < 0:
                return pi

        basic_angle = atan(self.y/self.x)
        if self.quadrant in (3,4):
            return basic_angle + pi
        return basic_angle

    @property
    def unit_vector(self):
        magnitude = sqrt(self.x**2 + self.y**2)
        return Vector(self.x / magnitude, self.y / magnitude)


    def to_pg(self, screen_height):
        return pg.Vector2(self.x, screen_height - self.y)

    def transform(self, matrix):
        # define prime variables because otherwise y prime uses the value of x prime instead of x since x prime is defined first
        x_prime = (self.x * matrix.i_x) + (self.y * matrix.j_x)
        y_prime = (self.x * matrix.i_y) + (self.y * matrix.j_y)
        self.x = x_prime
        self.y = y_prime

    def rotate(self, angle):
        cosine = cos(angle)
        sine = sin(angle)
        rotation_matrix = TransformationMatrix(cosine, sine, -sine, cosine)
        self.transform(rotation_matrix)

    def rotate2(self, angle):
        try:
            rotation_matrix = self._rotation_matrix_cache[angle]
            #print("I RETRIEVED FROM DICT")
        except KeyError:
            cosine = cos(angle)
            sine = sin(angle)
            rotation_matrix = TransformationMatrix(cosine, sine, -sine, cosine)
            self._rotation_matrix_cache[angle] = rotation_matrix
            #print(self._rotation_matrix_cache)

        self.transform(rotation_matrix)

"""

