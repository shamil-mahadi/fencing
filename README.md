# Introduction
This is the software behind the AI fencing trainer robot being built for the Robotics Olympiad.
This code is not the final code. It's prototype code that will be translated to C later on so
for better speed. A high-level overview of the robot's logic is given below:

1. Measure information about the opponent's sword, for example position, velocity, acceleration, angle, etc.
2. Represent the information as a 3D `Vector`.
3. Feed the `Vector` to the Reinforcement-learning based AI.
4. Get a response action back in the form of an acceleration vector, and a rotational velocity and a rotational duration.
5. Convert to a vector.
6. (Optional) Check mathematically to make sure AI's decision causes a collision between the two swords.
7. Translate acceleration and rotation vectors to motor rotation speed, direction and time.
8. Measure opponent sword's deflection after the collision.
9. Confirm whether hit was successful.
10. (Optional depending on available processing power) Feed a reward or penalty back into the AI based on deflection.

# General instructions about API usage
## Vectors
The backbone of the linear algebra engine is the `Vector` object. It represents an element of R^n vector space
by using a numpy array of shape (1, n). Vectors can easily be created like this:
```py
vector = Vector(x, y, z...)
```
All vector components can be accessed by indexing the numpy array `vector.coords`. The first three components have
shorthands defined: `vector.x`, `vector.y` and `vector.z`.

`Vector`s have all the standard vector operations defined. They can be added, subtracted, multiplied, and divided by
using standard operators (i.e +/-, etc). Note that multiplication and division is only defined for scalars. For taking
the dot product of a `Vector` with a matrix, use:
```py
vector.transform(matrix)
```
Vector rotation is very robust. 3D vectors can be rotated about the principal axes, or about any arbitrary axis,
or even to lie on a plane.

An example of a rotation about a principal axis is given below:
```py
from math import pi
vector.rotate_about_axis('z', pi)
```
> Note: Angles should be given in radians.
An example of a rotation about an arbitrary axis is given below:
```py
from math import sqrt, pi
axis_vector = Vector(1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
vector.rotate_about_arbitrary_axis(axis_vector, pi/2)
```
> Note: Although not strictly necessary, it is best to use a unit vector to represent the axis.

An example of a rotation to lie on a plane is given below:
```py
rotation_matrix = get_rotation_to_plane(vector, "yz")
vector.transform(rotation_matrix)
```
> Note: As of now, only rotation to XZ plane rotation can be guaranteed to work perfectly. This is due to an issue with the rotation of a
> normal vector onto the plane, where the inverse tangent function stops working. In the XZ plane, this is correctly by handling the exception
> and setting the angle equal to pi/2. This solution could be applied to all the planes, but was not deemed necessary because rotation to plane 
> is mostly an internal procedure, and only the XZ plane is ever used by default.

Vectors have some dimension-specific methods. 2D Vectors can be converted to `pygame.Vector2` types via `vector.to_pg(screen_height)`.
3D Vectors can also be projected into 2D Vectors via the `vector.project(camera_plane_pos)`, where `camera_plane_pos` is the positive
Z offset of the camera plane from  the eye/focus. When displaying 3D vectors, they must first be projected to 2D vectors, and then converted to
`pg.Vector2`.

## Wireframes
Wireframes represent solid 3D objects. A wireframe is created as such:
```py
vertex0 = Vector(0, 0)
vertex1 = Vector(1, 5)
vertex2 = Vector(3, 9)
wireframe = Wireframe(((0, 1), (1,2), (2,0)), vertex0, vertex1, vertex2)
```
Quite  a lot to take in. The first argument is a tuple, containing other, ordered subtuples. Each subtuple represents an edge. The numbers represent
the index of the vertex. The remaining arguments are the vertices, and they are indexed in the order they are written. The edges are stored in `wireframe.edges`
and the vertices are stored as a list in `wireframe.vertices`.

Wireframes are modeled after real objects. Using the `wireframe.apply_acceleration` method, forces can be simulated. The use is as follows:
```py
acc = Vector(2, 0, 3)
wireframe.apply_acceleration(acc, 2, dt)
```
The first argument is an acceleration vector. The second argument is the duration for which the acceleration acts. `dt` is the tick speed, which will 
be explained later on.

Angular velocity can also be applied as follows:
```py
wireframe.apply_angular_velocity(axis_vector, 3pi/4, 10, dt)
```
The first argument is an axis vector, just like the ones in Vector rotation. The second argument is an angular velocity measured in radians per second.
The third is duration.

Wireframes also have addition defined but using it is inefficient and highly discouraged.

The key method of a Wireframe is `update`. It is used as follows:
```py
wireframe.update(dt)
```
`dt` is the tick speed. It can be thought of as how often the vectors change per second. It is intended for use with a loop. Right now, it's
being used with pygame's event loop. In the future, it'll be used in a custom loop in the ESP32.

Wireframes also support pygame displaying. The `render_to_pg(screen_height, origin=Vector(x,y)` method returns a list of tuples, where each tuple represents a line
by using an ordered set of two vectors: the start point and the end point of the line. These vectors are projected versions of the 3D position vectors
of the vertices. The `camera_plane_pos` is set to 100 by default. The origin argument indicates the origin of the coordinate plane in the pygame window.
By default  it's in the top left corner. It is recommended to set it to the centre of the screen.

# Committing
To ensure highest productivity, please follow the standard git workflow. Don't commit to the main branch directly. Create a new branch with a
descriptive name clearly showing the change you're working on. Make sure the branch is always up-to-date with the remote repo. Before
merging into main, please ask for the consent of the other teammates first. If conflicts occur, please consult with other teammates 
to ensure a proper conflict resolution, instead of just doing a hard rebase.

# Code writing
Please try to follow the same coding style which is already being used. The current code is compliant with PEP8 standards.
Please use descriptive variable names. Feel free to use comments liberally, whenever complex code is written. In general, try
to keep code as simple as possible. If you find any possibility of optimization, leave a comment detailing the possible optimization.
Keeping in mind that we will have limited processing power and memory in the ESP32, try to avoid long loops or large data structures.
Try to make new code fully compatible with the current codebase. Divide code into appropriate files if necessary at your own discretion.
Try as best as you can not to depend on external libraries, as they will be unavailable in the ESP32.
