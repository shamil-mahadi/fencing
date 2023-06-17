import numpy as np
import copy
from math import sqrt
from timeit import timeit


def solve_3d_intersection_matrix(a, b):
    for row in a:
        if not any(row):
            row_index = a.index(
                row
            )  # Find index this way instead of enumerate since indexes differ from the for loop as items decrease
            if not b[row_index]:
                return False  # If equation of the form 0a_1 + 0a_2 = C where C is non-zero, no solution.

            a = np.delete(a, row_index)
            b = np.delete(b, row_index)

    if len(a_filtered) < 2:  # If not enough information
        return False

    # Reduce to square matrix in case no null rows
    if len(b_filtered) == 3:
        a_filtered = a_filtered[:2]
        b_filtered = b_filtered[:2]

    solutions = np.linalg.solve(a_filtered, b_filtered)
    if any(solutions) and (
        solutions[0] * a[2, 0] + solutions[1] * a[2, 1] == b[2]
    ):  # Assumes 3x3 matrix at all times
        return solutions
    else:
        return False


def check_naive_collision(line1, line2, tolerance):
    line1_start, line1_end = line1.vertices
    line2_start, line2_end = line2.vertices
    line1_vector = line1_end - line1_start
    line2_vector = line2_end - line2_start
    end_distance = (line2_end - line1_end).magnitude
    max_distance = (
        (line1_end - line1_start).magnitude
        + (line2_end - line1_start).magnitude
        + tolerance
    )
    if end_distance <= max_distance:
        coefficient_matrix = np.array(
            [
                [line1_vector.x, -line2_vector.x],
                [line1_vector.y, -line2_vector.y],
                [line1_vector.z, -line2_vector.z],
            ]
        )
        ordinate_matrix = np.array(
            [
                line2_start.x - line1_start.x,
                line2_start.y - line1_start.y,
                line2_start.z - line1_start.z,
            ]
        )
        scale_factors = solve_3d_intersection_matrix(
            coefficient_matrix, ordinate_matrix
        )
        if type(scale_factors) == np.ndarray:
            intersection_position = line1_start + line1_vector * scale_factors[0]
            if all(
                [
                    (
                        intersection_position.coords[i] >= line1_start.coords[i]
                        and intersection_position.coords[i] <= line1_end.coords[i]
                    )
                    for i in range(3)
                ]
            ):
                return intersection_position

    return False


def check_point_collision(line1, line2, tolerance):
    line1_start, line1_end = line1.vertices
    line2_start, line2_end = line2.vertices
    end_distance = (line2_end - line1_end).magnitude
    max_distance = (
        (line1_end - line1_start).magnitude
        + (line2_end - line1_start).magnitude
        + tolerance
    )
    if end_distance <= max_distance:
        line1_deltax = line1_end.x - line1_start.x
        line2_deltax = line2_end.x - line2_start.x

        line1_grad_y = (line1_end.y - line1_start.y) / line1_deltax
        line2_grad_y = (line2_end.y - line2_start.y) / line2_deltax

        line1_grad_z = (line1_end.z - line1_start.y) / line1_deltax
        line2_grad_z = (line2_end.z - line2_start.z) / line2_deltax

        c1 = line1_end.y - (line1_grad_y * line1_end.x)
        c2 = line1_end.z - (line1_grad_z * line1_end.x)

        c3 = line2_end.y - (line2_grad_y * line2_end.x)
        c4 = line2_end.z - (line2_grad_z * line2_end.x)

        # The following substitutions are not found in the math paper.
        a = 1 + line2_grad_y * line1_grad_y + line2_grad_z * line1_grad_z
        c5 = c3 - c1
        c6 = c4 - c2
        coefficient_matrix = np.array(
            [
                [1 + line1_grad_y**2 + line1_grad_z**2, a],
                [a, 1 + line2_grad_y**2 + line2_grad_z**2],
            ]
        )

        ordinate_matrix = np.array(
            [
                line1_grad_y * c5 + line1_grad_z * c6,
                line2_grad_y * c5 + line2_grad_z * c6,
            ]
        )

        x1, x2 = np.linalg.solve(coefficient_matrix, ordinate_matrix)

        distance = sqrt(
            (x2 - x1) ** 2
            + (line2_grad_y * x2 - line1_grad_y * x1 + c5) ** 2
            + (line2_grad_z * x2 - line1_grad_z * x1 + c6) ** 2
        )

        return distance <= tolerance


def _check_point_collision_fast(line1, line2, tolerance):
    line1_start, line1_end = line1.vertices
    line2_start, line2_end = line2.vertices
    end_distance = (line2_end - line1_end).magnitude
    max_distance = (
        (line1_end - line1_start).magnitude
        + (line2_end - line1_start).magnitude
        + tolerance
    )
    if end_distance <= max_distance:
        line1_start_x = line1_start.x
        line1_start_y = line1_start.y
        line1_start_z = line1_start.z

        line2_start_x = line2_start.x
        line2_start_y = line2_start.y
        line2_start_z = line2_start.z

        line1_end_x = line1_end.x
        line1_end_y = line1_end.y
        line1_end_z = line1_end.z

        line2_end_x = line2_end.x
        line2_end_y = line2_end.y
        line2_end_z = line2_end.z

        line1_deltax = line1_end_x - line1_start_x
        line2_deltax = line2_end_x - line2_start_x

        line1_grad_y = (line1_end_y - line1_start_y) / line1_deltax
        line2_grad_y = (line2_end_y - line2_start_y) / line2_deltax

        line1_grad_z = (line1_end_z - line1_start_y) / line1_deltax
        line2_grad_z = (line2_end_z - line2_start_z) / line2_deltax

        c1 = line1_end_y - (line1_grad_y * line1_end_x)
        c2 = line1_end_z - (line1_grad_z * line1_end_x)

        c3 = line2_end_y - (line2_grad_y * line2_end_x)
        c4 = line2_end_z - (line2_grad_z * line2_end_x)

        # The following substitutions are not found in the math paper.
        a = 1 + line2_grad_y * line1_grad_y + line2_grad_z * line1_grad_z
        c5 = c3 - c1
        c6 = c4 - c2
        coefficient_matrix = np.array(
            [
                [1 + line1_grad_y**2 + line1_grad_z**2, a],
                [a, 1 + line2_grad_y**2 + line2_grad_z**2],
            ]
        )

        ordinate_matrix = np.array(
            [
                line1_grad_y * c5 + line1_grad_z * c6,
                line2_grad_y * c5 + line2_grad_z * c6,
            ]
        )

        x1, x2 = np.linalg.solve(coefficient_matrix, ordinate_matrix)
        distance = sqrt(
            (x2 - x1) ** 2
            + (line2_grad_y * x2 - line1_grad_y * x1 + c5) ** 2
            + (line2_grad_z * x2 - line1_grad_z * x1 + c6) ** 2
        )
        return distance <= tolerance


def check_vector_collision(line1, line2, tolerance):
    line1_start, line1_end = line1.vertices
    line2_start, line2_end = line2.vertices
    line1_vector = line1_end - line1_start
    line2_vector = line2_end - line2_start
    end_distance = (line2_end - line1_end).magnitude
    max_distance = line1_vector.magnitude + line2_vector.magnitude + tolerance

    if end_distance <= max_distance:
        x1 = line1_start.x
        x2 = line1_vector.x
        x3 = line2_start.x
        x4 = line2_vector.x

        y1 = line1_start.y
        y2 = line1_vector.y
        y3 = line2_start.y
        y4 = line2_vector.y

        z1 = line1_start.z
        z2 = line1_vector.z
        z3 = line2_start.z
        z4 = line2_vector.z

        c1 = -x1 + x3
        c2 = -y1 + y3
        c3 = -z1 + z3

        c4 = -x2 * x4 - y2 * y4 - z2 * z4

        coefficient_matrix = np.array(
            [[c4, -(x4**2) - y4**2 - z4**2], [x2**2 + y2**2 + z2**2, c4]]
        )

        ordinate_matrix = np.array(
            [x4 * c1 + y4 * c2 + z4 * c3, x2 * c1 + y2 * c2 + z2 * c3]
        )

        try:
            scale_factors = np.linalg.solve(coefficient_matrix, ordinate_matrix)
        except np.linalg.LinAlgError:
            return None

        if any([scale_factors[i] > 1 for i in range(2)]):
            return None
        distance_start = line1_start + line1_vector * scale_factors[0]
        distance_end = line2_start + line2_vector * scale_factors[1]

        return (distance_end - distance_start).magnitude <= tolerance
