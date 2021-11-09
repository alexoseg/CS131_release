from typing import Tuple

import numpy as np


def camera_from_world_transform(d: float = 1.0) -> np.ndarray:
    """Define a transformation matrix in homogeneous coordinates that
    transforms coordinates from world space to camera space, according
    to the coordinate systems in Question 1.


    Args:
        d (float, optional): Total distance of displacement between world and camera
            origins. Will always be greater than or equal to zero. Defaults to 1.0.

    Returns:
        T (np.ndarray): Left-hand transformation matrix, such that c = Tw
            for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
    """
    T = np.eye(4)
    # YOUR CODE HERE
    T = np.array([
        [-1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
        [0, 1, 0, 0],
        [-1/np.sqrt(2), 0, -1/np.sqrt(2), d],
        [0, 0, 0,  1]
    ])
    # END YOUR CODE
    assert T.shape == (4, 4)
    return T


def apply_transform(T: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray]:
    """Apply a transformation matrix to a set of points.

    Hint: You'll want to first convert all of the points to homogeneous coordinates.
    Each point in the (3,N) shape edges is a length 3 vector for x, y, and z, so
    appending a 1 after z to each point will make this homogeneous coordinates.

    You shouldn't need any loops for this function.

    Args:
        T (np.ndarray):
            Left-hand transformation matrix, such that c = Tw
                for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
        points (np.ndarray):
            Shape = (3,N) where 3 means 3D and N is the number of points to transform.

    Returns:
        points_transformed (np.ndarray):
            Transformed points.
            Shape = (3,N) where 3 means 3D and N is the number of points.
    """
    N = points.shape[1]
    assert points.shape == (3, N)

    # You'll replace this!
    points_transformed = np.vstack([points, np.ones(N)])

    # YOUR CODE HERE
    points_transformed = T @ points_transformed
    points_transformed = points_transformed[:-1, :]
    # END YOUR CODE

    assert points_transformed.shape == (3, N)
    return points_transformed


def intersection_from_lines(
    a_0: np.ndarray, a_1: np.ndarray, b_0: np.ndarray, b_1: np.ndarray
) -> np.ndarray:
    """Find the intersection of two lines (infinite length), each defined by a
    pair of points.

    Args:
        a_0 (np.ndarray): First point of first line; shape `(2,)`.
        a_1 (np.ndarray): Second point of first line; shape `(2,)`.
        b_0 (np.ndarray): First point of second line; shape `(2,)`.
        b_1 (np.ndarray): Second point of second line; shape `(2,)`.

    Returns:
        np.ndarray: the intersection of the two lines definied by (a0, a1)
                    and (b0, b1).
    """
    # Validate inputs
    assert a_0.shape == a_1.shape == b_0.shape == b_1.shape == (2,)
    assert a_0.dtype == a_1.dtype == b_0.dtype == b_1.dtype == np.float

    # Intersection point between lines
    out = np.zeros(2)

    # YOUR CODE HERE
    line_one = np.cross(np.append(a_1, [[1]]), np.append(a_0, [[1]]))
    line_two = np.cross(np.append(b_1, [[1]]), np.append(b_0, [[1]]))
    out = np.cross(line_one, line_two)
    out = out / out[-1]
    out = out[:-1]
    # END YOUR CODE

    assert out.shape == (2,)
    assert out.dtype == np.float

    return out


def optical_center_from_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Compute the optical center of our camera intrinsics from three vanishing
    points corresponding to mutually orthogonal directions.

    Hints:
    - Your `intersection_from_lines()` implementation might be helpful here.
    - It might be worth reviewing vector projection with dot products.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v2 (np.ndarray): Vanishing point in image space; shape `(2,)`.

    Returns:
        np.ndarray: Optical center; shape `(2,)`.
    """
    assert v0.shape == v1.shape == v2.shape == (2,), "Wrong shape!"

    optical_center = np.zeros(2)

    # YOUR CODE HERE
    line_one = np.cross(np.append(v0.T, [[1]]), np.append(v1.T, [[1]]))
    line_two = np.cross(np.append(v1.T, [[1]]), np.append(v2.T, [[1]]))
    
    m_line_one_perp = line_one[1] / line_one[0]
    m_line_two_perp = line_two[1] / line_two[0]
                                              
    line_one_perp_b = v2[1] - (m_line_one_perp * v2[0])
    line_two_perp_b = v0[1] - (m_line_two_perp * v0[0])
    
    line_one_perp = np.array([-m_line_one_perp, 1, -line_one_perp_b])
    line_two_perp = np.array([-m_line_two_perp, 1, -line_two_perp_b])
    
    final = np.cross(line_one_perp, line_two_perp)
    final_final = final / final[-1]
    optical_center = final_final[:-1] 
    # END YOUR CODE

    assert optical_center.shape == (2,)
    return optical_center


def focal_length_from_two_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, optical_center: np.ndarray
) -> np.ndarray:
    """Compute focal length of camera, from two vanishing points and the
    calibrated optical center.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        optical_center (np.ndarray): Calibrated optical center; shape `(2,)`.

    Returns:
        float: Calibrated focal length.
    """
    assert v0.shape == v1.shape == optical_center.shape == (2,), "Wrong shape!"

    f = None

    # YOUR CODE HERE
    x_comp = (v0[0] - optical_center[0]) * (v1[0] - optical_center[0])
    y_comp = (v0[1] - optical_center[1]) * (v1[1] - optical_center[1]) 
    f = np.sqrt((x_comp + y_comp) / -1)
    # END YOUR CODE

    return float(f)


def physical_focal_length_from_calibration(
    f: float, sensor_diagonal_mm: float, image_diagonal_pixels: float
) -> float:
    """Compute the physical focal length of our camera, in millimeters.

    Args:
        f (float): Calibrated focal length, using pixel units.
        sensor_diagonal_mm (float): Length across the diagonal of our camera
            sensor, in millimeters.
        image_diagonal_pixels (float): Length across the diagonal of the
            calibration image, in pixels.

    Returns:
        float: Calibrated focal length, in millimeters.
    """
    f_mm = None

    # YOUR CODE HERE
    f_mm = (sensor_diagonal_mm / image_diagonal_pixels) * f 
    # END YOUR CODE

    return f_mm
