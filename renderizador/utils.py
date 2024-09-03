import math
import numpy as np


class Transform:
    """Class to handle 3D transformations."""

    def __init__(self):
        """Initialize the transformation matrix as an identity matrix."""
        self.transformation_matrix = np.identity(4)

    def apply_translation(self, translation):
        """Apply translation to the transformation matrix."""
        dx, dy, dz = translation
        translation_matrix = np.array(
            [
                [1, 0, 0, dx],
                [0, 1, 0, dy],
                [0, 0, 1, dz],
                [0, 0, 0, 1],
            ]
        )
        self.transformation_matrix = np.matmul(
            self.transformation_matrix, translation_matrix
        )

    def apply_scale(self, scale):
        """Apply scaling to the transformation matrix."""
        sx, sy, sz = scale
        scale_matrix = np.array(
            [
                [sx, 0, 0, 0],
                [0, sy, 0, 0],
                [0, 0, sz, 0],
                [0, 0, 0, 1],
            ]
        )
        self.transformation_matrix = np.matmul(self.transformation_matrix, scale_matrix)

    def apply_screen_scale(self, scale):
        """Apply screen scaling to the transformation matrix."""
        sx, sy = scale
        scale_matrix = np.array(
            [
                [sx, 0, 0, 1],
                [0, sy, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self.transformation_matrix = np.matmul(self.transformation_matrix, scale_matrix)

    def apply_rotation(self, rotation, inverse=False):
        """Apply rotation to the transformation matrix."""
        axis = rotation[:3] / np.linalg.norm(rotation[:3])
        angle = rotation[3]
        ux, uy, uz = axis
        
        qx = math.sin(angle / 2) * ux
        qy = math.sin(angle / 2) * uy
        qz = math.sin(angle / 2) * uz
        qr = math.cos(angle / 2)

        rotation_matrix = np.array(
            [
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qz * qr),
                    2 * (qx * qz + qy * qr),
                    0,
                ],
                [
                    2 * (qx * qy + qz * qr),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qx * qr),
                    0,
                ],
                [
                    2 * (qx * qz - qy * qr),
                    2 * (qy * qz + qx * qr),
                    1 - 2 * (qx**2 + qy**2),
                    0,
                ],
                [0, 0, 0, 1],
            ]
        )
        if inverse:
            rotation_matrix = np.transpose(rotation_matrix)

        self.transformation_matrix = np.matmul(self.transformation_matrix, rotation_matrix)

    def apply_perspective(self, directions, near, far):
        """Apply perspective to the transformation matrix."""
        top, bottom, right, left = directions

        perspective_matrix = np.array(
            [
                [near/right, 0, 0, 0],
                [0, near/top, 0, 0],
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, -1, 0],
            ]
        )
        self.transformation_matrix = np.matmul(
            self.transformation_matrix, perspective_matrix
        )

    def apply_mirror(self, axis):
        """Apply mirror to the transformation matrix."""
        dict = {
            "x": np.array([[-1, 0, 0, 0], 
                           [0, 1, 0, 0], 
                           [0, 0, 1, 0], 
                           [0, 0, 0, 1]]),
            "y": np.array([[1, 0, 0, 0], 
                           [0, -1, 0, 0], 
                           [0, 0, 1, 0], 
                           [0, 0, 0, 1]]),
            "z": np.array([[1, 0, 0, 0], 
                           [0, 1, 0, 0], 
                           [0, 0, -1, 0], 
                           [0, 0, 0, 1]]),
        }

        self.transformation_matrix = np.matmul(self.transformation_matrix, dict[axis])


    def get_transformation_matrix(self):
        """Return the resulting transformation matrix."""
        return self.transformation_matrix
