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
        self.transformation_matrix = np.dot(
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
        self.transformation_matrix = np.dot(self.transformation_matrix, scale_matrix)

    def apply_rotation(self, rotation):
        """Apply rotation to the transformation matrix."""
        ux, uy, uz, angle = rotation
        qr = math.cos(angle / 2)
        qx = math.sin(angle / 2) * ux
        qy = math.sin(angle / 2) * uy
        qz = math.sin(angle / 2) * uz

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
        self.transformation_matrix = np.dot(self.transformation_matrix, rotation_matrix)

    def get_transformation_matrix(self):
        """Return the resulting transformation matrix."""
        return self.transformation_matrix
