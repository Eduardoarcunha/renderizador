import math
import numpy as np


class Point:
    def __init__(self, x, y, z_camera, z_ndc):
        self.x = x
        self.y = y
        self.z_camera = z_camera
        self.z_ndc = z_ndc


class Triangle:
    def __init__(self, p0: Point, p1: Point, p2: Point):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

        self.area = self.get_area()

        self.line0_coeficients = self.l_coef(self.p0.x, self.p0.y, self.p1.x, self.p1.y)
        self.line1_coeficients = self.l_coef(self.p1.x, self.p1.y, self.p2.x, self.p2.y)
        self.line2_coeficients = self.l_coef(self.p2.x, self.p2.y, self.p0.x, self.p0.y)

    def get_area(self):
        return abs(
            (
                self.p0.x * (self.p1.y - self.p2.y)
                + self.p1.x * (self.p2.y - self.p0.y)
                + self.p2.x * (self.p0.y - self.p1.y)
            )
            / 2
        )

    def get_bounds_within_screen(self, width, height):
        min_x = max(0, min(self.p0.x, self.p1.x, self.p2.x))
        max_x = min(width, max(self.p0.x, self.p1.x, self.p2.x) + 1)

        min_y = max(0, min(self.p0.y, self.p1.y, self.p2.y))
        max_y = min(height, max(self.p0.y, self.p1.y, self.p2.y) + 1)

        return int(min_x), int(max_x), int(min_y), int(max_y)

    def l_coef(self, x0, y0, x1, y1):
        A = y1 - y0
        B = -(x1 - x0)
        C = y0 * (x1 - x0) - x0 * (y1 - y0)
        return A, B, C

    def l_eval(self, la, lb, lc, p: Point):
        return la * (p.x + 0.5) + lb * (p.y + 0.5) + lc

    def is_inside(self, point: Point):
        l0 = self.l_eval(
            self.line0_coeficients[0],
            self.line0_coeficients[1],
            self.line0_coeficients[2],
            point,
        )
        l1 = self.l_eval(
            self.line1_coeficients[0],
            self.line1_coeficients[1],
            self.line1_coeficients[2],
            point,
        )
        l2 = self.l_eval(
            self.line2_coeficients[0],
            self.line2_coeficients[1],
            self.line2_coeficients[2],
            point,
        )

        return l0 >= 0 and l1 >= 0 and l2 >= 0

    def get_z(self, alpha, beta, gamma):

        z0, z1, z2 = self.p0.z_ndc, self.p1.z_ndc, self.p2.z_ndc
        d = (alpha / z0) + (beta / z1) + (gamma / z2)
        z = 1 / d

        return z

    def get_weights(self, point: Point):
        x, y = point.x, point.y
        x0, y0 = self.p0.x, self.p0.y
        x1, y1 = self.p1.x, self.p1.y
        x2, y2 = self.p2.x, self.p2.y

        a0 = abs(
            ((x + 0.5) * (y1 - y2) + x1 * (y2 - (y + 0.5)) + x2 * ((y + 0.5) - y1)) / 2
        )
        a1 = abs(
            ((x + 0.5) * (y2 - y0) + x2 * (y0 - (y + 0.5)) + x0 * ((y + 0.5) - y2)) / 2
        )

        alpha = min(max(a0 / self.area, 0), 1)
        beta = min(max(a1 / self.area, 0), 1)
        gamma = 1 - alpha - beta

        return alpha, beta, gamma

    def get_weights_and_z(self, point: Point):
        alpha, beta, gamma = self.get_weights(point)
        z = self.get_z(alpha, beta, gamma)
        return alpha, beta, gamma, z

    def get_normal(self):
        v0 = (
            self.p1.x - self.p0.x,
            self.p1.y - self.p0.y,
            self.p1.z_ndc - self.p0.z_ndc,
        )
        v1 = (
            self.p2.x - self.p0.x,
            self.p2.y - self.p0.y,
            self.p2.z_ndc - self.p0.z_ndc,
        )

        n = np.cross(v0, v1)
        magnitude = np.linalg.norm(n)
        if magnitude != 0:
            n = n / magnitude
        return n


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

        self.transformation_matrix = np.matmul(
            self.transformation_matrix, rotation_matrix
        )

    def apply_perspective(self, f, z_far, z_near, aspect_ratio):
        """Apply perspective to the transformation matrix."""

        perspective_matrix = np.array(
            [
                [f / aspect_ratio, 0, 0, 0],
                [0, f, 0, 0],
                [
                    0,
                    0,
                    (z_far + z_near) / (z_near - z_far),
                    (2 * z_far * z_near) / (z_near - z_far),
                ],
                [0, 0, -1, 0],
            ]
        )
        self.transformation_matrix = np.matmul(
            self.transformation_matrix, perspective_matrix
        )

    def apply_mirror(self, axis):
        """Apply mirror to the transformation matrix."""
        dict = {
            "x": np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            "y": np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            "z": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
        }

        self.transformation_matrix = np.matmul(self.transformation_matrix, dict[axis])

    def get_transformation_matrix(self):
        """Return the resulting transformation matrix."""
        return self.transformation_matrix


class DirectionalLight:
    def __init__(
        self,
        ambient_intensity: float,
        color: list[float],
        intensity: float,
        direction: list[int],
    ):
        self.ambient_intensity = ambient_intensity
        self.color = color
        self.intensity = intensity
        self.direction = direction
