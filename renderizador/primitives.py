from abc import ABC, abstractmethod
import math
from typing import List, Tuple, Dict, Set
import numpy as np


class Primitive(ABC):
    """Abstract base class for 3D geometric primitives."""

    def __init__(self):
        self.vertices: List[Tuple[float, float, float]] = self._get_vertices()
        self.faces: List[List[int]] = self._get_faces()
        self._vertex_face_map: Dict[int, Set[int]] = self._calculate_vertex_face_map()
        self._face_normals: List[np.ndarray] = self._calculate_face_normals()
        self._vertex_normals: Dict[int, np.ndarray] = self._calculate_vertex_normals()

    def _calculate_vertex_face_map(self) -> Dict[int, Set[int]]:
        """
        Create a mapping of vertex indices to the faces they belong to.
        Returns a dictionary where keys are vertex indices and values are sets of face indices.
        """
        vertex_face_map = {i: set() for i in range(len(self.vertices))}
        for face_idx, face in enumerate(self.faces):
            for vertex_idx in face:
                vertex_face_map[vertex_idx].add(face_idx)
        return vertex_face_map

    def _calculate_vertex_normals(self) -> Dict[int, np.ndarray]:
        """
        Calculate normal vectors for all vertices by averaging the normals of adjacent faces.
        Returns a dictionary mapping vertex indices to their normalized normal vectors.
        """
        vertex_normals = {}

        for vertex_idx in range(len(self.vertices)):
            # Get all faces that share this vertex
            adjacent_faces = self._vertex_face_map[vertex_idx]

            if not adjacent_faces:
                # If vertex has no faces, use a default normal pointing up
                vertex_normals[vertex_idx] = np.array([0.0, 1.0, 0.0])
                continue

            # Calculate ∑Ni (sum of adjacent face normals)
            normal_sum = sum(
                self._face_normals[face_idx] for face_idx in adjacent_faces
            )

            # Calculate ||∑Ni|| (magnitude of the sum)
            magnitude = np.linalg.norm(normal_sum)

            # Normalize: Nv = ∑Ni / ||∑Ni||
            if magnitude > 0:
                vertex_normals[vertex_idx] = normal_sum / magnitude
            else:
                vertex_normals[vertex_idx] = np.array([0.0, 1.0, 0.0])

        return vertex_normals

    @abstractmethod
    def _get_vertices(self) -> List[Tuple[float, float, float]]:
        """Generate vertex coordinates for the primitive."""
        pass

    @abstractmethod
    def _get_faces(self) -> List[List[int]]:
        """Generate face indices for the primitive."""
        pass

    def get_primitive(self) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
        """Return the primitive's vertices and faces."""
        return self.vertices, self.faces

    def _calculate_face_normals(self) -> List[np.ndarray]:
        """
        Calculate normal vectors for all faces.
        Returns a list of normalized normal vectors for each face.
        """
        self._face_normals = []
        for face in self.faces:
            v0 = np.array(self.vertices[face[0]])
            v1 = np.array(self.vertices[face[1]])
            v2 = np.array(self.vertices[face[2]])

            # Calculate vectors from vertex 0 to vertex 1 and vertex 2
            edge1 = v1 - v0
            edge2 = v2 - v0

            # Calculate cross product
            normal = np.cross(edge1, edge2)

            # Normalize the normal vector
            length = np.linalg.norm(normal)
            if length > 0:
                normal = normal / length

            self._face_normals.append(normal)

        return self._face_normals

    def get_vertex_normal(self, vertex_idx: int) -> np.ndarray:
        """
        Get the normal vector for a specific vertex.

        Args:
            vertex_idx: Index of the vertex

        Returns:
            numpy.ndarray: Normalized normal vector for the vertex
        """
        return self._vertex_normals[vertex_idx]

    def get_face_normal(self, face_idx: int) -> np.ndarray:
        """
        Get the normal vector for a specific face.

        Args:
            face_idx: Index of the face

        Returns:
            numpy.ndarray: Normalized normal vector for the face
        """
        return self._face_normals[face_idx]


class Cube(Primitive):
    def __init__(self, size: float):
        self.size = size
        super().__init__()

    def _get_vertices(self) -> List[Tuple[float, float, float]]:
        half_size = self.size / 2
        return [
            (-half_size, half_size, -half_size),  # v0
            (-half_size, half_size, half_size),  # v1
            (half_size, half_size, half_size),  # v2
            (half_size, half_size, -half_size),  # v3
            (-half_size, -half_size, -half_size),  # v4
            (-half_size, -half_size, half_size),  # v5
            (half_size, -half_size, half_size),  # v6
            (half_size, -half_size, -half_size),  # v7
        ]

    def _get_faces(self) -> List[List[int]]:
        return [
            [0, 1, 3],
            [1, 2, 3],  # top
            [0, 4, 1],
            [4, 5, 1],  # left
            [1, 5, 2],
            [5, 6, 2],  # front
            [2, 6, 3],
            [6, 7, 3],  # right
            [3, 7, 0],
            [7, 4, 0],  # back
            [4, 7, 5],
            [7, 6, 5],  # bottom
        ]


class Cone(Primitive):
    def __init__(self, bottom_radius: float, height: float, samples: int = 30):
        self.bottom_radius = bottom_radius
        self.height = height
        self.samples = samples
        super().__init__()

    def _get_vertices(self) -> List[Tuple[float, float, float]]:
        vertices = [(0, self.height / 2, 0)]  # apex

        for i in range(self.samples):
            theta = 2 * math.pi * i / self.samples
            x = self.bottom_radius * math.cos(theta)
            z = self.bottom_radius * math.sin(theta)
            vertices.append((x, -self.height / 2, z))

        return vertices

    def _get_faces(self) -> List[List[int]]:
        faces = []
        # Side faces
        for i in range(1, len(self.vertices) - 1):
            faces.append([0, i + 1, i])
        faces.append([0, 1, len(self.vertices) - 1])

        # Base face (optional, uncomment if needed)
        # base = list(range(1, len(self.vertices)))
        # faces.append(base)

        return faces


class Cylinder(Primitive):
    def __init__(self, radius: float, height: float, samples: int = 30):
        self.radius = radius
        self.height = height
        self.samples = samples
        super().__init__()

    def _get_vertices(self) -> List[Tuple[float, float, float]]:
        vertices = [
            (0, self.height / 2, 0),  # top center
            (0, -self.height / 2, 0),  # bottom center
        ]

        for i in range(self.samples):
            theta = 2 * math.pi * i / self.samples
            x = self.radius * math.cos(theta)
            z = self.radius * math.sin(theta)
            vertices.extend(
                [
                    (x, self.height / 2, z),  # top rim
                    (x, -self.height / 2, z),  # bottom rim
                ]
            )

        return vertices

    def _get_faces(self) -> List[List[int]]:
        faces = []
        vertex_count = len(self.vertices)

        # Side faces
        for i in range(2, vertex_count - 2, 2):
            next_i = (i + 2) if i + 2 < vertex_count else 2
            faces.extend(
                [
                    [i + 1, i, next_i],  # first triangle
                    [i + 1, next_i, next_i + 1],  # second triangle
                ]
            )

        # Top cap
        for i in range(2, vertex_count, 2):
            next_i = (i + 2) if i + 2 < vertex_count else 2
            faces.append([0, next_i, i])

        # Bottom cap
        for i in range(3, vertex_count + 1, 2):
            next_i = (i + 2) if i + 2 < vertex_count else 3
            faces.append([1, i, next_i])

        return faces


class Sphere(Primitive):
    def __init__(self, radius: float, samples: int = 10):
        self.radius = radius
        self.samples = samples
        super().__init__()

    def _get_vertices(self) -> List[Tuple[float, float, float]]:
        vertices = [(0, self.radius, 0)]  # top vertex

        # Generate vertices for each latitude ring
        for i in range(1, self.samples):
            phi = math.pi * i / self.samples
            y = self.radius * math.cos(phi)
            ring_radius = self.radius * math.sin(phi)

            for j in range(self.samples):
                theta = 2 * math.pi * j / self.samples
                x = ring_radius * math.cos(theta)
                z = ring_radius * math.sin(theta)
                vertices.append((x, y, z))

        vertices.append((0, -self.radius, 0))  # bottom vertex
        return vertices

    def _get_faces(self) -> List[List[int]]:
        faces = []
        vertices_per_ring = self.samples

        # Top cap
        top_vertex = 0
        first_ring_start = 1
        for i in range(vertices_per_ring):
            v1 = first_ring_start + i
            v2 = first_ring_start + ((i + 1) % vertices_per_ring)
            faces.append([top_vertex, v2, v1])

        # Middle rings
        for ring in range(self.samples - 2):
            ring_start = 1 + ring * vertices_per_ring
            next_ring_start = ring_start + vertices_per_ring

            for i in range(vertices_per_ring):
                v1 = ring_start + i
                v2 = ring_start + ((i + 1) % vertices_per_ring)
                v3 = next_ring_start + i
                v4 = next_ring_start + ((i + 1) % vertices_per_ring)
                faces.extend(
                    [[v1, v2, v4], [v1, v4, v3]]  # first triangle  # second triangle
                )

        # Bottom cap
        bottom_vertex = len(self.vertices) - 1
        last_ring_start = bottom_vertex - vertices_per_ring
        for i in range(vertices_per_ring):
            v1 = last_ring_start + i
            v2 = last_ring_start + ((i + 1) % vertices_per_ring)
            faces.append([bottom_vertex, v1, v2])

        return faces
