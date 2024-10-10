import math

class Cube:
    def __init__(self, size):
        self.size = size
        self.vertices = self._get_vertices()

    def _get_vertices(self) -> list[tuple]:
        half_size = self.size / 2
        v0 = (-half_size, half_size, -half_size)
        v1 = (-half_size, half_size, half_size)
        v2 = (half_size, half_size, half_size)
        v3 = (half_size, half_size, -half_size)
        v4 = (-half_size, -half_size, -half_size)
        v5 = (-half_size, -half_size, half_size)
        v6 = (half_size, -half_size, half_size)
        v7 = (half_size, -half_size, -half_size)

        return [v0, v1, v2, v3, v4, v5, v6, v7]

    def get_triangles(self) -> list[list[tuple[float, float, float]]]:
        t0 = [self.vertices[0], self.vertices[1], self.vertices[3]]
        t1 = [self.vertices[1], self.vertices[2], self.vertices[3]]
        t2 = [self.vertices[0], self.vertices[4], self.vertices[1]]
        t3 = [self.vertices[4], self.vertices[5], self.vertices[1]]
        t4 = [self.vertices[1], self.vertices[5], self.vertices[2]]
        t5 = [self.vertices[5], self.vertices[6], self.vertices[2]]
        t6 = [self.vertices[2], self.vertices[6], self.vertices[3]]
        t7 = [self.vertices[6], self.vertices[7], self.vertices[3]]
        t8 = [self.vertices[3], self.vertices[7], self.vertices[0]]
        t9 = [self.vertices[7], self.vertices[4], self.vertices[0]]
        t10 = [self.vertices[4], self.vertices[7], self.vertices[5]]
        t11 = [self.vertices[7], self.vertices[6], self.vertices[5]]

        return [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]

class Cone:

    def __init__(self, bottom_radius, height, sample=20):
        self.bottom_radius = bottom_radius
        self.height = height
        self.sample = sample
        self.vertices = self._get_vertices()

    def _get_vertices(self) -> list[tuple]:
        v0 = (0, self.height/2, 0)

        vertices = [v0]

        theta = 0
        while theta < 2*math.pi:
            x, z = self.bottom_radius * math.cos(theta), self.bottom_radius * math.sin(theta)
            vertices.append((x, -self.height/2, z))
            theta += 2*math.pi/self.sample

        return vertices
    
    def get_triangles(self) -> list[list[tuple[float, float, float]]]:
        triangles = []
        v0 = self.vertices[0]
        
        for i in range(1, len(self.vertices)-1):
            v1, v2 = self.vertices[i], self.vertices[i+1]
            triangles.append([v0, v2, v1])

        self.vertices.append(self.vertices[0])
        self.vertices.append(self.vertices[1])
        
        center = (0, -self.height/2, 0)
        for i in range(0, len(self.vertices)-1):
            v1, v2 = self.vertices[i], self.vertices[i+1]
            if i % 2 == 0:
                v1, v2 = v2, v1

            triangles.append([center, v1, v2])
            
        return triangles


class Cilinder:
    def __init__(self, radius, height, sample=20):
        self.radius = radius
        self.height = height
        self.sample = sample
        self.vertices = self._get_vertices()

    def _get_vertices(self) -> list[tuple]:
        vertices = []
        
        theta = 0
        while theta < 2*math.pi:
            x, z = self.radius * math.cos(theta), self.radius * math.sin(theta)
            vertices.append((x, self.height/2, z))
            vertices.append((x, -self.height/2, z))
            theta += 2*math.pi/self.sample

        return vertices

    def get_triangles(self) -> list[list[tuple[float, float, float]]]:
        triangles = []
        # Cilinder Walls
        for i in range(len(self.vertices)-2):
            v0, v1, v2 = self.vertices[i], self.vertices[i+1], self.vertices[i+2]

            if i % 2 == 0:
                v1, v2 = v2, v1
            triangles.append([v0, v1, v2])

        upper_center = (0, self.height / 2, 0)
        lower_center = (0, -self.height / 2, 0)

        self.vertices.append(self.vertices[0])
        self.vertices.append(self.vertices[1])
        
        for i in range(0, len(self.vertices)-3, 2):
            upper_v1, upper_v2 = self.vertices[i], self.vertices[i+2]
            lower_v1, lower_v2 = self.vertices[i+1], self.vertices[i+3]

            if i % 2 == 0:
                upper_v1, upper_v2 = upper_v2, upper_v1
                lower_v1, lower_v2 = lower_v2, lower_v1

            triangles.append([upper_center, upper_v1, upper_v2])
            triangles.append([lower_center, lower_v1, lower_v2])


        return triangles
