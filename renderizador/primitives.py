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

    