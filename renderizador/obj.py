class OBJModel:
    def __init__(self, file_path):
        self.vertices, self.faces = self.parse_obj(file_path)

    def parse_obj(self, file_path):
        vertices = []
        faces = []
        
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('v '):  # Vertex
                    vertex = list(map(float, line.split()[1:4]))
                    vertices.append(vertex)
                elif line.startswith('f '):  # Face
                    face = []
                    for v in line.split()[1:]:
                        w = v.split('/')
                        face.append(int(w[0]) - 1)
                    faces.append(face)
        
        return vertices, faces

def load_obj(file_path):
    return OBJModel(file_path)