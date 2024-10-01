#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time  # Para operações com tempo
import gpu  # Simula os recursos de uma GPU
import math  # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

from utils import Transform, Point, Triangle


class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.01  # plano de corte próximo
    far = 1000  # plano de corte distante

    two_d_width = 30
    two_d_height = 20


    perspective_matrix = None
    transformation_stack = []

    @staticmethod
    def setup(width, height, supersampling_factor, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.supersampling_factor = supersampling_factor
        GL.near = near
        GL.far = far
        
    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Polypoint2D : pontos = {0}".format(point)) # imprime no terminal pontos
        # print("Polypoint2D : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo:
        # pos_x = GL.width//2
        # pos_y = GL.height//2
        # gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 0])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

        for i in range(0, len(point), 2):
            pos_x = int(point[i]) * GL.supersampling_factor
            pos_y = int(point[i + 1]) * GL.supersampling_factor
            point_color = colors["emissiveColor"]
            # print("Ponto ({0}, {1}) com a cor {2}".format(pos_x, pos_y, point_color))
            gpu.GPU.draw_pixel(
                [pos_x, pos_y],
                gpu.GPU.RGB8,
                [
                    int(point_color[0] * 255),
                    int(point_color[1] * 255),
                    int(point_color[2] * 255),
                ],
            )

    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""

        def bresenham_line(x1, y1, x2, y2, color):
            if x1 > x2:  # Garante que x1 seja sempre menor que x2
                x1, x2 = x2, x1
                y1, y2 = y2, y1

            dy = y2 - y1
            dx = x2 - x1

            if dx == 0:
                for y in range(y1, y2 + 1):
                    if 0 <= x1 < GL.width and 0 <= y < GL.height:
                        gpu.GPU.draw_pixel(
                            [x1, y],
                            gpu.GPU.RGB8,
                            [
                                int(color[0] * 255),
                                int(color[1] * 255),
                                int(color[2] * 255),
                            ],
                        )
                return

            if dy == 0:
                for x in range(x1, x2 + 1):
                    if 0 <= x < GL.width and 0 <= y1 < GL.height:
                        gpu.GPU.draw_pixel(
                            [x, y1],
                            gpu.GPU.RGB8,
                            [
                                int(color[0] * 255),
                                int(color[1] * 255),
                                int(color[2] * 255),
                            ],
                        )
                return

            dydx = dy / dx

            if -1 < dydx <= 1:
                for x in range(x1, x2 + 1):
                    y = round(y1 + dydx * (x - x1))
                    if 0 <= x < GL.width and 0 <= y < GL.height:
                        gpu.GPU.draw_pixel(
                            [x, y],
                            gpu.GPU.RGB8,
                            [
                                int(color[0] * 255),
                                int(color[1] * 255),
                                int(color[2] * 255),
                            ],
                        )

            else:
                if y1 > y2:  # Garante que y1 seja sempre menor que y2
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                dy = y2 - y1
                dx = x2 - x1
                dxdy = dx / dy
                for y in range(y1, y2 + 1):
                    x = round(x1 + dxdy * (y - y1))
                    if 0 <= x < GL.width and 0 <= y < GL.height:
                        gpu.GPU.draw_pixel(
                            [x, y],
                            gpu.GPU.RGB8,
                            [
                                int(color[0] * 255),
                                int(color[1] * 255),
                                int(color[2] * 255),
                            ],
                        )
            return

        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        # print("Polyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal
        # print("Polyline2D : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo:
        # pos_x = GL.width//2
        # pos_y = GL.height//2
        # gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

        # print("Polyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal

        pos_x = int(lineSegments[0]) * GL.supersampling_factor
        pos_y = int(lineSegments[1]) * GL.supersampling_factor
        for i in range(2, len(lineSegments), 2):
            pos_x2 = int(lineSegments[i]) * GL.supersampling_factor
            pos_y2 = int(lineSegments[i + 1]) * GL.supersampling_factor
            line_color = colors["emissiveColor"]
            bresenham_line(pos_x, pos_y, pos_x2, pos_y2, line_color)
            pos_x = pos_x2
            pos_y = pos_y2

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        print("Circle2D : radius = {0}".format(radius))  # imprime no terminal
        print("Circle2D : colors = {0}".format(colors))  # imprime no terminal as cores

        xc, yc = 0, 0
        r = int(radius) * GL.supersampling_factor
        for x in range(-r, r + 1):
            y1 = int(math.sqrt(r**2 - x**2))
            y2 = -y1
            print("Ponto ({0}, {1})".format(xc + x, yc + y1))
            print("Ponto ({0}, {1})".format(xc + x, yc + y2))
            if 0 <= x <= GL.width and 0 <= y1 <= GL.height:
                gpu.GPU.draw_pixel(
                    [xc + x, yc + y1],
                    gpu.GPU.RGB8,
                    [
                        int(colors["emissiveColor"][0] * 255),
                        int(colors["emissiveColor"][1] * 255),
                        int(colors["emissiveColor"][2] * 255),
                    ],
                )
            if 0 <= x <= GL.width and 0 <= y2 <= GL.height:
                gpu.GPU.draw_pixel(
                    [xc + x, yc + y2],
                    gpu.GPU.RGB8,
                    [
                        int(colors["emissiveColor"][0] * 255),
                        int(colors["emissiveColor"][1] * 255),
                        int(colors["emissiveColor"][2] * 255),
                    ],
                )

        for y in range(-r, r + 1):
            x1 = int(math.sqrt(r**2 - y**2))
            x2 = -x1
            print("Ponto ({0}, {1})".format(xc + x1, yc + y))
            print("Ponto ({0}, {1})".format(xc + x2, yc + y))
            if 0 <= x1 <= GL.width and 0 <= y <= GL.height:
                gpu.GPU.draw_pixel(
                    [xc + x1, yc + y],
                    gpu.GPU.RGB8,
                    [
                        int(colors["emissiveColor"][0] * 255),
                        int(colors["emissiveColor"][1] * 255),
                        int(colors["emissiveColor"][2] * 255),
                    ],
                )
            if 0 <= x2 <= GL.width and 0 <= y <= GL.height:
                gpu.GPU.draw_pixel(
                    [xc + x2, yc + y],
                    gpu.GPU.RGB8,
                    [
                        int(colors["emissiveColor"][0] * 255),
                        int(colors["emissiveColor"][1] * 255),
                        int(colors["emissiveColor"][2] * 255),
                    ],
                )

    @staticmethod
    def triangleSet2D(vertices, colors, textures=None, three_d=False):
        """Função usada para renderizar TriangleSet2D."""

        def get_pixel_texture(alpha, beta, gamma, textures, uvw_primes):
            u0_prime, u1_prime, u2_prime = uvw_primes['u_primes']
            v0_prime, v1_prime, v2_prime = uvw_primes['v_primes']
            w0_prime, w1_prime, w2_prime = uvw_primes['w_primes']

            u_prime = alpha * u0_prime + beta * u1_prime + gamma * u2_prime
            v_prime = alpha * v0_prime + beta * v1_prime + gamma * v2_prime
            w_prime = alpha * w0_prime + beta * w1_prime + gamma * w2_prime

            u = u_prime / w_prime
            v = v_prime / w_prime

            img_width, img_height = len(textures["image"]), len(textures["image"][0])
            u = int(u * img_width) % img_width
            v = int((1 - v) * img_height) % img_height

            return textures["image"][u][v][:3]



        def get_pixel_color(triangle, alpha, beta, gamma, colors, i):
            z0, z1, z2 = triangle.p0.z_camera, triangle.p1.z_camera, triangle.p2.z_camera

            A = alpha / z0 if z0 != 0 else alpha
            B = beta / z1 if z1 != 0 else beta
            C = gamma / z2 if z2 != 0 else gamma

            z = 1 / (A + B + C) if A + B + C != 0 else 1

            if "color_per_vertex" not in colors:
                return [colors["emissiveColor"][0] * 255, colors["emissiveColor"][1] * 255, colors["emissiveColor"][2] * 255]

            c0 = colors["color_per_vertex"][i // 2]
            c1 = colors["color_per_vertex"][i // 2 + 1]
            c2 = colors["color_per_vertex"][i // 2 + 2]

            color = [
                z * (A * c0[0] + B * c1[0] + C * c2[0]), 
                z * (A * c0[1] + B * c1[1] + C * c2[1]), 
                z * (A * c0[2] + B * c1[2] + C * c2[2])
            ]

            return [
                max(min(int(color[0] * 255), 255), 0),
                max(min(int(color[1] * 255), 255), 0),
                max(min(int(color[2] * 255), 255), 0),
            ]

        step = 15 if three_d else 6
        for i in range(0, len(vertices), step):
            if three_d:
                x0, y0, z0_camera, z0_ndc, w0 = vertices[i], vertices[i + 1], vertices[i + 2], vertices[i + 3], vertices[i + 4]
                x1, y1, z1_camera, z1_ndc, w1 = vertices[i + 5], vertices[i + 6], vertices[i + 7], vertices[i + 8], vertices[i + 9]
                x2, y2, z2_camera, z2_ndc, w2 = vertices[i + 10], vertices[i + 11], vertices[i + 12], vertices[i + 13], vertices[i + 14]
                # print(f'Zs: {z0}, {z1}, {z2}')
            else:
                x0, y0, z0_camera, z0_ndc, w0 = int(vertices[i]) * GL.supersampling_factor, int(vertices[i + 1]) * GL.supersampling_factor, 1, 1, 1
                x1, y1, z1_camera, z1_ndc, w1 = int(vertices[i + 2]) * GL.supersampling_factor, int(vertices[i + 3]) * GL.supersampling_factor, 1, 1, 1
                x2, y2, z2_camera, z2_ndc, w2 = int(vertices[i + 4]) * GL.supersampling_factor, int(vertices[i + 5]) * GL.supersampling_factor, 1, 1, 1

            p0, p1, p2 = Point(x0, y0, z0_camera, z0_ndc), Point(x1, y1, z1_camera, z1_ndc), Point(x2, y2, z2_camera, z2_ndc)
            triangle = Triangle(p0, p1, p2)
            
            min_x, max_x, min_y, max_y = triangle.get_bounds_within_screen(GL.width, GL.height)

            if textures:
                # Get UV coordinates at vertices
                uv0, uv1, uv2 = textures["uvs"]
                u0, v0 = uv0[0], uv0[1]
                u1, v1 = uv1[0], uv1[1]
                u2, v2 = uv2[0], uv2[1]

                # Compute u', v', w' for each vertex
                u0_prime = u0 / w0
                v0_prime = v0 / w0
                w0_prime = 1 / w0

                u1_prime = u1 / w1
                v1_prime = v1 / w1
                w1_prime = 1 / w1

                u2_prime = u2 / w2
                v2_prime = v2 / w2
                w2_prime = 1 / w2

                # Store these values for access in get_pixel_texture
                uvw_primes = {
                    'u_primes': (u0_prime, u1_prime, u2_prime),
                    'v_primes': (v0_prime, v1_prime, v2_prime),
                    'w_primes': (w0_prime, w1_prime, w2_prime)
                }

            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    p = Point(x, y, 1, 1) # TODO: Z value
                    if triangle.is_inside(p):
                        alpha, beta, gamma, z = triangle.get_weights_and_z(p)
                        p.z_ndc = z

                        if p.z_ndc > gpu.GPU.read_pixel([int(x), int(y)], gpu.GPU.DEPTH_COMPONENT32F):
                            continue

                        if textures:
                            color = get_pixel_texture(alpha, beta, gamma, textures, uvw_primes)
                        else:
                            color = get_pixel_color(triangle, alpha, beta, gamma, colors, i)
                            if "transparency" in colors:
                                transparency = colors["transparency"]
                                old_color = gpu.GPU.read_pixel([int(x), int(y)], gpu.GPU.RGB8)

                                for c in range(3):
                                    color[c] = old_color[c] * transparency + color[c] * (1-transparency)

                        gpu.GPU.draw_pixel(
                            [int(x), int(y)],
                            gpu.GPU.RGB8,
                            [color[0], color[1], color[2]],
                        )

                        gpu.GPU.draw_pixel(
                            [int(x), int(y)],
                            gpu.GPU.DEPTH_COMPONENT32F,
                            [p.z_ndc],
                        )

    @staticmethod
    def triangleSet(point, colors, textures=None):
        """Função usada para renderizar TriangleSet."""

        # pix = gpu.GPU.read_pixel(
        #     [0, 0], 
        #     gpu.GPU.DEPTH_COMPONENT32F
        # )
        # print(f"Depth: {pix}")

        for i in range(0, len(point), 9):
            x0, y0, z0 = point[i], point[i + 1], point[i + 2]
            x1, y1, z1 = point[i + 3], point[i + 4], point[i + 5]
            x2, y2, z2 = point[i + 6], point[i + 7], point[i + 8]


            points_matrix = np.array([
                [x0, x1, x2], 
                [y0, y1, y2], 
                [z0, z1, z2], 
                [1, 1, 1]])
            
            transformed_points = np.matmul(GL.transformation_stack[-1], points_matrix)
            z_camera = transformed_points[2]
            ndc = np.matmul(GL.perspective_matrix, transformed_points)
            w_values = ndc[3]
            ndc = ndc / ndc[3]
            z_ndc = ndc[2]

            transform = Transform()
            transform.apply_scale([GL.width / 2, GL.height / 2, 1])
            transform.apply_translation([1, 1, 0])
            transform.apply_mirror("y")
            transform_matrix = transform.get_transformation_matrix()

            screen_points = np.matmul(transform_matrix, ndc)
            screen_points = screen_points / screen_points[3]
            
            points = []
            for j in range(0, 3):
                points.append(screen_points[0][j])  # x
                points.append(screen_points[1][j])  # y
                points.append(z_camera[j])          # z (Colors)
                points.append(z_ndc[j])             # z (Z Buffer)
                points.append(w_values[j])          # w

            GL.triangleSet2D(points, colors, textures, three_d=True)


    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # Instantiate the Transform class
        transform = Transform()

        # Define perspective directions
        fovY = 2 * math.atan(math.tan(fieldOfView / 2) * GL.height / math.sqrt((GL.height**2 + GL.width**2)))
        top = GL.near * math.tan(fovY)
        bottom = -top
        right = top * (GL.width / GL.height)
        left = -right

        # Apply perspective transformation
        # The perspective matrix handles how objects are projected onto the screen
        directions = (top, bottom, right, left)
        transform.apply_perspective(directions, GL.near, GL.far)
        # print(f'Perspec: {transform.get_transformation_matrix()}')


        # Apply rotation transformation
        # The camera's orientation is converted from the axis-angle form into a rotation matrix.
        # This matrix is responsible for rotating the scene according to the camera's orientation.
        transform.apply_rotation(orientation, inverse=True)

        # Apply translation transformation
        # The camera’s position is used to create a translation matrix that moves the entire scene
        # in the opposite direction of the camera's position.
        tx, ty, tz = position
        translation = (-tx, -ty, -tz)
        transform.apply_translation(translation)

        GL.perspective_matrix = transform.get_transformation_matrix()

        print("Viewpoint : ", end="")
        print("position = {0} ".format(position), end="")
        print("orientation = {0} ".format(orientation), end="")
        print("fieldOfView = {0} ".format(fieldOfView))
        print(f"perspective matrix: {GL.perspective_matrix}")

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Transform : ", end="")

        # Instantiate the Transform class
        transform = Transform()

        # Apply the transformations
        if translation:
            print("translation = {0} ".format(translation), end="")
            transform.apply_translation(translation)

        if rotation:
            print("rotation = {0} ".format(rotation), end="")
            transform.apply_rotation(rotation)

        if scale:
            print("scale = {0} ".format(scale), end="")
            transform.apply_scale(scale)

        # Get the final transformation matrix and append it to the stack
        transformation_matrix = transform.get_transformation_matrix()
        print(f"\ntransformation: {transformation_matrix}")

        if len(GL.transformation_stack) > 0:
            transformation_matrix = np.matmul(GL.transformation_stack[-1], transformation_matrix)

        GL.transformation_stack.append(transformation_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Saindo de Transform")
        GL.transformation_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TriangleStripSet : pontos = {0} ".format(point), end="")
        for i, strip in enumerate(stripCount):
            print("strip[{0}] = {1} ".format(i, strip), end="")
        print("")
        print(
            "TriangleStripSet : colors = {0}".format(colors)
        )  # imprime no terminal as cores

        vertex_index = 0
        for strip in stripCount:        
            for i in range(strip - 2):       
                x0, y0, z0 = point[vertex_index * 3], point[vertex_index * 3 + 1], point[vertex_index * 3 + 2]         
                x1, y1, z1 = point[(vertex_index + 1) * 3], point[(vertex_index + 1) * 3 + 1], point[(vertex_index + 1) * 3 + 2]
                x2, y2, z2 = point[(vertex_index + 2) * 3], point[(vertex_index + 2) * 3 + 1], point[(vertex_index + 2) * 3 + 2]

                if i % 2 == 0:
                    points = [x0, y0, z0, x1, y1, z1, x2, y2, z2]
                else:
                    points = [x0, y0, z0, x2, y2, z2, x1, y1, z1]
                
                GL.triangleSet(points, colors)                
                vertex_index += 1
            
            vertex_index += 2
        

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(f'IndexedTriangleStripSet : pontos = {point}, index = {index}')
        print(f'IndexedTriangleStripSet : colors = {colors}')

        all_points = []
        for i in range(0, len(point) - 2, 3):
            x = point[i]
            y = point[i + 1]
            z = point[i + 2]
            all_points.append((x, y, z))

        i = 0
        while i < len(index) - 2:
            if index[i+2] == -1:
                i += 3
            else:
                p0 = all_points[index[i]]
                p1 = all_points[index[i+1]]
                p2 = all_points[index[i+2]]

                if i % 2 == 0:
                    points = [p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]]
                else:
                    points = [p0[0], p0[1], p0[2], p2[0], p2[1], p2[2], p1[0], p1[1], p1[2]]

                GL.triangleSet(points, colors)
                i += 1


    @staticmethod
    def indexedFaceSet(
        coord,
        coordIndex,
        colorPerVertex,
        color,
        colorIndex,
        texCoord,
        texCoordIndex,
        colors,
        current_texture,
    ):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        # print("IndexedFaceSet: ")
        textures = None
        # if coord: print(f'Pontos: {coord}, coordIndex: {coordIndex}')
        # if colorPerVertex and color and colorIndex: print(f'Cores: {color}, colorIndex: {colorIndex}')
        # if texCoord and texCoordIndex: print(f'Texturas: {texCoord}, texCoordIndex: {texCoordIndex}')
        if current_texture:
            image = gpu.GPU.load_texture(current_texture[0])
            # print(f'Matriz com imagem = {image}')
            # print(f'Dimensões da image = {image.shape}')
            textures = {"image": image}


        # print(f'IndexedFaceSet : colors = {colors}')


        all_points, all_colors, all_uvs = [], [], []

        for i in range(0, len(coord) - 2, 3):
            x, y, z = coord[i], coord[i + 1], coord[i + 2]
            all_points.append((x, y, z))

            if colorPerVertex and color and colorIndex:
                c0, c1, c2 = color[i], color[i + 1], color[i + 2]
                all_colors.append((c0, c1, c2))

        if texCoord and texCoordIndex:
            for i in range(0, len(texCoord) - 1, 2):
                u, v = texCoord[i], texCoord[i + 1]
                all_uvs.append((u, v))

        i, origin_point = 0, None
        while i < len(coordIndex) - 1:
            if coordIndex[i] == -1 or coordIndex[i + 1] == -1:
                origin_point = None
                i += 1
                continue
            
            if origin_point is None:
                origin_point = coordIndex[i]
                i += 1
                continue
            
            p0, p1, p2 = all_points[origin_point], all_points[coordIndex[i]], all_points[coordIndex[i + 1]]
            points = [p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]]

            if colorPerVertex and color and colorIndex:
                c0, c1, c2 = all_colors[origin_point], all_colors[coordIndex[i]], all_colors[coordIndex[i + 1]]
                colors["color_per_vertex"] = [c0, c1, c2]

            if texCoord and texCoordIndex:
                uv0, uv1, uv2 = all_uvs[origin_point], all_uvs[coordIndex[i]], all_uvs[coordIndex[i + 1]]
                textures["uvs"] = [uv0, uv1, uv2]

            
            GL.triangleSet(points, colors, textures)
            i += 1



    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "Sphere : radius = {0}".format(radius)
        )  # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors))  # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "NavigationInfo : headlight = {0}".format(headlight)
        )  # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color))  # imprime no terminal
        print(
            "DirectionalLight : intensity = {0}".format(intensity)
        )  # imprime no terminal
        print(
            "DirectionalLight : direction = {0}".format(direction)
        )  # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color))  # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity))  # imprime no terminal
        print("PointLight : location = {0}".format(location))  # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color))  # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "TimeSensor : cycleInterval = {0}".format(cycleInterval)
        )  # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = (
            time.time()
        )  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print(
            "SplinePositionInterpolator : key = {0}".format(key)
        )  # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]

        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key))  # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
