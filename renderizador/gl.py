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

from utils import Transform


class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.01  # plano de corte próximo
    far = 1000  # plano de corte distante

    perspective_matrix = None
    transformation_stack = []

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
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
            pos_x = int(point[i])
            pos_y = int(point[i + 1])
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

        pos_x = int(lineSegments[0])
        pos_y = int(lineSegments[1])
        for i in range(2, len(lineSegments), 2):
            pos_x2 = int(lineSegments[i])
            pos_y2 = int(lineSegments[i + 1])
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
        r = int(radius)
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
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""

        def l_coef(x0, y0, x1, y1):
            A = y1 - y0
            B = -(x1 - x0)
            C = y0 * (x1 - x0) - x0 * (y1 - y0)
            return A, B, C

        def l_eval(la, lb, lc, x, y):
            return la * x + lb * y + lc

        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        

        # Exemplo:
        # gpu.GPU.draw_pixel([6, 8], gpu.GPU.RGB8, [255, 255, 0])  # altera pixel (u, v, tipo, r, g, b)

        for i in range(0, len(vertices), 6):
            pos_x1 = int(vertices[i])
            pos_y1 = int(vertices[i + 1])
            pos_x2 = int(vertices[i + 2])
            pos_y2 = int(vertices[i + 3])
            pos_x3 = int(vertices[i + 4])
            pos_y3 = int(vertices[i + 5])
            GL.polyline2D(
                [pos_x1, pos_y1, pos_x2, pos_y2, pos_x3, pos_y3, pos_x1, pos_y1], colors
            )  # Colorir o triângulo

            l1a, l1b, l1c = l_coef(pos_x1, pos_y1, pos_x2, pos_y2)
            l2a, l2b, l2c = l_coef(pos_x2, pos_y2, pos_x3, pos_y3)
            l3a, l3b, l3c = l_coef(pos_x3, pos_y3, pos_x1, pos_y1)

            for x in range(GL.width):
                for y in range(GL.height):
                    l1 = l_eval(l1a, l1b, l1c, x, y)
                    l2 = l_eval(l2a, l2b, l2c, x, y)
                    l3 = l_eval(l3a, l3b, l3c, x, y)
                    if l1 >= 0 and l2 >= 0 and l3 >= 0:
                        gpu.GPU.draw_pixel(
                            [int(x), int(y)],
                            gpu.GPU.RGB8,
                            [
                                int(colors["emissiveColor"][0] * 255),
                                int(colors["emissiveColor"][1] * 255),
                                int(colors["emissiveColor"][2] * 255),
                            ],
                        )

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("TriangleSet : pontos = {0}".format(point))  # imprime no terminal pontos
        # print(
        #     "TriangleSet : colors = {0}".format(colors)
        # ) 

        for i in range(0, len(point), 9):
            pos_x1 = point[i]
            pos_y1 = point[i + 1]
            pos_z1 = point[i + 2]
            pos_x2 = point[i + 3]
            pos_y2 = point[i + 4]
            pos_z2 = point[i + 5]
            pos_x3 = point[i + 6]
            pos_y3 = point[i + 7]
            pos_z3 = point[i + 8]

            points_matrix = np.array([
                [pos_x1, pos_x2, pos_x3], 
                [pos_y1, pos_y2, pos_y3], 
                [pos_z1, pos_z2, pos_z3], 
                [1, 1, 1]])
            
            transformed_points = np.matmul(GL.transformation_stack[-1], points_matrix)
            ndc = np.matmul(GL.perspective_matrix, transformed_points)
            ndc = ndc / ndc[3]

            transform = Transform()
            transform.apply_scale([GL.width / 2, GL.height / 2, 1])
            transform.apply_translation([1, 1, 0])
            transform.apply_mirror("y")
            transform_matrix = transform.get_transformation_matrix()

            screen_points = np.matmul(transform_matrix, ndc)
            screen_points = screen_points / screen_points[3]
            
            points = []
            for j in range(0, 3):
                points.append(screen_points[0][j])
                points.append(screen_points[1][j])

            # print(f'Points: {points}')
            GL.triangleSet2D(points, colors)


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
                pos_x1 = point[vertex_index * 3]
                pos_y1 = point[vertex_index * 3 + 1]
                pos_z1 = point[vertex_index * 3 + 2]

                pos_x2 = point[(vertex_index + 1) * 3]
                pos_y2 = point[(vertex_index + 1) * 3 + 1]
                pos_z2 = point[(vertex_index + 1) * 3 + 2]

                pos_x3 = point[(vertex_index + 2) * 3]
                pos_y3 = point[(vertex_index + 2) * 3 + 1]
                pos_z3 = point[(vertex_index + 2) * 3 + 2]

                if i % 2 == 0:
                    points = [pos_x1, pos_y1, pos_z1, pos_x2, pos_y2, pos_z2, pos_x3, pos_y3, pos_z3]
                else:
                    points = [pos_x1, pos_y1, pos_z1, pos_x3, pos_y3, pos_z3, pos_x2, pos_y2, pos_z2]
                
                GL.triangleSet(points, colors)                
                vertex_index += 1
            
            vertex_index += 2  # Move to the next set of vertices after this strip


        # for i in range(0, len(point) - 8, 3):
        #     pos_x1 = point[i]
        #     pos_y1 = point[i + 1]
        #     pos_z1 = point[i + 2]
        #     pos_x2 = point[i + 3]
        #     pos_y2 = point[i + 4]
        #     pos_z2 = point[i + 5]
        #     pos_x3 = point[i + 6]
        #     pos_y3 = point[i + 7]
        #     pos_z3 = point[i + 8]

        #     if i % 2 == 0:
        #         points = [pos_x1, pos_y1, pos_z1, pos_x2, pos_y2, pos_z2, pos_x3, pos_y3, pos_z3]
        #     else:
        #         points = [pos_x1, pos_y1, pos_z1, pos_x3, pos_y3, pos_z3, pos_x2, pos_y2, pos_z2]

        #     GL.triangleSet(points, colors)
        

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
        print(
            "IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index)
        )
        print(
            "IndexedTriangleStripSet : colors = {0}".format(colors)
        )  # imprime as cores

        all_points = []
        for i in range(0, len(point) - 2, 3):
            pos_x = point[i]
            pos_y = point[i + 1]
            pos_z = point[i + 2]
            all_points.append((pos_x, pos_y, pos_z))

        for i in range(len(index)-3):

            p0 = all_points[index[i]]
            p1 = all_points[index[i+1]]
            p2 = all_points[index[i+2]]

            if i % 2 == 0:
                points = [p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]]
            else:
                points = [p0[0], p0[1], p0[2], p2[0], p2[1], p2[2], p1[0], p1[1], p1[2]]

            GL.triangleSet(points, colors)


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
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        # Os prints abaixo são só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("IndexedFaceSet : ")
        if coord:
            print("\tpontos(x, y, z) = {0}, coordIndex = {1}".format(coord, coordIndex))
        print("colorPerVertex = {0}".format(colorPerVertex))
        if colorPerVertex and color and colorIndex:
            print("\tcores(r, g, b) = {0}, colorIndex = {1}".format(color, colorIndex))
        if texCoord and texCoordIndex:
            print(
                "\tpontos(u, v) = {0}, texCoordIndex = {1}".format(
                    texCoord, texCoordIndex
                )
            )
        if current_texture:
            image = gpu.GPU.load_texture(current_texture[0])
            print("\t Matriz com image = {0}".format(image))
            print("\t Dimensões da image = {0}".format(image.shape))
        print(
            "IndexedFaceSet : colors = {0}".format(colors)
        )  # imprime no terminal as cores

        pos_x0 = coord[0]
        pos_y0 = coord[1]
        pos_z0 = coord[2]

        for i in range(3, len(coord) - 5, 3):
            pos_x1 = coord[i]
            pos_y1 = coord[i + 1]
            pos_z1 = coord[i + 2]
            pos_x2 = coord[i + 3]
            pos_y2 = coord[i + 4]
            pos_z2 = coord[i + 5]

            GL.triangleSet([pos_x0, pos_y0, pos_z0, pos_x1, pos_y1, pos_z1, pos_x2, pos_y2, pos_z2], colors)

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
