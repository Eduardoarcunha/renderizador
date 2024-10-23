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


from geometry import Point, Triangle, Transform, DirectionalLight
from primitives import Cube, Cone, Cylinder, Sphere
from utils import downsample_matrix_with_channels, vector_module


class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.01  # plano de corte próximo
    far = 1000  # plano de corte distante

    two_d_width = 30
    two_d_height = 20

    transformation_stack = []
    view_matrix = None
    view_transform_matrix = None
    perspective_matrix = None

    directional_light = None

    @staticmethod
    def setup(width, height, supersampling_factor, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.supersampling_factor = supersampling_factor
        GL.near = near
        GL.far = far

        transform = Transform()
        transform.apply_scale([GL.width / 2, GL.height / 2, 1])
        transform.apply_translation([1, 1, 0])
        transform.apply_mirror("y")
        GL.screen_transform = transform.get_transformation_matrix()

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
        # #print("Polypoint2D : pontos = {0}".format(point)) # imprime no terminal pontos
        # #print("Polypoint2D : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo:
        # pos_x = GL.width//2
        # pos_y = GL.height//2
        # gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 0])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

        for i in range(0, len(point), 2):
            pos_x = int(point[i]) * GL.supersampling_factor
            pos_y = int(point[i + 1]) * GL.supersampling_factor
            point_color = colors["emissiveColor"]
            # #print("Ponto ({0}, {1}) com a cor {2}".format(pos_x, pos_y, point_color))
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

        # #print("Polyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal
        # #print("Polyline2D : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo:
        # pos_x = GL.width//2
        # pos_y = GL.height//2
        # gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

        # #print("Polyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal

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

        # print("Circle2D : radius = {0}".format(radius))  # imprime no terminal
        # print("Circle2D : colors = {0}".format(colors))  # imprime no terminal as cores

        xc, yc = 0, 0
        r = int(radius) * GL.supersampling_factor
        for x in range(-r, r + 1):
            y1 = int(math.sqrt(r**2 - x**2))
            y2 = -y1
            # print("Ponto ({0}, {1})".format(xc + x, yc + y1))
            # print("Ponto ({0}, {1})".format(xc + x, yc + y2))
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
            # print("Ponto ({0}, {1})".format(xc + x1, yc + y))
            # print("Ponto ({0}, {1})".format(xc + x2, yc + y))
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
    def triangleSet2D(
        vertices, colors, textures=None, three_d=False, normal=None, normals=None
    ):
        """Função usada para renderizar TriangleSet2D."""

        def get_uv(alpha, beta, gamma, triangle):
            uv0, uv1, uv2 = textures["uvs"]
            u0, v0 = uv0[0], uv0[1]
            u1, v1 = uv1[0], uv1[1]
            u2, v2 = uv2[0], uv2[1]

            z0 = triangle.p0.z_camera
            z1 = triangle.p1.z_camera
            z2 = triangle.p2.z_camera

            Z = 1 / (alpha / z0 + beta / z1 + gamma / z2)

            u = Z * (u0 * alpha / z0 + u1 * beta / z1 + u2 * gamma / z2)
            v = Z * (v0 * alpha / z0 + v1 * beta / z1 + v2 * gamma / z2)

            return u, v

        def get_pixel_texture(x, y, uv_matrix):
            u00, v00 = uv_matrix[y][x]

            if y + 1 >= len(uv_matrix) or x + 1 >= len(uv_matrix[y]):
                img_width, img_height = len(textures["images"][0]), len(
                    textures["images"][0][0]
                )
                u = int(u00 * img_width) % img_width
                v = int((1 - v00) * img_height) % img_height
                return textures["images"][0][u][v][:3]

            u01, v01 = uv_matrix[y][x + 1]
            u10, v10 = uv_matrix[y + 1][x]

            dudx = (u01 - u00) * textures["images"][0].shape[0]
            dudy = (u10 - u00) * textures["images"][0].shape[0]
            dvdx = (v01 - v00) * textures["images"][0].shape[1]
            dvdy = (v10 - v00) * textures["images"][0].shape[1]

            L = max(math.sqrt(dudx**2 + dvdx**2), math.sqrt(dudy**2 + dvdy**2))
            D = max(0, int(math.log2(L)))
            D = min(D, len(textures["images"]) - 1)

            img_width, img_height = len(textures["images"][D]), len(
                textures["images"][D][0]
            )
            u = int(u00 * img_width) % img_width
            v = int((1 - v00) * img_height) % img_height

            return textures["images"][D][u][v][:3]

        def get_pixel_color(triangle, alpha, beta, gamma, colors, i):
            z0, z1, z2 = (
                triangle.p0.z_camera,
                triangle.p1.z_camera,
                triangle.p2.z_camera,
            )

            A = alpha / z0
            B = beta / z1
            C = gamma / z2

            z = 1 / (A + B + C) if A + B + C != 0 else 1

            if "color_per_vertex" not in colors:
                return [
                    colors["emissiveColor"][0],
                    colors["emissiveColor"][1],
                    colors["emissiveColor"][2],
                ]

            c0 = colors["color_per_vertex"][i // 2]
            c1 = colors["color_per_vertex"][i // 2 + 1]
            c2 = colors["color_per_vertex"][i // 2 + 2]

            color = [
                z * (A * c0[0] + B * c1[0] + C * c2[0]),
                z * (A * c0[1] + B * c1[1] + C * c2[1]),
                z * (A * c0[2] + B * c1[2] + C * c2[2]),
            ]

            return color

        def interpolate_normal(alpha, beta, gamma, normals):
            n0, n1, n2 = normals

            # Basic linear interpolation for testing
            interpolated = np.array(
                [
                    alpha * n0[0] + beta * n1[0] + gamma * n2[0],
                    alpha * n0[1] + beta * n1[1] + gamma * n2[1],
                    alpha * n0[2] + beta * n1[2] + gamma * n2[2],
                ]
            )

            # Normalize
            length = np.linalg.norm(interpolated)
            if length > 0:
                interpolated = interpolated / length

            return interpolated

        def apply_light(oergb, normal_to_use):
            # print(colors)
            odrgb = colors["diffuseColor"] if "diffuseColor" in colors else [0, 0, 0]
            oa = colors["ambientIntensity"] if "ambientIntensity" in colors else 0
            osrgb = colors["specularColor"] if "specularColor" in colors else [0, 0, 0]
            shiness = colors["shininess"] if "shininess" in colors else 0

            ilrgb = GL.directional_light.color
            ii = GL.directional_light.intensity
            iia = GL.directional_light.ambient_intensity

            L = -np.array(GL.directional_light.direction)
            v = normal_to_use

            lv = (L + v) / vector_module(L + v)

            ambient_i = iia * np.array(odrgb) * np.array(oa)
            diffuse_i = ii * np.array(odrgb) * np.dot(normal_to_use, L)
            specular_i = (
                ii * np.array(osrgb) * (np.dot(normal_to_use, lv)) ** (shiness * 128)
            )
            if np.isnan(specular_i).any():
                specular_i = [0, 0, 0]

            # print(f"Oergb: {oergb}")
            # print(f"{ilrgb * (ambient_i + diffuse_i + specular_i)}")

            irgb = oergb + ilrgb * (ambient_i + diffuse_i + specular_i)

            return irgb

        step = 15 if three_d else 6
        for i in range(0, len(vertices), step):
            if three_d:
                x0, y0, z0_camera, z0_ndc = (
                    vertices[i],
                    vertices[i + 1],
                    vertices[i + 2],
                    vertices[i + 3],
                )
                x1, y1, z1_camera, z1_ndc = (
                    vertices[i + 4],
                    vertices[i + 5],
                    vertices[i + 6],
                    vertices[i + 7],
                )
                x2, y2, z2_camera, z2_ndc = (
                    vertices[i + 8],
                    vertices[i + 9],
                    vertices[i + 10],
                    vertices[i + 11],
                )
            else:
                x0, y0, z0_camera, z0_ndc = (
                    int(vertices[i]) * GL.supersampling_factor,
                    int(vertices[i + 1]) * GL.supersampling_factor,
                    1,
                    1,
                )
                x1, y1, z1_camera, z1_ndc = (
                    int(vertices[i + 2]) * GL.supersampling_factor,
                    int(vertices[i + 3]) * GL.supersampling_factor,
                    1,
                    1,
                )
                x2, y2, z2_camera, z2_ndc = (
                    int(vertices[i + 4]) * GL.supersampling_factor,
                    int(vertices[i + 5]) * GL.supersampling_factor,
                    1,
                    1,
                )

            p0, p1, p2 = (
                Point(x0, y0, z0_camera, z0_ndc),
                Point(x1, y1, z1_camera, z1_ndc),
                Point(x2, y2, z2_camera, z2_ndc),
            )
            triangle = Triangle(p0, p1, p2)

            min_x, max_x, min_y, max_y = triangle.get_bounds_within_screen(
                GL.width, GL.height
            )

            if textures:
                uv_matrix = []
                for y in range(min_y, max_y):
                    row = []
                    for x in range(min_x, max_x):
                        p = Point(x, y, 1, 1)
                        alpha, beta, gamma, z = triangle.get_weights_and_z(p)
                        u, v = get_uv(alpha, beta, gamma, triangle)
                        row.append([u, v])
                    uv_matrix.append(row)

            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    p = Point(x, y, 1, 1)  # TODO: Z value
                    if triangle.is_inside(p):
                        alpha, beta, gamma, z = triangle.get_weights_and_z(p)
                        p.z_ndc = z

                        if p.z_ndc > gpu.GPU.read_pixel(
                            [int(x), int(y)], gpu.GPU.DEPTH_COMPONENT32F
                        ):
                            continue

                        if normals is not None:
                            normal_to_use = interpolate_normal(
                                alpha, beta, gamma, normals
                            )
                        else:
                            normal_to_use = normal

                        if textures:
                            color = get_pixel_texture(x - min_x, y - min_y, uv_matrix)
                        else:
                            color = get_pixel_color(
                                triangle, alpha, beta, gamma, colors, i
                            )
                            if "transparency" in colors:
                                transparency = colors["transparency"]
                                old_color = gpu.GPU.read_pixel(
                                    [int(x), int(y)], gpu.GPU.RGB8
                                )

                                for c in range(3):
                                    color[c] = old_color[c] * transparency + color[
                                        c
                                    ] * (1 - transparency)

                            if GL.directional_light and normal_to_use is not None:
                                color = apply_light(color, normal_to_use)

                            color = [
                                max(min(int(color[0] * 255), 255), 0),
                                max(min(int(color[1] * 255), 255), 0),
                                max(min(int(color[2] * 255), 255), 0),
                            ]

                        # normal_map_color = [
                        #     (normal_to_use[0] + 1) * 127.5,
                        #     (normal_to_use[1] + 1) * 127.5,
                        #     (normal_to_use[2] + 1) * 127.5,
                        # ]

                        # gpu.GPU.draw_pixel(
                        #     [int(x), int(y)],
                        #     gpu.GPU.RGB8,
                        #     [
                        #         normal_map_color[0],
                        #         normal_map_color[1],
                        #         normal_map_color[2],
                        #     ],
                        # )

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
    def triangleSet(point, colors, textures=None, primitive=None, face_idx=None):
        """Função usada para renderizar TriangleSet."""

        def transform_normal(normal, transformation_matrix):
            """Transform normal using the inverse transpose of the transformation matrix."""
            # Get the 3x3 portion of the transformation matrix (remove translation)
            m = transformation_matrix[:3, :3]

            # Calculate inverse transpose
            try:
                m_inv_transpose = np.linalg.inv(m).T
            except np.linalg.LinAlgError:
                # Fallback if matrix is not invertible
                return normal

            # Transform normal
            transformed_normal = np.dot(m_inv_transpose, normal)

            # Renormalize
            length = np.linalg.norm(transformed_normal)
            if length > 0:
                transformed_normal = transformed_normal / length

            return transformed_normal

        for i in range(0, len(point), 9):
            x0, y0, z0 = point[i], point[i + 1], point[i + 2]
            x1, y1, z1 = point[i + 3], point[i + 4], point[i + 5]
            x2, y2, z2 = point[i + 6], point[i + 7], point[i + 8]

            points_matrix = np.array(
                [[x0, x1, x2], [y0, y1, y2], [z0, z1, z2], [1, 1, 1]]
            )

            if GL.view_transform_matrix is None:
                GL.view_transform_matrix = np.matmul(
                    GL.view_matrix, GL.transformation_stack[-1]
                )

            view_points = np.matmul(GL.view_transform_matrix, points_matrix)
            z_camera = view_points[2]

            if primitive and not isinstance(primitive, Cube):
                face = primitive.faces[face_idx]

                n0 = primitive.get_vertex_normal(face[0])
                n1 = primitive.get_vertex_normal(face[1])
                n2 = primitive.get_vertex_normal(face[2])

                n0 = transform_normal(n0, GL.view_transform_matrix)
                n1 = transform_normal(n1, GL.view_transform_matrix)
                n2 = transform_normal(n2, GL.view_transform_matrix)

                face_normal = None
                vertex_normals = [n0, n1, n2]

            else:
                p0 = Point(
                    view_points[0][0], view_points[1][0], None, view_points[2][0]
                )
                p1 = Point(
                    view_points[0][1], view_points[1][1], None, view_points[2][1]
                )
                p2 = Point(
                    view_points[0][2], view_points[1][2], None, view_points[2][2]
                )
                face_normal = Triangle(p0, p1, p2).get_normal()
                vertex_normals = None

            # Apply Perspective
            ndc = np.matmul(GL.perspective_matrix, view_points)
            ndc = ndc / ndc[3]
            z_ndc = ndc[2]

            # Transform to screen space
            screen_points = np.matmul(GL.screen_transform, ndc)
            screen_points = screen_points / screen_points[3]

            points = []
            for j in range(0, 3):
                points.append(screen_points[0][j])  # x
                points.append(screen_points[1][j])  # y
                points.append(z_camera[j])  # z (Colors)
                points.append(z_ndc[j])  # z (Z Buffer)

            GL.triangleSet2D(
                points,
                colors,
                textures,
                three_d=True,
                normal=face_normal,
                normals=vertex_normals,
            )

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # Instantiate the Transform class
        perspective_transform = Transform()
        view_transform = Transform()

        # Apply perspective transformation
        # The perspective matrix handles how objects are projected onto the screen

        f = 1.0 / np.tan(fieldOfView / 2)
        perspective_transform.apply_perspective(
            f, GL.far, GL.near, GL.width / GL.height
        )
        GL.perspective_matrix = perspective_transform.get_transformation_matrix()

        # #print(f'Perspec: {transform.get_transformation_matrix()}')

        # Apply rotation transformation
        # The camera's orientation is converted from the axis-angle form into a rotation matrix.
        # This matrix is responsible for rotating the scene according to the camera's orientation.
        view_transform.apply_rotation(orientation, inverse=True)

        # Apply translation transformation
        # The camera’s position is used to create a translation matrix that moves the entire scene
        # in the opposite direction of the camera's position.
        tx, ty, tz = position
        translation = (-tx, -ty, -tz)
        view_transform.apply_translation(translation)
        GL.view_matrix = view_transform.get_transformation_matrix()

        # print(GL.perspective_matrix, GL.perspective_matrix @ GL.view_matrix)

        # print("Viewpoint : ", end="")
        # print("position = {0} ".format(position), end="")
        # print("orientation = {0} ".format(orientation), end="")
        # print("fieldOfView = {0} ".format(fieldOfView))
        # print(f"perspective matrix: {GL.perspective_matrix}")

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
        # print("Transform : ", end="")

        GL.view_transform_matrix = None

        # Instantiate the Transform class
        transform = Transform()

        # Apply the transformations
        if translation:
            # print("translation = {0} ".format(translation), end="")
            transform.apply_translation(translation)

        if rotation:
            # print("rotation = {0} ".format(rotation), end="")
            transform.apply_rotation(rotation)

        if scale:
            # print("scale = {0} ".format(scale), end="")
            transform.apply_scale(scale)

        # Get the final transformation matrix and append it to the stack
        transformation_matrix = transform.get_transformation_matrix()
        # print(f"\ntransformation: {transformation_matrix}")

        if len(GL.transformation_stack) > 0:
            transformation_matrix = np.matmul(
                GL.transformation_stack[-1], transformation_matrix
            )

        GL.transformation_stack.append(transformation_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Saindo de Transform")
        GL.transformation_stack.pop()
        GL.view_transform_matrix = None

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
        # print("TriangleStripSet : pontos = {0} ".format(point), end="")
        # for i, strip in enumerate(stripCount):
        # print("strip[{0}] = {1} ".format(i, strip), end="")
        # print("")
        # print(
        #     "TriangleStripSet : colors = {0}".format(colors)
        # )  # imprime no terminal as cores

        vertex_index = 0
        for strip in stripCount:
            for i in range(strip - 2):
                x0, y0, z0 = (
                    point[vertex_index * 3],
                    point[vertex_index * 3 + 1],
                    point[vertex_index * 3 + 2],
                )
                x1, y1, z1 = (
                    point[(vertex_index + 1) * 3],
                    point[(vertex_index + 1) * 3 + 1],
                    point[(vertex_index + 1) * 3 + 2],
                )
                x2, y2, z2 = (
                    point[(vertex_index + 2) * 3],
                    point[(vertex_index + 2) * 3 + 1],
                    point[(vertex_index + 2) * 3 + 2],
                )

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
        # print(f'IndexedTriangleStripSet : pontos = {point}, index = {index}')
        # print(f'IndexedTriangleStripSet : colors = {colors}')

        all_points = []
        for i in range(0, len(point) - 2, 3):
            x = point[i]
            y = point[i + 1]
            z = point[i + 2]
            all_points.append((x, y, z))

        i = 0
        while i < len(index) - 2:
            if index[i + 2] == -1:
                i += 3
            else:
                p0 = all_points[index[i]]
                p1 = all_points[index[i + 1]]
                p2 = all_points[index[i + 2]]

                if i % 2 == 0:
                    points = [
                        p0[0],
                        p0[1],
                        p0[2],
                        p1[0],
                        p1[1],
                        p1[2],
                        p2[0],
                        p2[1],
                        p2[2],
                    ]
                else:
                    points = [
                        p0[0],
                        p0[1],
                        p0[2],
                        p2[0],
                        p2[1],
                        p2[2],
                        p1[0],
                        p1[1],
                        p1[2],
                    ]

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

        # #print("IndexedFaceSet: ")
        textures = None
        # if coord: #print(f'Pontos: {coord}, coordIndex: {coordIndex}')
        # if colorPerVertex and color and colorIndex: #print(f'Cores: {color}, colorIndex: {colorIndex}')
        # if texCoord and texCoordIndex: #print(f'Texturas: {texCoord}, texCoordIndex: {texCoordIndex}')]

        if current_texture:
            images = []
            images.append(
                gpu.GPU.load_texture(current_texture[0])
            )  # Load the original texture

            while len(images[-1]) > 1:  # Continue creating mipmap levels
                new_image = downsample_matrix_with_channels(
                    images[-1], 2
                )  # Downsample by a factor of 2
                images.append(new_image)

            textures = {"images": images}

        # #print(f'IndexedFaceSet : colors = {colors}')

        all_points, all_colors, all_uvs = [], [], []

        for i in range(0, len(coord) - 2, 3):
            x, y, z = coord[i], coord[i + 1], coord[i + 2]
            all_points.append((x, y, z))

            if colorPerVertex and color and colorIndex:
                c0, c1, c2 = color[i], color[i + 1], color[i + 2]
                all_colors.append((c0, c1, c2))

        if texCoord:
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

            p0, p1, p2 = (
                all_points[origin_point],
                all_points[coordIndex[i]],
                all_points[coordIndex[i + 1]],
            )
            points = [p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]]

            if colorPerVertex and color and colorIndex:
                c0, c1, c2 = (
                    all_colors[origin_point],
                    all_colors[coordIndex[i]],
                    all_colors[coordIndex[i + 1]],
                )
                colors["color_per_vertex"] = [c0, c1, c2]

            if texCoord:
                uv0, uv1, uv2 = (
                    all_uvs[origin_point],
                    all_uvs[coordIndex[i]],
                    all_uvs[coordIndex[i + 1]],
                )
                textures["uvs"] = [uv0, uv1, uv2]

            GL.triangleSet(points, colors, textures)
            i += 1

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box

        cube = Cube(size[0])
        vertices, faces = cube.get_primitive()
        for idx, face in enumerate(faces):
            i1, i2, i3 = face[0], face[1], face[2]
            v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
            points = list(v1) + list(v2) + list(v3)
            GL.triangleSet(points, colors, primitive=cube, face_idx=idx)

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere

        # print(f"Sphere: {radius}")
        sphere = Sphere(radius)
        vertices, faces = sphere.get_primitive()
        for idx, face in enumerate(faces):
            i1, i2, i3 = face[0], face[1], face[2]
            v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
            points = list(v1) + list(v2) + list(v3)
            GL.triangleSet(points, colors, primitive=sphere, face_idx=idx)

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone

        cone = Cone(bottomRadius, height)
        vertices, faces = cone.get_primitive()

        for idx, face in enumerate(faces):
            i1, i2, i3 = face[0], face[1], face[2]
            v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
            points = list(v1) + list(v2) + list(v3)
            GL.triangleSet(points, colors, primitive=cone, face_idx=idx)

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder

        cylinder = Cylinder(radius, height)
        vertices, faces = cylinder.get_primitive()

        for idx, face in enumerate(faces):
            i1, i2, i3 = face[0], face[1], face[2]
            v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
            points = list(v1) + list(v2) + list(v3)
            GL.triangleSet(points, colors, primitive=cylinder, face_idx=idx)

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

        if headlight:
            GL.directionalLight(0.0, (1, 1, 1), 1, (0, 0, -1))

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
        # print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        # print("DirectionalLight : color = {0}".format(color))  # imprime no terminal
        # print(
        #     "DirectionalLight : intensity = {0}".format(intensity)
        # )  # imprime no terminal
        # print(
        #     "DirectionalLight : direction = {0}".format(direction)
        # )  # imprime no terminal

        GL.directional_light = DirectionalLight(
            ambientIntensity, color, intensity, direction
        )

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
        # print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        # print("PointLight : color = {0}".format(color))  # imprime no terminal
        # print("PointLight : intensity = {0}".format(intensity))  # imprime no terminal
        # print("PointLight : location = {0}".format(location))  # imprime no terminal

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
        # print("Fog : color = {0}".format(color))  # imprime no terminal
        # print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """
        Gera eventos conforme o tempo passa, com controle de velocidade.

        Args:
            cycleInterval (float): Duração de um ciclo completo em segundos
            loop (bool): Se a animação deve repetir
        """
        # Get current time
        epoch = time.time()

        # Add a speed factor to make the animation slower
        speed_factor = 1  # Adjust this value to control speed (smaller = slower)

        # Calculate fraction with speed adjustment
        scaled_time = epoch * speed_factor
        fraction_changed = (scaled_time % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """
        Optimized Hermite/Catmull-Rom spline interpolation.
        """
        try:
            # Find segment index using binary search instead of linear search
            segment_index = min(
                len(key) - 2,
                max(
                    0,
                    next(
                        (i for i, k in enumerate(key) if k > set_fraction), len(key) - 1
                    )
                    - 1,
                ),
            )

            # Calculate local parameter s
            t_i = key[segment_index]
            t_i1 = key[segment_index + 1]
            s = (set_fraction - t_i) / (t_i1 - t_i) if t_i1 != t_i else 0.0

            # Pre-compute s powers (faster than numpy array creation)
            s2 = s * s
            s3 = s2 * s

            # Pre-compute basis functions (avoiding matrix multiplication)
            h00 = 2 * s3 - 3 * s2 + 1
            h10 = s3 - 2 * s2 + s
            h01 = -2 * s3 + 3 * s2
            h11 = s3 - s2

            # Get points for current segment
            i = segment_index * 3
            p0x, p0y, p0z = keyValue[i], keyValue[i + 1], keyValue[i + 2]
            p1x, p1y, p1z = keyValue[i + 3], keyValue[i + 4], keyValue[i + 5]

            # Calculate tangents for current segment
            if segment_index == 0:
                if closed:
                    i_prev = len(keyValue) - 3
                    m0x = (p1x - keyValue[i_prev]) * 0.5
                    m0y = (p1y - keyValue[i_prev + 1]) * 0.5
                    m0z = (p1z - keyValue[i_prev + 2]) * 0.5
                else:
                    m0x = m0y = m0z = 0.0
            else:
                m0x = (p1x - keyValue[i - 3]) * 0.5
                m0y = (p1y - keyValue[i - 2]) * 0.5
                m0z = (p1z - keyValue[i - 1]) * 0.5

            if segment_index >= len(keyValue) // 3 - 2:
                if closed:
                    m1x = (keyValue[0] - p0x) * 0.5
                    m1y = (keyValue[1] - p0y) * 0.5
                    m1z = (keyValue[2] - p0z) * 0.5
                else:
                    m1x = m1y = m1z = 0.0
            else:
                m1x = (keyValue[i + 6] - p0x) * 0.5
                m1y = (keyValue[i + 7] - p0y) * 0.5
                m1z = (keyValue[i + 8] - p0z) * 0.5

            # Compute interpolated position (faster than matrix multiplication)
            x = h00 * p0x + h10 * m0x + h01 * p1x + h11 * m1x
            y = h00 * p0y + h10 * m0y + h01 * p1y + h11 * m1y
            z = h00 * p0z + h10 * m0z + h01 * p1z + h11 * m1z

            return [x, y, z]

        except Exception as e:
            return [0.0, 0.0, 0.0]

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpolates between a list of specific rotation values."""
        # Edge cases
        if set_fraction <= key[0]:
            rotation = keyValue[0:4]
            return rotation
        elif set_fraction >= key[-1]:
            rotation = keyValue[-4:]
            return rotation
        else:
            # Find index i such that key[i] <= set_fraction <= key[i+1]
            for i in range(len(key) - 1):
                if key[i] <= set_fraction <= key[i + 1]:
                    break

            t = (
                (set_fraction - key[i]) / (key[i + 1] - key[i])
                if key[i + 1] != key[i]
                else 0.0
            )

            # Get rotations at keyValue[i] and keyValue[i+1]
            idx0 = 4 * i
            idx1 = 4 * (i + 1)
            rotation0 = keyValue[idx0 : idx0 + 4]
            rotation1 = keyValue[idx1 : idx1 + 4]

            axis0 = rotation0[:3]
            angle0 = rotation0[3]
            axis1 = rotation1[:3]
            angle1 = rotation1[3]

            # Convert axis-angle to quaternions
            q0 = GL.axis_angle_to_quaternion(axis0, angle0)
            q1 = GL.axis_angle_to_quaternion(axis1, angle1)

            # Perform slerp
            q = GL.slerp(q0, q1, t)

            # Convert back to axis-angle
            axis, angle = GL.quaternion_to_axis_angle(q)

            value_changed = list(axis) + [angle]

            return value_changed

    @staticmethod
    def axis_angle_to_quaternion(axis, angle):
        axis = np.array(axis, dtype=np.float64)
        axis_length = np.linalg.norm(axis)
        if axis_length == 0:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / axis_length
        half_angle = angle / 2.0
        s = np.sin(half_angle)
        w = np.cos(half_angle)
        x, y, z = axis * s
        return np.array([w, x, y, z], dtype=np.float64)

    @staticmethod
    def quaternion_to_axis_angle(q):
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        angle = 2 * np.arccos(w)
        s = np.sqrt(1 - w * w)
        if s < 1e-8:
            # If s is close to zero, axis direction is not important
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([x, y, z]) / s
        return axis.tolist(), angle

    @staticmethod
    def slerp(q0, q1, t):
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        DOT_THRESHOLD = 0.9995
        if dot > DOT_THRESHOLD:
            # Quaternions are close; use linear interpolation
            result = q0 + t * (q1 - q0)
            result = result / np.linalg.norm(result)
            return result
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta = theta_0 * t
            sin_theta = np.sin(theta)

            s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0

            result = (s0 * q0) + (s1 * q1)
            result = result / np.linalg.norm(result)
            return result

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""

    @staticmethod
    def obj(model, colors):

        for face in model.faces:
            points = []
            for vertex_index in face:
                points.extend(model.vertices[vertex_index])
            GL.triangleSet(points, colors)

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
