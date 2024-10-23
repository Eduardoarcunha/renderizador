#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Renderizador X3D.

Desenvolvido por: Luciano Soares <lpsoares@insper.edu.br>
Disciplina: Computação Gráfica
Data: 28 de Agosto de 2020
"""

import numpy as np

import os  # Para rotinas do sistema operacional
import argparse  # Para tratar os parâmetros da linha de comando

import gl  # Recupera rotinas de suporte ao X3D

import interface  # Janela de visualização baseada no Matplotlib
import gpu  # Simula os recursos de uma GPU

import x3d  # Faz a leitura do arquivo X3D, gera o grafo de cena e faz traversal
import scenegraph  # Imprime o grafo de cena no console

LARGURA = 60  # Valor padrão para largura da tela
ALTURA = 40  # Valor padrão para altura da tela


class Renderizador:
    """Realiza a renderização da cena informada."""

    def __init__(self, supersampling_factor=2):
        """Definindo valores padrão."""
        self.supersampling_factor = supersampling_factor
        self.width = LARGURA
        self.height = ALTURA

        self.x3d_file = ""
        self.image_file = "tela.png"
        self.scene = None
        self.framebuffers = {}

    def setup(self):
        """Configura o sistema para a renderização."""
        # Configurando color buffers para exibição na tela

        # Cria uma (1) posição de FrameBuffer na GPU
        fbo = gpu.GPU.gen_framebuffers(3)
        print(f"Framebuffers: {fbo}")

        # Define o atributo FRONT como o FrameBuffe principal
        self.framebuffers["FRONT"] = fbo[0]
        self.framebuffers["BACK"] = fbo[1]
        self.framebuffers["SUPERSAMPLING"] = fbo[2]

        gpu.GPU.bind_framebuffer(
            gpu.GPU.FRAMEBUFFER, self.framebuffers["SUPERSAMPLING"]
        )

        for buffer in ["FRONT", "BACK"]:
            gpu.GPU.framebuffer_storage(
                self.framebuffers[buffer],
                gpu.GPU.COLOR_ATTACHMENT,
                gpu.GPU.RGB8,
                self.width,
                self.height,
            )

        gpu.GPU.framebuffer_storage(
            self.framebuffers["SUPERSAMPLING"],
            gpu.GPU.COLOR_ATTACHMENT,
            gpu.GPU.RGB8,
            self.width * self.supersampling_factor,
            self.height * self.supersampling_factor,
        )

        gpu.GPU.framebuffer_storage(
            self.framebuffers["SUPERSAMPLING"],
            gpu.GPU.DEPTH_ATTACHMENT,
            gpu.GPU.DEPTH_COMPONENT32F,
            self.width * self.supersampling_factor,
            self.height * self.supersampling_factor,
        )

        # Initially bind to the supersampling buffer
        gpu.GPU.bind_framebuffer(
            gpu.GPU.FRAMEBUFFER, self.framebuffers["SUPERSAMPLING"]
        )

        # Define cor que ira apagar o FrameBuffer quando clear_buffer() invocado
        gpu.GPU.clear_color([0, 0, 0])

        gpu.GPU.clear_depth(1.0)

        # Definindo tamanho do Viewport para renderização
        self.scene.viewport(width=self.width, height=self.height)

    def pre(self):
        print(f"Pre renderização: {self.width}x{self.height}")
        gpu.GPU.bind_framebuffer(
            gpu.GPU.FRAMEBUFFER, self.framebuffers["SUPERSAMPLING"]
        )
        """Rotinas pré renderização."""
        # Função invocada antes do processo de renderização iniciar.

        # Limpa o frame buffers atual
        gpu.GPU.clear_buffer()

        # Recursos que podem ser úteis:
        # Define o valor do pixel no framebuffer: draw_pixel(coord, mode, data)
        # Retorna o valor do pixel no framebuffer: read_pixel(coord, mode)

    def pos(self):
        """Rotinas pós renderização."""
        # Get the supersampled buffer
        supersampled_framebuffer = gpu.GPU.get_frame_buffer()

        # Switch to the back buffer for downsampling
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["BACK"])
        gpu.GPU.clear_buffer()
        self.downsample(supersampled_framebuffer)

        # Swap the buffers to display the new frame
        gpu.GPU.swap_buffers()

        # Prepare supersampling buffer for next frame
        # gpu.GPU.bind_framebuffer(
        #     gpu.GPU.FRAMEBUFFER, self.framebuffers["SUPERSAMPLING"]
        # )

    def downsample(self, supersampled_framebuffer):
        """Performs downsampling of the framebuffer using vectorized operations."""
        # print(f"Framebuffers shape: {supersampled_framebuffer.shape}")
        # print(f"Downsample: {self.width}x{self.height}")

        ssf = self.supersampling_factor
        height_ss, width_ss = supersampled_framebuffer.shape[:2]

        # Ensure the framebuffer has the expected shape
        assert height_ss == self.height * ssf
        assert width_ss == self.width * ssf

        # Reshape and compute the mean over the supersampling dimensions
        framebuffer = supersampled_framebuffer.reshape(
            self.height, ssf, self.width, ssf, 3
        ).mean(axis=(1, 3))

        # Convert to integers within the valid range [0, 255]
        framebuffer = np.clip(framebuffer, 0, 255).astype(np.uint8)

        # Draw the downsampled pixels to the GPU framebuffer
        for y in range(self.height):
            for x in range(self.width):
                r, g, b = framebuffer[y, x]
                gpu.GPU.draw_pixel((x, y), gpu.GPU.RGB8, [int(r), int(g), int(b)])

    def mapping(self):
        """Mapeamento de funções para as rotinas de renderização."""
        # Rotinas encapsuladas na classe GL (Graphics Library)
        x3d.X3D.renderer["Polypoint2D"] = gl.GL.polypoint2D
        x3d.X3D.renderer["Polyline2D"] = gl.GL.polyline2D
        x3d.X3D.renderer["Circle2D"] = gl.GL.circle2D
        x3d.X3D.renderer["TriangleSet2D"] = gl.GL.triangleSet2D
        x3d.X3D.renderer["TriangleSet"] = gl.GL.triangleSet
        x3d.X3D.renderer["Viewpoint"] = gl.GL.viewpoint
        x3d.X3D.renderer["Transform_in"] = gl.GL.transform_in
        x3d.X3D.renderer["Transform_out"] = gl.GL.transform_out
        x3d.X3D.renderer["TriangleStripSet"] = gl.GL.triangleStripSet
        x3d.X3D.renderer["IndexedTriangleStripSet"] = gl.GL.indexedTriangleStripSet
        x3d.X3D.renderer["IndexedFaceSet"] = gl.GL.indexedFaceSet
        x3d.X3D.renderer["Box"] = gl.GL.box
        x3d.X3D.renderer["Sphere"] = gl.GL.sphere
        x3d.X3D.renderer["Cone"] = gl.GL.cone
        x3d.X3D.renderer["Cylinder"] = gl.GL.cylinder
        x3d.X3D.renderer["OBJ"] = gl.GL.obj
        x3d.X3D.renderer["NavigationInfo"] = gl.GL.navigationInfo
        x3d.X3D.renderer["DirectionalLight"] = gl.GL.directionalLight
        x3d.X3D.renderer["PointLight"] = gl.GL.pointLight
        x3d.X3D.renderer["Fog"] = gl.GL.fog
        x3d.X3D.renderer["TimeSensor"] = gl.GL.timeSensor
        x3d.X3D.renderer["SplinePositionInterpolator"] = (
            gl.GL.splinePositionInterpolator
        )
        x3d.X3D.renderer["OrientationInterpolator"] = gl.GL.orientationInterpolator

    def render(self):
        """Laço principal de renderização."""
        self.pre()  # executa rotina pré renderização
        self.scene.render()  # faz o traversal no grafo de cena
        self.pos()  # executa rotina pós renderização
        return gpu.GPU.get_frame_buffer()

    def main(self):
        """Executa a renderização."""
        # Tratando entrada de parâmetro
        parser = argparse.ArgumentParser(add_help=False)  # parser para linha de comando
        parser.add_argument("-i", "--input", help="arquivo X3D de entrada")
        parser.add_argument("-o", "--output", help="arquivo 2D de saída (imagem)")
        parser.add_argument("-w", "--width", help="resolução horizonta", type=int)
        parser.add_argument("-h", "--height", help="resolução vertical", type=int)
        parser.add_argument(
            "-g", "--graph", help="imprime o grafo de cena", action="store_true"
        )
        parser.add_argument(
            "-p", "--pause", help="começa simulação em pausa", action="store_true"
        )
        parser.add_argument(
            "-q", "--quiet", help="não exibe janela", action="store_true"
        )
        args = parser.parse_args()  # parse the arguments
        if args.input:
            self.x3d_file = args.input
        if args.output:
            self.image_file = args.output
        if args.width:
            self.width = args.width
        if args.height:
            self.height = args.height

        path = os.path.dirname(os.path.abspath(self.x3d_file))

        # Iniciando simulação de GPU
        gpu.GPU(self.image_file, path)

        # Abre arquivo X3D
        self.scene = x3d.X3D(self.x3d_file)

        # Iniciando Biblioteca Gráfica
        gl.GL.setup(
            self.width * self.supersampling_factor,
            self.height * self.supersampling_factor,
            self.supersampling_factor,
            near=0.01,
            far=1000,
        )

        # Funções que irão fazer o rendering
        self.mapping()

        # Se no modo silencioso não configurar janela de visualização
        if not args.quiet:
            window = interface.Interface(self.width, self.height, self.x3d_file)
            self.scene.set_preview(window)

        # carrega os dados do grafo de cena
        if self.scene:
            self.scene.parse()
            if args.graph:
                scenegraph.Graph(self.scene.root)

        # Configura o sistema para a renderização.
        self.setup()

        # Se no modo silencioso salvar imagem e não mostrar janela de visualização
        if args.quiet:
            gpu.GPU.save_image()  # Salva imagem em arquivo
        else:
            window.set_saver(gpu.GPU.save_image)  # pasa a função para salvar imagens
            window.preview(args.pause, self.render)  # mostra visualização


if __name__ == "__main__":
    renderizador = Renderizador()
    renderizador.main()
