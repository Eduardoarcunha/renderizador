#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Renderizador X3D.

Desenvolvido por: Luciano Soares <lpsoares@insper.edu.br>
Disciplina: Computação Gráfica
Data: 28 de Agosto de 2020
"""

import numpy as np

import os           # Para rotinas do sistema operacional
import argparse     # Para tratar os parâmetros da linha de comando

import gl           # Recupera rotinas de suporte ao X3D

import interface    # Janela de visualização baseada no Matplotlib
import gpu          # Simula os recursos de uma GPU

import x3d          # Faz a leitura do arquivo X3D, gera o grafo de cena e faz traversal
import scenegraph   # Imprime o grafo de cena no console

LARGURA = 60  # Valor padrão para largura da tela
ALTURA = 40   # Valor padrão para altura da tela


class Renderizador:
    """Realiza a renderização da cena informada."""

    def __init__(self, supersampling_factor=4):
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
        fbo = gpu.GPU.gen_framebuffers(2)
        print(f'Framebuffers: {fbo}')

        # Define o atributo FRONT como o FrameBuffe principal
        self.framebuffers["FRONT"] = fbo[0]
        self.framebuffers["SUPERSAMPLING"] = fbo[1]

        # Define que a posição criada será usada para desenho e leitura
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["SUPERSAMPLING"])
        # Opções:
        # - DRAW_FRAMEBUFFER: Faz o bind só para escrever no framebuffer
        # - READ_FRAMEBUFFER: Faz o bind só para leitura no framebuffer
        # - FRAMEBUFFER: Faz o bind para leitura e escrita no framebuffer

        # Aloca memória no FrameBuffer para um tipo e tamanho especificado de buffer

        # Memória de Framebuffer para canal de cores
        gpu.GPU.framebuffer_storage(
            self.framebuffers["FRONT"],
            gpu.GPU.COLOR_ATTACHMENT,
            gpu.GPU.RGB8,
            self.width,
            self.height
        )

        gpu.GPU.framebuffer_storage(
            self.framebuffers["SUPERSAMPLING"],
            gpu.GPU.COLOR_ATTACHMENT,
            gpu.GPU.RGB8,
            self.width * self.supersampling_factor,
            self.height * self.supersampling_factor
        )

        # Descomente as seguintes linhas se for usar um Framebuffer para profundidade
        gpu.GPU.framebuffer_storage(
            self.framebuffers["SUPERSAMPLING"],
            gpu.GPU.DEPTH_ATTACHMENT,
            gpu.GPU.DEPTH_COMPONENT32F,
            self.width * self.supersampling_factor,
            self.height * self.supersampling_factor
        )
    
        # Opções:
        # - COLOR_ATTACHMENT: alocações para as cores da imagem renderizada
        # - DEPTH_ATTACHMENT: alocações para as profundidades da imagem renderizada
        # Obs: Você pode chamar duas vezes a rotina com cada tipo de buffer.

        # Tipos de dados:
        # - RGB8: Para canais de cores (Vermelho, Verde, Azul) 8bits cada (0-255)
        # - RGBA8: Para canais de cores (Vermelho, Verde, Azul, Transparência) 8bits cada (0-255)
        # - DEPTH_COMPONENT16: Para canal de Profundidade de 16bits (half-precision) (0-65535)
        # - DEPTH_COMPONENT32F: Para canal de Profundidade de 32bits (single-precision) (float)

        # Define cor que ira apagar o FrameBuffer quando clear_buffer() invocado
        gpu.GPU.clear_color([0, 0, 0])

        # Define a profundidade que ira apagar o FrameBuffer quando clear_buffer() invocado
        # Assuma 1.0 o mais afastado e -1.0 o mais próximo da camera
        gpu.GPU.clear_depth(1.0)

        # Definindo tamanho do Viewport para renderização
        self.scene.viewport(width=self.width, height=self.height)

    def pre(self):
        print(f'Pre renderização: {self.width}x{self.height}')
        """Rotinas pré renderização."""
        # Função invocada antes do processo de renderização iniciar.

        # Limpa o frame buffers atual
        gpu.GPU.clear_buffer()

        # Recursos que podem ser úteis:
        # Define o valor do pixel no framebuffer: draw_pixel(coord, mode, data)
        # Retorna o valor do pixel no framebuffer: read_pixel(coord, mode)

    def pos(self):
        print('Pós renderização')
    
        """Rotinas pós renderização."""
        # Função invocada após o processo de renderização terminar.

        supersampled_framebuffer = gpu.GPU.get_frame_buffer() # Get no super sample
        # with open("supersampled_framebuffer.txt", "w") as file:
        #     for y in range(self.height * self.supersampling_factor):
        #         for x in range(self.width * self.supersampling_factor):
        #             file.write(f'{supersampled_framebuffer[y][x]}\n')

        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["FRONT"])
        gpu.GPU.clear_buffer()
        self.downsample(supersampled_framebuffer)
        
        normal_framebuffer = gpu.GPU.get_frame_buffer()
        # with open("normal_framebuffer.txt", "w") as file:
        #     for y in range(self.height):
        #         for x in range(self.width):
        #             file.write(f'{normal_framebuffer[y][x]}\n')


        # Essa é uma chamada conveniente para manipulação de buffers
        # ao final da renderização de um frame. Como por exemplo, executar
        # downscaling da imagem.

        # Método para a troca dos buffers (NÃO IMPLEMENTADO)
        # Esse método será utilizado na fase de implementação de animações
        gpu.GPU.swap_buffers()


    def downsample(self, supersampled_framebuffer):
        """Realiza o downsample do framebuffer."""
        print(f'Framebuffers shape: {supersampled_framebuffer.shape}')
        print(f'Downsample: {self.width}x{self.height}')
        for y in range(self.height):
            for x in range(self.width):
                # Calcula a média dos pixels vizinhos usando um tipo de dado maior
                r = 0
                g = 0
                b = 0
                
                # Temporarily use a larger data type for accumulation
                r_acc = np.int32(0)
                g_acc = np.int32(0)
                b_acc = np.int32(0)
                
                for i in range(self.supersampling_factor):
                    for j in range(self.supersampling_factor):
                        r_acc += np.int32(supersampled_framebuffer[y * self.supersampling_factor + i][x * self.supersampling_factor + j][0])
                        g_acc += np.int32(supersampled_framebuffer[y * self.supersampling_factor + i][x * self.supersampling_factor + j][1])
                        b_acc += np.int32(supersampled_framebuffer[y * self.supersampling_factor + i][x * self.supersampling_factor + j][2])
                
                # Perform averaging and cast back to the original data type
                r = int(r_acc / (self.supersampling_factor ** 2))
                g = int(g_acc / (self.supersampling_factor ** 2))
                b = int(b_acc / (self.supersampling_factor ** 2))
                
                # Ensure values stay within [0, 255]
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                # Define o pixel no framebuffer
                gpu.GPU.draw_pixel((x, y), gpu.GPU.RGB8, [r, g, b])

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
        x3d.X3D.renderer["NavigationInfo"] = gl.GL.navigationInfo
        x3d.X3D.renderer["DirectionalLight"] = gl.GL.directionalLight
        x3d.X3D.renderer["PointLight"] = gl.GL.pointLight
        x3d.X3D.renderer["Fog"] = gl.GL.fog
        x3d.X3D.renderer["TimeSensor"] = gl.GL.timeSensor
        x3d.X3D.renderer["SplinePositionInterpolator"] = gl.GL.splinePositionInterpolator
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
        parser = argparse.ArgumentParser(add_help=False)   # parser para linha de comando
        parser.add_argument("-i", "--input", help="arquivo X3D de entrada")
        parser.add_argument("-o", "--output", help="arquivo 2D de saída (imagem)")
        parser.add_argument("-w", "--width", help="resolução horizonta", type=int)
        parser.add_argument("-h", "--height", help="resolução vertical", type=int)
        parser.add_argument("-g", "--graph", help="imprime o grafo de cena", action='store_true')
        parser.add_argument("-p", "--pause", help="começa simulação em pausa", action='store_true')
        parser.add_argument("-q", "--quiet", help="não exibe janela", action='store_true')
        args = parser.parse_args() # parse the arguments
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
            far=1000
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

if __name__ == '__main__':
    renderizador = Renderizador()
    renderizador.main()
