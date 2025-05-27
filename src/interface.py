
import pygame
import torch
import sys
import os
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy.spatial import Delaunay

from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import matplotlib.cm as cm

class Interface:
    def __init__(self, input_parameters, ic_data, title='GNN Based Flows'):
        self.input_parameters = input_parameters
        self.title = title
        self.cmap = cm.viridis
        pygame.init()

        self.width = input_parameters['screen_parameters']['width']
        self.height = input_parameters['screen_parameters']['height']

        self.init_screen()
        self.init_buffers()
        self.compile_shader()

        self.base_line_width = input_parameters['screen_parameters']['line_width']
        self.base_point_radius = input_parameters['screen_parameters']['point_radius']
        self.selection_radius = 0.02
        self.mode = "points"
        self.dirty = True

        # undo redo
        self.undo_stack = []
        self.redo_stack = []

        # save and load interface
        self.input_mode = None
        self.input_text = ""

        # mouse interactions
        self.selected_point = None
        self.hovered_point = None
        self.moving_point = False

        # screen and update loop
        self.mesh = False
        self.rerender = True
        self.update_state = False
        self.graph_change = False
        self.running = True
        self.node_update = False

        # self.points_color = list(mcolors.TABLEAU_COLORS)[:n.shape[1]-1]
        self.points_color = list(mcolors.TABLEAU_COLORS)[:3]

        # screen parameters
        self.center = torch.zeros(2)
        self.move_vrt = 0
        self.move_hrz = 0
        self.zoom = 1

        # field parameters
        self.pos = ic_data.q_0
        self.field = ic_data.x
        self.n = ic_data.n
        self.edges = ic_data.edge_index

        # crete triangles for contour plotting
        points_np = self.pos.numpy()
        self.triangles, _, self.tri = self.build_triangle_edges(points_np)

        # dislay fields
        self.field_idx = 0
        self.field_name = 'X velocity'
        self.model_type = input_parameters['model']['model_type']

    def update_parameters(self,field,E=None,S=None,n=None,edges=None):
        self.field = field
        if not(n is None):
            self.n = n
        if not(edges is None):
            self.edges = edges
        if not(E is None):
            self.E = E
        if not(S is None):
            self.S = S

    def update_mesh(self,pos,update_edges=False):
        self.pos = pos
        if update_edges:
            self.triangles, self.edges, self.tri = self.build_triangle_edges(self.pos.numpy())

    def update_vbos(self):
        # Upload points
        render_pos = self.pos2render(self.pos).numpy().astype(np.float32)

        if self.field_idx < 3:
            values_np = self.field[:, self.field_idx]
        elif self.field_idx == 3:
            values_np = self.E.squeeze(-1)
        elif self.field_idx == 4:
            values_np = self.S.squeeze(-1)
            
        vmin, vmax = values_np.min(), values_np.max()

        vmean = values_np.mean()
        vstd = values_np.std()
        vmin, vmax = vmean-2*vstd,vmean+2*vstd

        if self.mode == 'points':
            # if self.dirty:
            glBindBuffer(GL_ARRAY_BUFFER, self.points_vbo)
            glBufferData(GL_ARRAY_BUFFER, render_pos, GL_DYNAMIC_DRAW)

            colors = np.zeros((len(render_pos),3),dtype=np.float32)
            for i in range(len(self.points_color)):
                color = mcolors.to_rgb(self.points_color[i])
                colors[self.n[:,i+1] == 1] = color
            
            glBindBuffer(GL_ARRAY_BUFFER, self.points_colors_vbo)
            glBufferData(GL_ARRAY_BUFFER, colors, GL_DYNAMIC_DRAW)   

            # Upload edges if needed
            if self.edges.shape[1] > 0:
                edges_pos = np.concatenate([
                    render_pos[self.edges[0,:self.edges.shape[1]].numpy()],
                    render_pos[self.edges[1,:self.edges.shape[1]].numpy()]
                ], axis=1).astype(np.float32).reshape(-1, 2)

                vnorm_src = self.normalize(values_np[self.edges[0]],vmin,vmax)
                vnorm_dst = self.normalize(values_np[self.edges[1]],vmin,vmax)
                vnorm = np.empty(len(vnorm_src)*2)
                vnorm[::2] = vnorm_src
                vnorm[1::2] = vnorm_dst
                
                colors = self.cmap(vnorm)[:,:3].astype(np.float32)

                glBindBuffer(GL_ARRAY_BUFFER, self.edges_vbo)
                glBufferData(GL_ARRAY_BUFFER, edges_pos, GL_DYNAMIC_DRAW)

                glBindBuffer(GL_ARRAY_BUFFER, self.edge_colors_vbo)
                glBufferData(GL_ARRAY_BUFFER, colors, GL_DYNAMIC_DRAW)


        if self.mode == 'contours':
            if self.mesh:
                triangles = self.filter_skewed_triangles(render_pos, self.tri.simplices, skew_threshold=0.001)
            else:
                # if self.moving_point or not(hasattr(self,'tri')):
                #     tri = Delaunay(self.pos.numpy())
                #     self.tri = tri
                triangles = self.tri.simplices

            mask = np.isin(triangles,np.arange(self.pos.shape[0])[self.n[:,3]==1])
            mask = np.all(mask,axis=1)
            triangles = triangles[~mask]
            n_tri = len(triangles)

            vertices = render_pos[triangles].reshape(n_tri*3,2)
            vals = values_np[triangles]
            norm_val = self.normalize(vals.mean(1),vmin,vmax)
            color = self.cmap(norm_val)[:,:3]
            colors = np.repeat(color,3,axis=0).astype(np.float32)

            # Upload vertex positions
            glBindBuffer(GL_ARRAY_BUFFER, self.triangle_vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)

            # Upload vertex colors
            glBindBuffer(GL_ARRAY_BUFFER, self.triangle_color_vbo)
            glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_DYNAMIC_DRAW)
            self.num_render_triangles = n_tri*3

    def compile_indiv_shader(self,vert_fname,frag_fname,geom_fname=None):
        shader_program = glCreateProgram()

        vert_src =  open(vert_fname).read()
        vert_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vert_shader, vert_src)
        glCompileShader(vert_shader)
        if not glGetShaderiv(vert_shader, GL_COMPILE_STATUS):
            raise Exception(glGetShaderInfoLog(vert_shader))
        glAttachShader(shader_program, vert_shader)
        
        frag_src =  open(frag_fname).read()
        frag_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(frag_shader, frag_src)
        glCompileShader(frag_shader)
        if not glGetShaderiv(frag_shader, GL_COMPILE_STATUS):
            raise Exception(glGetShaderInfoLog(frag_shader))
        glAttachShader(shader_program, frag_shader)
        
        if not(geom_fname is None): 
            geom_src =  open(geom_fname).read()
            geom_shader = glCreateShader(GL_GEOMETRY_SHADER)
            glShaderSource(geom_shader, geom_src)
            glCompileShader(geom_shader)
            if not glGetShaderiv(geom_shader, GL_COMPILE_STATUS):
                raise Exception(glGetShaderInfoLog(geom_shader))
            glAttachShader(shader_program, geom_shader)

        glLinkProgram(shader_program) 
        return shader_program
         

    def compile_shader(self):
        # Compile shaders
        self.shader_program = self.compile_indiv_shader(
            os.path.join(self.input_parameters['shader_dir'],'vertex_shader.glsl'),
            os.path.join(self.input_parameters['shader_dir'],'fragment_shader.glsl'))

        self.line_shader_program = self.compile_indiv_shader(
            os.path.join(self.input_parameters['shader_dir'],'vertex_line.glsl'),
            os.path.join(self.input_parameters['shader_dir'],'frag_line.glsl'), 
            os.path.join(self.input_parameters['shader_dir'],'geometry_line.glsl'))

        self.fxaa_shader = self.compile_indiv_shader(
            os.path.join(self.input_parameters['shader_dir'],'aa_vertex.glsl'),
            os.path.join(self.input_parameters['shader_dir'],'aa_frag.glsl'))

        self.colorbar_shader = self.compile_indiv_shader(
            os.path.join(self.input_parameters['shader_dir'],'colorbar_vert.glsl'),
            os.path.join(self.input_parameters['shader_dir'],'colorbar_frag.glsl'))
        
        self.texture_shader = self.compile_indiv_shader(
            os.path.join(self.input_parameters['shader_dir'],'texture_quad.glsl'),
            os.path.join(self.input_parameters['shader_dir'],'texture_frag.glsl'))
        
        self.contour_shader = self.compile_indiv_shader(
             os.path.join(self.input_parameters['shader_dir'],'contour_vertex.glsl'),
            os.path.join(self.input_parameters['shader_dir'],'contour_frag.glsl'))
        
        self.text_shader = self.compile_indiv_shader( 
            os.path.join(self.input_parameters['shader_dir'],'text_vertex.glsl'),
            os.path.join(self.input_parameters['shader_dir'],'text_frag.glsl'))

    def save_state(self):
        # self.undo_stack.append((self.points.clone(), self.edges.clone()))
        self.undo_stack.append((self.pos.clone(), self.field.clone(), self.n.clone(), self.edges.clone()))
        self.redo_stack.clear()

        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    def normalize(self, val, vmin, vmax):
        return (val - vmin) / (vmax - vmin + 1e-8)
    
    def recreate_fbo(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # Delete previous texture
        glDeleteTextures([self.fbo_texture])

        # Create new texture with new size
        self.fbo_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.fbo_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.fbo_texture, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Framebuffer not complete!")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def init_screen(self):
        self.render_surface = pygame.Surface((self.width, self.height))

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption(self.title)

        glViewport(0, 0, self.width, self.height)

        self.clock = pygame.time.Clock()

        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def init_buffers(self):

        self.create_fbo()
        self.create_nodes()
        self.create_edges()
        self.create_quad()
        self.create_colorbar()
        # self.create_fullscreen_vao()
        self.create_contours()
        self.create_text()



    def create_fbo(self):
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        self.fbo_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.fbo_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.fbo_texture, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Framebuffer is not complete!")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def create_nodes(self):
        self.points_vbo = glGenBuffers(1)
        self.edges_vbo = glGenBuffers(1)

        self.points_vao = glGenVertexArrays(1)
        glBindVertexArray(self.points_vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.points_vbo)
        glEnableVertexAttribArray(0)  # location = 0 in shader
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        self.points_colors_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.points_colors_vbo)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindVertexArray(0)


    def create_edges(self):
        self.edges_vao = glGenVertexArrays(1)
        glBindVertexArray(self.edges_vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.edges_vbo)
        glEnableVertexAttribArray(0)  # location = 0 in shader
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        self.edge_colors_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.edge_colors_vbo)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glBindVertexArray(0)

    def create_quad(self):
        self.quad_vao = glGenVertexArrays(1)
        glBindVertexArray(self.quad_vao)

        quad_vertices = np.array([
            -1.0, -1.0,
            3.0, -1.0,
            -1.0,  3.0
        ], dtype=np.float32)

        self.quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        glBindVertexArray(0)

    def create_colorbar(self):
        colorbar_vertices = np.array([
            [0.8, -0.8],
            [0.9, -0.8],
            [0.9,  0.8],
            [0.8,  0.8],
        ], dtype=np.float32)

        self.colorbar_vao = glGenVertexArrays(1)
        self.colorbar_vbo = glGenBuffers(1)

        glBindVertexArray(self.colorbar_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.colorbar_vbo)
        glBufferData(GL_ARRAY_BUFFER, colorbar_vertices.nbytes, colorbar_vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)



    def create_fullscreen_vao(self):
        self.quad_vertices = np.array([
            # positions   # tex coords
            -1.0,  1.0,   0.0, 1.0,   # top-left
            -1.0, -1.0,   0.0, 0.0,   # bottom-left
            1.0, -1.0,   1.0, 0.0,   # bottom-right
            1.0,  1.0,   1.0, 1.0,   # top-right
        ], dtype=np.float32)

        self.quad_indices = np.array([
            0, 1, 2,
            2, 3, 0,
        ], dtype=np.uint32)

        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)
        self.quad_ebo = glGenBuffers(1)

        glBindVertexArray(self.quad_vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.quad_vertices.nbytes, self.quad_vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.quad_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.quad_indices.nbytes, self.quad_indices, GL_STATIC_DRAW)

        # position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        # texture coord attribute
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))

        glBindVertexArray(0)

    def create_contours(self):
        self.triangle_vao = glGenVertexArrays(1)
        glBindVertexArray(self.triangle_vao)

        # Positions
        self.triangle_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.triangle_vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        # Colors
        self.triangle_color_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.triangle_color_vbo)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindVertexArray(0)

    def create_text(self):
        self.text_vao = glGenVertexArrays(1)
        self.text_vbo = glGenBuffers(1)
        self.text_ebo = glGenBuffers(1)

        self.font = pygame.font.SysFont(None, 24)


    def text_to_texture(self, text, font_size=24, color=(0, 0, 0)):
        font = pygame.font.SysFont('Arial', font_size)
        text_surface = font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        width, height = text_surface.get_size()

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        return texture_id, width, height
    
    def draw_texture(self, texture_id, x, y, w, h):
        # Compute opengl coordinates
        screen_width = self.width
        screen_height = self.height

        left   = (x / screen_width) * 2.0 - 1.0
        right  = ((x + w) / screen_width) * 2.0 - 1.0
        top    = 1.0 - (y / screen_height) * 2.0
        bottom = 1.0 - ((y + h) / screen_height) * 2.0

        vertices = np.array([
            left,  top,    0.0, 1.0,
            left,  bottom, 0.0, 0.0,
            right, bottom, 1.0, 0.0,
            right, top,    1.0, 1.0,
        ], dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

        glUseProgram(self.texture_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glUniform1i(glGetUniformLocation(self.texture_shader, "texture1"), 0)

        glBindVertexArray(self.quad_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glDisable(GL_BLEND)


    def render2pos(self,pos):
        ar = self.width/self.height
        pos_max,_ = self.pos.max(dim=0)
        pos_min,_ = self.pos.min(dim=0)
        position_range = (pos_max-pos_min).max()*0.6

        pos[0] = 2*(pos[0]/self.width) - 1
        pos[1] = -2*(pos[1]/self.height) + 1

        pos[0] += 0.1

        pos /= self.zoom
        pos += self.center

        if ar<1:
            pos[1] /= ar
        else:
            pos[0] *= ar
        

        pos = pos*position_range+self.pos.mean(dim=0)
        self.position_range = position_range
        return pos

    def pos2render(self,pos):
        ar = self.width/self.height
        pos_max,_ = pos.max(dim=0)
        pos_min,_ = pos.min(dim=0)
        position_range = (pos_max-pos_min).max()*0.6
        screen_scaling = min(self.width,self.height)*0.8
        pos = ((pos - pos.mean(dim=0))/position_range)
        self.point_radius = int(self.base_point_radius*screen_scaling/480)
        self.line_width = int(self.base_line_width*screen_scaling/480)

        if ar<1:
            pos[:,1] *= ar
        else:
            pos[:,0] /= ar

        pos -= self.center
        pos *= self.zoom

        pos[:,0] -= 0.1

        return pos

    def point_line_distance(self, p, a, b):
        pa = p - a
        ba = b - a
        t = np.dot(pa, ba) / (np.dot(ba, ba) + 1e-8)
        t = np.clip(t, 0, 1)
        closest = a + t * ba
        return np.linalg.norm(p - closest)
    
    def edges_to_triangles(self):
        adj = {}
        for u, v in self.edges.T:
            if u not in adj:
                adj[u] = set()
            if v not in adj:
                adj[v] = set()
            adj[u].add(v)
            adj[v].add(u)

        triangles = set()
        for u in adj:
            for v in adj[u]:
                for w in adj[v]:
                    if w in adj[u] and u < v < w:
                        triangles.add(tuple(sorted((u, v, w))))
        return list(triangles)


    @staticmethod
    def filter_skewed_triangles(points_np, triangles, skew_threshold=0.1):
        if skew_threshold == 0:
            return triangles
        filtered = []
        for tri in triangles:
            verts = points_np[tri]
            a, b, c = np.linalg.norm(verts[0] - verts[1]), np.linalg.norm(verts[1] - verts[2]), np.linalg.norm(verts[2] - verts[0])
            s = (a + b + c) / 2
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
            if area == 0:
                continue
            inradius = area / s
            circumradius = (a * b * c) / (4 * area)
            if inradius / circumradius > skew_threshold:
                filtered.append(tri)
        return np.array(filtered)

    @staticmethod
    def build_triangle_edges(points_np,skew_threshold=0.1):
        tri = Delaunay(points_np)
        triangles = Interface.filter_skewed_triangles(points_np, tri.simplices,skew_threshold)

        edge_set = set()
        for triangle in triangles:
            for i in range(3):
                a, b = triangle[i], triangle[(i + 1) % 3]
                edge_set.add((a, b))
                edge_set.add((b, a))
        if edge_set:
            edges = torch.tensor(list(edge_set), dtype=torch.long).t()
        else:
            edges = torch.empty((2, 0), dtype=torch.long)
        return triangles, edges, tri

    def draw_colorbar(self, vmin, vmax, highlight_value=None):
        """
        colors: (N, 3) float32 array, values between 0-1
        """
        colors = np.array([self.cmap(i / 255.0)[:3] for i in range(256)], dtype=np.float32)

        self.colormap_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, self.colormap_tex)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, colors.shape[0], 0, GL_RGB, GL_FLOAT, colors)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_1D, 0)

        glUseProgram(self.colorbar_shader)

        glUniform1f(glGetUniformLocation(self.colorbar_shader, "vmin"), vmin)
        glUniform1f(glGetUniformLocation(self.colorbar_shader, "vmax"), vmax)

        if highlight_value is None:
            highlight_value = -1e10
            draw_highlight = False
        else:
            vdist = (highlight_value-vmin)/(vmax-vmin)
            vdist = np.clip(vdist,a_min=0,a_max=1)

            draw_highlight = True

        # print(highlight_value)
        glUniform1f(glGetUniformLocation(self.colorbar_shader, "highlight_value"), np.clip(highlight_value,a_min=vmin,a_max=vmax))

        # Bind the colormap
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_1D, self.colormap_tex)
        glUniform1i(glGetUniformLocation(self.colorbar_shader, "colormap"), 0)  # texture unit 0

        glBindVertexArray(self.colorbar_vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)

        self.draw_text('%2.2f'%(vmax),self.width*0.9,0.09*self.height,24,align_hrz='center',align_vrt='bot')
        self.draw_text('%2.2f'%(vmin),self.width*0.9,0.91*self.height,24,align_hrz='center',align_vrt='top')

        if draw_highlight:
            self.draw_text('%2.2f'%(highlight_value),self.width*0.89,(0.9-vdist*0.8)*self.height,24,align_hrz='right',align_vrt='bot')

    def draw_fullscreen_quad(self):
        glBindVertexArray(self.quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

    def handle_input_box(self, event):
        if event.key == pygame.K_RETURN:
            filename = os.path.join("./", self.input_text)
            if self.input_mode == "Save":
                self.save_state()
                with open(filename, 'wb') as f:
                    pickle.dump({'pos': self.pos, 'field':self.field, 'n':self.n, 'edges': self.edges}, f)
                print(f"Saved to {filename}")
            elif self.input_mode == "Load":
                try:
                    with open(filename, 'rb') as f:
                        data = pickle.load(f)
                        self.pos = data['pos']
                        self.field = data['field']
                        self.n = data['n']
                        self.edges = data['edges']
                    print(f"Loaded from {filename}")
                except FileNotFoundError:
                    print(f"File {filename} not found.")
            self.input_mode = None
            self.input_text = ""
        elif event.key == pygame.K_BACKSPACE:
            self.input_text = self.input_text[:-1]
        elif event.key == pygame.K_ESCAPE:
            self.input_mode = None
            self.input_text = ""
        else:
            self.input_text += event.unicode

    def handle_events(self):

        self.mouse_pos = self.render2pos(torch.tensor(pygame.mouse.get_pos(), dtype=torch.float32))

        distances = torch.norm(self.pos - self.mouse_pos, dim=1)
        self.hovered_point = torch.argmin(distances) if torch.any(distances < self.selection_radius*self.position_range/self.zoom) else None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.size
                self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, width, height)
                self.width,self.height = width,height
                self.recreate_fbo()
            elif event.type == pygame.KEYDOWN:
                if self.input_mode:
                    self.handle_input_box(event)
                else:
                    self.handle_keydown(event)
            elif event.type == pygame.KEYUP:
                if not(self.input_mode):
                    self.handle_keyup(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_down(event, self.mouse_pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.handle_mouse_up(event)
            if event.type == pygame.MOUSEWHEEL:
                self.handle_scroll(event)
        
        if self.move_hrz != 0:
            self.center[0] += self.move_hrz/40/self.zoom
        if self.move_vrt != 0:
            self.center[1] += self.move_vrt/40/self.zoom
    
    def handle_scroll(self,event):
        
        if event.y == 1:
            self.zoom *= 1.2
        elif event.y == -1:
            self.zoom /= 1.2

    def handle_keydown(self, event):
        if event.key == pygame.K_TAB:
            self.mode = "contours" if self.mode == "points" else "points"
            self.dirty = True
        elif event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_1:
            self.field_idx = 0
            self.cmap = cm.viridis
            self.field_name = 'X velocity'
        elif event.key == pygame.K_2:
            self.field_idx = 1
            self.cmap = cm.viridis
            self.field_name = 'Y velocity'
        elif event.key == pygame.K_3:
            self.field_idx = 2
            self.cmap = cm.plasma
            self.field_name = 'Pressure'
        elif event.key == pygame.K_4 and self.model_type == 'tignn':
            self.field_idx = 3
            self.cmap = cm.cividis
            self.field_name = 'Energy'
        elif event.key == pygame.K_5 and self.model_type == 'tignn':
            self.field_idx = 4
            self.cmap = cm.cividis
            self.field_name = 'Entropy'
        elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
            self.input_mode = "Save"
            self.input_text = ""
        elif event.key == pygame.K_l and pygame.key.get_mods() & pygame.KMOD_CTRL:
            self.input_mode = "Load"
            self.input_text = ""
        elif event.key == pygame.K_SPACE:
            self.update_state = False if self.update_state else True
        elif event.key == pygame.K_w:
            self.center[1] += 0.1
            self.move_vrt = 1
        elif event.key == pygame.K_s:
            self.move_vrt = -1
        elif event.key == pygame.K_a:
            self.move_hrz = -1
        elif event.key == pygame.K_d:
            self.move_hrz = 1
        elif event.key == pygame.K_g:
            self.center = torch.zeros(2)
        elif event.key == pygame.K_m:
            self.mesh = False if self.mesh else True
        elif event.key == pygame.K_n:
            self.node_update = True 
        elif event.key == pygame.K_c and self.hovered_point is not None:
            self.save_state()
            self.n[self.hovered_point] = torch.nn.functional.one_hot(
                (torch.argmax(self.n[self.hovered_point]) + 1) % self.n.shape[1], num_classes=self.n.shape[1]
            ).float()
            self.dirty = True
        elif event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
            if self.undo_stack:
                self.redo_stack.append((self.pos.clone().detach(), self.field.clone().detach(), self.n.clone().detach(), self.edges.clone().detach()))
                self.pos, self.field, self.n, self.edges = self.undo_stack.pop()
                self.hovered_point = None
                self.dirty = True
        elif event.key == pygame.K_y and pygame.key.get_mods() & pygame.KMOD_CTRL:
            if self.redo_stack:
                self.undo_stack.append((self.pos.clone().detach(), self.field.clone().detach(), self.n.clone().detach(), self.edges.clone().detach()))

                self.pos, self.field, self.n, self.edges = self.redo_stack.pop()
                self.hovered_point = None
                self.dirty = True


    def handle_keyup(self,event):
        if event.key == pygame.K_w or event.key == pygame.K_s or event.key == pygame.K_a or event.key == pygame.K_d:
            self.move_vrt = 0
            self.move_hrz = 0

    def handle_mouse_down(self, event, mouse_pos):
        if event.button == 1:
            if self.mode == "points" and self.hovered_point is not None:
                self.save_state()
                self.selected_point = self.hovered_point
                self.moving_point = True
            elif self.mode == "points":
                self.save_state()
                new_pos = mouse_pos.clone().detach()
                new_field = self.field.mean(0)
                new_n = torch.nn.functional.one_hot(torch.zeros((1,),dtype=torch.long), num_classes=self.n.shape[1]).float().squeeze(0)
                self.pos = torch.cat((self.pos,new_pos.unsqueeze(0)),dim=0)
                self.field = torch.cat((self.field,new_field.unsqueeze(0)),dim=0)
                self.n = torch.cat((self.n,new_n.unsqueeze(0)),dim=0)
                self.dirty = True
        elif event.button == 3:
            self.handle_right_click(mouse_pos)


    def handle_right_click(self, mouse_pos):
        if self.mode == "points":
            if self.hovered_point is not None:
                self.save_state()
                mask = (self.edges[0] != self.hovered_point) & (self.edges[1] != self.hovered_point)
                self.edges = self.edges[:, mask]
                # self.points = torch.cat((self.points[:self.hovered_point], self.points[self.hovered_point + 1:]), dim=0)
                self.pos = torch.cat((self.pos[:self.hovered_point], self.pos[self.hovered_point + 1:]), dim=0)
                self.field = torch.cat((self.field[:self.hovered_point], self.field[self.hovered_point + 1:]), dim=0)
                self.n = torch.cat((self.n[:self.hovered_point], self.n[self.hovered_point + 1:]), dim=0)

                self.edges = self.edges.clone()
                self.edges[self.edges > self.hovered_point] -= 1
                self.dirty = True
                self.hovered_point = None
            elif self.edges.numel() > 0:
                # p1s, p2s = self.points[:, :2][self.edges[0]], self.points[:, :2][self.edges[1]]
                p1s, p2s = self.pos[self.edges[0]], self.pos[self.edges[1]]
                dists = torch.tensor([
                    self.point_line_distance(mouse_pos.numpy(), p1.numpy(), p2.numpy())
                    for p1, p2 in zip(p1s, p2s)
                ])
                min_dist, min_idx = torch.min(dists, dim=0)
                # if min_dist < self.selection_radius*self.position_range/self.zoom:
                if min_dist < self.selection_radius*self.position_range/self.zoom:
                    self.save_state()
                    a, b = self.edges[0, min_idx], self.edges[1, min_idx]
                    keep = ~(((self.edges[0] == a) & (self.edges[1] == b)) | ((self.edges[0] == b) & (self.edges[1] == a)))
                    self.edges = self.edges[:, keep]

            # tri = Delaunay(self.pos.numpy())
            # self.tri = tri

    def handle_mouse_up(self, event):
        if event.button == 1:
            # if self.moving_point:
            #     print('save sat')
            #     self.save_state()
            # print('here')
            self.selected_point = None
            self.moving_point = False
            # self.save_state()

    def update(self):
        # check if mouse is moving points
        if self.selected_point is not None and self.mode == "points" and self.moving_point:
            self.pos[self.selected_point] = torch.tensor(self.mouse_pos, dtype=torch.float32)
            self.dirty = True

        # update edges if points are moving
        points_np = self.pos.numpy()
        if self.dirty and len(points_np) >= 3:
            if self.mesh:
                self.triangles, self.edges, self.tri = self.build_triangle_edges(points_np)
            # self.dirty = False

        # update coutours or point opengl buffers
        self.update_vbos()

    def export_state(self):
        return self.field,self.pos,self.edges,self.n


    def draw(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.94, 0.94, 0.94, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # determine field type and min/max of colorbar
        if self.field_idx < 3:
            values_np = self.field[:, self.field_idx].numpy()
        elif self.field_idx == 3:
            values_np = self.E.numpy()
        elif self.field_idx == 4:
            values_np = self.S.numpy()
        vmean = values_np.mean()
        vstd = values_np.std()
        vmin, vmax = vmean-2*vstd,vmean+2*vstd

        # draw type
        if self.mode == "points":
            self.draw_edges(values_np, vmin, vmax)
            self.draw_points()
        elif self.mode == "contours" and len(self.field) >= 3:
            self.draw_contours()

        # determine highlighting of colorbar
        if not(self.hovered_point is None):
            if self.field_idx < 3:
                highlight_value = self.field[self.hovered_point, self.field_idx].item()
            elif self.field_idx == 3:
                highlight_value = self.E[self.hovered_point].item()
            elif self.field_idx == 4:
                highlight_value = self.S[self.hovered_point].item()
        else:
            highlight_value = None

        self.draw_colorbar(vmin, vmax, highlight_value)

        # draw text info
        self.draw_text('FPS: %i'%(self.clock.get_fps()),50,50,24)
        self.draw_text('Auto Edge: %s'%(self.mesh),50,50+24+5,24)
        self.draw_text('Mode: %s'%(self.mode),50,50+2*(24+5),24)
        self.draw_text('Field: %s'%(self.field_name),50,50+3*(24+5),24)
        
        # for mesh saves and loading
        if self.input_mode:
            self.draw_text('%s: %s'%(self.input_mode,self.input_text),50,self.height-50,24)

        # Bind screen framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.94, 0.94, 0.94, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Bind FXAA shader
        glUseProgram(self.fxaa_shader)

        # Bind the FBO texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.fbo_texture)
        glUniform1i(glGetUniformLocation(self.fxaa_shader, "screenTexture"), 0)

        # Set resolution uniform
        glUniform2f(glGetUniformLocation(self.fxaa_shader, "resolution"), float(self.width), float(self.height))

        glUniform1f(glGetUniformLocation(self.fxaa_shader, "fxaaAmount"), 1)
        glUniform1f(glGetUniformLocation(self.fxaa_shader, "clampStrength"), 100.0)

        # Draw fullscreen triangle
        glBindVertexArray(self.quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        pygame.display.flip()



    def draw_edges(self, values_np, vmin, vmax):
        glBindVertexArray(self.edges_vao)

        glUseProgram(self.line_shader_program)

        # Line thickness
        thickness_loc = glGetUniformLocation(self.line_shader_program, "uThickness")
        glUniform1f(thickness_loc, self.line_width) 

        # Screen size
        screen_size_loc = glGetUniformLocation(self.line_shader_program, "uScreenSize")
        glUniform2f(screen_size_loc, self.width, self.height)
        glDrawArrays(GL_LINES, 0, 2*self.edges.shape[1])


    def draw_points(self):
        glUseProgram(self.shader_program)
        glBindVertexArray(self.points_vao)

        glUniform1f(glGetUniformLocation(self.shader_program, "uPointSize"), self.point_radius)

        glDrawArrays(GL_POINTS, 0, len(self.pos))

        glBindVertexArray(0)


    def draw_contours(self):
        glUseProgram(self.contour_shader)  
        glBindVertexArray(self.triangle_vao) 

        glDrawArrays(GL_TRIANGLES, 0, self.num_render_triangles)

        glBindVertexArray(0)


    def render_text_texture(self,text, font_size=32, color=(255, 255, 255)):
        pygame.font.init()
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        width, height = text_surface.get_size()

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        return texture, width, height
    
    def draw_text(self,text,x,y,font_size=16,color=(0,0,0),align_hrz='left',align_vrt='bot'):
        # Generate texture from text
        texture, tex_width, tex_height = self.render_text_texture(text, font_size=font_size, color=color)
        if align_hrz == 'right':
            x -= tex_width
        elif align_hrz == 'left':
            pass
        elif align_hrz == 'center':
            y -= tex_height/2
        else:
            raise ValueError()


        if align_vrt == 'bot':
            y -= tex_height/2
        elif align_vrt == 'top':
            y += tex_height/2
        elif align_vrt == 'center':
            pass
        else:
            raise ValueError()

    
        # Quad vertices and texture coords
        vertices = np.array([
            # Position      # Texture Coord
            0.0,         0.0,          0.0, 1.0,  # Bottom-left
            tex_width,   0.0,          1.0, 1.0,  # Bottom-right
            tex_width,   tex_height,   1.0, 0.0,  # Top-right
            0.0,         tex_height,   0.0, 0.0,  # Top-left
        ], dtype=np.float32)

        indices = np.array([
            0, 1, 2,
            2, 3, 0
        ], dtype=np.uint32)

        glBindVertexArray(self.text_vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.text_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.text_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Attribute pointers
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(2 * vertices.itemsize))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

        # Uniform locations
        uResolutionLoc = glGetUniformLocation(self.text_shader, "uResolution")
        uTextPosLoc = glGetUniformLocation(self.text_shader, "uTextPos")
        uTextureLoc = glGetUniformLocation(self.text_shader, "uTexture")

        glUseProgram(self.text_shader)
        glUniform2f(uResolutionLoc, self.width, self.height)
        glUniform2f(uTextPosLoc, x, y)
        glUniform1i(uTextureLoc, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)

        glBindVertexArray(self.text_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glDeleteTextures([texture])

    def exit(self):
        pygame.quit()
        sys.exit()
