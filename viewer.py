#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle
import sys

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import assimpcy                     # 3D resource loader
from PIL import Image               # load images for textures

from transform import Trackball, identity, vec, translate, rotate, scale, lerp, vec, sincos
from transform import quaternion, quaternion_from_euler, quaternion_matrix, quaternion_slerp

from bisect import bisect_left


# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object


class VertexArray:
    """ helper class to create and self destroy OpenGL vertex array objects."""
    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            Attributes should be list of arrays with one row per vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load buffer per vertex attribute (in list with index = shader layout)
        for loc, data in enumerate(attributes):
            if data is not None:
                # bind a new vbo, upload its data to GPU, declare size and type
                self.buffers.append(GL.glGenBuffers(1))
                data = np.array(data, np.float32, copy=False)  # ensure format
                nb_primitives, size = data.shape
                GL.glEnableVertexAttribArray(loc)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
                GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)  # good format
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

    def execute(self, primitive):
        """ draw a vertex array, either as direct array or indexed array """
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)


class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        if time <= self.times[0]:
            return self.values[0]
        if time >= self.times[-1]:
            return self.values[-1]
        # 2. search for closest index entry in self.times, using bisect_left function
        index = bisect_left(self.times, time)
        # 3. using the retrieved index, interpolate between the two neighboring values
        # in self.values, using the initially stored self.interpolate function
        f = (time - self.times[index - 1]) / (self.times[index] - self.times[index - 1])
        return self.interpolate(self.values[index - 1], self.values[index], f)


class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.translate_keyframes = KeyFrames(translate_keys)
        self.rotate_keyframes = KeyFrames(rotate_keys, quaternion_slerp)
        self.scale_keyframes = KeyFrames(scale_keys)

    def value(self, time, max=0):
        """ Compute each component's interpolation and compose TRS matrix """
        if max != 0:
            time = time % max
        translation = translate(self.translate_keyframes.value(time))
        rotation = quaternion_matrix(self.rotate_keyframes.value(time))
        scaling = scale(self.scale_keyframes.value(time))

        return translation @ rotation @ scaling

# -------------- OpenGL Texture Wrapper ---------------------------------------
class Texture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, file, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        try:
            # imports image as a numpy array in exactly right format
            tex = np.asarray(Image.open(file).convert('RGBA'))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.glid)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, tex.shape[1],
                            tex.shape[0], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            # message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            # print(message % (file, tex.shape, wrap_mode, min_filter, mag_filter))
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)


# ------------  Scene object classes ------------------------------------------
class Node:
    """ Scene graph transform and parameter broadcast node """
    def __init__(self, children=(), transform=identity()):
        self.premiere = transform
        self.transform = transform
        self.children = list(iter(children))
        self.rotation = 0

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model):
        """ Recursive draw, passing down updated model matrix. """
        for child in self.children:
            child.draw(projection, view, model @ self.transform)

    def tranforme(self,transform):
        self.transform = self.transform @ transform

    def key_handler(self, key):
        """ Dispatch keyboard events to children """
        for child in self.children:
            if hasattr(child, 'key_handler'):
                child.key_handler(key)

    def tourne(self,direction):
        if direction=='droite':
            rotation = rotate((0,1,0),5)
            self.rotation += 5
        if direction=='gauche':
            rotation = rotate((0,1,0),-5)
            self.rotation -= 5
        if direction=='haut':
            rotation = rotate((-np.cos(self.rotation*np.pi /180),0,np.sin(self.rotation*np.pi / 180)),5)
        if direction=='bas':
            rotation = rotate((-np.cos(self.rotation* np.pi /180),0,np.sin(self.rotation * np.pi /180)),-5)
        if direction=='origine':
            for child in self.children:
                child.transform = child.premiere
                self.rotation = 0
        else:
            for child in self.children:
                child.tranforme(rotation)

class Cylinder(Node):
    """ Very simple cylinder based on practical 2 load function """
    def __init__(self, shader):
        super().__init__()
        self.figure = load_textured('cylinder.obj', shader, "rouille.jpg")[0]  # just load cylinder from file
        self.add(self.figure)

class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, translate_keys, rotate_keys, scale_keys, max=0):
        super().__init__()
        self.keyframes = TransformKeyFrames(translate_keys, rotate_keys, scale_keys)
        self.max = max

    def draw(self, projection, view, model):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time(), self.max)
        super().draw(projection, view, model)
# -------------- Phong rendered Mesh class -----------------------------------
# mesh to refactor all previous classes

class Mesh:

    def __init__(self, shader, attributes, index=None):
        self.shader = shader
        names = ['view', 'projection', 'model']
        self.loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        GL.glUniformMatrix4fv(self.loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(self.loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(self.loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.execute(primitives)

class MeshCube:

    def __init__(self, shader, texture, attributes=None, index=None):
        self.shader = shader
        names = ['view', 'projection', 'model']
        vertices = [
                    -1.0,  1.0, -1.0,
                    -1.0, -1.0, -1.0,
                     1.0, -1.0, -1.0,
                     1.0, -1.0, -1.0,
                     1.0,  1.0, -1.0,
                    -1.0,  1.0, -1.0,

                    -1.0, -1.0,  1.0,
                    -1.0, -1.0, -1.0,
                    -1.0,  1.0, -1.0,
                    -1.0,  1.0, -1.0,
                    -1.0,  1.0,  1.0,
                    -1.0, -1.0,  1.0,

                     1.0, -1.0, -1.0,
                     1.0, -1.0,  1.0,
                     1.0,  1.0,  1.0,
                     1.0,  1.0,  1.0,
                     1.0,  1.0, -1.0,
                     1.0, -1.0, -1.0,

                    -1.0, -1.0,  1.0,
                    -1.0,  1.0,  1.0,
                     1.0,  1.0,  1.0,
                     1.0,  1.0,  1.0,
                     1.0, -1.0,  1.0,
                    -1.0, -1.0,  1.0,

                    -1.0,  1.0, -1.0,
                     1.0,  1.0, -1.0,
                     1.0,  1.0,  1.0,
                     1.0,  1.0,  1.0,
                    -1.0,  1.0,  1.0,
                    -1.0,  1.0, -1.0,

                    -1.0, -1.0, -1.0,
                    -1.0, -1.0,  1.0,
                     1.0, -1.0, -1.0,
                     1.0, -1.0, -1.0,
                    -1.0, -1.0,  1.0,
                     1.0, -1.0,  1.0]
        vertexes = []
        for i in range(0,len(vertices), 3):
            vertexes.append((vertices[i], vertices[i+1], vertices[i+2]))
        self.attributes = np.array(vertexes, 'f')

        self.loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.vertex_array = VertexArray([self.attributes], index)
        self.texture = texture

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glUniformMatrix4fv(self.loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(self.loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(self.loc['model'], 1, True, model)
        self.vertex_array.execute(primitives)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.texture.glid)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 36)
        GL.glDepthMask(GL.GL_TRUE)


        # draw triangle as GL_TRIANGLE vertex array, draw array call
        # self.vertex_array.execute(primitives)

class Axis(Mesh):
    """ Axis object useful for debugging coordinate frames """
    def __init__(self, shader):
        pos = ((0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 1))
        col = ((1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1))
        super().__init__(shader, [pos, col])

    def draw(self, projection, view, model, primitives=GL.GL_LINES):
        super().draw(projection, view, model, primitives)


class SimpleTriangle(Mesh):
    """Hello triangle object"""

    def __init__(self, shader):

        # triangle position buffer
        position = np.array(((0, .5, 0), (.5, -.5, 0), (-.5, -.5, 0)), 'f')
        color = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 'f')

        super().__init__(shader, [position, color])


# -------------- Example texture plane class ----------------------------------
class TexturedMesh(Mesh):
    """ Simple first textured object """

    def __init__(self, shader, texture, attributes, index):
        super().__init__(shader, attributes, index)

        loc = GL.glGetUniformLocation(shader.glid, 'diffuse_map')
        self.loc['diffuse_map'] = loc

        # setup texture and upload it to GPU
        self.texture = texture

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        # texture access setups
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(self.loc['diffuse_map'], 0)
        super().draw(projection, view, model, primitives)

class TexturedPhongMesh(Mesh):
    """ Mesh with Phong illumination """

    def __init__(self, shader, attributes, texture, index=None,
                 lights=[[(0, 10, 0), (70, 30, 30)], [(10, 0, 0), (80, 10, 40)]],
                 k_a=(0, 0, 0), k_d=(1, 1, 0), k_s=(1, 1, 1), s=16.):
        super().__init__(shader, attributes, index)
        self.lights = lights
        self.k_a, self.k_d, self.k_s, self.s = k_a, k_d, k_s, s


        # retrieve OpenGL locations of shader variables at initialization
        names = ['k_a', 's', 'k_s', 'k_d', 'w_camera_position', 'diffuse_map', 'normal_matrix']
        for i in range(len(self.lights)):
            names.append('lights[{}].position'.format(i))
            names.append('lights[{}].intensity'.format(i))

        loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.loc.update(loc)
        self.texture = texture

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        # texture access setups
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)

        GL.glUniform1i(self.loc['diffuse_map'], 0)

        # setup light parameters
        for i, light in enumerate(self.lights):
            position, intensity = light
            GL.glUniform3fv(self.loc['lights[{}].position'.format(i)], 1, position)
            GL.glUniform3fv(self.loc['lights[{}].intensity'.format(i)], 1, intensity)

        normal_matrix = np.linalg.inv(model).T
        GL.glUniformMatrix4fv(self.loc['normal_matrix'], 1, True, normal_matrix)

        # setup material parameters
        GL.glUniform3fv(self.loc['k_a'], 1, self.k_a)
        GL.glUniform3fv(self.loc['k_d'], 1, self.k_d)
        GL.glUniform3fv(self.loc['k_s'], 1, self.k_s)
        GL.glUniform1f(self.loc['s'], max(self.s, 0.001))

        # world camera position for Phong illumination specular component
        w_camera_position = np.linalg.inv(view)[0:3, 3]
        GL.glUniform3fv(self.loc['w_camera_position'], 1, w_camera_position)

        super().draw(projection, view, model, primitives)


def load_textured(file, shader, tex_file=None):
    """ load resources from file using assimp, return list of TexturedMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_FlipUVs
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if not tex_file:
            if 'TEXTURE_BASE' in mat.properties:  # texture token
                name = os.path.basename(mat.properties['TEXTURE_BASE'])
                # search texture in file's whole subdir since path often screwed up
                paths = os.walk(path, followlinks=True)
                found = [os.path.join(d, f) for d, _, n in paths for f in n
                         if name.startswith(f) or f.startswith(name)]
                assert found, 'Cannot find texture %s in %s subtree' % (name, path)
                tex_file = found[0]
            elif mat != scene.mMaterials[0]:
                for filename in os.listdir(path):
                    if filename.endswith('_Base_Color.png') or filename.endswith('_Base Color.png'):
                        tex_file = path + '/' +filename
                        break
                if not tex_file:
                    for filename in os.listdir(path):
                        if filename.endswith('Normal.png'):
                            tex_file = path + '/' +filename
                            break
        if tex_file:
            mat.properties['diffuse_map'] = Texture(file=tex_file)
    # prepare textured mesh
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        assert mat['diffuse_map'], "Trying to map using a textureless material"
        attributes = [mesh.mVertices, mesh.mTextureCoords[0]]
        mesh = TexturedMesh(shader, mat['diffuse_map'], attributes, mesh.mFaces)
        meshes.append(mesh)

    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    # print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes

def load_phong_textured(file, shader, lights, tex_file=None):
    """ load resources from file using assimp, return list of ColorMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_GenSmoothNormals
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if not tex_file:
            if 'TEXTURE_BASE' in mat.properties:  # texture token
                name = os.path.basename(mat.properties['TEXTURE_BASE'])
                # search texture in file's whole subdir since path often screwed up
                paths = os.walk(path, followlinks=True)
                found = [os.path.join(d, f) for d, _, n in paths for f in n
                         if name.startswith(f) or f.startswith(name)]
                assert found, 'Cannot find texture %s in %s subtree' % (name, path)
                tex_file = found[0]
            elif mat != scene.mMaterials[0]:
                for filename in os.listdir(path):
                    if filename.endswith('_Base_Color.png') or filename.endswith('_Base Color.png'):
                        tex_file = path + '/' +filename
                        break
                if not tex_file:
                    for filename in os.listdir(path):
                        if filename.endswith('Normal.png'):
                            tex_file = path + '/' +filename
                            break
        if tex_file:
            mat.properties['diffuse_map'] = Texture(file=tex_file)

    # prepare mesh nodes
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        attributes = [mesh.mVertices, mesh.mNormals, mesh.mTextureCoords[0]]
        mesh = TexturedPhongMesh(shader, attributes,
                         mat['diffuse_map'],
                         mesh.mFaces,
                         k_d=mat.get('COLOR_DIFFUSE', (1, 1, 1)),
                         k_s=mat.get('COLOR_SPECULAR', (1, 1, 1)),
                         k_a=mat.get('COLOR_AMBIENT', (0, 0, 0)),
                         s=mat.get('SHININESS', 16.),
                         lights=lights)
        meshes.append(mesh)
    return meshes


class CubemapTexture:
    def __init__(self, file, wrap_mode=GL.GL_CLAMP_TO_EDGE):
        self.glid = GL.glGenTextures(1)
        try:
            # imports image as a numpy array in exactly right format
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.glid)
            for i in range(6):
                tex = np.asarray(Image.open(file[i]).convert('RGBA'))
                GL.glTexImage2D(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL.GL_RGBA, tex.shape[1],
                                tex.shape[0], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            # GL.glGenerateMipmap(GL.GL_TEXTURE_CUBE_MAP)

            # print("texture Loaded")
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)

# ------------  Viewer class & window management ------------------------------
class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__()
        self.mouse = (0, 0)
    #     glfw.set_cursor_pos_callback(win, self.on_mouse_move)
    #     # glfw.set_scroll_callback(win, self.on_scroll)
    #
    # def on_mouse_move(self, win, xpos, ypos):
    #     """ Rotate on left-click & drag, pan on right-click & drag """
    #     old = self.mouse
    #     self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
    #     if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
    #         self.drag(old, self.mouse, glfw.get_window_size(win))
        # if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
        #     self.pan(old, self.mouse)

    # def on_scroll(self, win, _deltax, deltay):
    #     """ Scroll controls the camera distance to trackball center """
    #     self.zoom(deltay, glfw.get_window_size(win)[1])


class Viewer(Node):
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):
        super().__init__()

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        GL.glEnable(GL.GL_DEPTH_TEST)    # depth test now enabled (TP2)
        GL.glEnable(GL.GL_CULL_FACE)     # backface culling enabled (TP2)

        # initialize trackball
        self.trackball = GLFWTrackball(self.win)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            win_size = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(win_size)

            # draw our scene objects
            self.draw(projection, view, identity())

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            if key==glfw.KEY_RIGHT:
                super().tourne('droite')
            if key==glfw.KEY_LEFT:
                super().tourne('gauche')
            if key==glfw.KEY_UP:
                super().tourne('haut')
            if key==glfw.KEY_DOWN:
                super().tourne('bas')
            if key==glfw.KEY_SPACE:
                super().tourne('origine')
            self.key_handler(key)


# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()
    shader = Shader("shader.vert", "shader.frag")
    # shaderPoisson = Shader("texture.vert", "texture.frag")
    phongShader = Shader("phong.vert", "phong.frag")

    faces =["right.jpg","left.jpg","top.jpg","bottom.jpg","front.jpg","back.jpg"]
    for i in range(len(faces)):
        faces[i] = "underwater" + "/" + faces[i]

    texture = CubemapTexture(faces)

    noeudTexture = Node(transform=translate(0,0,1))
    noeudTexture.add(MeshCube(shader,texture))

    poisson = Node(transform=translate(0.3,0,0)@scale(0.1,0.1,0.1))

    fichier = "./Fish/BottlenoseDolphin/BottleNoseDolphin.obj"
    if len(sys.argv)> 1 :
        fichier = sys.argv[1]
        print(fichier[-3:])
    if fichier[-3:] != "obj":
        print("Veuillez entrer un fichier au format '.obj'")
        sys.exit()


    origine = vec(5,8,-5,0)
    origine1 = (origine[0], origine[1], origine[2])
    origine2 = (5, -8, 5)
    lights = [[origine1,(95,95,150)],[origine2, (30,30,30)]]

    translate_keys={}
    rotate_keys={}
    scale_keys={}
    translate_keys = {0: vec(10, 0, 0)}
    rotate_keys = {0: quaternion()}
    scale_keys = {0: 1}
    for i in range(1,24,1):
        angle = 15 *i
        scale_keys[i] = 1
        rotate_keys[i] = quaternion_from_euler(0,-angle, 0)
        sinus, cosinus = sincos(angle)
        sinusleve, cosinusleve = sincos(2*angle)
        translate_keys[i] = vec(10*cosinus, sinusleve, 10*sinus)
    keynode = KeyFrameControlNode(translate_keys, rotate_keys, scale_keys,24)
    keynode.add(*[mesh for mesh in load_phong_textured(fichier, phongShader, lights)])
    poisson.add(keynode)

    for k in range(4):
        translate_keys = {}
        rotate_keys = {}
        scale_keys = {}
        for i in range(200):
            if i < 10:
                scale_keys[i] = i * 0.1
            elif i >= 190:
                scale_keys[i] = (199 - i) * 0.1
            else:
                scale_keys[i] = 1
            translate_keys[i] = vec(20 + k*3, k*3 - 5, 10*i - 100)
            rotate_keys[i] = quaternion()
        keynode = KeyFrameControlNode(translate_keys, rotate_keys, scale_keys,400)
        keynode.add(*[mesh for mesh in load_phong_textured("./Fish/ClownFish2/ClownFish2.obj", phongShader, lights)])
        poisson.add(keynode)

    noeudTexture.add(poisson)
    viewer.add(noeudTexture)

    # affichage commandes
    print("Voici la liste des commandes disponibles :")
    print("Ctrl + Z ==> change le mode d'affichage : remplissage, bordures ou par points")
    print("les fleches de votre clavier permettent de faire tourner la caméra pour observer l'ensemble du décor")
    print("la touche espace vous permet de replacer la caméra a l'origine")
    print("")
    print("vous pouvez entrer en argument un fichier objet (le chemin entier a partir de ce repertoire) et il sera affiché a la place du dauphin en train de nager")

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
