#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coords;



uniform mat4 model, view, projection;
uniform mat4 normal_matrix;

// position and normal for the fragment shader, in WORLD coordinates
// (you can also compute in VIEW coordinates, your choice! rename variables)
out vec3 w_position, w_normal;   // in world coordinates
out vec2 frag_tex_coords;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);

    // compute the vertex position and normal in world coordinates
    w_position = vec3(model * vec4(position, 1.0));
    w_normal = normalize(vec3(normal_matrix * vec4(normal, 0.0)));
    frag_tex_coords = tex_coords;
}
