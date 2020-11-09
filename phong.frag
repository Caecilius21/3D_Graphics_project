#version 330 core

// fragment position and normal of the fragment, in WORLD coordinates
// (you can also compute in VIEW coordinates, your choice! rename variables)
in vec3 w_position, w_normal;   // in world coodinates

// lights, in world coordinates
#define NB_LIGHTS 2
uniform struct Light {
    vec3 position;
    vec3 intensity;
} lights[NB_LIGHTS];

// material properties
uniform vec3 k_a;
uniform vec3 k_d;
uniform vec3 k_s;
uniform float s;
uniform sampler2D diffuse_map;


in vec2 frag_tex_coords;

// world camera position
uniform vec3 w_camera_position;

out vec4 out_color;

vec3 phong(Light light, vec3 n, vec3 v) {
    vec3 l_unormalized = light.position - w_position;
    vec3 l = normalize(l_unormalized);
    vec3 r = reflect(-l, n);
    float distance_squared = dot(l_unormalized, l_unormalized);

    float diffuse_coeff = max(dot(n, l), 0.0);
    vec3 diffuse = k_d * diffuse_coeff;

    float specular_coeff = pow(max(dot(r, v), 0.0), s);
    vec3 specular = k_s * specular_coeff;

    return light.intensity * (diffuse + specular) / distance_squared;
}

void main() {
    vec3 n = normalize(w_normal);
    vec3 v = normalize(w_camera_position - w_position);

    vec4 color = vec4(k_a,1);
    for (int i = 0; i < NB_LIGHTS; ++i) {
        color += vec4(phong(lights[i], n, v),0);
    }

    out_color = color * texture(diffuse_map, frag_tex_coords);
}
