
#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;
uniform vec3 w_camera_position;

void main()
{
    FragColor = texture(skybox, TexCoords);
}
