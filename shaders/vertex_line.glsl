#version 330 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;

out vec3 vColor;  // Output to pass to geometry shader

void main()
{
    vColor = aColor;  // Pass color to geometry shader
    gl_Position = vec4(aPos, 0.0, 1.0);
}
