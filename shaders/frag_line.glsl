#version 330 core

out vec4 FragColor;
in vec3 fColor;  // Color passed from geometry shader

void main()
{
    FragColor = vec4(fColor,1.0);  // Use the color passed from geometry shader
    // FragColor = vec4(fColor.r,fColor.b,fColor.g,);  // Use the color passed from geometry shader
    // FragColor = vec4(fColor.g,0.0,0.0,0.5);  // Use the color passed from geometry shader
}
