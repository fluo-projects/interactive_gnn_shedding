#version 330 core
out vec4 FragColor;

in vec3 vColor;

void main()
{
    vec2 coord = gl_PointCoord * 2.0 - 1.0; // transform to [-1, 1]
    float dist = length(coord);

    if (dist > 1.0)
        discard; // outside the circle -> discard

    FragColor = vec4(vColor,1.0);
}
