#version 330 core
layout(location = 0) in vec2 aPos;
out vec2 vTexCoord;

void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);  // Transform position
    // Adjust the mapping to fully use the [0, 1] texture coordinate range
    vTexCoord = vec2((aPos.x + 1.0) * 0.5, (aPos.y + 1.0) * 0.5);  // Map to [0, 1]
}
