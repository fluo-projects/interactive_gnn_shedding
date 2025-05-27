#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform vec2 uResolution;
uniform vec2 uTextPos;

void main()
{
    vec2 pos = aPos + uTextPos;
    vec2 clipSpace = (pos / uResolution) * 2.0 - 1.0;
    gl_Position = vec4(clipSpace.x, -clipSpace.y, 0.0, 1.0);
    TexCoord = aTexCoord;
}