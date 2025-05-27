#version 330 core

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

uniform float uThickness;  // Line thickness in pixels
uniform vec2 uScreenSize;  // Screen width and height in pixels

in vec3 vColor[];  // Input color array from vertex shader
out vec3 fColor;  // Output color to fragment shader

void main()
{
    vec2 p0 = gl_in[0].gl_Position.xy;
    vec2 p1 = gl_in[1].gl_Position.xy;
    
    vec2 dir = normalize(p1 - p0);
    vec2 normal = vec2(-dir.y, dir.x); // Perpendicular vector

    // Convert pixel thickness to NDC units separately for X and Y
    vec2 pixel_to_clip = vec2(2.0 / uScreenSize.x, 2.0 / uScreenSize.y);
    vec2 offset = normal * pixel_to_clip * (uThickness * 0.5);

    // Emit quad vertices with color
    gl_Position = vec4(p0 + offset, 0.0, 1.0);
    fColor = vColor[0];  // Pass color from vertex 0 to fragment shader
    EmitVertex();

    gl_Position = vec4(p0 - offset, 0.0, 1.0);
    fColor = vColor[0];  // Pass color from vertex 0 to fragment shader
    EmitVertex();

    gl_Position = vec4(p1 + offset, 0.0, 1.0);
    fColor = vColor[1];  // Pass color from vertex 1 to fragment shader
    EmitVertex();

    gl_Position = vec4(p1 - offset, 0.0, 1.0);
    fColor = vColor[1];  // Pass color from vertex 1 to fragment shader
    EmitVertex();

    EndPrimitive();
}
