#version 330 core

in vec2 vTexCoord;
out vec4 FragColor;

uniform float vmin;
uniform float vmax;
uniform float highlight_value;
uniform sampler1D colormap;

const float epsilon = 0.01; // Tolerance for comparing highlight_value to vTexCoord.y

void main()
{
    // Adjust the mapping to ensure the full texture range is used (avoiding clipping at edges)
    float value = mix(vmin, vmax, vTexCoord.y);
    float norm_value = (value - vmin) / (vmax - vmin);

    // Get the color from the colormap based on the normalized value
    vec3 color = texture(colormap, norm_value).rgb;

    // Normalize the highlight value to the [0, 1] range
    float highlight_norm = (highlight_value - vmin) / (vmax - vmin);

    // Check if the vTexCoord.y is near the highlight_value, considering epsilon tolerance
    // Allow the highlight to reach the edges of the texture coordinate range
    // if (abs(vTexCoord.y - highlight_norm) < epsilon || 
    //     vTexCoord.y <= epsilon + 0.02 || vTexCoord.y >= 1.0 - epsilon - 0.02) {
    if (abs((vTexCoord.y-0.5)*1.25+0.5 - highlight_norm) < epsilon) { 
        // Apply a distinctive color (e.g., black) for highlighting
        color = vec3(0.0, 0.0, 0.0);  // Or use any other color for highlighting
    }

    FragColor = vec4(color, 1.0);
}
