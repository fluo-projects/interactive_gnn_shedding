#version 330 core
in vec2 vTexCoords;
out vec4 FragColor;

uniform sampler2D screenTexture;
uniform vec2 resolution;   // (width, height)
uniform float fxaaAmount;  // 0.0 = no FXAA, 1.0 = full FXAA
uniform float clampStrength; // how strong the smoothing search is

void main()
{
    vec2 texelSize = 1.0 / resolution;

    vec3 rgbNW = texture(screenTexture, vTexCoords + texelSize * vec2(-1.0, -1.0)).rgb;
    vec3 rgbNE = texture(screenTexture, vTexCoords + texelSize * vec2( 1.0, -1.0)).rgb;
    vec3 rgbSW = texture(screenTexture, vTexCoords + texelSize * vec2(-1.0,  1.0)).rgb;
    vec3 rgbSE = texture(screenTexture, vTexCoords + texelSize * vec2( 1.0,  1.0)).rgb;
    vec3 rgbM  = texture(screenTexture, vTexCoords).rgb;

    vec3 luma = vec3(0.299, 0.587, 0.114);

    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.5, 1.0/128.0);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);

    dir = clamp(dir * rcpDirMin, -clampStrength, clampStrength) * texelSize;

    vec3 result1 = texture(screenTexture, vTexCoords + dir * (1.0/3.0 - 0.5)).rgb;
    vec3 result2 = texture(screenTexture, vTexCoords + dir * (2.0/3.0 - 0.5)).rgb;

    vec3 fxaaColor = (result1 + result2) * 0.5;

    vec3 finalColor = mix(rgbM, fxaaColor, fxaaAmount);  // blend based on fxaaAmount

    FragColor = vec4(finalColor, 1.0);
}
