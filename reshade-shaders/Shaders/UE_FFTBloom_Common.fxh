/*==============================================================================

    UE FFT Bloom - real life accurate lens bloom for ReShade


    Repo:    https://github.com/HCamphics/reshade-phototools
    Licence: CC BY-NC 4.0 - see LICENSE in the repository root.

    Convolution bloom evaluated as a multiplication in the frequency domain, so
    cost depends on FFT size instead of kernel size: any PSF - blades, rings,
    dust, anamorphic streaks - for the same runtime price.

    Shared core for two effects; only the source of the PSF differs:
        UE_FFTBloom.fx           PSF read from a DDS kernel (Textures/Kernels)
        UE_FFTBloom_Aperture.fx  PSF baked on the GPU from aperture controls
    Kernels must be declared here: a declared file that is missing makes ReShade
    skip the whole effect.

    Per frame:   prefilter -> FFT rows/cols -> multiply by kernel spectrum ->
                 inverse FFT -> composite (+ lens dirt)
    Per kernel:  pupil -> dispersion -> FFT -> energy + phase normalisation

    Intensity is Unreal-style: it mixes scene -> bloom, so 1.0 is pure bloom and
    above that does nothing. Beam Emphasis and Spike * are artistic, not optical;
    the rest of the aperture group is aperture geometry.

    Limits
        - Bloom runs on the tonemapped LDR frame, so clipped highlights have
          already lost their headroom. HDR input is a TODO (untestable here).
        - Requires compute shaders: D3D11/12, Vulkan, OpenGL 4.3+. D3D9/D3D10
          cannot load it.
        - Kernels must be square, power-of-two, 8-bit BGRA DDS.

    TODO
        - HDR input path (no HDR display to test on)
        - kernel spectrum cache: needs KernelPrep as a compute pass, because a
          pixel pass that returns early leaves its targets undefined
        - dust scatter term, one-sided anamorphic streak, depth-faded bloom

==============================================================================*/
#include "ReShade.fxh"
#include "ReShadeUI.fxh"

// ==========================================================================
//  preprocessor configuration
// ==========================================================================

#ifndef KERNEL_MODE

    #define KERNEL_MODE 0
#endif

// Dispersion samples used when baking the procedural PSF. Bake-time only.
#define BAKE_DISP_STEPS 16

#ifndef FFT_SIZE
    #define FFT_SIZE 512
#endif

#if FFT_SIZE == 256
    #define LOG2_FFT 8
#elif FFT_SIZE == 512
    #define LOG2_FFT 9
#elif FFT_SIZE == 1024
    #define LOG2_FFT 10
#else
    #error "FFT_SIZE must be 256, 512 or 1024"
#endif

#define FFT_THREADS   (FFT_SIZE / 2)
#define RCP_FFT_SIZE  (1.0 / FFT_SIZE)
#define INV_SQRT_N    rsqrt((float)FFT_SIZE)

#ifndef BAKE_OVERSAMPLE
    #define BAKE_OVERSAMPLE 2
#endif

#if BAKE_OVERSAMPLE != 2
    #error "BAKE_OVERSAMPLE must be 2 (LOG2_BAKE assumes one extra bit)"
#endif

#define BAKE_SIZE        (FFT_SIZE * BAKE_OVERSAMPLE)
#define LOG2_BAKE        (LOG2_FFT + 1)
#define BAKE_THREADS     (BAKE_SIZE / 2)
#define RCP_BAKE_SIZE    (1.0 / BAKE_SIZE)
#define INV_SQRT_BAKE    rsqrt((float)BAKE_SIZE)

#define EDGE_PADDING  0.15

#ifndef SPECTRUM_16F
    #define SPECTRUM_16F 0
#endif

#if SPECTRUM_16F
    #define SPEC_FMT RGBA16F
#else
    #define SPEC_FMT RGBA32F
#endif

#if KERNEL_MODE == 0

// Every path listed must exist. A missing source texture makes ReShade log
// "Source ... was not found" and skip the whole effect.
#ifndef KERNEL_TEXTURE
    #define KERNEL_TEXTURE_0 "Kernels/DecagonApertureDirty.dds"
    #define KERNEL_TEXTURE_1 "Kernels/DirtyStreak.dds"
    #define KERNEL_TEXTURE_2 "Kernels/HeptagonApertureDirty.dds"
    #define KERNEL_TEXTURE_3 "Kernels/HumanEye.dds"
    #define KERNEL_TEXTURE_4 "Kernels/MovieLens.dds"
    #define KERNEL_TEXTURE_5 "Kernels/PhoneCamera1.dds"
    #define KERNEL_TEXTURE_6 "Kernels/PhoneCamera2.dds"
    #define KERNEL_TEXTURE_7 "Kernels/Streak.dds"
    #define KERNEL_TEXTURE   KERNEL_TEXTURE_0
#endif
#endif

#ifndef BANK_SKEW
    #define BANK_SKEW 0
#endif

#if BANK_SKEW
    #define GFFT_SIZE (BAKE_SIZE + (BAKE_SIZE >> 5))
    #define GIDX(i)   ((i) + ((i) >> 5))
#else
    #define GFFT_SIZE BAKE_SIZE
    #define GIDX(i)   (i)
#endif

static const float TAU = 6.28318530718;
static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

// ==========================================================================
//  UI uniforms
// ==========================================================================

uniform float BloomIntensity <
    ui_type = "drag"; ui_min = 0.0; ui_max = 10.0; ui_step = 0.01;
    ui_category = "Bloom";
    ui_label = "Intensity";
    ui_tooltip = "Overall strength of the glow.";
> = 0.5;

uniform float Threshold <
    ui_type = "drag"; ui_min = -1.0; ui_max = 5.0; ui_step = 0.01;
    ui_category = "Bloom";
    ui_label = "Threshold";
    ui_tooltip = "Luminance above which pixels bloom. -1 removes the threshold so everything contributes, which is the physically based behaviour of Unreal convolution bloom.";
> = -1.0;

uniform float SoftKnee <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Bloom";
    ui_label = "Soft Knee";
    ui_tooltip = "Softens the threshold transition. Inactive while Threshold is -1.";
> = 0.5;

uniform float ExposureEV <
    ui_type = "drag"; ui_min = -5.0; ui_max = 5.0; ui_step = 0.01;
    ui_category = "Bloom";
    ui_label = "Exposure Bias";
    ui_tooltip = "Stops of exposure applied to the bloom input before the threshold, so it decides which pixels qualify. The scene is untouched.";
> = 0.0;

uniform float HighlightShoulder <
    ui_type = "drag"; ui_min = 0.0; ui_max = 0.9; ui_step = 0.01;
    ui_category = "Bloom";
    ui_label = "Highlight Shoulder";
    ui_tooltip = "Rolls off the bloom input with c/(1 + s*c) before the threshold, so a clipped white pixel and a much brighter source no longer bloom identically. 0 = off; 0.2-0.5 is a sensible range for LDR games.";
> = 0.0;

uniform int BlendMode <
    ui_type = "combo";
    ui_items = "Lerp (Unreal Style)\0Additive\0Screen\0";
    ui_category = "Bloom";
    ui_label = "Blend Mode";
    ui_tooltip = "Lerp blends scene and glow by Intensity, the way Unreal's bloom composite does; Intensity is the blend amount, so 1 means glow only. Additive adds the glow and lets Intensity go above 1. Screen brightens without clipping.";
> = 0;

uniform float KernelScale <
    ui_type = "drag"; ui_min = 0.05; ui_max = 3.0; ui_step = 0.01;
    ui_category = "Kernel";
    ui_label = "Scale";
    ui_tooltip = "Magnifies the PSF inside the transform grid: above 1 is a larger, softer glow but crops the outer rays, below 1 is tighter with the whole PSF visible.";
> = 1.0;

uniform float KernelRotation <
    ui_type = "drag"; ui_min = -180.0; ui_max = 180.0; ui_step = 0.1;
    ui_category = "Kernel";
    ui_tooltip = "Degrees. Rotating the PSF never triggers a re-bake.";
    ui_label = "Rotation";
> = 0.0;

uniform float AnamorphicStretch <
    ui_type = "drag"; ui_min = 0.25; ui_max = 4.0; ui_step = 0.01;
    ui_category = "Kernel";
    ui_label = "Anamorphic Stretch";
    ui_tooltip = "Stretches the PSF vertically, like an anamorphic lens. 1 = off.";
> = 1.0;

#if KERNEL_MODE == 0
uniform int KernelFile <
    ui_type = "combo";
    ui_items = "DecagonApertureDirty\0DirtyStreak\0HeptagonApertureDirty\0HumanEye\0MovieLens\0PhoneCamera1\0PhoneCamera2\0Streak\0";
    ui_category = "Kernel";
    ui_label = "Kernel File";
    ui_tooltip = "DDS file to convolve with, from reshade-shaders/Textures/Kernels. The list is edited at the top of this shader. Aperture effect bakes its own PSF, so this only affects this effect.";
> = 0;
#endif

#if KERNEL_MODE == 1
uniform int AperturePreset <
    ui_type = "combo";
    ui_items = "Custom\0Circle\0Hexagon\0Heptagon\0Octagon\0Nonagon\0Decagon\0Hendecagon\0Dodecagon\0";
    ui_category = "Aperture";
    ui_label = "Preset";
    ui_tooltip = "Built-in blade counts, 6 to 12 blades. Custom uses the Sides slider, 3 to 24.";
> = 0;

uniform int ApertureSides <
    ui_type = "drag"; ui_min = 3; ui_max = 24; ui_step = 1;
    ui_category = "Aperture";
    ui_label = "Sides";
    ui_tooltip = "Blade count, used only when Preset is Custom. The named presets set their own count.";
> = 6;

uniform float ApertureBladeCurvature <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Aperture";
    ui_label = "Blade Curvature";
    ui_tooltip = "0 = straight blades with sharp polygon corners, 1 = fully curved and round.";
> = 0.0;

uniform float ApertureSoftness <
    ui_type = "drag"; ui_min = 0.002; ui_max = 0.9; ui_step = 0.002;
    ui_category = "Aperture";
    ui_label = "Edge Softness";
    ui_tooltip = "Width of the fade at the aperture edge. Below about 0.03 the rays stop getting brighter (measured: they saturate near 1.8% of the peak) and only the haze grows, so 0.03 is the useful floor.";
> = 0.03;

uniform float ApertureFalloff <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Aperture";
    ui_label = "Falloff";
    ui_tooltip = "0 = flat bokeh disc, 1 = soft Gaussian glow.";
> = 0.15;

uniform float ApertureRing <
    ui_type = "drag"; ui_min = 0.0; ui_max = 0.95; ui_step = 0.01;
    ui_category = "Aperture";
    ui_label = "Ring";
    ui_tooltip = "Removes the aperture centre, giving the donut bokeh of a catadioptric lens.";
> = 0.0;

uniform int ApertureSpikes <
    ui_type = "drag"; ui_min = 0; ui_max = 24; ui_step = 1;
    ui_category = "Aperture";
    ui_label = "Spike Count";
    ui_tooltip = "Adds radial diffraction spikes on top of the aperture shape. 0 = off. Sharp blades already produce spikes at low Edge Softness.";
> = 0;

uniform float ApertureSpikeStrength <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Aperture";
    ui_tooltip = "Brightness of the added spikes.";
    ui_label = "Spike Strength";
> = 0.25;

uniform float ApertureSpikeLength <
    ui_type = "drag"; ui_min = 0.5; ui_max = 8.0; ui_step = 0.1;
    ui_category = "Aperture";
    ui_tooltip = "How far the added spikes reach from the centre.";
    ui_label = "Spike Length";
> = 2.5;

uniform float3 ApertureTint <
    ui_type = "color";
    ui_category = "Aperture";
    ui_label = "Tint";
    ui_tooltip = "Colours the aperture. DDS kernels carry their own colour, this does the same for the analytic shapes.";
> = float3(1.0, 1.0, 1.0);

uniform float ApertureDispersion <
    ui_type = "drag"; ui_min = 0.0; ui_max = 0.5; ui_step = 0.005;
    ui_category = "Aperture";
    ui_label = "Dispersion";
    ui_tooltip = "Channel-dependent aperture radius: red and blue see a different pupil size, so each ray fans into a spectrum. 0 = achromatic, 0.1-0.2 is a visible rainbow, higher smears the rays apart.";
> = 0.12;

uniform float ApertureSize <
    ui_type = "drag"; ui_min = 0.005; ui_max = 0.6; ui_step = 0.001;
    ui_category = "Aperture";
    ui_label = "Size";
    ui_tooltip = "Aperture radius relative to the transform grid. Smaller = wider diffraction and a bigger glow, like stopping down, but below about 0.2 the pupil is under-resolved on the 512 grid and the pattern aliases into a lattice. 0.35-0.5 gives a clean star.";
> = 0.35;

uniform float ApertureBeamGain <
    ui_type = "drag"; ui_min = 1.0; ui_max = 8.0; ui_step = 0.05;
    ui_category = "Aperture";
    ui_label = "Beam Emphasis";
    ui_tooltip = "A clean 7-blade pupil puts its rays at roughly 2e-5 of the core peak, which is invisible on an LDR game. 2, the default, lifts them to about 0.5% (15/255) while the far field stays black. 3+ starts lifting the whole tail into a grey haze.";
> = 2.0;
#endif

uniform float3 BloomTint <
    ui_type = "color";
    ui_category = "Color";
    ui_label = "Tint";
    ui_tooltip = "Colours the glow. White keeps the kernel's own chromatic diffraction intact.";
> = float3(1.0, 1.0, 1.0);

#if KERNEL_MODE == 1

uniform float LensDirtStrength <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Lens Dirt";
    ui_label = "Lens Dirt";
    ui_tooltip = "How strongly the dust map modulates the pupil before it is diffracted, so each speck scatters light the way real dust on the front element does. It can only dim inside the aperture, never add glow outside it. 0 = clean lens.";
> = 0.0;

uniform float LensDirtGain <
    ui_type = "drag"; ui_min = 1.0; ui_max = 8.0; ui_step = 0.1;
    ui_category = "Lens Dirt";
    ui_label = "Dust Gain";
    ui_tooltip = "Brightens the dust photo before use. Dust maps are dark, dirt1.png averages 0.21, so 4 puts it near 0.85.";
> = 4.0;

uniform float LensDirtTiling <
    ui_type = "drag"; ui_min = 0.25; ui_max = 4.0; ui_step = 0.05;
    ui_category = "Lens Dirt";
    ui_tooltip = "Repeats the dust map across the aperture.";
    ui_label = "Dust Tiling";
> = 1.0;
#endif

uniform int DebugView <
    ui_type = "combo";
    ui_items = "Final Composite\0Bloom Only\0Pre-Filtered Input\0Kernel\0Spectrum Magnitude\0Quadrants\0Capture Test\0Pupil\0Bake Spectrum\0";
    ui_category = "Debug";
    ui_category_closed = true;
    ui_label = "Debug View";
    ui_tooltip = "Kernel shows the PSF at a normalised exposure. Leave at Final Composite for normal play.";
> = 0;

// ==========================================================================
//  textures, samplers, storage
// ==========================================================================

texture TexImageA   { Width = FFT_SIZE; Height = FFT_SIZE; Format = SPEC_FMT; };
texture TexImageB   { Width = FFT_SIZE; Height = FFT_SIZE; Format = SPEC_FMT; };
texture TexKernelA  { Width = FFT_SIZE; Height = FFT_SIZE; Format = SPEC_FMT; };
texture TexKernelB  { Width = FFT_SIZE; Height = FFT_SIZE; Format = SPEC_FMT; };

sampler sImageA  { Texture = TexImageA;  AddressU = CLAMP; AddressV = CLAMP; };
sampler sImageB  { Texture = TexImageB;  AddressU = CLAMP; AddressV = CLAMP; };
sampler sKernelA { Texture = TexKernelA; AddressU = CLAMP; AddressV = CLAMP; };
sampler sKernelB { Texture = TexKernelB; AddressU = CLAMP; AddressV = CLAMP; };

storage2D stImageA  { Texture = TexImageA;  };
storage2D stImageB  { Texture = TexImageB;  };
storage2D stKernelA { Texture = TexKernelA; };
storage2D stKernelB { Texture = TexKernelB; };

texture TexKernelEnergy { Width = 1; Height = 1; Format = RGBA32F; };
sampler sKernelEnergy   { Texture = TexKernelEnergy; };
storage2D stKernelEnergy { Texture = TexKernelEnergy; };

#if KERNEL_MODE == 0

texture TexBloomKernel0 < source = KERNEL_TEXTURE_0; > { Width = FFT_SIZE; Height = FFT_SIZE; Format = RGBA8; };
texture TexBloomKernel1 < source = KERNEL_TEXTURE_1; > { Width = FFT_SIZE; Height = FFT_SIZE; Format = RGBA8; };
texture TexBloomKernel2 < source = KERNEL_TEXTURE_2; > { Width = FFT_SIZE; Height = FFT_SIZE; Format = RGBA8; };
texture TexBloomKernel3 < source = KERNEL_TEXTURE_3; > { Width = FFT_SIZE; Height = FFT_SIZE; Format = RGBA8; };
texture TexBloomKernel4 < source = KERNEL_TEXTURE_4; > { Width = FFT_SIZE; Height = FFT_SIZE; Format = RGBA8; };
texture TexBloomKernel5 < source = KERNEL_TEXTURE_5; > { Width = FFT_SIZE; Height = FFT_SIZE; Format = RGBA8; };
texture TexBloomKernel6 < source = KERNEL_TEXTURE_6; > { Width = FFT_SIZE; Height = FFT_SIZE; Format = RGBA8; };
texture TexBloomKernel7 < source = KERNEL_TEXTURE_7; > { Width = FFT_SIZE; Height = FFT_SIZE; Format = RGBA8; };

sampler sBloomKernel0 { Texture = TexBloomKernel0; AddressU = WRAP; AddressV = WRAP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sBloomKernel1 { Texture = TexBloomKernel1; AddressU = WRAP; AddressV = WRAP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sBloomKernel2 { Texture = TexBloomKernel2; AddressU = WRAP; AddressV = WRAP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sBloomKernel3 { Texture = TexBloomKernel3; AddressU = WRAP; AddressV = WRAP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sBloomKernel4 { Texture = TexBloomKernel4; AddressU = WRAP; AddressV = WRAP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sBloomKernel5 { Texture = TexBloomKernel5; AddressU = WRAP; AddressV = WRAP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sBloomKernel6 { Texture = TexBloomKernel6; AddressU = WRAP; AddressV = WRAP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sBloomKernel7 { Texture = TexBloomKernel7; AddressU = WRAP; AddressV = WRAP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
#endif

#ifndef LENS_DIRT_TEXTURE

    #define LENS_DIRT_TEXTURE "LensDirt/dust.png"
#endif

#if KERNEL_MODE == 1
texture TexBakeA     { Width = BAKE_SIZE; Height = BAKE_SIZE; Format = SPEC_FMT; };

texture TexAperture  { Width = BAKE_SIZE; Height = BAKE_SIZE; Format = SPEC_FMT; };
texture TexPupil     { Width = BAKE_SIZE; Height = BAKE_SIZE; Format = RGBA16F; };
texture TexBakeState { Width = 1; Height = 1; Format = R32F; };

sampler sBakeState { Texture = TexBakeState; };

sampler sApertureKernel
{
    Texture   = TexAperture;
    AddressU  = WRAP;
    AddressV  = WRAP;
    MagFilter = LINEAR;
    MinFilter = LINEAR;
    MipFilter = POINT;
};

sampler sApertureKernelPoint
{
    Texture   = TexAperture;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
};

sampler sBakeSpec
{
    Texture   = TexBakeA;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
};

sampler sPupil
{
    Texture   = TexPupil;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
};

storage2D stBakeA       { Texture = TexBakeA; };
storage2D stAperture    { Texture = TexAperture; };
storage2D stPupil       { Texture = TexPupil; };
storage2D stBakeState   { Texture = TexBakeState; };
#endif

#if KERNEL_MODE == 1

// Width/Height must be declared: a source texture is created at the annotated size
// (1x1 by default) and the image is resized to it.
texture TexLensDirt < source = LENS_DIRT_TEXTURE; > { Width = 512; Height = 512; Format = RGBA8; };

sampler sLensDirt
{
    Texture   = TexLensDirt;
    AddressU  = WRAP;
    AddressV  = WRAP;
    MagFilter = LINEAR;
    MinFilter = LINEAR;
    MipFilter = POINT;
};
#endif

// ==========================================================================
//  maths and helpers
// ==========================================================================

float2 ComplexMul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

uint BitRev(uint i)
{
    return reversebits(i) >> (32u - LOG2_FFT);
}

uint BitRevBake(uint i)
{
    return reversebits(i) >> (32u - LOG2_BAKE);
}

float Luminance(float3 c)
{
    return dot(c, LUMA);
}

float3 LinearToSRGB(float3 c)
{
    c = max(c, 0.0);
    return pow(c, 1.0 / 2.2);
}

float3 SRGBToLinear(float3 c)
{
    c = max(c, 0.0);
    return pow(c, 2.2);
}

float2 ContentSize()
{
    float innerX = 1.0 - 2.0 * EDGE_PADDING;
    return float2(innerX, innerX / ReShade::AspectRatio);
}

float2 ContentMin()
{
    float2 size = ContentSize();
    return float2(EDGE_PADDING, 0.5 - size.y * 0.5);
}

float2 GridToScreen(float2 gridUV)
{
    return (gridUV - ContentMin()) / ContentSize();
}

float2 ScreenToGrid(float2 screenUV)
{
    return ContentMin() + screenUV * ContentSize();
}

float3 SampleSceneRaw(float2 uv)
{

    return tex2Dlod(ReShade::BackBuffer, float4(uv, 0.0, 0.0)).rgb;
}

float3 SceneLinear(float2 uv)
{
#ifdef SELFTEST_PATTERN

    return (length(uv - 0.5) < 0.004) ? float3(1.0, 1.0, 1.0) : float3(0.0, 0.0, 0.0);
#else
    float3 c = SampleSceneRaw(uv);
    return SRGBToLinear(c);
#endif
}

float3 PrefilterLinear(float3 linearColor, float2 uv)
{
    float3 c = linearColor * exp2(ExposureEV);

    if (HighlightShoulder > 0.0)
        c = c / (1.0 + HighlightShoulder * c);

    float L = Luminance(c);
    float w = 1.0;

    if (Threshold >= 0.0)
    {

        float knee = max(Threshold * SoftKnee, 1e-4);
        float rq = clamp(L - (Threshold - knee), 0.0, 2.0 * knee);
        rq = (0.25 / knee) * rq * rq;
        w = max(rq, L - Threshold) / max(L, 1e-5);
    }

    return max(c * w, 0.0);
}

#if KERNEL_MODE == 1
// ==========================================================================
//  procedural aperture
// ==========================================================================

void ApertureParams(out float sides, out float curvature)
{
    sides = (float)ApertureSides;
    curvature = ApertureBladeCurvature;

    if (AperturePreset == 1)
    {
        sides = 32.0;
        curvature = 1.0;
    }
    else if (AperturePreset >= 2)
    {
        sides = (float)(AperturePreset + 4);
    }

    sides = clamp(sides, 3.0, 64.0);
}

float ApertureMask(float2 d)
{
    float sides, curvature;
    ApertureParams(sides, curvature);

    float sector = TAU / sides;

    float radius = length(d) * 2.0 / max(ApertureSize, 1e-4);
    float th = atan2(d.y, d.x);

    float a = th - sector * floor(th / sector) - 0.5 * sector;

    float aa = a * (1.0 - 0.995 * saturate(curvature));
    float edge = cos(0.5 * sector) / max(cos(aa), 1e-3);

    float rn = radius / max(edge, 1e-4);

    float pixSoft = 3.0 / max(ApertureSize * (float)FFT_SIZE, 1.0);
    float soft = clamp(max(ApertureSoftness, pixSoft), 0.002, 0.9);
    float mask = 1.0 - smoothstep(1.0 - soft, 1.0 + 0.5 * soft, rn);

    float ring = saturate(ApertureRing);
    if (ring > 0.001)
        mask *= smoothstep(ring * 0.85, ring + 1e-4, rn);

    float falloff = saturate(ApertureFalloff);
    if (falloff > 0.0)
        mask *= lerp(1.0, exp(-2.5 * rn * rn), falloff);

    if (ApertureSpikes > 0)
    {
        float sn = (float)clamp(ApertureSpikes, 1, 32);
        float sSector = TAU / sn;
        float sa = th - sSector * floor(th / sSector + 0.5);
        float t = sa / (0.12 * sSector);
        float angFall = exp(-t * t);
        float radial = exp(-max(ApertureSpikeLength, 0.1) * radius);
        mask += saturate(ApertureSpikeStrength) * angFall * radial;
    }

    return max(mask, 0.0);
}

// ==========================================================================
//  kernel signature and sampling
// ==========================================================================

float BakeSignature()
{
    float s = 1.0;
    s = s * 3.17 + (float)AperturePreset;
    s = s * 3.17 + (float)ApertureSides;
    s = s * 3.17 + ApertureBladeCurvature;
    s = s * 3.17 + ApertureSoftness;
    s = s * 3.17 + ApertureFalloff;
    s = s * 3.17 + ApertureRing;
    s = s * 3.17 + (float)ApertureSpikes;
    s = s * 3.17 + ApertureSpikeStrength;
    s = s * 3.17 + ApertureSpikeLength;
    s = s * 3.17 + ApertureDispersion;
    s = s * 3.17 + ApertureSize;
    s = s * 3.17 + ApertureBeamGain;

    s = s * 3.17 + LensDirtStrength;
    s = s * 3.17 + LensDirtGain;
    s = s * 3.17 + LensDirtTiling;
    s = s * 3.17 + (float)BAKE_DISP_STEPS;
    return s;
}
#endif

bool BakeNeeded()
{
#if KERNEL_MODE == 1
    return tex2Dlod(sBakeState, float4(0.5, 0.5, 0.0, 0.0)).x != BakeSignature();
#else
    return false;
#endif
}

// Must list every uniform that changes the kernel grid, otherwise a control will
// appear to do nothing.
float KernelSignature()
{
    float s = 1.0;
    s = s * 3.17 + KernelScale;
    s = s * 3.17 + KernelRotation;
    s = s * 3.17 + AnamorphicStretch;
#if KERNEL_MODE == 0
    s = s * 3.17 + (float)KernelFile;
#else
    s = s * 3.17 + ApertureTint.r;
    s = s * 3.17 + ApertureTint.g;
    s = s * 3.17 + ApertureTint.b;

    s = s * 3.17 + BakeSignature();
#endif
    return s;
}

bool KernelNeeded()
{
    // TEMPORARY: returns true unconditionally. The cache is only safe once KernelPrep is a
    // compute pass - a pixel shader that returns without writing leaves its render targets
    // undefined, which zeroed the kernel spectrum while the cached passes skipped.
    return true;
}

float3 SampleKernel(float2 gridUV)
{
    float2 d = gridUV - 0.5;

    float ang = radians(-KernelRotation);
    float ca = cos(ang);
    float sa = sin(ang);
    d = float2(d.x * ca - d.y * sa, d.x * sa + d.y * ca);

    d /= max(KernelScale, 1e-4);
    d.y /= max(AnamorphicStretch, 1e-4);

#if KERNEL_MODE == 1

    float3 k = tex2Dlod(sApertureKernel, float4(frac(0.5 + d), 0.0, 0.0)).rgb;
    return max(k, 0.0) * max(ApertureTint, 0.0);
#else
    float2 kuv = frac(0.5 + d);

    float3 k;
    if      (KernelFile == 1) k = tex2Dlod(sBloomKernel1, float4(kuv, 0.0, 0.0)).rgb;
    else if (KernelFile == 2) k = tex2Dlod(sBloomKernel2, float4(kuv, 0.0, 0.0)).rgb;
    else if (KernelFile == 3) k = tex2Dlod(sBloomKernel3, float4(kuv, 0.0, 0.0)).rgb;
    else if (KernelFile == 4) k = tex2Dlod(sBloomKernel4, float4(kuv, 0.0, 0.0)).rgb;
    else if (KernelFile == 5) k = tex2Dlod(sBloomKernel5, float4(kuv, 0.0, 0.0)).rgb;
    else if (KernelFile == 6) k = tex2Dlod(sBloomKernel6, float4(kuv, 0.0, 0.0)).rgb;
    else if (KernelFile == 7) k = tex2Dlod(sBloomKernel7, float4(kuv, 0.0, 0.0)).rgb;
    else                      k = tex2Dlod(sBloomKernel0, float4(kuv, 0.0, 0.0)).rgb;
    return pow(max(k, 0.0), 2.2);
#endif
}

groupshared float2 g_fft0[GFFT_SIZE];
groupshared float2 g_fft1[GFFT_SIZE];

// ==========================================================================
//  FFT
// ==========================================================================

void FFTStages1N(uint lane, uint n, uint log2n, bool forward)
{
    uint halfSize = 1;
    for (uint s = 0; s < log2n; ++s)
    {
        for (uint idx = lane; idx < (n >> 1); idx += (n >> 1))
        {
            uint j  = idx & (halfSize - 1u);
            uint i0 = (idx << 1) - j;
            uint i1 = i0 + halfSize;

            float2 u = g_fft0[GIDX(i0)];
            float2 v = g_fft0[GIDX(i1)];

            float ang = (forward ? -TAU : TAU) * (float)j / (float)(halfSize << 1);
            float sn, cs;
            sincos(ang, sn, cs);

            float2 t = ComplexMul(float2(cs, sn), v);
            g_fft0[GIDX(i0)] = u + t;
            g_fft0[GIDX(i1)] = u - t;
        }
        halfSize <<= 1;
        barrier();
    }
}

void FFTStages2N(uint lane, uint n, uint log2n, bool forward)
{
    uint halfSize = 1;
    for (uint s = 0; s < log2n; ++s)
    {
        for (uint idx = lane; idx < (n >> 1); idx += (n >> 1))
        {
            uint j  = idx & (halfSize - 1u);
            uint i0 = (idx << 1) - j;
            uint i1 = i0 + halfSize;

            float ang = (forward ? -TAU : TAU) * (float)j / (float)(halfSize << 1);
            float sn, cs;
            sincos(ang, sn, cs);
            float2 w = float2(cs, sn);

            float2 u0 = g_fft0[GIDX(i0)];
            float2 v0 = g_fft0[GIDX(i1)];
            float2 t0 = ComplexMul(w, v0);
            g_fft0[GIDX(i0)] = u0 + t0;
            g_fft0[GIDX(i1)] = u0 - t0;

            float2 u1 = g_fft1[GIDX(i0)];
            float2 v1 = g_fft1[GIDX(i1)];
            float2 t1 = ComplexMul(w, v1);
            g_fft1[GIDX(i0)] = u1 + t1;
            g_fft1[GIDX(i1)] = u1 - t1;
        }
        halfSize <<= 1;
        barrier();
    }
}

void FFTStages1(uint lane, bool forward) { FFTStages1N(lane, FFT_SIZE, LOG2_FFT, forward); }
void FFTStages2(uint lane, bool forward) { FFTStages2N(lane, FFT_SIZE, LOG2_FFT, forward); }
void FFTStages2Bake(uint lane, bool forward) { FFTStages2N(lane, BAKE_SIZE, LOG2_BAKE, forward); }

// ==========================================================================
//  kernel spectrum
// ==========================================================================

void KernelPrepPS(float4 vpos : SV_Position, float2 uv : TEXCOORD0,
                  out float4 o0 : SV_Target0, out float4 o1 : SV_Target1)
{
    if (!KernelNeeded())
        return;

    float3 k = SampleKernel(uv);
    o0 = float4(k.r, 0.0, k.g, 0.0);
    o1 = float4(k.b, 0.0, 0.0, 0.0);
}

groupshared float2 g_kernel_stats[256];

[shader("compute")]
void KernelSumCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    if (!KernelNeeded())
        return;

    uint tid = gtid.x;

    float acc = 0.0;
    float mx  = 0.0;
    const uint count = FFT_SIZE * FFT_SIZE;
    for (uint i = tid; i < count; i += 256u)
    {
        int2 p = int2((int)(i % FFT_SIZE), (int)(i / FFT_SIZE));
        float4 a = tex2Dfetch(stKernelA, p);
        float4 b = tex2Dfetch(stKernelB, p);
        float v = (a.x + a.z + b.x) / 3.0;
        acc += v;
        mx = max(mx, max(a.x, max(a.z, b.x)));
    }

    g_kernel_stats[tid] = float2(acc, mx);
    barrier();

    for (uint s = 128u; s > 0u; s >>= 1)
    {
        if (tid < s)
        {
            g_kernel_stats[tid].x += g_kernel_stats[tid + s].x;
            g_kernel_stats[tid].y  = max(g_kernel_stats[tid].y, g_kernel_stats[tid + s].y);
        }
        barrier();
    }

    if (tid == 0)
    {
        float mean = g_kernel_stats[0].x / (float)count;
        float energy = (mean > 1e-8) ? (1.0 / ((float)FFT_SIZE * mean)) : 1.0;

        float peak = max(g_kernel_stats[0].y, 1e-8);
        tex2Dstore(stKernelEnergy, int2(0, 0), float4(energy, peak, KernelSignature(), 0.0));
    }
}

[shader("compute")]
void KernelFFTXCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    if (!KernelNeeded())
        return;

    uint row  = dtid.y;
    uint lane = gtid.x;

    for (uint k = lane; k < FFT_SIZE; k += FFT_THREADS)
    {
        float2 g = float2(((float)k + 0.5) * RCP_FFT_SIZE, ((float)row + 0.5) * RCP_FFT_SIZE);
        float3 kern = SampleKernel(g);
        g_fft0[GIDX(BitRev(k))] = float2(kern.r, 0.0);
        g_fft1[GIDX(BitRev(k))] = float2(kern.g, 0.0);
    }
    barrier();

    FFTStages2(lane, true);

    for (uint k2 = lane; k2 < FFT_SIZE; k2 += FFT_THREADS)
    {
        float2 r = g_fft0[GIDX(k2)] * INV_SQRT_N;
        float2 gg = g_fft1[GIDX(k2)] * INV_SQRT_N;
        tex2Dstore(stKernelA, int2((int)k2, (int)row), float4(r.x, r.y, gg.x, gg.y));
    }
    barrier();

    for (uint k3 = lane; k3 < FFT_SIZE; k3 += FFT_THREADS)
    {
        float2 g = float2(((float)k3 + 0.5) * RCP_FFT_SIZE, ((float)row + 0.5) * RCP_FFT_SIZE);
        float3 kern = SampleKernel(g);
        g_fft0[GIDX(BitRev(k3))] = float2(kern.b, 0.0);
    }
    barrier();

    FFTStages1(lane, true);

    for (uint k4 = lane; k4 < FFT_SIZE; k4 += FFT_THREADS)
    {
        float2 b = g_fft0[GIDX(k4)] * INV_SQRT_N;
        tex2Dstore(stKernelB, int2((int)k4, (int)row), float4(b.x, b.y, 0.0, 0.0));
    }
    barrier();
}

[shader("compute")]
void KernelFFTYCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    if (!KernelNeeded())
        return;

    uint col  = dtid.x;
    uint lane = gtid.y;

    for (uint k = lane; k < FFT_SIZE; k += FFT_THREADS)
    {
        float4 v = tex2Dfetch(stKernelA, int2((int)col, (int)k));
        g_fft0[GIDX(BitRev(k))] = v.xy;
        g_fft1[GIDX(BitRev(k))] = v.zw;
    }
    barrier();

    FFTStages2(lane, true);

    for (uint k2 = lane; k2 < FFT_SIZE; k2 += FFT_THREADS)
    {
        float2 r  = g_fft0[GIDX(k2)] * INV_SQRT_N;
        float2 gg = g_fft1[GIDX(k2)] * INV_SQRT_N;
        tex2Dstore(stKernelA, int2((int)col, (int)k2), float4(r.x, r.y, gg.x, gg.y));
    }
    barrier();

    for (uint k3 = lane; k3 < FFT_SIZE; k3 += FFT_THREADS)
        g_fft0[GIDX(BitRev(k3))] = tex2Dfetch(stKernelB, int2((int)col, (int)k3)).xy;
    barrier();

    FFTStages1(lane, true);

    for (uint k4 = lane; k4 < FFT_SIZE; k4 += FFT_THREADS)
    {
        float2 b = g_fft0[GIDX(k4)] * INV_SQRT_N;
        tex2Dstore(stKernelB, int2((int)col, (int)k4), float4(b.x, b.y, 0.0, 0.0));
    }
    barrier();
}

#if KERNEL_MODE == 1
[shader("compute")]
// ==========================================================================
//  aperture bake
// ==========================================================================

void BakeApertureCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    if (!BakeNeeded())
        return;

    int2 p = int2((int)dtid.x, (int)dtid.y);
    float2 d = ((float2)p + 0.5) * RCP_BAKE_SIZE - 0.5;

    float dirt = 1.0;
    if (LensDirtStrength > 0.0)
    {

        float2 duv = d / max(ApertureSize, 1e-4) * max(LensDirtTiling, 0.05) + 0.5;
        float dust = Luminance(tex2Dlod(sLensDirt, float4(duv, 0.0, 0.0)).rgb);

        dirt = lerp(1.0, saturate(dust * LensDirtGain), saturate(LensDirtStrength));
    }

    float2 o = RCP_BAKE_SIZE * 0.25;
    float2 dA = d + float2(-o.x, -o.y);
    float2 dB = d + float2( o.x, -o.y);
    float2 dC = d + float2(-o.x,  o.y);
    float2 dD = d + float2( o.x,  o.y);

    float mR = (ApertureMask(dA) + ApertureMask(dB)
              + ApertureMask(dC) + ApertureMask(dD)) * 0.25 * dirt;
    float mG = (ApertureMask(dA) + ApertureMask(dB)
              + ApertureMask(dC) + ApertureMask(dD)) * 0.25 * dirt;
    float mB = (ApertureMask(dA) + ApertureMask(dB)
              + ApertureMask(dC) + ApertureMask(dD)) * 0.25 * dirt;

    tex2Dstore(stBakeA, p, float4(mR, 0.0, mG, 0.0));

    tex2Dstore(stPupil, p, float4(mR, mG, mB, dirt));
}

[shader("compute")]
void BakeFFTXCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    if (!BakeNeeded())
        return;

    uint row  = dtid.y;
    uint lane = gtid.x;

    for (uint k = lane; k < BAKE_SIZE; k += BAKE_THREADS)
    {
        float4 v = tex2Dfetch(stBakeA, int2((int)k, (int)row));
        g_fft0[GIDX(BitRevBake(k))] = v.xy;
        g_fft1[GIDX(BitRevBake(k))] = v.zw;
    }
    barrier();

    FFTStages2Bake(lane, true);

    for (uint k2 = lane; k2 < BAKE_SIZE; k2 += BAKE_THREADS)
    {
        float2 r  = g_fft0[GIDX(k2)] * INV_SQRT_BAKE;
        float2 gg = g_fft1[GIDX(k2)] * INV_SQRT_BAKE;
        tex2Dstore(stBakeA, int2((int)k2, (int)row), float4(r.x, r.y, gg.x, gg.y));
    }
    barrier();
}

[shader("compute")]
void BakeFFTYCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    if (!BakeNeeded())
        return;

    uint col  = dtid.x;
    uint lane = gtid.y;

    for (uint k = lane; k < BAKE_SIZE; k += BAKE_THREADS)
    {
        float4 v = tex2Dfetch(stBakeA, int2((int)col, (int)k));
        g_fft0[GIDX(BitRevBake(k))] = v.xy;
        g_fft1[GIDX(BitRevBake(k))] = v.zw;
    }
    barrier();

    FFTStages2Bake(lane, true);

    for (uint k2 = lane; k2 < BAKE_SIZE; k2 += BAKE_THREADS)
    {
        float2 r  = g_fft0[GIDX(k2)] * INV_SQRT_BAKE;
        float2 gg = g_fft1[GIDX(k2)] * INV_SQRT_BAKE;
        tex2Dstore(stBakeA, int2((int)col, (int)k2), float4(r.x, r.y, gg.x, gg.y));
    }
    barrier();
}

float3 DispWeight(float t)
{
    float3 w = float3(saturate(1.0 - 2.0 * abs(t - 1.0)),
                      saturate(1.0 - 2.0 * abs(t - 0.5)),
                      saturate(1.0 - 2.0 * abs(t)));
    return w / max(w.r + w.g + w.b, 1e-3);
}

[shader("compute")]
void BakePSFCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    if (!BakeNeeded())
        return;

    int2  o   = int2((int)dtid.x, (int)dtid.y);
    float2 uvC = (float2(o) + 0.5) * RCP_BAKE_SIZE;
    float2 dC  = uvC - 0.5;

    float  disp = saturate(ApertureDispersion);
    float3 acc  = 0.0;

    int steps = BAKE_DISP_STEPS;
    for (int i = 0; i < steps; i++)
    {
        // Dispersion is a radial rescale of the PSF, so the finished pattern is
        // resampled instead of recomputed per wavelength.
        float t = (steps > 1) ? ((float)i / (float)(steps - 1)) : 0.5;
        float s = 1.0 + disp * (2.0 * t - 1.0);
        s = max(s, 0.05);

        float2 uvS = frac(dC / s);
        int2   b2  = int2(uvS * BAKE_SIZE) & int2(BAKE_SIZE - 1, BAKE_SIZE - 1);

        float4 a = tex2Dfetch(stBakeA, b2);
        float  m2 = dot(a.xy, a.xy) / (s * s);

        acc += DispWeight(t) * m2;
    }

    float3 psf = pow(max(acc, 0.0), 1.0 / max(ApertureBeamGain, 1.0));

    tex2Dstore(stAperture, o, float4(psf, 1.0));
}

[shader("compute")]
void BakeCommitCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    if (!BakeNeeded())
        return;

    tex2Dstore(stBakeState, int2(0, 0), float4(BakeSignature(), 0.0, 0.0, 0.0));
}
#endif

[shader("compute")]
// ==========================================================================
//  image FFT
// ==========================================================================

void ImageFFTXCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    uint row  = dtid.y;
    uint lane = gtid.x;

    for (uint k = lane; k < FFT_SIZE; k += FFT_THREADS)
    {
        float2 g = float2(((float)k + 0.5) * RCP_FFT_SIZE, ((float)row + 0.5) * RCP_FFT_SIZE);
        float2 s = GridToScreen(g);

        float3 c = float3(0.0, 0.0, 0.0);
        if (s.x >= 0.0 && s.x <= 1.0 && s.y >= 0.0 && s.y <= 1.0)
            c = PrefilterLinear(SceneLinear(s), s);

        g_fft0[GIDX(BitRev(k))] = float2(c.r, 0.0);
        g_fft1[GIDX(BitRev(k))] = float2(c.g, 0.0);
    }
    barrier();

    FFTStages2(lane, true);

    for (uint k2 = lane; k2 < FFT_SIZE; k2 += FFT_THREADS)
    {
        float2 r  = g_fft0[GIDX(k2)] * INV_SQRT_N;
        float2 gg = g_fft1[GIDX(k2)] * INV_SQRT_N;
        tex2Dstore(stImageA, int2((int)k2, (int)row), float4(r.x, r.y, gg.x, gg.y));
    }
    barrier();


    for (uint k3 = lane; k3 < FFT_SIZE; k3 += FFT_THREADS)
    {
        float2 g = float2(((float)k3 + 0.5) * RCP_FFT_SIZE, ((float)row + 0.5) * RCP_FFT_SIZE);
        float2 s = GridToScreen(g);

        float3 c = float3(0.0, 0.0, 0.0);
        if (s.x >= 0.0 && s.x <= 1.0 && s.y >= 0.0 && s.y <= 1.0)
            c = PrefilterLinear(SceneLinear(s), s);

        g_fft0[GIDX(BitRev(k3))] = float2(c.b, 0.0);
    }
    barrier();

    FFTStages1(lane, true);

    for (uint k4 = lane; k4 < FFT_SIZE; k4 += FFT_THREADS)
    {
        float2 b = g_fft0[GIDX(k4)] * INV_SQRT_N;
        tex2Dstore(stImageB, int2((int)k4, (int)row), float4(b.x, b.y, 0.0, 0.0));
    }
    barrier();
}

[shader("compute")]
void ImageFFTYCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    uint col  = dtid.x;
    uint lane = gtid.y;

    for (uint k = lane; k < FFT_SIZE; k += FFT_THREADS)
    {
        float4 v = tex2Dfetch(stImageA, int2((int)col, (int)k));
        g_fft0[GIDX(BitRev(k))] = v.xy;
        g_fft1[GIDX(BitRev(k))] = v.zw;
    }
    barrier();

    FFTStages2(lane, true);

    for (uint k2 = lane; k2 < FFT_SIZE; k2 += FFT_THREADS)
    {
        float2 r  = g_fft0[GIDX(k2)] * INV_SQRT_N;
        float2 gg = g_fft1[GIDX(k2)] * INV_SQRT_N;
        tex2Dstore(stImageA, int2((int)col, (int)k2), float4(r.x, r.y, gg.x, gg.y));
    }
    barrier();

    for (uint k3 = lane; k3 < FFT_SIZE; k3 += FFT_THREADS)
        g_fft0[GIDX(BitRev(k3))] = tex2Dfetch(stImageB, int2((int)col, (int)k3)).xy;
    barrier();

    FFTStages1(lane, true);

    for (uint k4 = lane; k4 < FFT_SIZE; k4 += FFT_THREADS)
    {
        float2 b = g_fft0[GIDX(k4)] * INV_SQRT_N;
        tex2Dstore(stImageB, int2((int)col, (int)k4), float4(b.x, b.y, 0.0, 0.0));
    }
    barrier();
}

[shader("compute")]
void IFFTXCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    uint row  = dtid.y;
    uint lane = gtid.x;

    float energy = tex2Dlod(sKernelEnergy, float4(0.5, 0.5, 0.0, 0.0)).x;

    for (uint k = lane; k < FFT_SIZE; k += FFT_THREADS)
    {
        float4 ia = tex2Dfetch(stImageA, int2((int)k, (int)row));
        float4 ka = tex2Dfetch(stKernelA, int2((int)k, (int)row));

        float sgn = (((k + row) & 1u) == 0u) ? 1.0 : -1.0;
        g_fft0[GIDX(BitRev(k))] = ComplexMul(ia.xy, ka.xy) * (energy * sgn);
        g_fft1[GIDX(BitRev(k))] = ComplexMul(ia.zw, ka.zw) * (energy * sgn);
    }
    barrier();

    FFTStages2(lane, false);

    for (uint k2 = lane; k2 < FFT_SIZE; k2 += FFT_THREADS)
    {
        float2 r  = g_fft0[GIDX(k2)] * INV_SQRT_N;
        float2 gg = g_fft1[GIDX(k2)] * INV_SQRT_N;
        tex2Dstore(stImageA, int2((int)k2, (int)row), float4(r.x, r.y, gg.x, gg.y));
    }
    barrier();

    for (uint k3 = lane; k3 < FFT_SIZE; k3 += FFT_THREADS)
    {
        float4 ib = tex2Dfetch(stImageB, int2((int)k3, (int)row));
        float4 kb = tex2Dfetch(stKernelB, int2((int)k3, (int)row));
        float sgn = (((k3 + row) & 1u) == 0u) ? 1.0 : -1.0;
        g_fft0[GIDX(BitRev(k3))] = ComplexMul(ib.xy, kb.xy) * (energy * sgn);
    }
    barrier();

    FFTStages1(lane, false);

    for (uint k4 = lane; k4 < FFT_SIZE; k4 += FFT_THREADS)
    {
        float2 b = g_fft0[GIDX(k4)] * INV_SQRT_N;
        tex2Dstore(stImageB, int2((int)k4, (int)row), float4(b.x, b.y, 0.0, 0.0));
    }
    barrier();
}

[shader("compute")]
void IFFTYCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    uint col  = dtid.x;
    uint lane = gtid.y;

    for (uint k = lane; k < FFT_SIZE; k += FFT_THREADS)
    {
        float4 v = tex2Dfetch(stImageA, int2((int)col, (int)k));
        g_fft0[GIDX(BitRev(k))] = v.xy;
        g_fft1[GIDX(BitRev(k))] = v.zw;
    }
    barrier();

    FFTStages2(lane, false);

    for (uint k2 = lane; k2 < FFT_SIZE; k2 += FFT_THREADS)
    {
        float2 r  = g_fft0[GIDX(k2)] * INV_SQRT_N;
        float2 gg = g_fft1[GIDX(k2)] * INV_SQRT_N;
        tex2Dstore(stImageA, int2((int)col, (int)k2), float4(r.x, r.y, gg.x, gg.y));
    }
    barrier();

    for (uint k3 = lane; k3 < FFT_SIZE; k3 += FFT_THREADS)
        g_fft0[GIDX(BitRev(k3))] = tex2Dfetch(stImageB, int2((int)col, (int)k3)).xy;
    barrier();

    FFTStages1(lane, false);

    for (uint k4 = lane; k4 < FFT_SIZE; k4 += FFT_THREADS)
    {
        float2 b = g_fft0[GIDX(k4)] * INV_SQRT_N;
        tex2Dstore(stImageB, int2((int)col, (int)k4), float4(b.x, b.y, 0.0, 0.0));
    }
    barrier();
}

// ==========================================================================
//  composite
// ==========================================================================

float3 BloomAt(float2 gridUV)
{
    float4 a = tex2D(sImageA, gridUV);
    float4 b = tex2D(sImageB, gridUV);
    return float3(a.x, a.z, b.x);
}

float3 BloomAtFiltered(float2 gridUV)
{

    return BloomAt(gridUV);
}

void CompositePS(float4 vpos : SV_Position, float2 uv : TEXCOORD0, out float4 o : SV_Target0)
{
    float2 gridUV = ScreenToGrid(uv);

    float3 bloom = BloomAtFiltered(gridUV);
    bloom *= exp2(-ExposureEV);
    bloom *= BloomTint;

    float3 scene = SceneLinear(uv);
    float3 outc;

    if (BlendMode == 0)
        outc = lerp(scene, bloom, saturate(BloomIntensity));
    else if (BlendMode == 1)
        outc = scene + bloom * BloomIntensity;
    else
        outc = 1.0 - (1.0 - scene) * (1.0 - saturate(bloom * BloomIntensity));

    if (DebugView == 1)
    {
        outc = bloom;
    }
    else if (DebugView == 2)
    {
        float2 s = GridToScreen(uv);
        outc = (s.x >= 0.0 && s.x <= 1.0 && s.y >= 0.0 && s.y <= 1.0)
             ? PrefilterLinear(SceneLinear(s), s)
             : float3(0.0, 0.0, 0.0);
    }
    else if (DebugView == 3)
    {

#if KERNEL_MODE == 1
        float2 kuv = 0.5 + (uv - 0.5) * 0.5;
        float3 ks  = tex2Dlod(sApertureKernelPoint, float4(kuv, 0.0, 0.0)).rgb;
        float3 kc  = tex2Dlod(sApertureKernelPoint, float4(0.5, 0.5, 0.0, 0.0)).rgb;
        float  kp  = max(max(kc.r, max(kc.g, kc.b)), 1e-12);
        float3 kv  = saturate(ks / kp);
        kv = pow(max(kv, 0.0), max(ApertureBeamGain, 1.0));
        outc = pow(saturate(kv), 1.0 / 2.2);
#else
        // Same framing as the aperture view: 2x zoom in kernel space, normalised by the
        // kernel peak. Sampling with screen UV stretched the pattern by the content rect,
        // which is why the DDS kernel looked wrong in this view.
        float2 kuv = 0.5 + (uv - 0.5) * 0.5;
        float  kp  = max(tex2Dlod(sKernelEnergy, float4(0.5, 0.5, 0.0, 0.0)).y, 1e-12);
        float3 kv  = saturate(max(SampleKernel(kuv), 0.0) / kp);
        outc = pow(saturate(kv), 1.0 / 2.2);
#endif
    }
    else if (DebugView == 4)
    {
        float4 a = tex2D(sImageA, uv);
        float4 b = tex2D(sImageB, uv);
        float m = length(a.xy) + length(a.zw) + length(b.xy);
        outc = log(1.0 + 5.0 * m).xxx;
    }
    else if (DebugView == 7)
    {
#if KERNEL_MODE == 1

        outc = saturate(tex2D(sPupil, ScreenToGrid(uv)).rgb);
#else
        outc = float3(0.0, 0.0, 0.0);
#endif
    }
    else if (DebugView == 8)
    {
#if KERNEL_MODE == 1

        float2 uv8 = frac(uv + 0.5);
        float4 s8  = tex2Dlod(sBakeSpec, float4(uv8, 0.0, 0.0));
        float  m8  = length(s8.xy) + length(s8.zw);
        outc = saturate(log(1.0 + 200.0 * m8) / log(201.0)).xxx;
#else
        outc = float3(0.0, 0.0, 0.0);
#endif
    }
    else if (DebugView == 6)
    {

        float2 q = uv * 2.0;
        if (q.x < 1.0 && q.y < 1.0)      outc = float3(1.0, 0.0, 0.0);
        else if (q.x >= 1.0 && q.y < 1.0) outc = float3(0.0, 1.0, 0.0);
        else if (q.x < 1.0 && q.y >= 1.0) outc = float3(0.0, 0.0, 1.0);
        else                              outc = float3(1.0, 1.0, 1.0);
    }
    else if (DebugView == 5)
    {

        float2 q5 = uv * 2.0;
        float2 l = frac(q5);

        if (q5.x < 1.0 && q5.y < 1.0)
        {
#if KERNEL_MODE == 1

            float2 kuv5 = 0.5 + (l - 0.5) * 0.5;
            float3 ks5  = tex2Dlod(sApertureKernelPoint, float4(kuv5, 0.0, 0.0)).rgb;
            float3 kc5  = tex2Dlod(sApertureKernelPoint, float4(0.5, 0.5, 0.0, 0.0)).rgb;
            float  kp5  = max(max(kc5.r, max(kc5.g, kc5.b)), 1e-12);
            float3 kv5  = pow(saturate(ks5 / kp5), max(ApertureBeamGain, 1.0));
            outc = pow(saturate(kv5), 1.0 / 2.2);
#else
            outc = pow(saturate(max(SampleKernel(l), 0.0)), 1.0 / 2.2);
#endif
        }
        else if (q5.x >= 1.0 && q5.y < 1.0)
            outc = max(BloomAt(l), 0.0);
        else if (q5.x < 1.0 && q5.y >= 1.0)
        {
            float4 a = tex2D(sImageA, l);
            float4 b = tex2D(sImageB, l);
            float m = length(a.xy) + length(a.zw) + length(b.xy);
            outc = log(1.0 + 5.0 * m).xxx;
        }
        else
        {

            float pk5 = tex2Dlod(sKernelEnergy, float4(0.5, 0.5, 0.0, 0.0)).y;
            outc = saturate((log10(max(pk5, 1e-9)) + 9.0) / 9.0).xxx;
        }
    }

    outc = max(outc, 0.0);
    outc = LinearToSRGB(outc);

    o = float4(outc, 1.0);
}

#if KERNEL_MODE == 1
    #define TECHNIQUE_NAME  UE_FFTBloom_Aperture
    #define TECHNIQUE_LABEL "UE FFT Bloom - Procedural Aperture"
    #define TECHNIQUE_TIP \
        "Unreal-style FFT convolution bloom driven by an analytic aperture.\n" \
        "\n" \
        "The PSF (blade shape, curvature, ring, spikes, dispersion) is baked on\n" \
        "the GPU once per settings change, then the frame is convolved with it\n" \
        "in the frequency domain, so cost is independent of kernel size.\n" \
        "Threshold -1 = physically based (everything blooms).\n" \
        "\n" \
        "Debug view 3 shows the baked kernel, 2 the padding, 5 the quadrants."
#else
    #define TECHNIQUE_NAME  UE_FFTBloom
    #define TECHNIQUE_LABEL "UE FFT Convolution Bloom - Kernel Texture"
    #define TECHNIQUE_TIP \
        "Unreal-style FFT convolution bloom with a kernel texture.\n" \
        "\n" \
        "Convolves the frame with the PSF selected in 'Kernel File'\n" \
        "(reshade-shaders/Textures/Kernels) in the frequency domain, giving\n" \
        "arbitrary bloom shapes (starbursts, diffusion) at a cost independent\n" \
        "of kernel size. Threshold -1 = physically based (everything blooms).\n" \
        "\n" \
        "Debug views 2/3 verify padding and the fftshifted kernel.\n" \
        "For a fully procedural PSF enable UE_FFTBloom_Aperture instead."
#endif

// ==========================================================================
//  passes
// ==========================================================================

technique TECHNIQUE_NAME <
    ui_label = TECHNIQUE_LABEL;
    ui_tooltip = TECHNIQUE_TIP;
>
{
#if KERNEL_MODE == 1
    pass BakeAperture { ComputeShader = BakeApertureCS<16, 16, 1>; DispatchSizeX = BAKE_SIZE / 16; DispatchSizeY = BAKE_SIZE / 16; DispatchSizeZ = 1; }
    pass BakeFFTX     { ComputeShader = BakeFFTXCS<BAKE_THREADS, 1, 1>; DispatchSizeX = 1;         DispatchSizeY = BAKE_SIZE; DispatchSizeZ = 1; }
    pass BakeFFTY     { ComputeShader = BakeFFTYCS<1, BAKE_THREADS, 1>; DispatchSizeX = BAKE_SIZE; DispatchSizeY = 1;         DispatchSizeZ = 1; }
    pass BakePSF      { ComputeShader = BakePSFCS<16, 16, 1>; DispatchSizeX = BAKE_SIZE / 16; DispatchSizeY = BAKE_SIZE / 16; DispatchSizeZ = 1; }
    pass BakeCommit   { ComputeShader = BakeCommitCS<1, 1, 1>; DispatchSizeX = 1; DispatchSizeY = 1; DispatchSizeZ = 1; }
#endif

    pass KernelPrep  { VertexShader = PostProcessVS; PixelShader = KernelPrepPS; RenderTarget0 = TexKernelA; RenderTarget1 = TexKernelB; }
    pass KernelSum   { ComputeShader = KernelSumCS<256, 1, 1>; DispatchSizeX = 1; DispatchSizeY = 1; DispatchSizeZ = 1; }

    pass KernelFFTX  { ComputeShader = KernelFFTXCS<FFT_THREADS, 1, 1>; DispatchSizeX = 1;         DispatchSizeY = FFT_SIZE; DispatchSizeZ = 1; }
    pass KernelFFTY  { ComputeShader = KernelFFTYCS<1, FFT_THREADS, 1>; DispatchSizeX = FFT_SIZE;   DispatchSizeY = 1;        DispatchSizeZ = 1; }

    pass ImageFFTX   { ComputeShader = ImageFFTXCS<FFT_THREADS, 1, 1>;  DispatchSizeX = 1;         DispatchSizeY = FFT_SIZE; DispatchSizeZ = 1; }
    pass ImageFFTY   { ComputeShader = ImageFFTYCS<1, FFT_THREADS, 1>;  DispatchSizeX = FFT_SIZE;   DispatchSizeY = 1;        DispatchSizeZ = 1; }

    pass IFFTX       { ComputeShader = IFFTXCS<FFT_THREADS, 1, 1>;      DispatchSizeX = 1;         DispatchSizeY = FFT_SIZE; DispatchSizeZ = 1; }
    pass IFFTY       { ComputeShader = IFFTYCS<1, FFT_THREADS, 1>;      DispatchSizeX = FFT_SIZE;   DispatchSizeY = 1;        DispatchSizeZ = 1; }

    pass Composite   { VertexShader = PostProcessVS; PixelShader = CompositePS; }
}
