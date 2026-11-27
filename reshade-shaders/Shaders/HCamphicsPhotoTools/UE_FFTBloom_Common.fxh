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

    Blend Mode decides how the glow meets the scene: Additive (the default) is the
    physical one - scattered light adds to the image - while Lerp is Unreal's
    artistic composite, where Intensity mixes scene into glow. Beam Emphasis is
    artistic, not optical; the rest of the aperture group is aperture geometry.

    Limits
        - Bloom runs on the tonemapped LDR frame, so clipped highlights have
          already lost their headroom. HDR input is a TODO (untestable here).
        - Requires compute shaders: D3D11/12, Vulkan, OpenGL 4.3+. D3D9/D3D10
          cannot load it.
        - Kernels must be square, power-of-two, 8-bit BGRA DDS.

    TODO
        - HDR input path (no HDR display to test on)
        - one-sided anamorphic streak, depth-faded bloom

==============================================================================*/
#include "ReShade.fxh"
#include "ReShadeUI.fxh"

// ==========================================================================
//  preprocessor configuration
// ==========================================================================

#ifndef KERNEL_MODE

    #define KERNEL_MODE 0
#endif

// Quality tier. Anything that adds or removes passes, textures or dispatch sizes has to be gated
// here, because ReShade builds a fixed pass list at load time and a uniform can never skip a pass.
// Set it per effect: ReShade overlay -> the effect's ... menu -> Edit preprocessor definitions, or
// the preset line PreprocessorDefinitions=FTB_QUALITY=1,
#ifndef FTB_QUALITY
    #define FTB_QUALITY 0
#endif

// Dispersion samples used when baking the procedural PSF. Bake-time only, so a higher count costs
// a slower first bake and never frame time.
#if FTB_QUALITY >= 1
    #define BAKE_DISP_STEPS 64
#else
    #define BAKE_DISP_STEPS 16
#endif

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
// Threads in the kernel-sum reduction. Changing it means changing one line, but it has to stay in
// step everywhere it is used: the groupshared array, the loop stride, the reduction start and the
// group size in the pass below. A mismatch is silently wrong, not a compile error.
#define KERNEL_SUM_THREADS 256
#define DUST_SUM_THREADS 256
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
    ui_tooltip = "Luminance above which pixels bloom. -1 blooms everything, which is Unreal's default.";
> = -1.0;

uniform float SoftKnee <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Bloom";
    ui_label = "Soft Knee";
    ui_tooltip = "Softens the threshold transition. Inactive while Threshold is -1.";
> = 0.5;

uniform float HighlightShoulder <
    ui_type = "drag"; ui_min = 0.0; ui_max = 0.9; ui_step = 0.01;
    ui_category = "Bloom";
    ui_label = "Highlight Shoulder";
    ui_tooltip = "Softens how much a clipped white blooms compared to a brighter source. 0 = off; 0.2-0.5 suits LDR games.";
> = 0.0;

uniform int BlendMode <
    ui_type = "combo";
    ui_items = "Lerp (Unreal Style)\0Additive\0Screen\0";
    ui_category = "Bloom";
    ui_label = "Blend Mode";
    ui_tooltip = "Additive adds the glow, which is physical but clips to white early. Lerp mixes toward it. Screen brightens without clipping.";
> = 1;

uniform float KernelScale <
    ui_type = "drag"; ui_min = 0.05; ui_max = 3.0; ui_step = 0.01;
    ui_category = "Kernel";
    ui_label = "Scale";
    ui_tooltip = "Zooms the PSF in the grid: above 1 gives a larger, softer glow but crops the outer rays; below 1 keeps them all.";
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
    ui_tooltip = "DDS kernel to convolve with, from Textures/Kernels. The list is declared at the top of the shader.";
> = 0;
#endif

#if KERNEL_MODE == 1
uniform int ApertureSides <
    ui_type = "drag"; ui_min = 3; ui_max = 32; ui_step = 1;
    ui_category = "Aperture";
    ui_label = "Sides";
    ui_tooltip = "Blade count. Fewer sides with 0 curvature give sharp corners and stronger spikes; 32 with curvature 1 is a circle.";
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
    ui_tooltip = "Softness of the aperture edge. Below about 0.03 the rays stop getting brighter and only the haze grows.";
> = 0.03;

uniform float ApertureRim <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Aperture";
    ui_label = "Soap Bubble Rim";
    ui_tooltip = "Soap-bubble bokeh: weights the pupil toward its edge, giving a bright rim around a softer interior. 0 = flat disc.";
> = 0.0;

uniform float ApertureCentre <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Aperture";
    ui_label = "Pupil Centre";
    ui_tooltip = "Hot-centre bokeh: weights the pupil toward its middle, fading outward. 0 = flat disc.";
> = 0.0;

uniform float ApertureRing <
    ui_type = "drag"; ui_min = 0.0; ui_max = 0.95; ui_step = 0.01;
    ui_category = "Aperture";
    ui_label = "Ring";
    ui_tooltip = "Removes the aperture centre, giving the donut bokeh of a catadioptric lens.";
> = 0.0;

uniform float ApertureCatEye <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Aperture";
    ui_label = "Cat Eye";
    ui_tooltip = "Cat's-eye pupils, as a fast lens gives at the frame corners. 1 is about 60% round; stopping down removes it. 0 = round.";
> = 0.0;

uniform float ApertureCatEyeAngle <
    ui_type = "drag"; ui_min = -180.0; ui_max = 180.0; ui_step = 1.0;
    ui_category = "Aperture";
    ui_label = "Cat Eye Angle";
    ui_tooltip = "Direction the cat's-eye bites. One kernel serves the whole frame, so it cannot follow the field angle.";
> = 0.0;

uniform float ApertureDispersion <
    ui_type = "drag"; ui_min = 0.0; ui_max = 0.5; ui_step = 0.005;
    ui_category = "Aperture";
    ui_label = "Dispersion";
    ui_tooltip = "Splits the pupil radius per colour so rays fan into a spectrum. 0 = no colour; 0.1-0.2 is a visible rainbow.";
> = 0.12;

uniform float ApertureSize <
    ui_type = "drag"; ui_min = 0.005; ui_max = 0.6; ui_step = 0.001;
    ui_category = "Aperture";
    ui_label = "Size";
    ui_tooltip = "Pupil size on the grid. Smaller means a wider glow, like stopping down; below 0.2 it aliases. 0.35-0.5 is clean.";
> = 0.35;

uniform float ApertureBeamGain <
    ui_type = "drag"; ui_min = 1.0; ui_max = 8.0; ui_step = 0.05;
    ui_category = "Aperture";
    ui_label = "Beam Emphasis";
    ui_tooltip = "Lifts the diffraction rays out of the black. 2 puts them near 0.5% of the core; 3+ turns the tail into grey haze.";
> = 2.0;
#endif

uniform float3 BloomTint <
    ui_type = "color";
    ui_category = "Color";
    ui_label = "Tint";
    ui_tooltip = "Colours the glow. White keeps the kernel's own chromatic diffraction intact.";
> = float3(1.0, 1.0, 1.0);

// Post-processing of the finished kernel, so it sits next to Tint rather than with the controls that
// shape the pupil. Per channel: dispersion gives the kernel its colour, and a grey Levels curve does
// not flatten that - it moves each channel on its own, so saturation shifts as it does on a photo.
uniform float KernelDehaze <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Color";
    ui_label = "Kernel Dehaze";
    ui_tooltip = "Deepens the kernel with a Photoshop Levels midtone pull, per channel, so colour survives. Artistic. 0 = off.";
> = 0.0;

#if KERNEL_MODE == 1

uniform float LensDirtStrength <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Lens Dirt";
    ui_label = "Lens Dirt";
    ui_tooltip = "How strongly the dust map dims the pupil. With Dust Scatter up, the blocked light returns as glare. 0 = clean lens.";
> = 0.0;

uniform float LensDirtGain <
    ui_type = "drag"; ui_min = 1.0; ui_max = 8.0; ui_step = 0.1;
    ui_category = "Lens Dirt";
    ui_label = "Dust Gain";
    ui_tooltip = "Brightens the dust map before it dims the pupil. 4 keeps the clean field clear and darkens only the specks.";
> = 4.0;

uniform float LensDirtScatter <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Lens Dirt";
    ui_label = "Dust Scatter";
    ui_tooltip = "Turns the light the dust blocks into a faint veil plus a wide halo instead of only dimming. Baked once. 0 = off.";
> = 0.0;

uniform float LensDirtTiling <
    ui_type = "drag"; ui_min = 0.25; ui_max = 4.0; ui_step = 0.05;
    ui_category = "Lens Dirt";
    ui_tooltip = "Repeats the dust map. Above 1 each tile is shuffled so the repeat is hidden; 1 is one continuous pattern.";
    ui_label = "Dust Tiling";
> = 1.0;
#endif

uniform int DebugView <
    ui_type = "combo";
    ui_items = "Final Composite\0Bloom Only\0Pre-Filtered Input\0Pupil + Kernel\0Spectrum Magnitude\0Quadrants\0Bake Spectrum\0";
    ui_category = "Debug";
    ui_category_closed = true;
    ui_label = "Debug View";
    ui_tooltip = "3 shows the pupil and the kernel side by side. Leave at Final Composite for normal play.";
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

storage2D stImageA  { Texture = TexImageA;  };
storage2D stImageB  { Texture = TexImageB;  };
storage2D stKernelA { Texture = TexKernelA; };
storage2D stKernelB { Texture = TexKernelB; };

texture TexKernelEnergy { Width = 1; Height = 1; Format = RGBA32F; };
sampler sKernelEnergy   { Texture = TexKernelEnergy; };
storage2D stKernelEnergy { Texture = TexKernelEnergy; };

// The signature lives in its own 1x1 texture so no pass ever reads a texel it also writes.
// Having KernelFFTYCS read TexKernelEnergy and write it back in the same pass is undefined
// behaviour in D3D11 (the SRV is bound alongside the UAV) and zeroed energy and peak under
// DXVK, which made the whole bloom composite multiply by zero.
texture TexKernelState  { Width = 1; Height = 1; Format = RGBA32F; };
sampler sKernelState    { Texture = TexKernelState; };
storage2D stKernelState { Texture = TexKernelState; };

// Mean fraction of the aperture blocked by dust, reduced once per bake and used by the PSF pass to
// size the veiling-glare wing. Kept in its own texture because TexBakeState is read by BakeNeeded()
// in every bake pass, and writing a texel a pass also reads is undefined in D3D11.
texture TexDustMean { Width = 1; Height = 1; Format = RGBA16F; };
sampler sDustMean   { Texture = TexDustMean; };
storage2D stDustMean { Texture = TexDustMean; };

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
texture TexLensDirt < source = LENS_DIRT_TEXTURE; > { Width = 1024; Height = 1024; Format = RGBA8; };

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

// Signature hash for the caches below. A slider step must move the result by far more than float
// noise: the smallest step contributes ~2e-4, while a difference in FMA contraction between two
// entry points is ~1e-7. The comparison is tolerant, so exact bit reproducibility is not needed.
// The earlier s = s*3.17 + v chain grew to ~3e6, where one ulp is 0.25 and the last parameters
// moved the result by only 1-2 ulp - too close to the noise floor for a cache to compare.
float HashStep(float s, float v)
{
    return frac(s * 3.17 + v * 0.7331);
}

bool SignatureChanged(float stored, float fresh)
{
    return abs(stored - fresh) > 1e-5;
}

float2 ComplexMul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

// Bit reversal by hand rather than through the reversebits intrinsic: ReShade 6.8.0.2's parser
// rejects it ("Function "reversebits" is not defined") even though its code generator emits it, and
// the FFT only ever needs the low LOG2_FFT bits reversed. Unrolled, so it costs nothing at runtime.
uint BitRev(uint i)
{
    uint r = 0u;
    [unroll]
    for (uint b = 0u; b < LOG2_FFT; ++b)
    {
        r = (r << 1u) | (i & 1u);
        i >>= 1u;
    }
    return r;
}

uint BitRevBake(uint i)
{
    uint r = 0u;
    [unroll]
    for (uint b = 0u; b < LOG2_BAKE; ++b)
    {
        r = (r << 1u) | (i & 1u);
        i >>= 1u;
    }
    return r;
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
    // sopt found the clamp is dead code: on an 8-bit SDR swapchain every channel is already in
    // [0,1]. Kept behind the guard so an HDR swapchain still clamps.
#if BUFFER_COLOR_SPACE <= 1
    return pow(c, 2.2);
#else
    return pow(max(c, 0.0), 2.2);
#endif
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
    // No exposure gain here: multiplying before the threshold and dividing again in the composite
    // cancels exactly once Threshold is -1, so the control would look inert at its default.
    float3 c = linearColor;

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

    // sopt: c is in [0,1] and w in [0, 1.7e10], so the clamp can never trigger (bit-exact).
    return c * w;
}

#if KERNEL_MODE == 1
// ==========================================================================
//  procedural aperture
// ==========================================================================

void ApertureParams(out float sides, out float curvature)
{
    sides = clamp((float)ApertureSides, 3.0, 32.0);
    curvature = ApertureBladeCurvature;
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

    // Apodization: the pupil's transmission profile. Real bokeh kernels show a narrow bright band at
    // the pupil edge (under-corrected spherical aberration reads as a soap-bubble rim) and, on the
    // other end, a broad centre glow. A narrow rim band plus a wide centre glow fitted six measured
    // lens kernels to a mean profile error of 5.8%: band width 0.10 of the radius at 1.8x, centre
    // width 0.55 at up to full removal of the rim. Applied to the mask, so it composes with blades,
    // ring, cat's eye and dispersion for free. Both at zero leaves the disc flat and the shader inert.
    float rim = saturate(ApertureRim);
    float cen = saturate(ApertureCentre);
    if (rim > 0.0 || cen > 0.0)
    {
        float band = exp(-(1.0 - rn) * (1.0 - rn) * 100.0);   // exp(-((1-rn)/0.10)^2)
        float hot  = exp(-rn * rn * 3.3058);                  // exp(-(rn/0.55)^2)
        mask *= max(1.0 + 1.8 * rim * band - cen * (1.0 - hot), 0.02);
    }

    // Mechanical vignetting: at a field angle the barrel rims project onto the pupil, and what
    // survives is the overlap of the pupil with an offset circle of the same size - a vesica, not a
    // disc. lenssimulator.photo ray-traces published patents and measures the same thing: a Helios
    // 44-2 is 61% round in the 90% corner at f/2, and 95% at f/4, because stopping down pulls the
    // iris inside the rims.
    float catEye = saturate(ApertureCatEye);
    if (catEye > 0.0)
    {
        // Offset in pupil radii. The range covers the traced corners: 0.62 is a Helios 44-2 at f/2
        // (61% of the pupil left), 0.72 a Noct 58/0.95 (55%), 0.85 a Canon RF 85/1.2 (48%).
        // Stopping down pulls the iris inside the rims, so the bite falls with the pupil area: at
        // the traced fall-off that takes the Helios from 61% to 95% over two stops, which this
        // gives as 48% -> 87%. The clamp stops the strongest setting short of a sliver.
        float ap  = ApertureSize / 0.35;
        float u   = min(0.85 * catEye * ap * ap, 0.90);
        float ang = radians(ApertureCatEyeAngle);
        float2 q  = d * (2.0 / max(ApertureSize, 1e-4)) - float2(cos(ang), sin(ang)) * u;
        // The rim sits well away from the pupil plane, so its projected edge is defocused: wider
        // than the blade edge, which also keeps the cusp diffraction from reading as streaks.
        float rimSoft = clamp(3.0 * soft, 0.01, 0.35);
        mask *= 1.0 - smoothstep(1.0 - rimSoft, 1.0 + 0.5 * rimSoft, length(q));
    }

    return max(mask, 0.0);
}

// ==========================================================================
//  kernel signature and sampling
// ==========================================================================

float BakeSignature()
{
    // Every term is normalised to its slider range first, so the accumulator stays in [0, 1] and a
    // single-step change is always far above the comparison tolerance.
    float s = 0.5;
    s = HashStep(s, (float)ApertureSides / 32.0);
    s = HashStep(s, ApertureBladeCurvature);
    s = HashStep(s, ApertureSoftness / 0.9);
    s = HashStep(s, ApertureRim);
    s = HashStep(s, ApertureCentre);
    s = HashStep(s, ApertureRing / 0.95);
    s = HashStep(s, ApertureCatEye);
    s = HashStep(s, (ApertureCatEyeAngle + 180.0) / 360.0);
    s = HashStep(s, ApertureDispersion / 0.5);
    s = HashStep(s, ApertureSize / 0.6);
    s = HashStep(s, ApertureBeamGain / 8.0);

    s = HashStep(s, LensDirtStrength);
    s = HashStep(s, LensDirtGain / 8.0);
    s = HashStep(s, LensDirtTiling / 4.0);
    s = HashStep(s, LensDirtScatter);
    s = HashStep(s, (float)BAKE_DISP_STEPS / 64.0);

    // Offset away from zero: a freshly created (zeroed) state texture must always read as changed.
    return 0.5 + 0.5 * s;
}
#endif

bool BakeNeeded()
{
#if KERNEL_MODE == 1
    return SignatureChanged(tex2Dlod(sBakeState, float4(0.5, 0.5, 0.0, 0.0)).x, BakeSignature());
#else
    return false;
#endif
}

// Must list every uniform that changes the kernel grid, otherwise a control will
// appear to do nothing.
float KernelSignature()
{
    float s = 0.5;
    s = HashStep(s, KernelScale / 3.0);
    s = HashStep(s, (KernelRotation + 180.0) / 360.0);
    s = HashStep(s, AnamorphicStretch / 4.0);
    s = HashStep(s, KernelDehaze);
#if KERNEL_MODE == 0
    s = HashStep(s, (float)KernelFile / 8.0);
#else
    s = HashStep(s, BakeSignature());
#endif
    return 0.5 + 0.5 * s;
}

bool KernelNeeded()
{
    // True when any control that shapes the kernel changed since the spectrum was last built.
    // Every pass in the kernel chain is compute, so skipping its writes is safe: the previous
    // spectrum simply stays in TexKernelA/B. A pixel shader here would leave its render targets
    // undefined instead (AMD zeroes them), which is what broke the cache before.
    return SignatureChanged(tex2Dlod(sKernelState, float4(0.5, 0.5, 0.0, 0.0)).x, KernelSignature());
}

// Note: computing these two decisions once per frame in a dedicated pass was tried and MEASURED
// as a regression - the extra 1-thread dispatch cost 0.017 ms under DXVK, more than the ~0.014 ms
// of per-thread hashing it saved. Keep the hashes in the gated passes unless the dispatch cost
// changes.

// Photoshop's Levels midtone response, verified against a Photoshop LUT rather than derived:
// midtone 0.5 is exactly u^2 on 0..1 (every level of the reference LUT matched, 0 mismatches;
// floor(v^2/255) was off by one level on 55 of them). That is the u^(1/gamma) form PhotoCraft uses
// (crates/compose/src/adjust/tone.rs, levels_gamma); their shadow-toe branch only applies above
// gamma 1, and this control stops at 0.5 -> 1.0, so it is unreachable here.
float LevelsMidtone(float u, float midtone)
{
    return pow(saturate(u), 1.0 / max(midtone, 0.05));
}

// The kernel is convolved in linear light, but Levels is a curve pulled on a display-encoded image
// - which is also exactly what the DDS kernels are. Convert, curve, convert back; with Kernel Dehaze at 0
// none of this runs and SampleKernel is unchanged. Applied per channel, so the colour the dispersion
// bake put in the kernel is preserved as colour (each channel moves by a different amount) instead of
// being flattened to a single luminance curve.
float3 DehazeKernel(float3 k)
{
    if (KernelDehaze <= 0.0)
        return k;

    float  midtone = 1.0 - 0.5 * saturate(KernelDehaze);   // slider 0 -> 1.0, 1 -> 0.5
    float3 d = pow(saturate(k), 1.0 / 2.2);
    d = float3(LevelsMidtone(d.r, midtone),
               LevelsMidtone(d.g, midtone),
               LevelsMidtone(d.b, midtone));
    return pow(d, 2.2);
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
    return DehazeKernel(max(k, 0.0));
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
    // sopt: an 8-bit kernel texture read is in [0,1], so the clamp is dead code (bit-exact).
    return DehazeKernel(pow(k, 2.2));
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

groupshared float2 g_kernel_stats[KERNEL_SUM_THREADS];

[shader("compute")]
void KernelSumCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    if (!KernelNeeded())
        return;

    uint tid = gtid.x;

    float acc = 0.0;
    float mx  = 0.0;
    const uint count = FFT_SIZE * FFT_SIZE;
    for (uint i = tid; i < count; i += KERNEL_SUM_THREADS)
    {
        // Sample the PSF here instead of fetching a prepared copy. The old KernelPrep pixel pass
        // existed only to give this reduction something to read - a full grid pass plus a render
        // target pair - and it made the cache unsafe. The FFT passes sample the kernel the same way.
        float2 g = (float2((int)(i % FFT_SIZE), (int)(i / FFT_SIZE)) + 0.5) * RCP_FFT_SIZE;
        float3 k = SampleKernel(g);
        acc += (k.r + k.g + k.b) / 3.0;
        mx   = max(mx, max(k.r, max(k.g, k.b)));
    }

    g_kernel_stats[tid] = float2(acc, mx);
    barrier();

    for (uint s = KERNEL_SUM_THREADS / 2u; s > 0u; s >>= 1)
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
        // Energy and peak only. The signature is committed by the last pass of the chain, in its
        // own texture, so nothing here reads a texel this pass writes.
        tex2Dstore(stKernelEnergy, int2(0, 0), float4(energy, peak, 0.0, 0.0));
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

    // The signature commit lives in its own pass (KernelCommitCS) so this one never writes the
    // 1x1 texel that KernelNeeded() reads.
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

        float  tiling = max(LensDirtTiling, 0.05);
        float2 duv;

        if (tiling > 1.0)
        {
            // Tiling repeats the map, and at 2x the same specks show up four times across the pupil.
            // Give each tile its own hashed offset and quarter turn instead: the dust is sparse on a
            // clean field, so the boundary between two unrelated tiles reads as a fresh patch of
            // dirt rather than as a seam, and the repeat is gone. One hash per bake texel.
            float2 tiled = d / max(ApertureSize, 1e-4) * tiling + 0.5;
            float2 cell  = floor(tiled);
            float2 f     = frac(tiled) - 0.5;

            float2 h = frac(sin(float2(dot(cell, float2(127.1, 311.7)),
                                       dot(cell, float2(269.5, 183.3)))) * 43758.5453);

            float turn = floor(h.x * 4.0);
            if      (turn == 1.0) f = float2(-f.y,  f.x);
            else if (turn == 2.0) f = float2(-f.x, -f.y);
            else if (turn == 3.0) f = float2( f.y, -f.x);

            duv = frac(f + 0.5 + h.yx);
        }
        else
        {
            duv = d / max(ApertureSize, 1e-4) * tiling + 0.5;
        }

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

groupshared float2 g_dust_stats[DUST_SUM_THREADS];

// Mean fraction of the aperture that dust blocks. Scatter is incoherent, so the veiling wing needs
// a global figure - a per-texel value has no meaning in PSF space, which is the whole point of the
// transform. Reduced here, once per bake, so it costs nothing per frame.
void DustSumCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    if (!BakeNeeded())
        return;

    uint tid = gtid.x;

    if (LensDirtScatter <= 0.0)
    {
        if (tid == 0)
            tex2Dstore(stDustMean, int2(0, 0), float4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    float blocked = 0.0;
    float weight  = 0.0;

    const uint count = BAKE_SIZE * BAKE_SIZE;
    for (uint i = tid; i < count; i += DUST_SUM_THREADS)
    {
        int2   p   = int2((int)(i % BAKE_SIZE), (int)(i / BAKE_SIZE));
        float4 pup = tex2Dfetch(stPupil, p);                  // rgb = mask * dirt, a = dirt
        float  dust = pup.a;
        float  md   = max(pup.r, max(pup.g, pup.b));          // mask attenuated by the dust
        float  mask = (dust > 1e-4) ? saturate(md / dust) : 0.0;

        blocked += (1.0 - dust) * mask;
        weight  += mask;
    }

    g_dust_stats[tid] = float2(blocked, weight);
    barrier();

    for (uint s = DUST_SUM_THREADS / 2u; s > 0u; s >>= 1)
    {
        if (tid < s)
            g_dust_stats[tid] += g_dust_stats[tid + s];
        barrier();
    }

    if (tid == 0)
    {
        float mean = (g_dust_stats[0].y > 1e-6) ? (g_dust_stats[0].x / g_dust_stats[0].y) : 0.0;
        tex2Dstore(stDustMean, int2(0, 0), float4(saturate(mean), 0.0, 0.0, 0.0));
    }
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

    // Dust scatter: the specks do not absorb the light they block, they scatter it. Willson et al.,
    // "An optical model for image artifacts produced by dust particles on lenses" (iSAIRAS 2005)
    // is the reference for how dust behaves here: a particle smaller than the aperture produces an
    // artifact the size of the *aperture* whose depth is its area ratio, and dust at the pupil plane
    // attenuates the whole frame uniformly rather than forming a halo. So the wing has two parts -
    // a power-law skirt (stray light falls off with angle, never as fast as a Gaussian) and a flat
    // pedestal for the pupil-plane case.
    //
    // The constant is an energy ratio, not a visual one: the kernel is normalised to unit sum, so
    // what matters is the share of the bloom's energy the wing carries. Sum(1/(1+18r2) + 0.15) over
    // the kernel grid is 1.885e4, so 1.3e-5 gives the wing (blocked * 0.25) of the core energy: a
    // quarter of the blocked light lands inside the kernel, the rest goes to angles this transform
    // cannot represent. Scaling by the wing's peak instead would put 250x the core's energy into
    // the veil and turn the bloom into a flat grey wash. The shipped dust map blocks 17.6% of the
    // pupil, so this is ~4% of the bloom with Dust Scatter at 1.
    if (LensDirtScatter > 0.0)
    {
        float blocked = saturate(tex2Dfetch(stDustMean, int2(0, 0)).x);
        float r2   = dot(dC, dC) * 4.0;                    // 0 centre, 1 mid-edge, 2 corner
        float wing = 1.0 / (1.0 + 18.0 * r2) + 0.15;
        psf += (LensDirtScatter * blocked * 1.3e-5 * wing).xxx;
    }

    tex2Dstore(stAperture, o, float4(psf, 1.0));
}

[shader("compute")]
void BakeCommitCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    // Writes unconditionally. Gating this on BakeNeeded() means reading the very 1x1 texel the pass
    // writes, which is undefined in D3D11; re-writing an unchanged signature is a no-op anyway.
    tex2Dstore(stBakeState, int2(0, 0), float4(BakeSignature(), 0.0, 0.0, 0.0));
}
#endif

[shader("compute")]
// ==========================================================================
//  image FFT
// ==========================================================================

void KernelCommitCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    // Ungated and write-only: the single point that commits the kernel signature, so no pass has to
    // read and write the same texel. Re-writing an unchanged signature is a no-op.
    tex2Dstore(stKernelState, int2(0, 0), float4(KernelSignature(), 0.0, 0.0, 0.0));
}

// The four image FFT passes only produce something the composite actually reads when the glow is
// visible, or when a debug view shows the spectrum. Every one of them is compute, so returning
// early is safe: the previous frame's spectrum simply stays in the buffers, unread.
bool BloomNeeded()
{
    return BloomIntensity > 0.0 || DebugView == 1 || DebugView == 4;
}

void ImageFFTXCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    if (!BloomNeeded())
        return;
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
    if (!BloomNeeded())
        return;
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
    if (!BloomNeeded())
        return;
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
    if (!BloomNeeded())
        return;
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
    float3 scene = SceneLinear(uv);

    // Intensity 0 with no spectrum debug view means the image chain skipped its passes, so those
    // buffers hold stale data. Return the scene - and write it, never return without a value.
    if (!BloomNeeded() && DebugView == 0)
    {
        o = float4(scene, 1.0);
        return;
    }

    float2 gridUV = ScreenToGrid(uv);

    float3 bloom = BloomAtFiltered(gridUV);
    bloom *= BloomTint;

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
        // Pupil and kernel side by side. Each half is a square window in kernel space, so the
        // frame's aspect cannot stretch the pattern; both are normalised to their own peak and
        // never apply artistic controls (Beam Emphasis), or the view cannot be trusted.
        float2 px     = uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT);
        float  side   = min(BUFFER_WIDTH * 0.5, BUFFER_HEIGHT);
        bool   right  = uv.x >= 0.5;
        float2 centre = float2(right ? BUFFER_WIDTH * 0.75 : BUFFER_WIDTH * 0.25, BUFFER_HEIGHT * 0.5);
        float2 k      = (px - centre) / side + 0.5;
        float  inside = (k.x >= 0.0 && k.x <= 1.0 && k.y >= 0.0 && k.y <= 1.0) ? 1.0 : 0.0;

        float3 v = float3(0.0, 0.0, 0.0);

#if KERNEL_MODE == 1
        if (right)
        {
            float3 ks = tex2Dlod(sApertureKernelPoint, float4(k, 0.0, 0.0)).rgb;
            float3 kc = tex2Dlod(sApertureKernelPoint, float4(0.5, 0.5, 0.0, 0.0)).rgb;
            float  kp = max(max(kc.r, max(kc.g, kc.b)), 1e-12);
            v = DehazeKernel(saturate(ks / kp));
        }
        else
        {
            // The bake stores the per-channel aperture mask, so dispersion shows up here as colour.
            v = saturate(tex2Dlod(sPupil, float4(k, 0.0, 0.0)).rgb);
        }
#else
        // A DDS kernel has no pupil: full PSF on the left, its core at 2x on the right.
        float2 kk = right ? (0.5 + (k - 0.5) * 0.5) : k;
        float  kp = max(tex2Dlod(sKernelEnergy, float4(0.5, 0.5, 0.0, 0.0)).y, 1e-12);
        v = saturate(max(SampleKernel(kk), 0.0) / kp);
#endif

        outc = pow(saturate(v), 1.0 / 2.2) * inside;
        if (abs(uv.x - 0.5) * BUFFER_WIDTH < 1.5)
            outc = float3(0.08, 0.08, 0.08);
    }
    else if (DebugView == 4)
    {
        // Normalised to its own DC, so black means "the spectrum is empty" rather than "the numbers
        // are small": an unnormalised view reads as a dead pipeline even when it is working.
        float  m  = length(tex2D(sImageA, uv).xy) + length(tex2D(sImageA, uv).zw)
                  + length(tex2D(sImageB, uv).xy);
        float4 a0 = tex2Dlod(sImageA, float4(0.001, 0.001, 0.0, 0.0));
        float4 b0 = tex2Dlod(sImageB, float4(0.001, 0.001, 0.0, 0.0));
        float  m0 = max(length(a0.xy) + length(a0.zw) + length(b0.xy), 1e-20);
        outc = saturate(log(1.0 + 200.0 * (m / m0)) / log(201.0)).xxx;
    }
    else if (DebugView == 6)
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
        "The PSF (blade shape, curvature, ring, dispersion, dust) is baked on\n" \
        "the GPU once per settings change, then the frame is convolved with it\n" \
        "in the frequency domain, so cost is independent of kernel size.\n" \
        "Threshold -1 = physically based (everything blooms).\n" \
        "\n" \
        "Debug view 3 shows the pupil and kernel, 4 the image spectrum, 6 the bake."
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
        "Debug views 3 and 5 check the kernel and the padding.\n" \
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
    pass DustSum      { ComputeShader = DustSumCS<DUST_SUM_THREADS, 1, 1>; DispatchSizeX = 1; DispatchSizeY = 1; DispatchSizeZ = 1; }
    pass BakeFFTX     { ComputeShader = BakeFFTXCS<BAKE_THREADS, 1, 1>; DispatchSizeX = 1;         DispatchSizeY = BAKE_SIZE; DispatchSizeZ = 1; }
    pass BakeFFTY     { ComputeShader = BakeFFTYCS<1, BAKE_THREADS, 1>; DispatchSizeX = BAKE_SIZE; DispatchSizeY = 1;         DispatchSizeZ = 1; }
    pass BakePSF      { ComputeShader = BakePSFCS<16, 16, 1>; DispatchSizeX = BAKE_SIZE / 16; DispatchSizeY = BAKE_SIZE / 16; DispatchSizeZ = 1; }
    pass BakeCommit   { ComputeShader = BakeCommitCS<1, 1, 1>; DispatchSizeX = 1; DispatchSizeY = 1; DispatchSizeZ = 1; }
#endif

    pass KernelSum   { ComputeShader = KernelSumCS<KERNEL_SUM_THREADS, 1, 1>; DispatchSizeX = 1; DispatchSizeY = 1; DispatchSizeZ = 1; }

    pass KernelFFTX  { ComputeShader = KernelFFTXCS<FFT_THREADS, 1, 1>; DispatchSizeX = 1;         DispatchSizeY = FFT_SIZE; DispatchSizeZ = 1; }
    pass KernelFFTY  { ComputeShader = KernelFFTYCS<1, FFT_THREADS, 1>; DispatchSizeX = FFT_SIZE;   DispatchSizeY = 1;        DispatchSizeZ = 1; }
    pass KernelCommit { ComputeShader = KernelCommitCS<1, 1, 1>; DispatchSizeX = 1; DispatchSizeY = 1; DispatchSizeZ = 1; }

    pass ImageFFTX   { ComputeShader = ImageFFTXCS<FFT_THREADS, 1, 1>;  DispatchSizeX = 1;         DispatchSizeY = FFT_SIZE; DispatchSizeZ = 1; }
    pass ImageFFTY   { ComputeShader = ImageFFTYCS<1, FFT_THREADS, 1>;  DispatchSizeX = FFT_SIZE;   DispatchSizeY = 1;        DispatchSizeZ = 1; }

    pass IFFTX       { ComputeShader = IFFTXCS<FFT_THREADS, 1, 1>;      DispatchSizeX = 1;         DispatchSizeY = FFT_SIZE; DispatchSizeZ = 1; }
    pass IFFTY       { ComputeShader = IFFTYCS<1, FFT_THREADS, 1>;      DispatchSizeX = FFT_SIZE;   DispatchSizeY = 1;        DispatchSizeZ = 1; }

    pass Composite   { VertexShader = PostProcessVS; PixelShader = CompositePS; }
}
