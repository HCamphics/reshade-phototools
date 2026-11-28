/*==============================================================================

    HC Depth of Field
    Scatter-as-gather depth of field through the aperture shared with the bloom.

    Author:  HCamphics
    Repo:    https://github.com/HCamphics/reshade-phototools
    Licence: CC BY-NC 4.0 - see LICENSE in the repository root.

    Every pixel gathers over a disc whose radius is its own circle of confusion, so there are no holes
    to fill and each tap can apply its own depth test. The pupil shape, its apodization and the
    cat's-eye clip come from HC_Aperture.fxh and are evaluated per tap, so the clip can rotate with the
    pixel's field angle - which a single baked kernel cannot do.

    Pipeline, 8 passes, half resolution unless DOF_FULLRES=1

        DoFCoC        full     depth -> signed near and far CoC, in pixels
        DoFColour     half     2x2 downsample, weighted by CoC so a silhouette is not averaged
        DoFCoCHalf    half     near CoC reduced to the dilation grid (2x2 max)
        DoFCoCSmall   half     small max on the CoC, which stabilises thin geometry
        DoFDilate     eighth   dilated reach of the near CoC - this is what sets the halo's width
        DoFGather     half     the gather, into separate far and near layers with coverage
        DoFDenoise    half     edge-aware 3x3 over both layers
        DoFComposite  full     upsample, sub-pixel bridge, additive composite

    Depth is ReShade::GetLinearizedDepth divided by RESHADE_DEPTH_LINEARIZATION_FAR_PLANE, so set that
    define to the scene's far plane or the focus distance is off by that factor. DisplayDepth.fx is the
    tool for checking the RESHADE_DEPTH_INPUT_* flags first.

==============================================================================*/

#include "ReShade.fxh"
#include "ReShadeUI.fxh"
#include "HC_Aperture.fxh"

// Half resolution by default: the gather is the whole cost of this effect. Define DOF_FULLRES=1 in
// the effect's preprocessor definitions for a full-resolution gather at roughly four times the work.
// Like FTB_QUALITY this has to be a define, because it changes buffer sizes and dispatch counts.
#ifndef DOF_FULLRES
    #define DOF_FULLRES 0
#endif

#if DOF_FULLRES
    #define GATHER_DIV 1
#else
    #define GATHER_DIV 2
#endif

#define GATHER_W  ((BUFFER_WIDTH  + GATHER_DIV - 1) / GATHER_DIV)
#define GATHER_H  ((BUFFER_HEIGHT + GATHER_DIV - 1) / GATHER_DIV)
#define RCP_GATHER_W (1.0 / (float)GATHER_W)
#define RCP_GATHER_H (1.0 / (float)GATHER_H)

// Dilated-CoC grid. The near CoC is reduced to gather resolution, max-filtered, and stored at one eighth
// of the frame: a cell is 8 px, so a 7x7 window reaches about 32 px of halo - enough for any near CoC the
// cap allows - and reading it bilinearly still gives a continuous ramp rather than a stepped one.
#define DIL_W     ((BUFFER_WIDTH  + 7) / 8)
#define DIL_H     ((BUFFER_HEIGHT + 7) / 8)
#define SENSOR_WIDTH_MM 36.0     // full frame; only sets the mm-to-pixel scale of the CoC
#define GOLDEN_ANGLE    2.39996323
#define MAX_TAPS        96

// Soft clamp on the prefiltered colour, in linear luminance. Only matters for an HDR source: on a
// tonemapped LDR back buffer nothing exceeds 1.0 and this never fires.
#ifndef HIGHLIGHT_CLAMP
    #define HIGHLIGHT_CLAMP 64.0
#endif

// The Taps slider is the runtime control. Define DOF_TAPS to pin it at compile time instead, which
// lets the compiler unroll the gather - worth an A/B if the effect ever shows up in a profile.
#ifdef DOF_TAPS
    #define TAP_COUNT ((int)DOF_TAPS)
#else
    #define TAP_COUNT ((int)clamp((float)Quality, 4.0, (float)MAX_TAPS))
#endif

/*==============================================================================
    UI uniforms
==============================================================================*/


uniform float FocusDistance <
    ui_type = "drag"; ui_min = 0.05; ui_max = 100.0; ui_step = 0.01;
    ui_category = "Depth of Field";
    ui_label = "Focus Distance";
    ui_tooltip = "Distance that stays sharp, in the same units as RESHADE_DEPTH_LINEARIZATION_FAR_PLANE (metres, with the default 1000).";
> = 2.0;

uniform float FocalLength <
    ui_type = "drag"; ui_min = 8.0; ui_max = 300.0; ui_step = 1.0;
    ui_category = "Depth of Field";
    ui_label = "Focal Length";
    ui_tooltip = "Millimetres. Longer lenses blur the background harder for the same aperture.";
> = 50.0;

uniform float ApertureNumber <
    ui_type = "drag"; ui_min = 1.0; ui_max = 32.0; ui_step = 0.1;
    ui_category = "Depth of Field";
    ui_label = "Aperture";
    ui_tooltip = "f-number. Smaller means a faster lens and a wider disc of confusion.";
> = 2.8;

uniform float MaxBlurFar <
    ui_type = "drag"; ui_min = 0.0; ui_max = 64.0; ui_step = 0.5;
    ui_category = "Depth of Field";
    ui_label = "Max Blur Far";
    ui_tooltip = "Ceiling on the background disc radius in pixels. This is the knob that keeps the background from smearing.";
> = 20.0;

uniform float MaxBlurNear <
    ui_type = "drag"; ui_min = 0.0; ui_max = 64.0; ui_step = 0.5;
    ui_category = "Depth of Field";
    ui_label = "Max Blur Near";
    ui_tooltip = "Ceiling on the foreground disc radius in pixels. Foreground blur is usually kept lower than background.";
> = 12.0;

uniform float HaloSpread <
    ui_type = "drag"; ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Depth of Field";
    ui_label = "Halo Spread";
    ui_tooltip = "Scales how far a blurred foreground's halo reaches, as a multiple of the foreground's own circle of confusion. 1 is physical: a bokeh disc spreads its own radius past the silhouette. Lower it for a tighter, less aggressive halo.";
> = 1.0;

uniform int Quality <
    ui_type = "drag"; ui_min = 8; ui_max = 96; ui_step = 8;
    ui_category = "Depth of Field";
    ui_label = "Taps";
    ui_tooltip = "Samples per pixel. Grain falls as 1/N; 48 suits a detailed scene, 64 removes banding on very bright bokeh, 8 is for testing.";
> = 48;

uniform float Denoise <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Depth of Field";
    ui_label = "Denoise";
    ui_tooltip = "Edge-aware clean-up of the gathered layers. Removes the sampling grain of a wide disc without softening silhouettes or the detail inside them. 0 is the raw gather.";
> = 0.7;

uniform int LinearBlend <
    ui_type = "combo";
    ui_items = "Off\0On (sRGB buffer)\0";
    ui_category = "Depth of Field";
    ui_label = "Blend In Linear Light";
    ui_tooltip = "Average in linear light instead of in the buffer's encoded values. Mixing encoded pixels averages their transfer curves rather than their light, which darkens every sharp-to-blurred blend and leaves a dark fringe along hard bright edges. Leave it off for HDR output, which is already linear.";
> = 0;

uniform float HighlightBoost <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Bokeh";
    ui_label = "Highlight Boost";
    ui_tooltip = "Lets bright taps weigh more, the way a real disc brightens with its source. Weights are renormalised, so this cannot darken the frame.";
> = 0.25;

uniform float Anamorphic <
    ui_type = "drag"; ui_min = 0.25; ui_max = 4.0; ui_step = 0.01;
    ui_category = "Bokeh";
    ui_label = "Anamorphic";
    ui_tooltip = "Stretches the bokeh vertically, like an anamorphic lens. 1 = off.";
> = 1.0;

uniform float ApertureCatEye <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Bokeh";
    ui_label = "Cat Eye";
    ui_tooltip = "How strongly the barrel clips the pupil toward the frame corners. 1 is about 60% round at the corner; 0 = round everywhere.";
> = 0.5;

uniform float LoCA <
    ui_type = "drag"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Bokeh";
    ui_label = "Longitudinal CA";
    ui_tooltip = "Per-channel disc radius: magenta rims in front of the focus plane, green behind it. Vanishes at the plane itself, so in-focus detail stays sharp. 0 = off.";
> = 0.0;

uniform float CatEyeStart <
    ui_type = "drag"; ui_min = 0.0; ui_max = 0.9; ui_step = 0.01;
    ui_category = "Bokeh";
    ui_label = "Cat Eye Start";
    ui_tooltip = "Where the barrel starts clipping the pupil, between the frame centre (0) and the corner (0.9). Cat Eye sets how strong it gets at the corner.";
> = 0.35;

uniform int DebugView <
    ui_type = "combo";
    ui_items = "Final\0Circle of Confusion\0Dilated CoC\0Aperture At Pixel\0Near Field\0Far Field\0Colour Buffer\0Foreground Coverage\0";
    ui_category = "Debug";
    ui_category_closed = true;
    ui_label = "Debug View";
    ui_tooltip = "Leave at Final. Circle of Confusion is the first thing to check after setting the depth flags.";
> = 0;

/*==============================================================================
    Resources
==============================================================================*/


texture TexColour { Width = GATHER_W;     Height = GATHER_H;     Format = RGBA16F; };
texture TexCoC  { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
texture TexCoCHalf  { Width = GATHER_W; Height = GATHER_H; Format = R16F; };
texture TexCoCSmall { Width = GATHER_W; Height = GATHER_H; Format = R16F; };
texture TexDilate  { Width = DIL_W;    Height = DIL_H;    Format = R16F; };
// The gather radius each texel used. The layers' alpha carries coverage now, so the denoise and the
// upsample read their defocus signal from here.
texture TexGatherCoC { Width = GATHER_W; Height = GATHER_H; Format = R16F; };
texture TexFar  { Width = GATHER_W;     Height = GATHER_H;     Format = RGBA16F; };
texture TexNear { Width = GATHER_W;     Height = GATHER_H;     Format = RGBA16F; };

// Denoised layers, written by DoFDenoise_CS.
texture TexFarS  { Width = GATHER_W;    Height = GATHER_H;     Format = RGBA16F; };
texture TexNearS { Width = GATHER_W;    Height = GATHER_H;     Format = RGBA16F; };

sampler sColour { Texture = TexColour; AddressU = CLAMP; AddressV = CLAMP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sCoC  { Texture = TexCoC;  AddressU = CLAMP; AddressV = CLAMP; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };
sampler sCoCHalf  { Texture = TexCoCHalf;  AddressU = CLAMP; AddressV = CLAMP; MagFilter = POINT;  MinFilter = POINT;  MipFilter = POINT; };
sampler sCoCSmall { Texture = TexCoCSmall; AddressU = CLAMP; AddressV = CLAMP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sDilate  { Texture = TexDilate;  AddressU = CLAMP; AddressV = CLAMP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sGatherCoC { Texture = TexGatherCoC; AddressU = CLAMP; AddressV = CLAMP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sFar  { Texture = TexFar;  AddressU = CLAMP; AddressV = CLAMP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sNear { Texture = TexNear; AddressU = CLAMP; AddressV = CLAMP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sFarS  { Texture = TexFarS;  AddressU = CLAMP; AddressV = CLAMP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };
sampler sNearS { Texture = TexNearS; AddressU = CLAMP; AddressV = CLAMP; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = POINT; };

storage2D stColour { Texture = TexColour; };
storage2D stCoC  { Texture = TexCoC; };
storage2D stCoCHalf  { Texture = TexCoCHalf; };
storage2D stCoCSmall { Texture = TexCoCSmall; };
storage2D stDilate  { Texture = TexDilate; };
storage2D stGatherCoC { Texture = TexGatherCoC; };
storage2D stFar  { Texture = TexFar; };
storage2D stNear { Texture = TexNear; };
storage2D stFarS  { Texture = TexFarS; };
storage2D stNearS { Texture = TexNearS; };

/*==============================================================================
    Helpers
==============================================================================*/


// ReShade returns view distance divided by the linearisation far plane, so this is metres.
float LinearDepth(float2 uv)
{
    return ReShade::GetLinearizedDepth(uv) * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
}

// Thin lens: c = f^2 / (N (S1 - f)) * |S2 - S1| / S2 is a diameter on the sensor, so halve it once here.
// Returns (near radius, far radius) in pixels; only one is non-zero.
float2 CoCPixels(float2 uv)
{
    float S2 = max(LinearDepth(uv), 1e-3);
    float S1 = max(FocusDistance, 1e-3);
    float f  = FocalLength * 0.001;
    float N  = max(ApertureNumber, 1.0);

    // c = K (S1 / S2 - 1) with K = f^2 / (N (S1 - f)), so the per-pixel cost is one divide and an
    // add once K and the pixel scale are folded into scale/bias. Positive means in front of focus.
    float K       = (f * f) / (N * max(S1 - f, 1e-4));
    float pxPerM  = 0.5 * ((float)BUFFER_WIDTH / SENSOR_WIDTH_MM) * 1000.0;
    float bias    = -K * pxPerM;
    float scale   =  K * S1 * pxPerM;

    float signed_px = scale / S2 + bias;

    return float2(min(max( signed_px, 0.0), MaxBlurNear),
                  min(max(-signed_px, 0.0), MaxBlurFar));
}

// The pupil as this pixel sees it: clipped toward the frame corner, facing outward.
float2 CatEyeAt(float2 uv)
{
    float2 field  = (uv - 0.5) * float2(ReShade::AspectRatio, 1.0);
    float2 corner = float2(ReShade::AspectRatio, 1.0) * 0.5;
    float  fr     = saturate(length(field) / length(corner));

    float start = min(CatEyeStart, 0.89);
    // pow(s, 1.5) written as s * sqrt(s): a log2/exp2 pair becomes a sqrt, and sopt measured it as
    // marginally closer to exact than the pow (1.19e-07 against 4.11e-07).
    float edgeT = saturate((fr - start) / (1.0 - start));
    float amt   = saturate(ApertureCatEye) * edgeT * sqrt(edgeT);

    return float2(amt, atan2(field.y, field.x));
}

/*==============================================================================
    Passes
==============================================================================*/


// sRGB transfer curves for the optional linear-light blend.
float3 SRGBToLinear3(float3 c)
{
    c = max(c, 0.0);
    return (c <= 0.04045) ? c * 0.07739938 : pow(mad(c, 0.94786733, 0.052132703), 2.4);
}

float3 LinearToSRGB3(float3 c)
{
    c = max(c, 0.0);
    // 1.055 * pow(c, 1/2.4) - 0.055 as a mad, plus the reciprocal exponent as a literal.
    return (c <= 0.0031308) ? c * 12.92 : mad(pow(c, 0.41666666), 1.055, -0.055);
}

// Every read of the back buffer goes through here, so the blend happens in the space the switch selects
// and nothing can quietly bypass it.
float3 SampleSharp(float2 uv)
{
    float3 c = tex2Dlod(ReShade::BackBuffer, float4(uv, 0.0, 0.0)).rgb;
    return LinearBlend ? SRGBToLinear3(c) : c;
}

[shader("compute")]
void DoFColour_CS(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= GATHER_W || tid.y >= GATHER_H)
        return;

    float2 uv  = (float2(tid.xy) + 0.5) * float2(RCP_GATHER_W, RCP_GATHER_H);
    float2 tex = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT) * (float)GATHER_DIV * 0.5;

    // The four full-resolution texels under this gather texel are only averaged together when they
    // agree on defocus. A plain box that straddles a silhouette mixes two depths, which puts background
    // colour inside a foreground edge and vice versa - one of the ways a rim of the wrong colour appears
    // along every blurred boundary.
    float2 coc0 = tex2Dlod(sCoC, float4(uv, 0.0, 0.0)).xy;

    float3 c    = 0.0;
    float  wsum = 0.0;
    [unroll]
    for (int dy = -1; dy <= 1; dy += 2)
    {
        [unroll]
        for (int dx = -1; dx <= 1; dx += 2)
        {
            float2 t   = uv + float2((float)dx, (float)dy) * tex;
            float2 coc = tex2Dlod(sCoC, float4(t, 0.0, 0.0)).xy;
            float  d   = max(abs(coc.x - coc0.x), abs(coc.y - coc0.y));
            float  w   = exp2(-d * d * 0.09);        // half weight at 3.4 px of CoC difference
            c    += SampleSharp(t) * w;
            wsum += w;
        }
    }
    c = (wsum > 1e-5) ? c / wsum : SampleSharp(uv);

    float luma = max(c.r, max(c.g, c.b));
    c *= 1.0 / (1.0 + max(luma - HIGHLIGHT_CLAMP, 0.0) / max(HIGHLIGHT_CLAMP, 1.0));

    tex2Dstore(stColour, int2(tid.xy), float4(c, 1.0));

    // Zero coverage in both layers. The composite's leftover term is the sharp frame, so an untouched
    // texel needs no stand-in colour: nothing gathered means the sharp frame shows, and in-focus pixels
    // stay the real sharp pixels rather than the prefiltered copy of them.
    tex2Dstore(stFar,  int2(tid.xy), float4(0.0, 0.0, 0.0, 0.0));
    tex2Dstore(stNear, int2(tid.xy), float4(0.0, 0.0, 0.0, 0.0));
}

[shader("compute")]
void DoFCoC_CS(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= BUFFER_WIDTH || tid.y >= BUFFER_HEIGHT)
        return;

    float2 uv = (float2(tid.xy) + 0.5) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
    float2 coc = CoCPixels(uv);

    tex2Dstore(stCoC, int2(tid.xy), float4(coc, 0.0, 1.0));
}

// CoC at gather resolution: the max of the 2x2 full-resolution texels each covers, so a silhouette's
// defocus survives the reduction instead of being averaged away.
[shader("compute")]
void DoFCoCHalf_CS(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= GATHER_W || tid.y >= GATHER_H)
        return;

    float2 uv  = (float2(tid.xy) + 0.5) * float2(RCP_GATHER_W, RCP_GATHER_H);
    float2 tex = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT) * (float)GATHER_DIV * 0.5;

    float a = tex2Dlod(sCoC, float4(uv + float2(-tex.x, -tex.y), 0.0, 0.0)).x;
    float b = tex2Dlod(sCoC, float4(uv + float2( tex.x, -tex.y), 0.0, 0.0)).x;
    float c = tex2Dlod(sCoC, float4(uv + float2(-tex.x,  tex.y), 0.0, 0.0)).x;
    float d = tex2Dlod(sCoC, float4(uv + float2( tex.x,  tex.y), 0.0, 0.0)).x;

    tex2Dstore(stCoCHalf, int2(tid.xy), float4(max(max(a, b), max(c, d)), 0.0, 0.0, 0.0));
}

// A small max on the CoC, which is what the blend decision reads. Depth is one sample per pixel, so a
// thin feature - the edge of a leaf - alternates between its own depth and the background behind it,
// and its CoC alternates with it. The composite then decides sharp on one pixel and blurred on the next
// along the same silhouette, which is the jaggedness: it never shows in the near layer alone, because
// that layer has no sharp reference beside it. Taking the max over 4 px makes a thin feature inherit
// the blur around it, which is also what it physically does - its own blur spreads over its edge.
[shader("compute")]
void DoFCoCSmall_CS(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= GATHER_W || tid.y >= GATHER_H)
        return;

    int2   p   = int2(tid.xy);
    int2   hi  = int2(GATHER_W - 1, GATHER_H - 1);
    float2 rcp = float2(RCP_GATHER_W, RCP_GATHER_H);

    float m = 0.0;
    [unroll]
    for (int dy = -2; dy <= 2; ++dy)
    {
        [unroll]
        for (int dx = -2; dx <= 2; ++dx)
        {
            float2 t = (float2(clamp(p + int2(dx, dy), int2(0, 0), hi)) + 0.5) * rcp;
            m = max(m, tex2Dlod(sCoCHalf, float4(t, 0.0, 0.0)).r);
        }
    }

    tex2Dstore(stCoCSmall, p, float4(m, 0.0, 0.0, 0.0));
}

// Dilation of the near CoC, which is what lets a blurred foreground's halo reach pixels whose own CoC is
// zero. A max filter at one eighth of the frame, read back bilinearly: continuous, where a coarser grid
// quantises the halo's extent and leaves stair-steps along the silhouette.
[shader("compute")]
void DoFDilate_CS(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= DIL_W || tid.y >= DIL_H)
        return;

    int2   p   = int2(tid.xy);
    int2   hi  = int2(DIL_W - 1, DIL_H - 1);
    float2 rcp = float2(RCP_GATHER_W, RCP_GATHER_H);   // the source is at gather resolution

    float m = 0.0;
    [unroll]
    for (int dy = -3; dy <= 3; ++dy)
    {
        [unroll]
        for (int dx = -3; dx <= 3; ++dx)
        {
            // Each 8 px cell owns a 4x4 block of gather texels, whose centre is texel 4c + 2.
            float2 c = (float2(clamp(p + int2(dx, dy), int2(0, 0), hi)) * 4.0 + 2.0) * rcp;

            // Disc-limited: a foreground's blur reaches its own CoC past its silhouette, so a neighbour
            // only counts while the distance to it is within that CoC.
            float  dist = length(float2((float)dx, (float)dy)) * 8.0;
            float  near = tex2Dlod(sCoCSmall, float4(c, 0.0, 0.0)).x;
            m = max(m, near >= dist ? near : 0.0);
        }
    }

    tex2Dstore(stDilate, p, float4(max(m, 0.0), 0.0, 0.0, 0.0));
}

[shader("compute")]
void DoFGather_CS(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= GATHER_W || tid.y >= GATHER_H)
        return;

    float2 uv = (float2(tid.xy) + 0.5) * float2(RCP_GATHER_W, RCP_GATHER_H);
    float2 cocCentre = tex2Dlod(sCoC, float4(uv, 0.0, 0.0)).xy;

    // Dilated reach for the halo, bilinear and uncapped; HaloSpread scales it for taste.
    float nearDilated = tex2Dlod(sDilate, float4(uv, 0.0, 0.0)).r * HaloSpread;

    float gatherFull = max(max(cocCentre.x, cocCentre.y), nearDilated);
    float radiusPx   = gatherFull / (float)GATHER_DIV;

    // Below this CoC the layers would contribute nothing, so skip the gather.
    if (gatherFull < 1.5)
    {
        // Zero coverage: the composite falls through to the sharp frame.
        tex2Dstore(stFar,       int2(tid.xy), float4(0.0, 0.0, 0.0, 0.0));
        tex2Dstore(stNear,      int2(tid.xy), float4(0.0, 0.0, 0.0, 0.0));
        tex2Dstore(stGatherCoC, int2(tid.xy), float4(gatherFull, 0.0, 0.0, 0.0));
        return;
    }

    float2 cat        = CatEyeAt(uv);
    float  pupilScale = ApertureSize * 0.5;      // the module puts the pupil edge at |d| = size/2
    // Tap count follows the radius: a wide disc needs more samples to keep its rim smooth.
    float  taps       = clamp(floor((float)TAP_COUNT * sqrt(max(radiusPx, 1.0) / 6.0) + 0.5),
                              8.0, (float)MAX_TAPS);
    float  rcpR       = 1.0 / max(radiusPx, 1e-4);

    // Per-tap pupil softness, about half the tap spacing (1/sqrt(N) of the radius): a rim sharper than
    // that cannot be resolved and aliases the disc edge.
    float  pixSoft    = 0.6 * rsqrt(taps);

    float3 farSum = 0.0, nearSum = 0.0;
    float3 farW   = 0.0, nearW   = 0.0;
    float3 wAll   = 0.0;                      // every tap's weight, landed or not

    // Rotate the spiral per pixel. A fixed spiral puts its rings in the same place in every pixel,
    // which reads as concentric banding on small bright sources; a hashed rotation turns that into
    // something closer to grain.
    float rot = frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453) * 6.28318530718;

    [loop]
    for (int i = 0; i < (int)taps; ++i)
    {
        float  t   = ((float)i + 0.5) / taps;
        float  rr  = sqrt(t) * radiusPx;
        float  ang = (float)i * GOLDEN_ANGLE + rot;
        float2 off = float2(cos(ang), sin(ang)) * float2(1.0, Anamorphic) * rr;

        // Clamped, not skipped. A dropped tap contributes no weight, so coverage falls away at the
        // frame border and a blurred object at the edge of the screen loses part of its halo; repeating
        // the border texel keeps the weight, which is what the coverage term below depends on.
        float2 suv = saturate(uv + off * float2(RCP_GATHER_W, RCP_GATHER_H));

        float3 sc    = tex2Dlod(sColour, float4(suv, 0.0, 0.0)).rgb;
        float2 scoc  = tex2Dlod(sCoC, float4(suv, 0.0, 0.0)).xy;
        float  srad  = max(scoc.x, scoc.y) / (float)GATHER_DIV;

        // A sample only contributes if its own disc reaches this pixel, which is what stops a sharp
        // background from bleeding through the edge of a blurred foreground.
        // Centred on the disc edge with a 2 texel ramp, so a tap fades in as the disc grows instead
        // of switching on at a threshold.
        float occl = saturate((srad - rr) * 0.5 + 0.5);

        // How much of the pupil lets this direction through, including the cat's-eye clip.
        float  ap = PupilShape(off * rcpR * pupilScale, cat.x, cat.y, pixSoft);
        float  hl = 1.0 + HighlightBoost * saturate(max(sc.r, max(sc.g, sc.b)));

        // Longitudinal CA: each channel has its own disc radius, so the three soft edges land at
        // different distances from the centre. The sign of the sample's CoC decides which way round,
        // which is what puts magenta in front of the focus plane and green behind it.
        float  sgn = (scoc.y >= scoc.x) ? 1.0 : -1.0;
        float  dlt = LoCA * 0.18 * sgn * radiusPx;
        float3 lc  = (LoCA > 0.0)
                   ? saturate((radiusPx + dlt * float3(-1.0, 0.0, 1.0) - rr) * 3.0 + 0.5)
                   : float3(1.0, 1.0, 1.0);

        // The weight this tap would carry if nothing were in its way, and the weight it does carry.
        // Keeping both is what lets the composite know how much of the disc landed here.
        float3 wBase = ap * hl * lc;
        wAll += wBase;

        float3 wc = wBase * occl;
        if (wc.r + wc.g + wc.b <= 0.0)
            continue;

        if (sgn > 0.0)
        {
            farSum  += sc * wc;
            farW    += wc;
        }
        else
        {
            nearSum += sc * wc;
            nearW   += wc;
        }
    }

    float3 farOut  = farSum  / max(farW,  1e-6);
    float3 nearOut = nearSum / max(nearW, 1e-6);

    // Coverage: how much of the aperture actually landed on this pixel. 1 where the disc is fully
    // sampled, falling smoothly to 0 where a foreground's samples stop reaching - which is exactly the
    // distance its own blur spreads to, so the halo no longer needs a cap or a gate to end at the right
    // place. It replaces the per-pixel blend gate the composite used to apply.
    float3 wUsed  = farW + nearW;
    float3 v      = saturate(wUsed / max(wAll, 1e-6));
    float3 aNearC = saturate(nearW / max(wUsed, 1e-6));

    // Both layers leave premultiplied by their own coverage, so the composite can simply add them.
    float3 aNear = v * aNearC;
    float3 aFar  = v * (1.0 - aNearC);

    const float3 LW = float3(0.2126, 0.7152, 0.0722);

    tex2Dstore(stFar,       int2(tid.xy), float4(farOut * aFar, dot(aFar, LW)));
    tex2Dstore(stNear,      int2(tid.xy), float4(nearOut * aNear, dot(aNear, LW)));
    tex2Dstore(stGatherCoC, int2(tid.xy), float4(gatherFull, 0.0, 0.0, 0.0));
}

// Edge-aware 3x3 after the a-trous pass in iMMERSE's RTGI: taps weighted by agreement on defocus,
// coverage and luminance, so sampling noise averages out without crossing a silhouette or layer edge.

// One tap, written out eight times: a stacked [unroll] over a body with `continue` is a shape DXC
// refuses to unroll.
void DenoiseTap(float2 t, float4 f0, float4 n0, float coc0, float lc, float lcn, float lumStop,
                inout float4 fSum, inout float fW, inout float4 nSum, inout float nW,
                inout float2 lrange)
{
    const float3 LW = float3(0.2126, 0.7152, 0.0722);

    float4 ft = tex2Dlod(sFar,  float4(t, 0.0, 0.0));
    float4 nt = tex2Dlod(sNear, float4(t, 0.0, 0.0));

    // Defocus: a tap with a different CoC is a different surface. This is the stop that holds
    // silhouettes together, because depth is the only geometry a DoF pass can see. Half weight at
    // 2.5 px of CoC difference.
    float dc = tex2Dlod(sGatherCoC, float4(t, 0.0, 0.0)).r - coc0;
    float wc = exp2(-dc * dc * 0.16);

    // Coverage: the near layer's own boundary, which has nothing to do with depth. Half weight at
    // a quarter of full coverage.
    float da = nt.a - n0.a;
    float wa = exp2(-da * da * 16.0);

    float lf = dot(ft.rgb, LW);
    float ln = dot(nt.rgb, LW);

    float wf = exp2(-(lf - lc) * (lf - lc) / (lumStop * lumStop)) * wc;
    float wn = exp2(-(ln - lcn) * (ln - lcn) / (lumStop * lumStop)) * wa;

    fSum += ft * (wf * Denoise);
    fW   += wf * Denoise;
    nSum += nt * (wn * Denoise);
    nW   += wn * Denoise;

    lrange.x = min(lrange.x, lf);
    lrange.y = max(lrange.y, lf);
}

// The tap's uv. Uses the caller's p, hi and rcp, which exist only to keep eight calls short.
#define DN_TEXEL(dx, dy) ((float2(clamp(p + int2(dx, dy), int2(0, 0), hi)) + 0.5) * rcp)

[shader("compute")]
void DoFDenoise_CS(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= GATHER_W || tid.y >= GATHER_H)
        return;

    int2   p   = int2(tid.xy);
    int2   hi  = int2(GATHER_W - 1, GATHER_H - 1);
    float2 rcp = float2(RCP_GATHER_W, RCP_GATHER_H);
    float2 uv  = (float2(p) + 0.5) * rcp;

    const float3 LW = float3(0.2126, 0.7152, 0.0722);

    float4 f0 = tex2Dlod(sFar,  float4(uv, 0.0, 0.0));
    float4 n0 = tex2Dlod(sNear, float4(uv, 0.0, 0.0));
    float  coc0 = tex2Dlod(sGatherCoC, float4(uv, 0.0, 0.0)).r;

    float lc  = dot(f0.rgb, LW);
    float lcn = dot(n0.rgb, LW);

    // Luminance tolerance in linear luma. A wide disc is sampled by fewer taps per unit area, so it
    // has more variance and may look further afield for samples that agree with it.
    float lumStop = 0.08 + 0.02 * f0.a;

    float4 fSum = f0, nSum = n0;
    float  fW = 1.0, nW = 1.0;
    float2 lrange = float2(lc, lc);

    DenoiseTap(DN_TEXEL(-1, -1), f0, n0, coc0, lc, lcn, lumStop, fSum, fW, nSum, nW, lrange);
    DenoiseTap(DN_TEXEL( 0, -1), f0, n0, coc0, lc, lcn, lumStop, fSum, fW, nSum, nW, lrange);
    DenoiseTap(DN_TEXEL( 1, -1), f0, n0, coc0, lc, lcn, lumStop, fSum, fW, nSum, nW, lrange);
    DenoiseTap(DN_TEXEL(-1,  0), f0, n0, coc0, lc, lcn, lumStop, fSum, fW, nSum, nW, lrange);
    DenoiseTap(DN_TEXEL( 1,  0), f0, n0, coc0, lc, lcn, lumStop, fSum, fW, nSum, nW, lrange);
    DenoiseTap(DN_TEXEL(-1,  1), f0, n0, coc0, lc, lcn, lumStop, fSum, fW, nSum, nW, lrange);
    DenoiseTap(DN_TEXEL( 0,  1), f0, n0, coc0, lc, lcn, lumStop, fSum, fW, nSum, nW, lrange);
    DenoiseTap(DN_TEXEL( 1,  1), f0, n0, coc0, lc, lcn, lumStop, fSum, fW, nSum, nW, lrange);

    float4 fo = fSum / max(fW, 1e-6);
    float  lo = dot(fo.rgb, LW);

    // Averaging can overshoot the values around it, and an overshoot is the one thing a denoiser must
    // not add. This is the same clamp RTGI applies on its early a-trous iterations.
    if (lo > 1e-5)
        fo.rgb *= clamp(lo, lrange.x, lrange.y) / lo;

    tex2Dstore(stFarS,  p, fo);
    tex2Dstore(stNearS, p, nSum / max(nW, 1e-6));
}

// Validity-aware upsample: each corner weighted by defocus agreement, so a texel from the other side
// of a depth step is not used.
void UpSampleLayers(float2 uv, float cocPx, out float4 farOut, out float4 nearOut)
{
    float2 g  = uv * float2(GATHER_W, GATHER_H) - 0.5;
    float2 b  = floor(g);
    float2 f  = g - b;
    float2 hi = float2(GATHER_W - 1, GATHER_H - 1);

    float4 fs = 0.0, ns = 0.0;
    float  ws = 0.0;

    [unroll]
    for (int dy = 0; dy <= 1; ++dy)
    {
        [unroll]
        for (int dx = 0; dx <= 1; ++dx)
        {
            float2 c   = clamp(b + float2((float)dx, (float)dy), float2(0.0, 0.0), hi);
            float2 tuv = (c + 0.5) * float2(RCP_GATHER_W, RCP_GATHER_H);

            float4 ft = tex2Dlod(sFarS,  float4(tuv, 0.0, 0.0));
            float4 nt = tex2Dlod(sNearS, float4(tuv, 0.0, 0.0));
            float  tc = tex2Dlod(sGatherCoC, float4(tuv, 0.0, 0.0)).r;

            // Asymmetric on purpose: a texel blurred more than this pixel is a halo arriving, which is
            // exactly what has to be let through. A texel sharper than the pixel is what drags sharp
            // colour into a blurred edge. Comparing for equality rejected both.
            float2 bw = float2((dx == 0) ? (1.0 - f.x) : f.x, (dy == 0) ? (1.0 - f.y) : f.y);
            float  sharper = min(max(cocPx - tc, 0.0), 40.0);   // clamped so the weight cannot underflow
            float  w  = bw.x * bw.y * exp2(-sharper * sharper * 0.02);

            fs += ft * w;
            ns += nt * w;
            ws += w;
        }
    }

    farOut  = fs / max(ws, 1e-30);
    nearOut = ns / max(ws, 1e-30);
}

float4 DoFComposite_PS(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float3 sharp = SampleSharp(uv);
    float2 coc   = tex2Dlod(sCoC,  float4(uv, 0.0, 0.0)).xy;

    if (DebugView == 1)
        return float4(coc.x / max(MaxBlurNear, 1e-3), coc.y / max(MaxBlurFar, 1e-3), 0.0, 1.0);

    if (DebugView == 2)
        return float4(tex2Dlod(sDilate, float4(uv, 0.0, 0.0)).rrr / max(MaxBlurNear, 1e-3), 1.0);

    if (DebugView == 3)
    {
        float2 cat = CatEyeAt(uv);
        float2 d   = (uv - 0.5) * ApertureSize;
        return float4(PupilShape(d, cat.x, cat.y, 0.6 * rsqrt((float)TAP_COUNT)).xxx, 1.0);
    }

    // This pixel's own blur. It is only used for the sub-pixel bridge and the upsample's validity
    // check now: the layers carry their own coverage, so nothing downstream has to infer a blend
    // weight from this number.
    float  ownCoc = max(coc.x, coc.y);

    float4 farT, nearT;
    UpSampleLayers(uv, ownCoc, farT, nearT);

    if (DebugView == 6)
        return float4(tex2Dlod(sColour, float4(uv, 0.0, 0.0)).rgb, 1.0);

    if (DebugView == 4)
        return float4(nearT.rgb, 1.0);
    if (DebugView == 5)
        return float4(farT.rgb, 1.0);

    if (DebugView == 7)
        return float4(nearT.aaa, 1.0);

    // How much of this pixel is neither layer, which is the share the sharp frame keeps.
    float  aNear     = saturate(nearT.a);
    float  aFar      = saturate(farT.a);
    float  uncovered = saturate(1.0 - aNear - aFar);

    // A half-resolution gather cannot represent a disc under about 2 px, so cover that band with a
    // full-resolution 3x3. Only where the layers do not already own the pixel: the result is scaled by
    // `uncovered`, so inside a blurred region the fetches would be discarded.
    float3 colour = sharp;
    float  small  = saturate((ownCoc - 0.5) / 1.5);
    if (small > 0.01 && uncovered > 0.05)
    {
        float2 t = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
        float3 b = SampleSharp(uv + float2(-t.x, -t.y))
                 + SampleSharp(uv + float2( 0.0, -t.y))
                 + SampleSharp(uv + float2( t.x, -t.y))
                 + SampleSharp(uv + float2(-t.x,  0.0))
                 + sharp
                 + SampleSharp(uv + float2( t.x,  0.0))
                 + SampleSharp(uv + float2(-t.x,  t.y))
                 + SampleSharp(uv + float2( 0.0,  t.y))
                 + SampleSharp(uv + float2( t.x,  t.y));
        colour = lerp(colour, b * (1.0 / 9.0), small);
    }

    // One addition: both layers arrive premultiplied by complementary coverages, so the remainder is
    // the sharp frame. A halo ends where the foreground's samples stop reaching.
    colour = colour * uncovered + nearT.rgb + farT.rgb;

    // A NaN upstream would turn the whole frame black.
    colour = (dot(colour, colour) < 3.0e38) ? colour : sharp;

    if (LinearBlend)
        colour = LinearToSRGB3(colour);

    return float4(colour, 1.0);
}

/*==============================================================================
    Technique
==============================================================================*/

technique HC_DoF <
    ui_label = "HC Depth of Field";
    ui_tooltip = "Scatter-as-gather depth of field through the shared aperture, with the cat's-eye clip rotating toward the frame corners.\n\nNeeds depth: check the RESHADE_DEPTH_INPUT_* flags with DisplayDepth.fx first, and set RESHADE_DEPTH_LINEARIZATION_FAR_PLANE to the scene's far plane.";
>
{
    pass DoFCoC       { ComputeShader = DoFCoC_CS<16, 16, 1>;   DispatchSizeX = (BUFFER_WIDTH + 15) / 16;  DispatchSizeY = (BUFFER_HEIGHT + 15) / 16; DispatchSizeZ = 1; }
    pass DoFColour    { ComputeShader = DoFColour_CS<16, 16, 1>; DispatchSizeX = (GATHER_W + 15) / 16;     DispatchSizeY = (GATHER_H + 15) / 16;     DispatchSizeZ = 1; }
    pass DoFCoCHalf   { ComputeShader = DoFCoCHalf_CS<8, 8, 1>;   DispatchSizeX = (GATHER_W + 7) / 8;      DispatchSizeY = (GATHER_H + 7) / 8;      DispatchSizeZ = 1; }
    pass DoFCoCSmall  { ComputeShader = DoFCoCSmall_CS<8, 8, 1>;  DispatchSizeX = (GATHER_W + 7) / 8;      DispatchSizeY = (GATHER_H + 7) / 8;      DispatchSizeZ = 1; }
    pass DoFDilate    { ComputeShader = DoFDilate_CS<8, 8, 1>;  DispatchSizeX = (DIL_W + 7) / 8;         DispatchSizeY = (DIL_H + 7) / 8;         DispatchSizeZ = 1; }
    pass DoFGather    { ComputeShader = DoFGather_CS<8, 8, 1>;  DispatchSizeX = (GATHER_W + 7) / 8;        DispatchSizeY = (GATHER_H + 7) / 8;        DispatchSizeZ = 1; }
    pass DoFDenoise   { ComputeShader = DoFDenoise_CS<8, 8, 1>; DispatchSizeX = (GATHER_W + 7) / 8;        DispatchSizeY = (GATHER_H + 7) / 8;        DispatchSizeZ = 1; }
    pass DoFComposite { VertexShader = PostProcessVS; PixelShader = DoFComposite_PS; }
}
