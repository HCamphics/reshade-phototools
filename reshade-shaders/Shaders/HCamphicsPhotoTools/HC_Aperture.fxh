/*==============================================================================

    HC Aperture - the pupil shape, shared by the bloom bake and the depth-of-field gather.

    Repo:    https://github.com/HCamphics/reshade-phototools
    Licence: CC BY-NC 4.0 - see LICENSE in the repository root.

    Both effects ask the same question: given a point on the pupil, how much light gets through? The
    bloom bakes the answer into a PSF; the depth of field evaluates it per tap, per pixel, which lets
    it rotate the cat's-eye clip with the pixel's field angle - something a single baked kernel cannot
    do. Keeping one function means the two effects cannot drift apart.

    ApertureMask() in the bloom is now a thin wrapper; do not re-implement the shape there.

==============================================================================*/

#ifndef HC_APERTURE_INCLUDED
#define HC_APERTURE_INCLUDED

// Self-contained: the depth-of-field effect includes this without the bloom around it.
static const float HC_TAU = 6.28318530718;

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

uniform float ApertureSize <
    ui_type = "drag"; ui_min = 0.005; ui_max = 0.6; ui_step = 0.001;
    ui_category = "Aperture";
    ui_label = "Size";
    ui_tooltip = "Pupil size on the grid. Smaller means a wider glow, like stopping down; below 0.2 it aliases. 0.35-0.5 is clean.";
> = 0.35;

void ApertureParams(out float sides, out float curvature)
{
    sides = clamp((float)ApertureSides, 3.0, 32.0);
    curvature = ApertureBladeCurvature;
}

float PupilShape(float2 d, float catEye, float catEyeAngle, float pixSoft)
{
    float sides, curvature;
    ApertureParams(sides, curvature);

    float sector = HC_TAU / sides;

    float radius = length(d) * 2.0 / max(ApertureSize, 1e-4);
    float th = atan2(d.y, d.x);

    float a = th - sector * floor(th / sector) - 0.5 * sector;

    float aa = a * (1.0 - 0.995 * saturate(curvature));
    float edge = cos(0.5 * sector) / max(cos(aa), 1e-3);

    float rn = radius / max(edge, 1e-4);

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
    catEye = saturate(catEye);
    if (catEye > 0.0)
    {
        // Offset in pupil radii. The range covers the traced corners: 0.62 is a Helios 44-2 at f/2
        // (61% of the pupil left), 0.72 a Noct 58/0.95 (55%), 0.85 a Canon RF 85/1.2 (48%).
        // Stopping down pulls the iris inside the rims, so the bite falls with the pupil area: at
        // the traced fall-off that takes the Helios from 61% to 95% over two stops, which this
        // gives as 48% -> 87%. The clamp stops the strongest setting short of a sliver.
        float ap  = ApertureSize / 0.35;
        float u   = min(0.85 * catEye * ap * ap, 0.90);
        float ang = radians(catEyeAngle);
        float2 q  = d * (2.0 / max(ApertureSize, 1e-4)) - float2(cos(ang), sin(ang)) * u;
        // The rim sits well away from the pupil plane, so its projected edge is defocused: wider
        // than the blade edge, which also keeps the cusp diffraction from reading as streaks.
        float rimSoft = clamp(3.0 * soft, 0.01, 0.35);
        mask *= 1.0 - smoothstep(1.0 - rimSoft, 1.0 + 0.5 * rimSoft, length(q));
    }

    return max(mask, 0.0);
}

#endif // HC_APERTURE_INCLUDED
