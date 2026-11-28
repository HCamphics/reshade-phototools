# reshade phototools

Shaders for in-game photography: a physically shaped FFT bloom and a matching depth of field, both
driven by the same lens aperture.

<!-- TODO: screenshots -->

## Effects

| shader | what it is | needs |
|---|---|---|
| `HC_FFTBloom_Aperture.fx` | FFT bloom, point spread function baked from the aperture controls | compute shaders |
| `HC_FFTBloom.fx` | the same bloom with its PSF read from a DDS kernel | compute shaders, `Textures/Kernels` |
| `HC_DoF.fx` | scatter-as-gather depth of field through the same aperture | compute shaders, game depth buffer |

Enable one bloom, not both - they are the same core with a different kernel source.

All three share `HC_Aperture.fxh`, so blades, apodization and the cat's-eye clip cannot drift between
them. The DoF evaluates the pupil per tap, which lets the clip rotate with the pixel's field angle -
something a single baked kernel cannot do.

## Requirements

- ReShade 6.8 or newer.
- Any API with compute shader support: D3D11, D3D12, Vulkan, OpenGL 4.3+.
- `HC_DoF.fx` also needs a readable depth buffer.

## Install

1. Copy `reshade-shaders/Shaders/HCamphicsPhotoTools` into `<game>/reshade-shaders/Shaders`.
2. For `HC_FFTBloom.fx`, also copy `reshade-shaders/Textures` - the kernels and the dust map.
3. Enable the effect in ReShade's list.

Presets written for earlier versions set uniforms that no longer exist. ReShade falls back to the
defaults, so re-save the preset once to clear the log.

## Depth of field

Set up depth before anything else: run `DisplayDepth.fx` (in the same folder) to find the
`RESHADE_DEPTH_INPUT_*` flags your game needs, then set `RESHADE_DEPTH_LINEARIZATION_FAR_PLANE` to the
scene's far plane. The focus distance is in those units, so if that define is wrong, focus is scaled
with it.

Half resolution by default. Define `DOF_FULLRES=1` for a full-resolution gather, at about four times
the cost.

| control | what it does |
|---|---|
| `Focus Distance`, `Focal Length`, `Aperture` | the thin lens: what stays sharp, and how fast focus falls away |
| `Max Blur Near`, `Max Blur Far` | ceilings on the disc radius, in pixels |
| `Taps` | quality: the gather is Monte-Carlo, so grain falls as `1/Taps` |
| `Denoise` | edge-aware clean-up of the gathered layers, never across a silhouette; 0 is the raw gather |
| `Halo Spread` | how far a blurred foreground reaches past its silhouette, as a multiple of its own CoC |
| `Highlight Boost`, `LoCA` | bokeh highlights, and longitudinal chromatic aberration |
| `Blend In Linear Light` | average in light instead of the buffer's encoded values; leave off for HDR |
| `Focus Helper` | whites out what is inside the depth of field, at a fixed half, so the plane can be dialled by eye; nothing outside it is touched, and it never draws over the debug views |
| `Debug View` | CoC, dilated CoC, aperture, near and far layers, coverage |

## Bloom

The shared core transforms the point spread function once, then each frame is a transform, a multiply
by the kernel spectrum and the inverse. Cost follows the FFT size rather than the kernel size, so an
elaborate aperture costs the same as a plain disc. 15 passes in the aperture build, 10 in the kernel
build; each file's header lists them.

The bake only runs when a kernel control changes, so adjusting it costs one frame.

| control | what it does |
|---|---|
| `Intensity`, `Threshold`, `Soft Knee`, `Highlight Shoulder` | how much light blooms and how clipped whites are treated |
| `Kernel File` | which DDS kernel `HC_FFTBloom.fx` reads |
| `Cat Eye`, `Cat Eye Angle`, `Dispersion`, `Beam Emphasis` | aperture shape, used by the bake |
| `Scale`, `Rotation`, `Anamorphic Stretch` | kernel placement and stretch |
| `Lens Dirt`, `Dust Gain`, `Dust Scatter`, `Dust Tiling` | the dust map, and how its specks scatter light |
| `Blend Mode` | Additive is the physical one; Lerp is Unreal's artistic composite |

Define `FTB_QUALITY=1` for finer dispersion at a slower bake.

## Kernels and textures

- `Textures/Kernels` - square, power-of-two, 8-bit BGRA DDS, one per look. They are declared in
  `HC_FFTBloom_Common.fxh`, and a declared file that is missing makes ReShade skip the whole effect,
  so the folder has to travel with the shaders.
- `Textures/LensDirt` - a transmission map: white is a clean lens, black is dirt.

## Licence

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) - free to use, modify and share
for non-commercial purposes, with credit and a link back. See `LICENSE`. Developed with AI
assistance. Human-authored contributions: the design, control set, kernel set, tuning, testing,
and all modifications to generated code.

## Credits

Every reference this project used - optics and depth of field, denoising, the FFT bloom work, lens dust,
colour handling and tooling - is listed in [CREDITS.md](CREDITS.md).
