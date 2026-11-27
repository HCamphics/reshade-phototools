# reshade phototools

A ReShade shader collection for in-game photography.
<!-- TODO: screenshots -->

## Contents

**FFT Bloom** - a from-scratch implementation of Unreal Engine's bloom convolution, plus
extensions: a procedural aperture kernel, spectral dispersion, beam emphasis, and lens dust that
scatters the light.

Two effects share one implementation (`UE_FFTBloom_Common.fxh`) and differ only in where the
kernel comes from:

- `UE_FFTBloom_Aperture.fx` — the kernel is baked on the GPU from the aperture controls.
- `UE_FFTBloom.fx` — the kernel is read from a DDS kernel in `reshade-shaders/Textures/Kernels`.

Enable one or the other. The DDS effect needs the
`Kernels` folder; the aperture effect does not.

### Kernels and textures

The kernels in `Textures/Kernels` are included and made by yours truly, more planned. Adding or removing one means declaring it in the shader.

The dust map in `Textures/LensDirt` is a transmission map also made by yours truly, white is a clean lens, black is dirt.

## Requirements
- ReShade 6.8.0 minimum. Any graphics API with **compute shader** support: D3D11 / D3D12 / Vulkan / OpenGL 4.3+

## Install

Merge `reshade-shaders/` into the game's ReShade folder. The shaders sit in
`Shaders/HCamphicsPhotoTools/`.

## Licence

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free to use, modify and share
for non-commercial purposes, with credit and a link back. See `LICENSE`. Developed with AI
assistance. Human-authored contributions: the design, control set, kernel set, tuning, testing,
and all modifications to generated code.

## Credits
- Unreal Engine's documented Bloom Convolution (behavioural reference)
- [RealBloom](https://github.com/bean-mhm/realbloom) by bean-mhm (behavioral reference)
- [Unity high-performance convolution bloom](https://github.com/StellarWarp/High-Performance-Convolution-Bloom-On-Unity) (behavioral reference)
- [iMMERSE FFT bloom](https://guides.martysmods.com/shaders/immerseultimate/convolutionbloom/) by Pascal Gilcher (settings reference)
- [An optical model for image artifacts produced by dust
  particles on lenses](https://www-robotics.jpl.nasa.gov/media/documents/3_willson_11b.pdf) by Willson, Maimone, Johnson, and Scherr

