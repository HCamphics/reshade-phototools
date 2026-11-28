# reshade phototools

A ReShade shader collection for in-game photography.
<!-- TODO: screenshots -->

## Contents

**FFT Bloom** - a from-scratch implementation of Unreal Engine's bloom convolution, plus
extensions: a procedural aperture kernel (blades, cat's eye, soap-bubble and hot-centre pupil
weighting), spectral dispersion, beam emphasis, and lens dust that scatters the light.

Two effects share one implementation (`HC_FFTBloom_Common.fxh`) and differ only in where the
kernel comes from:

- `HC_FFTBloom_Aperture.fx` — the kernel is baked on the GPU from the aperture controls.
- `HC_FFTBloom.fx` — the kernel is read from a DDS kernel in `reshade-shaders/Textures/Kernels`.

Enable one or the other. The DDS effect needs the `Kernels` folder; the aperture effect does not.

**Depth of Field** - `HC_DoF.fx`, scatter-as-gather depth of field through the same aperture, with the
cat's-eye clip rotating toward the frame corners instead of being baked into one kernel. Needs the
game's depth buffer: check the `RESHADE_DEPTH_INPUT_*` flags with ReShade's DisplayDepth first, and set
`RESHADE_DEPTH_LINEARIZATION_FAR_PLANE` to your scene's far plane. Half resolution by default; define
`DOF_FULLRES=1` for a full-resolution gather.

Controls worth knowing: `Taps` is the quality knob, since the gather is Monte-Carlo and grain falls as
`1/Taps`. `Denoise` removes that grain with an edge-aware filter that refuses silhouettes and layer
edges; 0 gives the raw gather. `Halo Spread` scales how far a blurred foreground reaches past its own
silhouette, where 1 is physical and there is no pixel cap. `Blend In Linear Light` averages in light
instead of in the buffer's encoded values, which removes the fringe along hard bright edges; leave it
off for HDR output.

Both effects take their pupil shape from `HC_Aperture.fxh`, so blades, apodization and the cat's eye
cannot drift between them.

### Kernels and textures

The kernels in `Textures/Kernels` are included and made by yours truly, more planned. Adding or removing one means declaring it in the shader.

The dust map in `Textures/LensDirt` is a transmission map also made by yours truly, white is a clean lens, black is dirt.

## Requirements
- ReShade 6.8.0 minimum. Any graphics API with **compute shader** support: D3D11 / D3D12 / Vulkan / OpenGL 4.3+

## Install

Merge `reshade-shaders/` into the game's ReShade folder. The shaders sit in
`Shaders/HCamphicsPhotoTools/`.

Presets written for earlier versions set uniforms that no longer exist. ReShade logs them as unknown
and falls back to the defaults, so re-save the preset once to clear the log.

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

