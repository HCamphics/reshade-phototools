# Credits and references

Everything this project was built with, read, or measured against, grouped by the part of the code it
touched. Licences are noted where they impose something: nothing here is copied code unless said so.

## Optics and depth of field

- **Potmesil & Chakravarty, "Synthetic Image Generation with a Lens and Aperture Camera Model"** (1981) -
  the layered model this depth of field still follows: near and far layers, composited with complementary
  masks. <https://www.semanticscholar.org/paper/Synthetic-Image-Generation-with-a-Lens-and-Aperture-Potmesil-Chakravarty/c6994e503c563af565022157b358357e6ddf6246>
- **Dr.Bokeh - Sheng et al., "DiffeRentiable Occlusion-aware Bokeh Rendering"** (arXiv 2308.08843) -
  colour bleeding and partial occlusion at object boundaries, and the case for a layered RGBD
  representation over a single gather. <https://arxiv.org/abs/2308.08843>
- **AMD FidelityFX Depth of Field 1.1** - half-resolution gather, CoC dilation, bilateral downsampling
  and validity-aware upsampling.
- **Unreal Engine, Cinematic Depth of Field** - near/far split, dilated CoC, the `Occlusion` control that
  `Halo Spread` descends from, and Adaptive DOF.
  <https://dev.epicgames.com/documentation/unreal-engine/cinematic-depth-of-field-in-unreal-engine>
- **Guillaume Abadie, "A Life of a Bokeh"** (SIGGRAPH 2018) - UE4's DiaphragmDOF, and what a real
  aperture does at the edge of the frame.
- **Kleber Garcia, "Circular Separable Depth of Field"** (Frostbite/EA), with **Olli Niemitalo's** notes on
  circularly symmetric convolution - the separable family, and why a hard-edged disc does not suit it.
- **Erfan Ahmadi, BokehDepthOfField** - the three gather families implemented side by side, including the
  48-sample practical gather this one follows.
  <https://github.com/Erfan-Ahmadi/BokehDepthOfField>
- **GPU Zen, "Practical Gather-based Bokeh Depth of Field"** - the sampling pattern and weighting the
  gather is built on.
- **Dennis Gustafsson, "Bokeh depth of field in a single pass"** - the single-pass alternative, and the
  reasons it is not used here.
- **Bart Wronski** - separable bokeh analysis: why hard-edged discs need a high rank, and why a gather is
  the right family for a pre-made kernel.
- **OtisFX CinematicDOF** (BSD-3-Clause) - control set reference.
- **iMMERSE / MartysMods Depth of Field** by Pascal Gilcher - behaviour and layout reference.
  <https://guides.martysmods.com/shaders/immersepro/depthoffield> 
- **Scratchapixel, "Bokeh effect"** - the thin-lens derivation behind `CoCPixels`.
  <https://www.scratchapixel.com/lessons/digital-imaging/simple-image-manipulations/bookeh-effect.html>
- **Wikipedia, Bokeh - Emulation**, and **Stanford EE367 (Revalla & Thotakura), bokeh emulation report** -
  a survey of real-time methods and their artefacts. <https://en.wikipedia.org/wiki/Bokeh#Emulation>,
  <https://web.stanford.edu/class/ee367/Winter2020/report/revalla_thotakura_report.pdf>
- **lenssimulator.photo/bokeh** - interactive reference for what a given lens and aperture actually
  produce, used to sanity-check the pupil model. <https://lenssimulator.photo/bokeh>


## Bloom and FFT

- **Unreal Engine, Bloom Convolution** - the effect this implements, and the reference for its controls.
  <https://dev.epicgames.com/documentation/en-us/unreal-engine/bloom-in-unreal-engine>
- **Francisco Letterio, "FFT Bloom Optimized to the Bone in Nabla"** (Graphics Programming Discord) -
  Hermitian packing, the single-element butterfly exchange, preloading every channel, the zero-padding and
  FFTShift discussion, and the axis-order analysis. The largest single influence on this implementation.
- **RealBloom** by bean-mhm - behavioural reference. <https://github.com/bean-mhm/realbloom>
- **StellarWarp, High-Performance Convolution Bloom on Unity** - behavioural reference.
  <https://github.com/StellarWarp/High-Performance-Convolution-Bloom-On-Unity>
- **iMMERSE FFT Bloom** by Pascal Gilcher - settings reference.
  <https://guides.martysmods.com/shaders/immerseultimate/convolutionbloom/>
- **crosire, ReShade and reshade-shaders** - the platform, `ReShade.fxh` and the FX compiler. The offline
  checker in `_dev/tools/fxcheck` is built from ReShade's own `reshadefx` sources at v6.8.0, so a shader
  can be validated without launching a game. <https://reshade.me/docs>,
  <https://github.com/crosire/reshade-shaders/blob/slim/REFERENCE.md>

## Lens dust

- **Willson, Maimone, Johnson, Scherr, "An optical model for image artifacts produced by dust particles on
  lenses"** (iSAIRAS 2005) - a speck scatters the light it blocks over the whole aperture rather than
  dimming locally, which is the wing in `BakePSFCS`.
  <https://www-robotics.jpl.nasa.gov/media/documents/3_willson_11b.pdf>

## Colour

- **Lumenite, `LUMENITE_ColorManagement.fxh`** - a worked example of `BUFFER_COLOR_SPACE` handling for
  sRGB, scRGB and HDR buffers, which shaped `Blend In Linear Light`.

## Tooling

- **CeeJayDK, SweetOpt (sopt)** - the rewrite search that produced `pow(s, 1.5)` to `s * sqrt(s)` and the
  `mad` constant folds, and surveyed 141 regions of the depth of field for cheaper equivalents.
- **ReShade 6.8, DXVK/Proton and Mesa RADV** - the environment every measurement in these notes was taken
  in.

## Textures

- The kernels in `Textures/Kernels` and the dust map in `Textures/LensDirt` were made for this project.

