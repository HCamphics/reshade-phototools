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

All three share `HC_Aperture.fxh`, so aperture settings are the same between them.

## Requirements

- ReShade 6.8 or newer.
- Any API with compute shader support: D3D11, D3D12, Vulkan, OpenGL 4.3+.
- `HC_DoF.fx` also needs a readable depth buffer.

## Install

1. Merge `reshade-shaders/` into `<game>/reshade-shaders/`.

## Licence

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) - free to use, modify and share
for non-commercial purposes, with credit and a link back. See `LICENSE`. Developed with AI
assistance. Human-authored contributions: the design, control set, kernel set, tuning, testing,
and all modifications to generated code.

## Credits

Every reference this project used - optics and depth of field, denoising, the FFT bloom work, lens dust,
colour handling and tooling - is listed in [CREDITS.md](CREDITS.md).
