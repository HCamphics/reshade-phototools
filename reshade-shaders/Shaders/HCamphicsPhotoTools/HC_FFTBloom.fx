/*==============================================================================

    HC FFT Bloom - kernel texture
    The bloom with its point spread function read from a DDS kernel.

    Author:  HCamphics
    Repo:    https://github.com/HCamphics/reshade-phototools
    Licence: CC BY-NC 4.0 - see LICENSE in the repository root.

    Same core as HC_FFTBloom_Aperture.fx - HC_FFTBloom_Common.fxh holds the full pipeline. Here the PSF
    is a picture rather than a bake, so there is nothing to compute for it beyond its transform:

        DustSum       normalise the dust map
        KernelSum     energy and statistics of the kernel
        KernelFFTX/Y  transform it
        KernelCommit  store it with the same normalisation the bake path uses
        ImageFFTX/Y   transform the frame
        IFFTX/Y       multiply by the kernel spectrum and transform back
        Composite     add the glow to the frame

    Kernels are declared in HC_FFTBloom_Common.fxh and read from reshade-shaders/Textures/Kernels. A
    declared file that is missing makes ReShade skip the effect outright, so the folder has to travel
    with the shader. Kernels must be square, power-of-two, 8-bit BGRA DDS.

==============================================================================*/

// 0 = fast (default), 1 = high quality (slower bake, finer dispersion). Can be overridden from the
// preset's preprocessor definitions without editing this file.
#ifndef FTB_QUALITY
    #define FTB_QUALITY 0
#endif

#define KERNEL_MODE 0
#include "HC_FFTBloom_Common.fxh"
