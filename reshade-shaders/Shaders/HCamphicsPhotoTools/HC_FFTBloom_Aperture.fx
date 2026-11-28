/*==============================================================================

    HC FFT Bloom - procedural aperture
    The bloom with its point spread function baked from the aperture controls.

    Author:  HCamphics
    Repo:    https://github.com/HCamphics/reshade-phototools
    Licence: CC BY-NC 4.0 - see LICENSE in the repository root.

    Same core as HC_FFTBloom.fx - HC_FFTBloom_Common.fxh holds the full pipeline. Here the PSF comes
    from the pupil in HC_Aperture.fxh, so blades, apodization, cat's eye, dispersion, beam emphasis and
    lens dust all reach it. The bake only does work when one of those controls changes:

        BakeAperture  pupil and dust -> PSF field
        DustSum       normalise the dust map
        BakeFFTX/Y    transform the PSF
        BakePSF       spectral dispersion and beam emphasis
        BakeCommit    energy and phase normalisation, stores the signature
        KernelSum     energy and statistics
        KernelFFTX/Y  transform the kernel
        KernelCommit
        ImageFFTX/Y   transform the frame
        IFFTX/Y       multiply by the kernel spectrum and transform back
        Composite     add the glow to the frame

    The bake runs on a 2x oversampled grid so the PSF holds its shape at small radii; FTB_QUALITY
    trades bake time for dispersion detail. No texture folder is needed.

==============================================================================*/

// 0 = fast (default), 1 = high quality (slower bake, finer dispersion). Can be overridden from the
// preset's preprocessor definitions without editing this file.
#ifndef FTB_QUALITY
    #define FTB_QUALITY 0
#endif

#define KERNEL_MODE 1
#include "HC_FFTBloom_Common.fxh"
