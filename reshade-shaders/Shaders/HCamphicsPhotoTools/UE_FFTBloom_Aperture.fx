/*==============================================================================

    UE FFT Bloom - procedural aperture

    Licence: CC BY-NC 4.0 - see LICENSE in the repository root.
    TODO: overview (HCamphics)

==============================================================================*/

// 0 = fast (default), 1 = high quality (slower bake, finer dispersion). Can be overridden from the
// preset's preprocessor definitions without editing this file.
#ifndef FTB_QUALITY
    #define FTB_QUALITY 0
#endif

#define KERNEL_MODE 1
#include "UE_FFTBloom_Common.fxh"
