#!/usr/bin/env python
import SimpleITK as sitk

def N4_bias_correction(inputImage, shrinkFactor =2, n_iterations = 50, n_fittingLevels = 4):
    mask = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    image = sitk.Shrink(inputImage, [shrinkFactor] * inputImage.GetDimension())
    mask = sitk.Shrink(mask, [shrinkFactor] * inputImage.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([n_iterations] * n_fittingLevels)
    corrected_image = corrector.Execute(image, mask)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    corrected_image_full_resolution = inputImage / sitk.Exp( log_bias_field )
    
    return corrected_image_full_resolution