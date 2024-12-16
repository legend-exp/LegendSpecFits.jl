# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    autocal_energy(e::AbstractArray{<:Real}, photon_lines::Vector{<:Unitful.RealOrRealQuantity}; min_e::Real=100, max_e_binning_quantile::Real=0.5, σ::Real = 2.0, threshold::Real = 50.0, min_n_peaks::Int = length(photon_lines), max_n_peaks::Int = 4 * length(photon_lines), α::Real = 0.01, rtol::Real = 5e-3)

Compute an energy calibration from raw reconstructed energy deposition values based on a given number of known photon lines which are contained in the spectrum.

# Arguments
    * `e`: Calibrated energies
    * `photon_lines`: Known photon lines in the spectrum

# Keywords
    * `min_e`: Minimum energy
    * `max_e_binning_quantile`: Maximum energy binning quantile
    * `threshold`: Calibration theshold
    * `min_n_peaks`: Minimum number of peaks
    * `max_n_peaks`: Maximum number of peaks
    * `α: 
    * `rtol`: Relative tolerance

# Returns
    * `f_calib`: Calibration constant
    * `h_cal`: Calibrated histogram
    * `h_uncal`: Uncalibratd histogram
    * `c`: Constant
    * `c_precal`: 
    * `threshold`: calibration threshold
    
"""
function autocal_energy(e::AbstractArray{<:Real}, photon_lines::Vector{<:Unitful.RealOrRealQuantity}; min_e::Real=100, max_e_binning_quantile::Real=0.5, σ::Real = 2.0, threshold::Real = 50.0, min_n_peaks::Int = length(photon_lines), max_n_peaks::Int = 4 * length(photon_lines), α::Real = 0.01, rtol::Real = 5e-3)
    # binning based on fd with max cut off at certain quantile
    max_e_binning = quantile(e, max_e_binning_quantile)
    bin_width = get_friedman_diaconis_bin_width(filter(x -> min_e < x < max_e_binning, e))
    # get binned uncalibrated histogram
    h_uncal = fit(Histogram, e, min_e:bin_width:maximum(e))
    _, _, _, threshold, c, c_precal = RadiationSpectra.calibrate_spectrum(h_uncal, ustrip.(photon_lines); min_n_peaks = min_n_peaks, max_n_peaks = max_n_peaks, threshold=threshold, α=α, σ=σ, rtol=rtol)
    
    # generate calibration constant
    f_calib = Base.Fix1(*, c * u"keV")
    
    # generate calibrated energy values and calibrated histogram
    # 0.5 keV standard binning
    cal_hist_binning = 0:0.5:3000
    e_cal = ustrip.(f_calib.(e))
    h_cal = fit(Histogram, e_cal, cal_hist_binning)
    # return result and report
    result = (f_calib = f_calib, h_cal = h_cal, h_uncal = h_uncal, c = c, c_precal = c_precal, threshold = threshold)
    report = (h_cal = h_cal, h_uncal = h_uncal, c = c, c_precal = c_precal, threshold = threshold)
    return result, report
end
export autocal_energy