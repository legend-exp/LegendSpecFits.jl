# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    autocal_energy(e::AbstractArray{<:Real}, photon_lines::Vector{<:Unitful.RealOrRealQuantity}; min_e::Real=100, max_e_binning_quantile::Real=0.5, σ::Real = 2.0, threshold::Real = 50.0, min_n_peaks::Int = length(photon_lines), max_n_peaks::Int = 4 * length(photon_lines), α::Real = 0.01, rtol::Real = 5e-3)

Compute an energy calibration from raw reconstructed energy deposition values based on a given number of known photon lines which are contained in the spectrum
"""
function autocal_energy(e::AbstractArray{<:Real}, photon_lines::Vector{<:Unitful.Energy{<:Real}}; mode::Symbol=:fit, min_e::Real=100, max_e::Real=maximum(e), max_e_binning_quantile::Real=0.5, σ::Real = 2.0, threshold::Real = 50.0, min_n_peaks::Int = length(photon_lines), max_n_peaks::Int = 4 * length(photon_lines), α::Real = 0.01, rtol::Real = 5e-3)
    e_unit = u"keV"
    # binning based on fd with max cut off at certain quantile
    max_e_binning = quantile(e, max_e_binning_quantile)
    bin_width = get_friedman_diaconis_bin_width(filter(x -> min_e < x < max_e_binning, e))
    # get binned uncalibrated histogram
    h_uncal = fit(Histogram, e, min_e:bin_width:max_e)
    if mode == :fit
        @debug "Auto calibrate via peak fitting"
        _, _, peak_positions, threshold, c, _ = RadiationSpectra.calibrate_spectrum(h_uncal, ustrip.(e_unit, photon_lines); min_n_peaks = min_n_peaks, max_n_peaks = max_n_peaks, threshold=threshold, α=α, σ=σ, rtol=rtol)
    elseif mode == :ratio
        @debug "Auto calibrate via peak ratios"
        c, _, peak_positions, threshold = RadiationSpectra.determine_calibration_constant_through_peak_ratios(h_uncal, ustrip.(e_unit, photon_lines), min_n_peaks = min_n_peaks, max_n_peaks = max_n_peaks, threshold=threshold, α=α, σ=σ, rtol=rtol)
    else
        throw(ArgumentError("Unknown mode $mode, only `:fit` and `:ratio` are supported"))
    end
    
    # generate calibratio constant
    f_calib = Base.Fix1(*, c * e_unit)
    
    # generate calibrated energy values and calibrated histogram
    # 0.5 keV standard binning
    cal_hist_binning = 0:0.5:3000
    e_cal = ustrip.(f_calib.(e))
    h_cal = fit(Histogram, e_cal, cal_hist_binning)
    # return result and report
    result = (; f_calib, h_cal, h_uncal, c, peak_positions, threshold)
    report = result
    return result, report
end
export autocal_energy