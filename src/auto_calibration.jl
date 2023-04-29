# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


struct EnergyCalibrationDiagnostics{H<:Histogram{<:Integer,1}}
    cal_hist::H
end


"""
    autocal_energy(E_raw::AbstractArray{<:Real})

Compute an energy calibration from raw reconstructed energy deposition values.
"""
function autocal_energy(E_raw::AbstractArray{<:Real})
    window_size = 25.0
    n_bins = 15000
    th228_lines = [2614.50]
    cal_hist_binning = 0:0.5:3000
    h_calsimple, h_uncal, calib_constant, fep_guess, peakhists, peakstats = simpleCalibration(
        E_raw, th228_lines,;
        window_size = window_size, n_bins = n_bins, calib_type = "th228"
    )
    f_calib = Base.Fix1(*, calib_constant * u"keV")
    E_cal_keV = ustrip.(f_calib.(E_raw))
    cal_hist = fit(Histogram, E_cal_keV, cal_hist_binning)
    return (result = f_calib, diagnostics = EnergyCalibrationDiagnostics(cal_hist))
end

export autocal_energy
