# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


struct EnergyCalibrationDiagnostics{H<:Histogram{<:Integer,1}}
    cal_hist::H
end


"""
    autocal_energy(E_raw::AbstractArray{<:Real})

Compute an energy calibration from raw reconstructed energy deposition values.
"""
function autocal_energy(E_raw::AbstractArray{<:Real},; quantile_perc::Real=0.995)
    # 0.5 keV standard binning
    cal_hist_binning = 0:0.5:3000
    # eiher generate FEP guess by quantile or use NaN to use peakfinder
    if isnan(quantile_perc)
        h_uncal = fit(Histogram, E_raw, 0:1.0:maximum(E_raw))
        _, peakpos = RadiationSpectra.peakfinder(h_uncal, σ=5.0, backgroundRemove=true, threshold=10)
        fep_guess = sort(peakpos)[end]
    else
        fep_guess = quantile(E_raw, quantile_perc)
    end
    # generate calibratio constant
    calib_constant = 2614.5 / fep_guess
    # generate calibration function
    f_calib = Base.Fix1(*, calib_constant * u"keV")
    # generate calibrated energy values and calibrated histogram
    E_cal_keV = ustrip.(f_calib.(E_raw))
    cal_hist = fit(Histogram, E_cal_keV, cal_hist_binning)
    return (result = f_calib, diagnostics = EnergyCalibrationDiagnostics(cal_hist))
end

export autocal_energy
