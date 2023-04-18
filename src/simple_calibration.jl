"""
    simpleCalibration(e_uncal::Array, th228_lines::Array, window_size::Float64=25.0, n_bins::Int=15000, calib_type::String="th228")


Perform a simple calibration for the uncalibrated energy array `e_uncal` 
using the calibration type `calib_type` and the calibration lines `th228_lines`. 
The window size is the size of the window around the calibration line to use for the calibration. 
The number of bins is the number of bins to use for the histogram.

Returns 
    * `h_calsimple`: histogram of the calibrated energy array
    * `h_uncal`: histogram of the uncalibrated energy array
    * `c`: calibration factor
    * `fep_guess`: estimated full energy peak (FEP)
    * `peakhists`: array of histograms around the calibration lines
    * `peakstats`: array of statistics for the calibration line fits
"""
function simpleCalibration(e_uncal::Array, th228_lines::Array; window_size::Float64=25.0, n_bins::Int=15000, calib_type::String="th228")
    if calib_type == "th228"
        h_uncal = fit(Histogram, e_uncal, nbins=n_bins)
        fep_guess = quantile(e_uncal, 0.995)
        c = 2614.5 / fep_guess
        h_calsimple = fit(Histogram, e_uncal .* c, nbins=n_bins)
        peakhists = LegendSpecFits.subhist.(Ref(h_calsimple), (x -> (x-window_size, x+window_size)).(th228_lines))
        peakstats = StructArray(estimate_single_peak_stats.(peakhists))
        return h_calsimple, h_uncal, c, fep_guess, peakhists, peakstats
    else
        error("Calibration type not supported")
    end
end
export simpleCalibration

