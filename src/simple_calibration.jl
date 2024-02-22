"""
    simple_calibration(e_uncal::Array, th228_lines::Array, window_size::Float64=25.0, n_bins::Int=15000, calib_type::String="th228")


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
function simple_calibration end
export simple_calibration

function simple_calibration(e_uncal::Vector{<:Real}, th228_lines::Vector{<:Unitful.Energy{<:Real}}, window_sizes::Vector{<:Tuple{Unitful.Energy{<:Real}, Unitful.Energy{<:Real}}},; kwargs...)
    # remove calib type from kwargs
    @assert haskey(kwargs, :calib_type) "Calibration type not specified"
    calib_type = kwargs[:calib_type]
    # remove :calib_type from kwargs
    kwargs = pairs(NamedTuple(filter(k -> !(:calib_type in k), kwargs)))
    if calib_type == :th228
        @info "Use simple calibration for Th228 lines"
        return simple_calibration_th228(e_uncal, th228_lines, window_sizes,; kwargs...)
    else
        error("Calibration type not supported")
    end
end
simple_calibration(e_uncal::Vector{<:Real}, th228_lines::Vector{<:Unitful.Energy{<:Real}}, left_window_sizes::Vector{<:Unitful.Energy{<:Real}}, right_window_sizes::Vector{<:Unitful.Energy{<:Real}}; kwargs...) = simple_calibration(e_uncal, th228_lines, [(l,r) for (l,r) in zip(left_window_sizes, right_window_sizes)],; kwargs...)


function simple_calibration_th228(e_uncal::Vector{<:Real}, th228_lines::Vector{<:Unitful.Energy{<:Real}}, window_sizes::Vector{<:Tuple{Unitful.Energy{<:Real}, Unitful.Energy{<:Real}}},; n_bins::Int=15000, quantile_perc::Float64=NaN, proxy_binning_peak::Unitful.Energy{<:Real}=2103.5u"keV", proxy_binning_peak_window::Unitful.Energy{<:Real}=10.0u"keV")
    # create initial peak search histogram
    h_uncal = fit(Histogram, e_uncal, nbins=n_bins)
    # search all possible peak candidates
    _, peakpos = RadiationSpectra.peakfinder(h_uncal, σ=5.0, backgroundRemove=true, threshold=10)
    # the FEP ist the last peak in the list
    fep_guess = if isnan(quantile_perc)
        sort(peakpos)[end]
    else
        quantile(e_uncal, quantile_perc)
    end
    # get calibration constant for simple calibration
    c = 2614.5*u"keV" / fep_guess
    e_simple = e_uncal .* c
    bin_window_cut = proxy_binning_peak - proxy_binning_peak_window .< e_simple .< proxy_binning_peak + proxy_binning_peak_window
    # get optimal binning for simple calibration
    bin_width  = get_friedman_diaconis_bin_width(e_simple[bin_window_cut])
    # create histogram for simple calibration
    e_min, e_max = 0u"keV", 3000u"keV"
    e_unit = u"keV"
    h_calsimple = fit(Histogram, ustrip.(e_unit, e_simple), ustrip.(e_unit, e_min:bin_width:e_max))
    # get histograms around calibration lines and peakstats
    peakhists = LegendSpecFits.subhist.(Ref(h_calsimple), [ustrip.((peak-first(window), peak+last(window))) for (peak, window) in zip(th228_lines, window_sizes)])
    # peakhists = LegendSpecFits.subhist.([e_simple[peak-window .< e_simple .< peak+window] for (peak, window) in zip(th228_lines, window_sizes)])
    peakstats = StructArray(estimate_single_peak_stats.(peakhists))
    result = (
        h_calsimple = h_calsimple, 
        h_uncal = h_uncal, 
        c = c,
        bin_width = bin_width,
        fep_guess = fep_guess, 
        peakhists = peakhists, 
        peakstats = peakstats
        )
    report = (
        h_calsimple = result.h_calsimple,
        h_uncal = result.h_uncal,
        c = result.c,
        fep_guess = result.fep_guess,
        peakhists = result.peakhists,
        peakstats = result.peakstats
    )
    return result, report
end




