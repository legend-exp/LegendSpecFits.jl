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
        @debug "Use simple calibration for Th228 lines"
        return simple_calibration_th228(e_uncal, th228_lines, window_sizes,; kwargs...)
    else
        error("Calibration type not supported")
    end
end
simple_calibration(e_uncal::Vector{<:Real}, th228_lines::Vector{<:Unitful.Energy{<:Real}}, left_window_sizes::Vector{<:Unitful.Energy{<:Real}}, right_window_sizes::Vector{<:Unitful.Energy{<:Real}}; kwargs...) = simple_calibration(e_uncal, th228_lines, [(l,r) for (l,r) in zip(left_window_sizes, right_window_sizes)],; kwargs...)


function simple_calibration_th228(e_uncal::Vector{<:Real}, th228_lines::Vector{<:Unitful.Energy{<:Real}}, window_sizes::Vector{<:Tuple{Unitful.Energy{<:Real}, Unitful.Energy{<:Real}}},; n_bins::Int=15000, quantile_perc::Float64=NaN, binning_peak_window::Unitful.Energy{<:Real}=10.0u"keV")
    # initial binning
    bin_width = get_friedman_diaconis_bin_width(filter(in(quantile(e_uncal, 0.05)..quantile(e_uncal, 0.5)), e_uncal))
    # create initial peak search histogram
    h_uncal = fit(Histogram, e_uncal, 0:bin_width:maximum(e_uncal))
    fep_guess = if isnan(quantile_perc)
        # expect FEP in the last 10% of the data
        min_e_fep = quantile(e_uncal, 0.9)
        h_fepsearch = fit(Histogram, e_uncal, min_e_fep:bin_width:maximum(e_uncal))
        # search all possible peak candidates
        h_decon, peakpos = RadiationSpectra.peakfinder(h_fepsearch, σ=5.0, backgroundRemove=true, threshold=10)
        # the FEP is the most prominent peak in the deconvoluted histogram
        peakpos_idxs = StatsBase.binindex.(Ref(h_decon), peakpos)
        cts_peakpos = h_decon.weights[peakpos_idxs]
        peakpos[argmax(cts_peakpos)]
    else
        quantile(e_uncal, quantile_perc)
    end
    # get calibration constant for simple calibration
    c = 2614.5*u"keV" / fep_guess
    e_simple = e_uncal .* c
    e_unit = u"keV"
    # get peakhists and peakstats
    peakhists, peakstats, h_calsimple, bin_widths = get_peakhists_th228(e_simple, th228_lines, window_sizes; binning_peak_window=binning_peak_window, e_unit=e_unit)
    
    result = (
        h_calsimple = h_calsimple, 
        h_uncal = h_uncal, 
        c = c,
        unit = e_unit,
        bin_width = median(bin_widths),
        fep_guess = fep_guess,
        peakbinwidths = bin_widths,
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


"""
    get_peakhists_th228(e::Array, th228_lines::Array, window_sizes::Array, e_unit::String="keV", proxy_binning_peak::Float64=2103.5, proxy_binning_peak_window::Float64=10.0)

Create histograms around the calibration lines and return the histograms and the peak statistics.
# Returns
    * `peakhists`: array of histograms around the calibration lines
    * `peakstats`: array of statistics for the calibration line fits
"""
function get_peakhists_th228(e::Vector{<:Unitful.Energy{<:Real}}, th228_lines::Vector{<:Unitful.Energy{<:Real}}, window_sizes::Vector{<:Tuple{Unitful.Energy{<:Real}, Unitful.Energy{<:Real}}},; e_unit::Unitful.EnergyUnits=u"keV", binning_peak_window::Unitful.Energy{<:Real}=10.0u"keV")
    # get optimal binning for simple calibration
    bin_widths = [get_friedman_diaconis_bin_width(filter(in(peak - binning_peak_window..peak + binning_peak_window), e)) for peak in th228_lines]
    # Main.@infiltrate
    # create histogram for simple calibration
    e_min, e_max = 0u"keV", 3000u"keV"
    h = fit(Histogram, ustrip.(e_unit, e), ustrip(e_unit, e_min):ustrip(e_unit, median(bin_widths)):ustrip(e_unit, e_max))
    # get histograms around calibration lines and peakstats
    peakenergies = [ustrip.(e_unit, filter(in(peak - first(window)..peak + last(window)), e)) for (peak, window) in zip(th228_lines, window_sizes)]
    peakstats = StructArray(estimate_single_peak_stats.(peakenergies, ustrip.(e_unit, bin_widths)))
    peakhists = [fit(Histogram, ustrip.(e_unit, e), ustrip(e_unit, peak - first(window)):ps.bin_width:ustrip(e_unit, peak + last(window))) for (peak, window, ps) in zip(th228_lines, window_sizes, peakstats)]

    peakhists, peakstats, h, peakstats.bin_width .* e_unit
end
export get_peakhists_th228

get_peakhists_th228(e, th228_lines, left_window_sizes::Vector{<:Unitful.Energy{<:Real}}, right_window_sizes::Vector{<:Unitful.Energy{<:Real}}; kwargs...) = get_peakhists_th228(e, th228_lines, [(l,r) for (l,r) in zip(left_window_sizes, right_window_sizes)],; kwargs...)

