const e_unit=u"keV" # unit for calibrated energies. All energy values will be convert to this unit
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

function simple_calibration(e_uncal::Vector{<:Real}, gamma_lines::Vector{<:Unitful.Energy{<:Real}}, window_sizes::Vector{<:Tuple{Unitful.Energy{<:Real}, Unitful.Energy{<:Real}}},; kwargs...)
    # remove calib type from kwargs
    @assert haskey(kwargs, :calib_type) "Calibration type not specified"
    calib_type = kwargs[:calib_type]
    # remove :calib_type from kwargs
    kwargs = pairs(NamedTuple(filter(k -> !(:calib_type in k), kwargs)))
    if calib_type == :th228
        @debug "Use simple calibration for Th228 lines - fix peak-search parameters"
        kwargs = merge(NamedTuple(kwargs), (peaksearch_type = :th228, peakfinder_σ = 5.0, peakfinder_threshold = 10.0, peak_quantile = 0.9..1.0, bin_quantile = 0.05..0.5)) # give good default values for Th228
        return simple_calibration_gamma(e_uncal, gamma_lines, window_sizes,; kwargs...)
    elseif calib_type == :gamma
         @debug "Use simple calibration for generic gamma lines"
         return simple_calibration_gamma(e_uncal, gamma_lines, window_sizes,; kwargs...)
    else
        error("Calibration type not supported")
    end
end
simple_calibration(e_uncal::Vector{<:Real}, gamma_lines::Vector{<:Unitful.Energy{<:Real}}, left_window_sizes::Vector{<:Unitful.Energy{<:Real}}, right_window_sizes::Vector{<:Unitful.Energy{<:Real}}; kwargs...) = simple_calibration(e_uncal, gamma_lines, [(l,r) for (l,r) in zip(left_window_sizes, right_window_sizes)],; kwargs...)


function simple_calibration_gamma(e_uncal::Vector{<:Real}, gamma_lines::Vector{<:Unitful.Energy{<:Real}}, window_sizes::Vector{<:Tuple{Unitful.Energy{<:Real}, Unitful.Energy{<:Real}}},; peaksearch_type::Symbol = :most_prominent, binning_peak_window::Unitful.Energy{<:Real}=10.0u"keV", peakfinder_gamma::Int = 1, kwargs...)
    if peaksearch_type == :most_prominent
        # use most prominent peak in spectrum and use that one. This relies on the peakfinder to find all peaks in gamma_lines!
       result_peak = peak_search_gamma(e_uncal, gamma_lines; kwargs...)
   elseif peaksearch_type == :th228
       # use Tl208FEP for Th228 calibration - value hardcoded for back compatibility
       result_peak = peak_search_gamma(e_uncal, [2614.5*u"keV"]; kwargs...)
   elseif peaksearch_type == :single
       # use only 1 energy `gamma_lines[peaksearch_gamma]` in peak search
       result_peak = peak_search_gamma(e_uncal, gamma_lines[peakfinder_gamma]; kwargs...)
   end 
    e_simple = e_uncal .* result_peak.c

    # get peakhists and peakstats
    peakhists, peakstats, h_calsimple, bin_widths = peakhists_gamma(e_simple, gamma_lines, window_sizes; binning_peak_window=binning_peak_window)
    
    result = (
        h_calsimple = h_calsimple, 
        h_uncal = result_peak.h_uncal, 
        c = result_peak.c,
        unit = e_unit,
        bin_width = median(bin_widths),
        peak_guess = result_peak.peak_guess,
        peakbinwidths = bin_widths,
        peakhists = peakhists,
        peakstats = peakstats
        )
    report = (
        h_calsimple = result.h_calsimple,
        h_uncal = result.h_uncal,
        c = result.c,
        peak_guess = result.peak_guess,
        peakhists = result.peakhists,
        peakstats = result.peakstats
    )
    return result, report
end


"""
    peakhists_gamma(e::Vector{<:Unitful.Energy{<:Real}}, gamma_lines::Vector{<:Unitful.Energy{<:Real}}, window_sizes::Vector{<:Tuple{Unitful.Energy{<:Real}, Unitful.Energy{<:Real}}},; binning_peak_window::Unitful.Energy{<:Real}=10.0u"keV")
Create histograms around the calibration lines and return the histograms and the peak statistics.
# input
    * `e`: energy array
    * `gamma_lines`: array of gamma lines
    * `window_sizes`: array of tuples with left and right window sizes around the gamma lines
## keyword Arguments:
    * `binning_peak_window::Unitful.Energy{<:Real}`: energy window around each peak that is used to find optimal binning for peakhists
# output
    * `peakhists`: array of histograms around the calibration lines
    * `peakstats`: array of statistics for the calibration line fits
"""
function peakhists_gamma(e::Vector{<:Unitful.Energy{<:Real}}, gamma_lines::Vector{<:Unitful.Energy{<:Real}}, window_sizes::Vector{<:Tuple{Unitful.Energy{<:Real}, Unitful.Energy{<:Real}}},; binning_peak_window::Unitful.Energy{<:Real}=10.0u"keV")
    # get optimal binning for simple calibration
    bin_widths = [get_friedman_diaconis_bin_width(filter(in(peak - binning_peak_window..peak + binning_peak_window), e)) for peak in gamma_lines]
    # create histogram for simple calibration
    e_min, e_max = 0u"keV", 3000u"keV"
    h = fit(Histogram, ustrip.(e_unit, e), ustrip(e_unit, e_min):ustrip(e_unit, median(bin_widths)):ustrip(e_unit, e_max))
    # get histograms around calibration lines and peakstats
    peakenergies = [ustrip.(e_unit, filter(in(peak - first(window)..peak + last(window)), e)) for (peak, window) in zip(gamma_lines, window_sizes)]
    peakstats = StructArray(estimate_single_peak_stats.(peakenergies, ustrip.(e_unit, bin_widths)))
    peakhists = [fit(Histogram, ustrip.(e_unit, e), ustrip(e_unit, peak - first(window)):ps.bin_width:ustrip(e_unit, peak + last(window))) for (peak, window, ps) in zip(Measurements.value.(gamma_lines), window_sizes, peakstats)]
    peakhists, peakstats, h, peakstats.bin_width .* e_unit
end
export peakhists_gamma
peakhists_gamma(e, gamma_lines, left_window_sizes::Vector{<:Unitful.Energy{<:Real}}, right_window_sizes::Vector{<:Unitful.Energy{<:Real}}; kwargs...) = peakhists_gamma(e, gamma_lines, [(l,r) for (l,r) in zip(left_window_sizes, right_window_sizes)],; kwargs...)

"""
     peak_search_gamma(e_uncal::Vector{<:Real}, gamma_lines::Vector{<:Unitful.Energy{<:Real}}; peakfinder_σ::Real = 2.0, peakfinder_threshold::Real = 10.0, peak_quantile::ClosedInterval{<:Real} = 0.0..1.0, bin_quantile::ClosedInterval{<:Real} = peak_quantile, quantile_perc::Float64=NaN) 
peak search in gamma-ray spectrum
# inputs:
* `e_uncal`: uncalibrated energy array
* `gamma_lines`: array of gamma lines
* keyword arguments:
        * `peakfinder_σ::Real=2.0`: The expected sigma of a peak in the spectrum. In units of bins. (see RadiationSpectra.peakfinder)
        * `peakfinder_threshold::Real=10.0``: Threshold for being identified as a peak in the deconvoluted spectrum. A single bin is identified as an peak when its weight exceeds the threshold and the previous bin was not identified as an peak. (see RadiationSpectra.peakfinder)
        * `peak_quantile::ClosedInterval{<:Real}=0.5..1.0`: quantiles that define energy window that is considered peak search (default: 0.0..1.0 == whole range). All `gamma_lines` have to be within this window.
        * `bin_quantile::ClosedInterval{<:Real}=0.5..1.0`: quantiles that define energy window that is used to find optimal binning (default same as peak_quantile)
        * `quantile_perc::Float64=NaN`: If NaN the standard peakfinder is used. If not NaN, the peak for simple calibration is set to given quantile energy. Useful for debugging/testing
# output: `result` NamedTuple with the following fields:
* `h_uncal`: histogram of the uncalibrated energy array with the binning that is used in peak search 
* `c`: simple calibration factor
* `peak_guess`: estimated peak energy in units of the uncalibrated energy array
"""
function peak_search_gamma(e_uncal::Vector{<:Real}, gamma_lines::Vector{<:Unitful.Energy{<:Real}}; peakfinder_σ::Real = 2.0, peakfinder_threshold::Real = 10.0, peak_quantile::ClosedInterval{<:Real} = 0.0..1.0, bin_quantile::ClosedInterval{<:Real} = peak_quantile, quantile_perc::Float64=NaN) 
    # find optimal binning for peak search
    bin_min = quantile(e_uncal, leftendpoint(bin_quantile))
    bin_max = quantile(e_uncal, rightendpoint(bin_quantile))
    bin_width = get_friedman_diaconis_bin_width(filter(in(bin_min..bin_max), e_uncal))
    h_uncal = fit(Histogram, e_uncal, 0:bin_width:maximum(e_uncal)) # histogram over full energy range; stored for plot recipe 

    peak_guess, peak_idx = if isnan(quantile_perc)
        # create histogram for peak-search in peak-window
        peak_min = quantile(e_uncal, leftendpoint(peak_quantile))
        peak_max = quantile(e_uncal, rightendpoint(peak_quantile))
        h_peaksearch = fit(Histogram, e_uncal, peak_min:bin_width:peak_max) # histogram for peak search

        # search all possible peak candidates
        @debug "debug: peakfinder_σ = $peakfinder_σ, peakfinder_threshold = $peakfinder_threshold"
        @debug "debug: peak_quantile = $peak_quantile, bin_quantile = $bin_quantile"
        h_decon, peakpos = RadiationSpectra.peakfinder(h_peaksearch, σ=peakfinder_σ, backgroundRemove = true, threshold = peakfinder_threshold)
        # find the most prominent peak in the deconvoluted histogram
        sort!(peakpos)
        peakpos_idxs = StatsBase.binindex.(Ref(h_decon), peakpos)
        cts_peakpos = h_decon.weights[peakpos_idxs]
        @debug "Peaks found at $peakpos with intensity $cts_peakpos - literature values: $(gamma_lines)"
        if length(gamma_lines) > 1 && (length(gamma_lines) != length(peakpos))
            error("Number of gamma lines expected: $(length(gamma_lines)) vs. number of peaks found: $(length(peakpos)) -->  do not match \n try to adjust the peakfinder parameters: peakfinder_σ, peakfinder_threshold, or binning")
        end
        peakpos[argmax(cts_peakpos)], argmax(cts_peakpos)
    else
        quantile(e_uncal, quantile_perc), length(gamma_lines)
    end

    if length(gamma_lines) == 1 && peak_idx > 1
        @debug "Found more than one gamma lines - assume that  most prominent peak is the one we are looking for" 
        peak_idx = 1 
    end
    @debug "Identified most prominent peak at $(round(peak_guess, digits = 2)) - literature value: $(gamma_lines[peak_idx])"

    # get calibration constant for simple calibration
    c = gamma_lines[peak_idx] / peak_guess

    result = (h_uncal = h_uncal, c = c, peak_guess = peak_guess)
    return result
end