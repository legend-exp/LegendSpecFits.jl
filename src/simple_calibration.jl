"""
    simple_calibration(
        e_uncal::Vector{<:Real}, gamma_lines::Vector{<:Unitful.Energy{<:Real}},
        window_sizes::Vector{<:Tuple{Unitful.Energy{<:Real}, Unitful.Energy{<:Real}}};
        calib_type::Symbol, kwargs...
    )
    simple_calibration(
        e_uncal::Vector{<:Real}, gamma_lines::Vector{<:Unitful.Energy{<:Real}},
        left_window_sizes::Vector{<:Unitful.Energy{<:Real}},
        right_window_sizes::Vector{<:Unitful.Energy{<:Real}}; calib_type::Symbol, kwargs...
    )

Determines a first, linear energy calibration of the uncalibrated energies
`e_uncal` from the position of one prominent peak, and builds histograms and
peak statistics around all `gamma_lines`.

The calibration constant is the ratio of the known energy of the selected line
to the peak position found in the uncalibrated spectrum. The peak search
histogram uses the Freedman-Diaconis bin width.

## Arguments
* `e_uncal`: Uncalibrated energies.
* `gamma_lines`: Known gamma line energies, e.g. from 228Th.
* `window_sizes`: One tuple `(left, right)` per line giving the window below and above the line
  that is histogrammed. Alternatively, vectors of left and right window sizes can be passed separately.

## Keywords
* `calib_type::Symbol`: Required. `:th228` searches for the 2614.5 keV line and sets the
  peak-finder parameters `peakfinder_σ = 5.0`, `peakfinder_threshold = 10.0`,
  `peak_quantile = 0.9..1.0` and `bin_quantile = 0.05..0.5`, overriding any user values.
  `:gamma` uses the keywords below for generic gamma lines. Other values throw an error.
* `peaksearch_type::Symbol = :most_prominent`: (`:gamma` only) `:most_prominent` uses the highest
  peak found, which requires the peak finder to find all `gamma_lines`; `:single` uses only
  `gamma_lines[peakfinder_gamma]`.
* `peakfinder_gamma::Int = 1`: Index of the line used for `peaksearch_type = :single`.
* `binning_peak_window::Unitful.Energy = 10.0u"keV"`: Energy window around each line used to choose the bin width of the peak histograms.
* `peakfinder_σ::Real = 2.0`, `peakfinder_threshold::Real = 10.0`: Expected peak width (in bins)
  and detection threshold of `RadiationSpectra.peakfinder`.
* `peak_quantile`, `bin_quantile`: Quantile ranges (`ClosedInterval`, default `0.0..1.0`)
  of `e_uncal` for the peak search and for choosing the bin width.
* `quantile_perc::Float64 = NaN`: If not `NaN`, the peak is set to this quantile of `e_uncal` instead of being searched for.

## Returns
A tuple `(result, report)`. `result` is a `NamedTuple` with the fields
* `h_calsimple`: histogram of the roughly calibrated energies (0 to 3000 keV).
* `h_uncal`: histogram of the uncalibrated energies used in the peak search.
* `c`: calibration constant with energy unit (energy per uncalibrated unit).
* `unit`: energy unit of the calibrated values.
* `bin_width`, `peakbinwidths`: median and per-line bin width of the peak histograms.
* `peak_guess`: position of the identified peak in uncalibrated units.
* `peakhists`: vector of histograms around the lines (in `unit`).
* `peakstats`: `StructArray` of peak statistics from [`estimate_single_peak_stats`](@ref).

`report` contains `h_calsimple`, `h_uncal`, `c`, `peak_guess`, `peakhists` and
`peakstats`, for plotting with `LegendMakie.lplot`.
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
    peakhists_gamma(
        e::Vector{<:Unitful.Energy{<:Real}}, gamma_lines::Vector{<:Unitful.Energy{<:Real}},
        window_sizes::Vector{<:Tuple{Unitful.Energy{<:Real}, Unitful.Energy{<:Real}}};
        binning_peak_window::Unitful.Energy{<:Real} = 10.0u"keV"
    )
    peakhists_gamma(e, gamma_lines, left_window_sizes, right_window_sizes; binning_peak_window)

Creates histograms of the energies `e` around each of the `gamma_lines` and
estimates the peak statistics in each window.

## Arguments
* `e`: Calibrated energies.
* `gamma_lines`: Gamma line energies.
* `window_sizes`: One tuple `(left, right)` per line with the window sizes below
  and above the line. Alternatively, vectors of left and right window sizes can be passed separately.

## Keywords
* `binning_peak_window::Unitful.Energy = 10.0u"keV"`: Energy window around each
  line used to choose the Freedman-Diaconis bin width.

## Returns
A tuple `(peakhists, peakstats, h, bin_widths)` with
* `peakhists`: vector of histograms around the lines (values in `e_unit`, keV).
* `peakstats`: `StructArray` of the results of [`estimate_single_peak_stats`](@ref) for each window.
* `h`: histogram of all of `e` from 0 to 3000 keV with the median bin width.
* `bin_widths`: bin width of each peak histogram with energy unit.
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
    peak_search_gamma(
        e_uncal::Vector{<:Real}, gamma_lines::Vector{<:Unitful.Energy{<:Real}};
        peakfinder_σ::Real = 2.0, peakfinder_threshold::Real = 10.0,
        peak_quantile::ClosedInterval{<:Real} = 0.0..1.0,
        bin_quantile::ClosedInterval{<:Real} = peak_quantile, quantile_perc::Float64 = NaN
    )

Searches for the most prominent peak in the uncalibrated spectrum `e_uncal` and
returns a simple calibration constant that assigns it to the line in `gamma_lines`.

## Keywords
* `peakfinder_σ::Real = 2.0`: Expected standard deviation of a peak, in bins
  (see `RadiationSpectra.peakfinder`).
* `peakfinder_threshold::Real = 10.0`: Threshold in the deconvoluted spectrum above which a bin is
  identified as a peak if the previous bin was not (see `RadiationSpectra.peakfinder`).
* `peak_quantile = 0.0..1.0`: Quantile range of `e_uncal` that is searched for peaks.
  All `gamma_lines` must lie within this window.
* `bin_quantile = peak_quantile`: Quantile range of `e_uncal` used to choose the bin width.
* `quantile_perc::Float64 = NaN`: If not `NaN`, the peak is set to this quantile of
  `e_uncal` and no peak search is done.

If more than one line is given, the number of peaks found must equal the number
of lines, otherwise an error is thrown.

## Returns
A `NamedTuple` with the fields
* `h_uncal`: histogram of `e_uncal` with the bin width used in the peak search.
* `c`: calibration constant (energy of the line divided by `peak_guess`).
* `peak_guess`: position of the identified peak in units of `e_uncal`.
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