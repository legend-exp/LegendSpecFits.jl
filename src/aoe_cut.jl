
"""
    get_sf_after_aoe_cut(
        aoe_cut::Unitful.RealOrRealQuantity, aoe::Vector{<:Unitful.RealOrRealQuantity},
        e::Vector{<:T}, peak::T, window::Vector{T}, bin_width::T, result_before::NamedTuple;
        uncertainty::Bool = true, fit_func::Symbol = :gamma_def
    ) where T<:Unitful.Energy{<:Real}

Calculates the signal survival fraction of a peak after the A/E cut value
`aoe_cut` (events with `aoe >= aoe_cut` survive).

The energies `e` in the window `[peak - window[1], peak + window[2]]` are
histogrammed with the bin width `bin_width` separately for the surviving and
the cut events. Both histograms are fitted simultaneously with
[`fit_subpeaks_th228`](@ref), using the fit result `result_before` of the peak
before the cut as fixed parameters.

## Returns
The survival fraction `sf` as a fraction between 0 and 1 (a `Measurement` if `uncertainty` is `true`).
"""
function get_sf_after_aoe_cut(aoe_cut::Unitful.RealOrRealQuantity, aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peak::T, window::Vector{T}, bin_width::T, result_before::NamedTuple; uncertainty::Bool=true, fit_func::Symbol=:gamma_def) where T<:Unitful.Energy{<:Real}
    # get energy after cut and create histogram
    survived = fit(Histogram, ustrip.(e[aoe .>= aoe_cut]), ustrip.(peak-first(window):bin_width:peak+last(window)))
    cut = fit(Histogram, ustrip.(e[aoe .< aoe_cut]), ustrip.(peak-first(window):bin_width:peak+last(window)))
    # fit peak and return number of signal counts
    result, _ = fit_subpeaks_th228(survived, cut, result_before; uncertainty=uncertainty, fit_func=fit_func)
    return result.sf
end
export get_sf_after_aoe_cut



"""
    get_low_aoe_cut(
        aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T};
        dep::T = 1592.53u"keV", window::Vector{<:T} = [12.0, 10.0]u"keV", dep_sf::Float64 = 0.9,
        rtol::Float64 = 0.001, maxiters::Int = 300, sigma_high_sided::Float64 = Inf,
        cut_search_interval::Tuple = (-25.0 * unit(first(aoe)), 1.0 * unit(first(aoe))),
        bin_width_window::T = 3.0u"keV", max_e_plot::T = 3000.0u"keV",
        plot_window::Vector{<:T} = [12.0, 50.0]u"keV",
        fixed_position::Bool = true, fit_func::Symbol = :gamma_def, uncertainty::Bool = true
    ) where T<:Unitful.Energy{<:Real}

Determines the lower A/E cut value at which a fraction `dep_sf` of the events in
the double escape peak (DEP) survives.

The DEP at energy `dep` is fitted with [`fit_single_peak_th228`](@ref) in the
window `[dep - window[1], dep + window[2]]`. A bisection root search in
`cut_search_interval` with relative tolerance `rtol` then finds the cut value
for which the survival fraction from [`get_sf_after_aoe_cut`](@ref) equals
`dep_sf`.

## Keywords
* `dep::T = 1592.53u"keV"`: Energy of the double escape peak.
* `window::Vector{<:T} = [12.0, 10.0]u"keV"`: Lower and upper width of the fit window around `dep`.
* `dep_sf::Float64 = 0.9`: Target survival fraction of the peak.
* `rtol::Float64 = 0.001`, `maxiters::Int = 300`: Relative tolerance and maximum number of iterations of the root search.
* `sigma_high_sided::Float64 = Inf`: Upper A/E cut value (in the unit of `aoe`); `Inf` applies no upper cut.
* `cut_search_interval`: Interval of A/E values searched for the lower cut.
* `bin_width_window::T = 3.0u"keV"`: Half-width of the energy window around `dep` that is used to
  determine the Freedman-Diaconis bin width.
* `max_e_plot::T = 3000.0u"keV"`, `plot_window::Vector{<:T} = [12.0, 50.0]u"keV"`: Energy range of the
  full-spectrum histograms and lower and upper width of the DEP histograms in `report`.
* `fixed_position::Bool = true`: Fix the fitted peak position to `dep`.
* `fit_func::Symbol = :gamma_def`: Peak shape (see [`fit_single_peak_th228`](@ref)).
* `uncertainty::Bool = true`: Calculate fit uncertainties.

## Returns
A tuple `(result, report)`. `result` is a `NamedTuple` with the fields
* `lowcut`: lower A/E cut value, with `rtol * lowcut` as uncertainty.
* `highcut`: upper A/E cut value (`sigma_high_sided`, with the unit of `aoe`).
* `n0`: number of counts in the peak before the cut (from the fit).
* `nsf`: number of counts in the peak after the cut.
* `sf`: survival fraction after the cut, in percent.

`report` contains the histograms `h_before`, `h_after_low` (lower cut only) and
`h_after_ds` (lower and upper cut) of the full spectrum and `dep_h_before`,
`dep_h_after_low`, `dep_h_after_ds` around the DEP, together with `sf`, `n0`,
`lowcut`, `highcut`, `e_unit` and `bin_width`, for plotting with `LegendMakie.lplot`.
"""
function get_low_aoe_cut(aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T},; 
            dep::T=1592.53u"keV", window::Vector{<:T}=[12.0, 10.0]u"keV", dep_sf::Float64=0.9, rtol::Float64=0.001, maxiters::Int=300, sigma_high_sided::Float64=Inf,
            cut_search_interval::Tuple{<:Unitful.RealOrRealQuantity, <:Unitful.RealOrRealQuantity}=(-25.0*unit(first(aoe)), 1.0*unit(first(aoe))), 
            bin_width_window::T=3.0u"keV", max_e_plot::T=3000.0u"keV",  plot_window::Vector{<:T}=[12.0, 50.0]u"keV",
            fixed_position::Bool=true, fit_func::Symbol=:gamma_def, uncertainty::Bool=true) where T<:Unitful.Energy{<:Real}
    # cut window around peak
    e_mask = (dep-first(window) .< e .< dep+last(window))
    aoe_dep = aoe[e_mask]
    e_dep   =   e[e_mask]
    # estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e_dep[dep - bin_width_window .< e_dep .< dep + bin_width_window])
    # create histogram
    dephist = fit(Histogram, ustrip.(e_unit, e_dep), ustrip(e_unit, dep-first(window)):ustrip(e_unit, bin_width):ustrip(e_unit, dep+last(window)))
    # get peakstats
    depstats = estimate_single_peak_stats_th228(dephist)
    if fixed_position
        depstats = merge(depstats, (peak_pos = ustrip(e_unit, dep), ))
    end
    # fit before cut
    result_before, _ = fit_single_peak_th228(dephist, depstats; uncertainty=uncertainty, fixed_position=fixed_position, fit_func=fit_func)
    # get aoe cut
    sf_dep_f = cut -> get_sf_after_aoe_cut(cut, aoe_dep, e_dep, dep, window, bin_width, mvalue(result_before); uncertainty=false, fit_func=fit_func) - dep_sf
    aoe_cut = find_zero(sf_dep_f, cut_search_interval, Bisection(), rtol=rtol, maxiters=maxiters)
    # get sf after cut
    sf = get_sf_after_aoe_cut(aoe_cut, aoe_dep, e_dep, dep, window, bin_width, mvalue(result_before); uncertainty=uncertainty, fit_func=fit_func)
    result = (
        lowcut = measurement(aoe_cut, aoe_cut * rtol), 
        highcut = sigma_high_sided * unit(aoe_cut), 
        n0 = result_before.n, 
        nsf = result_before.n * sf, 
        sf = sf * 100*u"percent"
    )
    hist_binning = 0:ustrip(e_unit, bin_width):ustrip(e_unit, max_e_plot)
    dep_binning = ustrip(e_unit, dep-first(plot_window)):ustrip(e_unit, bin_width):ustrip(e_unit, dep+last(plot_window))
    report = (
        h_before = fit(Histogram, ustrip.(e_unit, e), hist_binning),
        h_after_low = fit(Histogram, ustrip.(e_unit, e[result.lowcut .< aoe]), hist_binning),
        h_after_ds = fit(Histogram, ustrip.(e_unit, e[result.lowcut .< aoe .< result.highcut]), hist_binning),
        dep_h_before = fit(Histogram, ustrip.(e_unit, e), dep_binning),
        dep_h_after_low = fit(Histogram, ustrip.(e_unit, e[result.lowcut .< aoe]), dep_binning),
        dep_h_after_ds = fit(Histogram, ustrip.(e_unit, e[result.lowcut .< aoe .< result.highcut]), dep_binning),
        sf = result.sf,
        n0 = result.n0,
        lowcut = result.lowcut,
        highcut = result.highcut,
        e_unit = e_unit,
        bin_width = bin_width,
    )
    return result, report
end
export get_low_aoe_cut


"""
    get_peaks_survival_fractions(e, peaks, peak_names, windows, survival_flag; uncertainty, bin_width_window, fit_funcs)
    get_peaks_survival_fractions(e, peaks, peak_names, left_window_sizes, right_window_sizes, survival_flag; kwargs...)
    get_peaks_survival_fractions(cut_parameter, e, peaks, peak_names, windows; low_cut, high_cut, selection, kwargs...)
    get_peaks_survival_fractions(cut_parameter, e, peaks, peak_names, left_window_sizes, right_window_sizes; kwargs...)

Fits the survival fraction of several peaks after an event selection, with
[`get_peak_survival_fraction`](@ref) for each peak. The peaks are processed in parallel on the available threads.

## Arguments
* `e`: Energies of all events, as `Unitful.Energy` values.
* `peaks`: Peak energies.
* `peak_names::AbstractVector{Symbol}`: Names used as keys of the result.
* `windows`: One tuple `(lower, upper)` of window widths below and above each peak. Alternatively, the
  vectors `left_window_sizes` and `right_window_sizes` hold the widths.
* `survival_flag::AbstractVector{Bool}`: `true` for each event that survives the selection.
* `cut_parameter`: Instead of `survival_flag`, a parameter (such as A/E) from which the selection is built:
  an event survives if `low_cut < cut_parameter < high_cut` and `selection` is `true`.

## Keywords
* `uncertainty::Bool = true`: Calculate fit uncertainties.
* `bin_width_window::T = 2.0u"keV"`: Half-width of the energy window around each peak that is used to
  determine the Freedman-Diaconis bin width.
* `fit_funcs::AbstractVector{Symbol}`: Peak shape for each peak (see [`fit_single_peak_th228`](@ref)).
  Default is `:gamma_def` for all peaks.
* `low_cut`, `high_cut`: Exclusive bounds on `cut_parameter` (cut-parameter methods). Default is no bound.
* `selection::AbstractVector{Bool}`: Additional selection (cut-parameter methods). Default selects all events.

## Returns
A tuple `(result, report)` of two `OrderedDict`s keyed by `peak_names`. The values are the
`result` and `report` `NamedTuple`s of [`get_peak_survival_fraction`](@ref).
"""
function get_peaks_survival_fractions(e::AbstractVector{<:T}, peaks::AbstractVector{<:T}, peak_names::AbstractVector{Symbol}, windows::AbstractVector{<:Tuple{T, T}}, survival_flag::AbstractVector{Bool};
    uncertainty::Bool=true, bin_width_window::T=2.0u"keV", fit_funcs::AbstractVector{Symbol}=fill(:gamma_def, length(peaks))) where T<:Unitful.Energy{<:Real}
    v_result = Vector{NamedTuple}(undef, length(peak_names))
    v_report = Vector{NamedTuple}(undef, length(peak_names))

    Threads.@threads for i in eachindex(peaks, peak_names, windows, fit_funcs)
        peak, window, fit_func = peaks[i], windows[i], fit_funcs[i]
        result_peak, report_peak = get_peak_survival_fraction(e, peak, window, survival_flag; uncertainty, bin_width_window, fit_func)
        v_result[i] = result_peak
        v_report[i] = report_peak
    end

    result = OrderedDict{Symbol, NamedTuple}(peak_names .=> v_result)
    report = OrderedDict{Symbol, NamedTuple}(peak_names .=> v_report)
    return result, report
end

get_peaks_survival_fractions(e::AbstractVector{<:T}, peaks::AbstractVector{<:T}, peak_names::AbstractVector{Symbol}, left_window_sizes::AbstractVector{<:T}, right_window_sizes::AbstractVector{<:T}, survival_flag::AbstractVector{Bool}; kwargs...) where T<:Unitful.Energy{<:Real} = get_peaks_survival_fractions(e, peaks, peak_names, collect(zip(left_window_sizes, right_window_sizes)), survival_flag; kwargs...)

function get_peaks_survival_fractions(cut_parameter::AbstractVector{<:Unitful.RealOrRealQuantity}, e::AbstractVector{<:T}, peaks::AbstractVector{<:T}, peak_names::AbstractVector{Symbol}, windows::AbstractVector{<:Tuple{T, T}};
    low_cut::Unitful.RealOrRealQuantity=-Inf*unit(first(cut_parameter)), high_cut::Unitful.RealOrRealQuantity=Inf*unit(first(cut_parameter)), selection::AbstractVector{Bool}=trues(length(cut_parameter)), kwargs...) where T<:Unitful.Energy{<:Real}
    survival_flag = (low_cut .< cut_parameter .< high_cut) .&& selection
    get_peaks_survival_fractions(e, peaks, peak_names, windows, survival_flag; kwargs...)
end

get_peaks_survival_fractions(cut_parameter::AbstractVector{<:Unitful.RealOrRealQuantity}, e::AbstractVector{<:T}, peaks::AbstractVector{<:T}, peak_names::AbstractVector{Symbol}, left_window_sizes::AbstractVector{<:T}, right_window_sizes::AbstractVector{<:T}; kwargs...) where T<:Unitful.Energy{<:Real} = get_peaks_survival_fractions(cut_parameter, e, peaks, peak_names, collect(zip(left_window_sizes, right_window_sizes)); kwargs...)

Base.@deprecate get_peaks_surrival_fractions(args...; kwargs...) get_peaks_survival_fractions(args...; kwargs...)
export get_peaks_survival_fractions, get_peaks_surrival_fractions

"""Deprecated alias of [`get_peaks_survival_fractions`](@ref)."""
get_peaks_surrival_fractions


"""
    get_peak_survival_fraction(h_before, h_survived, h_cut; uncertainty, fit_func)
    get_peak_survival_fraction(e, survival_flag; uncertainty, fit_func)
    get_peak_survival_fraction(e, peak, window, survival_flag; uncertainty, bin_width_window, fit_func)
    get_peak_survival_fraction(cut_parameter, e, peak, window; low_cut, high_cut, selection, kwargs...)

Fits a peak before and after an event selection and calculates the fraction of
the peak counts that survives.

The peak is first fitted with [`fit_single_peak_th228`](@ref) before the
selection and then the surviving and rejected events are fitted simultaneously
with [`fit_subpeaks_th228`](@ref).

## Arguments
* `h_before`, `h_survived`, `h_cut`: One-dimensional histograms (unitless energy) of all, surviving
  and rejected events with the same binning. The latter two must add up to the first.
* `e`: Energies of the events in the peak region, as `Unitful.Energy` values. Without `peak` and
  `window`, the whole range of `e` is histogrammed and fitted.
* `survival_flag::AbstractVector{Bool}`: `true` for each event that survives.
* `peak`, `window`: Peak energy and a tuple `(lower, upper)` of window widths below and above it.
* `cut_parameter`: Instead of `survival_flag`, a parameter (such as A/E) from which the selection is built:
  an event survives if `low_cut < cut_parameter < high_cut` and `selection` is `true`.

## Keywords
* `uncertainty::Bool = true`: Calculate fit uncertainties.
* `fit_func::Symbol = :gamma_def`: Peak shape (see [`fit_single_peak_th228`](@ref)).
* `bin_width_window::T = 2.0u"keV"`: Half-width of the energy window around `peak` that is used to
  determine the Freedman-Diaconis bin width.
* `low_cut`, `high_cut`: Exclusive bounds on `cut_parameter`. Default is no bound.
* `selection::AbstractVector{Bool}`: Additional selection. Default selects all events.

## Returns
A tuple `(result, report)`. `result` is a `NamedTuple` with the fields
* `peak`: the peak energy (methods with `peak` only).
* `fit_func`: the peak shape that was used.
* `n_before`, `n_after`: number of counts in the peak before and after the selection.
* `sf`: survival fraction in percent.
* `gof`: goodness-of-fit results `before` and `after`.

`report` contains `peak` (methods with `peak` only), `n_before`, `n_after`,
`sf` and the reports `before` and `after` of the fits, for plotting with `LegendMakie.lplot`.
"""
function get_peak_survival_fraction(e::AbstractVector{<:T}, peak::T, window::Tuple{T, T}, survival_flag::AbstractVector{Bool};
    uncertainty::Bool=true, bin_width_window::T=2.0u"keV", fit_func::Symbol=:gamma_def) where T<:Unitful.Energy{<:Real}
    bin_width = get_friedman_diaconis_bin_width(e[e .> peak - bin_width_window .&& e .< peak + bin_width_window])
    binning = ustrip(peak-first(window)):ustrip(bin_width):ustrip(peak+last(window))

    h_before = fit(Histogram, ustrip.(e), binning)
    h_survived = fit(Histogram, ustrip.(e[survival_flag]), binning)
    h_cut = fit(Histogram, ustrip.(e[.!survival_flag]), binning)
    result, report = get_peak_survival_fraction(h_before, h_survived, h_cut; uncertainty, fit_func)
    return merge((peak = peak,), result), merge((peak = peak,), report)
end

function get_peak_survival_fraction(e::AbstractVector{<:T}, survival_flag::AbstractVector{Bool};
    uncertainty::Bool=true, fit_func::Symbol=:gamma_def) where T<:Unitful.Energy{<:Real}
    @argcheck axes(e) == axes(survival_flag)
    bin_width = get_friedman_diaconis_bin_width(e)
    binning = ustrip(minimum(e)):ustrip(bin_width):ustrip(maximum(e) + bin_width)
    h_before = fit(Histogram, ustrip.(e), binning)
    h_survived = fit(Histogram, ustrip.(e[survival_flag]), binning)
    h_cut = fit(Histogram, ustrip.(e[.!survival_flag]), binning)
    get_peak_survival_fraction(h_before, h_survived, h_cut; uncertainty, fit_func)
end

function get_peak_survival_fraction(h_before::Histogram{<:Real, 1}, h_survived::Histogram{<:Real, 1}, h_cut::Histogram{<:Real, 1};
    uncertainty::Bool=true, fit_func::Symbol=:gamma_def)
    peakstats = estimate_single_peak_stats(h_before)
    result_before, report_before = fit_single_peak_th228(h_before, peakstats; uncertainty, fit_func)
    result_after, report_after = fit_subpeaks_th228(h_survived, h_cut, result_before; uncertainty, fit_func)

    result = (
        fit_func = fit_func,
        n_before = result_before.n,
        n_after = result_before.n * result_after.sf,
        sf = result_after.sf * 100u"percent",
        gof = (after = result_after.gof, before = result_before.gof),
    )
    report = (
        n_before = result.n_before,
        n_after = result.n_after,
        sf = result.sf,
        before = report_before,
        after = report_after,
    )
    return result, report
end

function get_peak_survival_fraction(cut_parameter::AbstractVector{<:Unitful.RealOrRealQuantity}, e::AbstractVector{<:T}, peak::T, window::Tuple{T, T};
    low_cut::Unitful.RealOrRealQuantity=-Inf*unit(first(cut_parameter)), high_cut::Unitful.RealOrRealQuantity=Inf*unit(first(cut_parameter)), selection::AbstractVector{Bool}=trues(length(cut_parameter)), kwargs...) where T<:Unitful.Energy{<:Real}
    survival_flag = (low_cut .< cut_parameter .< high_cut) .&& selection
    get_peak_survival_fraction(e, peak, window, survival_flag; kwargs...)
end

Base.@deprecate get_peak_surrival_fraction(args...; kwargs...) get_peak_survival_fraction(args...; kwargs...)
export get_peak_survival_fraction, get_peak_surrival_fraction

"""Deprecated alias of [`get_peak_survival_fraction`](@ref)."""
get_peak_surrival_fraction



"""
    get_continuum_survival_fraction(e, center, window, survival_flag; low_flag = survival_flag)
    get_continuum_survival_fraction(cut_parameter, e, center, window; low_cut, high_cut, selection)

Counts the events in a continuum window before and after an event selection and
calculates the survival fraction.

The window is `center ± window`. Unlike for peaks, no fit is made.

## Arguments
* `e`: Energies of all events, as `Unitful.Energy` values.
* `center`, `window`: Center and half-width of the energy window.
* `survival_flag::AbstractVector{Bool}`: `true` for each event that survives.
* `cut_parameter`: Instead of `survival_flag`, a parameter (such as A/E) from which the selection is built:
  an event survives if `low_cut < cut_parameter < high_cut` and `selection` is `true`.

## Keywords
* `low_flag::AbstractVector{Bool} = survival_flag`: Events that pass the lower cut only; used for the
  report histogram `h_after_low`.
* `low_cut`, `high_cut`: Exclusive bounds on `cut_parameter`. Default is no bound. `h_after_low` applies
  `low_cut` only, `h_after_ds` applies the final selection including `high_cut` and `selection`.
* `selection::AbstractVector{Bool}`: Additional selection. Default selects all events.

## Returns
A tuple `(result, report)`. `result` is a `NamedTuple` with the fields
* `window`: `center ± window`.
* `n_before`, `n_after`: event counts with Poisson uncertainty `√n`.
* `sf`: survival fraction in percent, with binomial uncertainty.

`report` contains the histograms `h_before`, `h_after_low` and `h_after_ds` (in keV)
and `window`, `n_before`, `n_after`, `sf`, `e_unit` and `bin_width`, for plotting with `LegendMakie.lplot`.
"""
function get_continuum_survival_fraction(e::AbstractVector{<:T}, center::T, window::T, survival_flag::AbstractVector{Bool};
    low_flag::AbstractVector{Bool}=survival_flag) where T<:Unitful.Energy{<:Real}
    e_wdw_idx = center - window .< e .< center + window
    e = e[e_wdw_idx]
    survival_flag = survival_flag[e_wdw_idx]
    low_flag = low_flag[e_wdw_idx]

    bin_width = get_friedman_diaconis_bin_width(e)
    n_before = length(e)
    n_after = count(survival_flag)
    sf = n_after / n_before
    result = (
        window = measurement(center, window),
        n_before = measurement(n_before, sqrt(n_before)),
        n_after = measurement(n_after, sqrt(n_after)),
        sf = measurement(sf, sqrt(sf * (1-sf) / n_before)) * 100.0 * u"percent",
    )

    e_unit = u"keV"
    hist_binning = ustrip(e_unit, center - window):ustrip(e_unit, bin_width):ustrip(e_unit, center + window)
    report = (
        h_before = fit(Histogram, ustrip.(e_unit, e), hist_binning),
        h_after_low = fit(Histogram, ustrip.(e_unit, e[low_flag]), hist_binning),
        h_after_ds = fit(Histogram, ustrip.(e_unit, e[survival_flag]), hist_binning),
        window = result.window,
        n_before = result.n_before,
        n_after = result.n_after,
        sf = result.sf,
        e_unit = e_unit,
        bin_width = bin_width,
    )
    return result, report
end

function get_continuum_survival_fraction(cut_parameter::AbstractVector{<:Unitful.RealOrRealQuantity}, e::AbstractVector{<:T}, center::T, window::T;
    low_cut::Unitful.RealOrRealQuantity=-Inf*unit(first(cut_parameter)), high_cut::Unitful.RealOrRealQuantity=Inf*unit(first(cut_parameter)), selection::AbstractVector{Bool}=trues(length(cut_parameter))) where T<:Unitful.Energy{<:Real}
    low_flag = low_cut .< cut_parameter .< Inf*unit(first(cut_parameter))
    survival_flag = (low_cut .< cut_parameter .< high_cut) .&& selection
    get_continuum_survival_fraction(e, center, window, survival_flag; low_flag)
end

Base.@deprecate get_continuum_surrival_fraction(args...; kwargs...) get_continuum_survival_fraction(args...; kwargs...)
export get_continuum_survival_fraction, get_continuum_surrival_fraction

"""Deprecated alias of [`get_continuum_survival_fraction`](@ref)."""
get_continuum_surrival_fraction

