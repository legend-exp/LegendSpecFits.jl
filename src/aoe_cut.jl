
"""
    get_sf_after_aoe_cut(aoe_cut::Unitful.RealOrRealQuantity, aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peak::T, window::Vector{T}, bin_width::T, result_before::NamedTuple; uncertainty::Bool=true, fit_func::Symbol=:gamma_def) where T<:Unitful.Energy{<:Real}

Get the survival fraction after a AoE cut value `aoe_cut` for a given `peak` and `window` size from a combined fit to the survived and cut histograms.

# Returns
- `sf`: Survival fraction after the cut
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
    get_low_aoe_cut(aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T},; 
            dep::T=1592.53u"keV", window::Vector{<:T}=[12.0, 10.0]u"keV", dep_sf::Float64=0.9, rtol::Float64=0.001, maxiters::Int=300, sigma_high_sided::Float64=Inf,
            cut_search_interval::Tuple{<:Unitful.RealOrRealQuantity, <:Unitful.RealOrRealQuantity}=(-25.0*unit(first(aoe)), 1.0*unit(first(aoe))), 
            bin_width_window::T=3.0u"keV", max_e_plot::T=3000.0u"keV",  plot_window::Vector{<:T}=[12.0, 50.0]u"keV",
            fixed_position::Bool=true, fit_func::Symbol=:gamma_def, uncertainty::Bool=true) where T<:Unitful.Energy{<:Real}
Get the AoE cut value for a given `dep` and `window` size while performing a peak fit with fixed position. The AoE cut value is determined by finding the cut value for which the number of counts after the cut is equal to `dep_sf` times the number of counts before the cut.
The algorhithm utilizes a root search algorithm to find the cut value with a relative tolerance of `rtol`.
# Returns
- `cut`: AoE cut value
- `n0`: Number of counts before the cut
- `nsf`: Number of counts after the cut
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
    get_peaks_survival_fractions(e, peaks, peak_names, windows, survival_flag; kwargs...)
    get_peaks_survival_fractions(cut_parameter, e, peaks, peak_names, windows; low_cut=-Inf, high_cut=Inf, selection=trues(length(cut_parameter)), kwargs...)

Fit survival fractions for several peaks. The primary method accepts the final event
selection directly. The interval-based method constructs that flag and forwards to
the primary method. Either bound may be omitted; with both omitted, finite cut
parameter values survive. Bounds are exclusive.
"""
function get_peaks_survival_fractions(e::AbstractVector{<:T}, peaks::AbstractVector{<:T}, peak_names::AbstractVector{Symbol}, windows::AbstractVector{<:Tuple{T, T}}, survival_flag::AbstractVector{Bool};
    uncertainty::Bool=true, bin_width_window::T=2.0u"keV", fit_funcs::AbstractVector{Symbol}=fill(:gamma_def, length(peaks))) where T<:Unitful.Energy{<:Real}
    v_result = Vector{NamedTuple}(undef, length(peak_names))
    v_report = Vector{NamedTuple}(undef, length(peak_names))

    Threads.@threads for i in eachindex(peaks, peak_names, windows, fit_funcs)
        peak, window, fit_func = peaks[i], windows[i], fit_funcs[i]
        result_peak, report_peak = get_peak_survival_fraction(e, peak, collect(window), survival_flag; uncertainty, bin_width_window, fit_func)
        v_result[i] = result_peak
        v_report[i] = report_peak
    end

    result = OrderedDict{Symbol, NamedTuple}(peak_names .=> v_result)
    report = OrderedDict{Symbol, NamedTuple}(peak_names .=> v_report)
    return result, report
end

get_peaks_survival_fractions(e, peaks, peak_names, left_window_sizes::AbstractVector{<:Unitful.Energy{<:Real}}, right_window_sizes::AbstractVector{<:Unitful.Energy{<:Real}}, survival_flag::AbstractVector{Bool}; kwargs...) = get_peaks_survival_fractions(e, peaks, peak_names, collect(zip(left_window_sizes, right_window_sizes)), survival_flag; kwargs...)

function get_peaks_survival_fractions(cut_parameter::AbstractVector{<:Unitful.RealOrRealQuantity}, e::AbstractVector{<:T}, peaks::AbstractVector{<:T}, peak_names::AbstractVector{Symbol}, windows::AbstractVector{<:Tuple{T, T}};
    low_cut::Unitful.RealOrRealQuantity=-Inf*unit(first(cut_parameter)), high_cut::Unitful.RealOrRealQuantity=Inf*unit(first(cut_parameter)), selection::AbstractVector{Bool}=trues(length(cut_parameter)), kwargs...) where T<:Unitful.Energy{<:Real}
    survival_flag = (low_cut .< cut_parameter .< high_cut) .&& selection
    get_peaks_survival_fractions(e, peaks, peak_names, windows, survival_flag; kwargs...)
end

get_peaks_survival_fractions(cut_parameter::AbstractVector{<:Unitful.RealOrRealQuantity}, e, peaks, peak_names, left_window_sizes::AbstractVector{<:Unitful.Energy{<:Real}}, right_window_sizes::AbstractVector{<:Unitful.Energy{<:Real}}; kwargs...) = get_peaks_survival_fractions(cut_parameter, e, peaks, peak_names, collect(zip(left_window_sizes, right_window_sizes)); kwargs...)

Base.@deprecate get_peaks_surrival_fractions(args...; kwargs...) get_peaks_survival_fractions(args...; kwargs...)
export get_peaks_survival_fractions, get_peaks_surrival_fractions


"""
    get_peak_survival_fraction(peakhist, survived_hist, cut_hist; uncertainty=true, fit_func=:gamma_def)
    get_peak_survival_fraction(e, survival_flag; uncertainty=true, fit_func=:gamma_def)
    get_peak_survival_fraction(e, peak, window, survival_flag; kwargs...)
    get_peak_survival_fraction(cut_parameter, e, peak, window; low_cut=-Inf, high_cut=Inf, selection=trues(length(cut_parameter)), kwargs...)

Fit a peak before and after an arbitrary event selection. The histogram method accepts
count histograms for all, surviving, and rejected events with identical, unitless
energy binning; the latter two must partition the first. It returns fit results and
reports without a nominal `peak` label.

The two-vector method bins the supplied peak-region energies and returns the
histogram method's outputs. The center/window method accepts exactly two lower
and upper widths and adds the supplied `peak` to both outputs. The interval-based
method constructs the selection and forwards to the center/window method. Bounds
are exclusive.
"""
function get_peak_survival_fraction(e::AbstractVector{<:T}, peak::T, window::Union{AbstractVector{T}, Tuple{T, T}}, survival_flag::AbstractVector{Bool};
    uncertainty::Bool=true, bin_width_window::T=2.0u"keV", fit_func::Symbol=:gamma_def) where T<:Unitful.Energy{<:Real}
    @argcheck length(window) == 2
    bin_width = get_friedman_diaconis_bin_width(e[e .> peak - bin_width_window .&& e .< peak + bin_width_window])
    binning = ustrip(peak-first(window)):ustrip(bin_width):ustrip(peak+last(window))

    peakhist = fit(Histogram, ustrip.(e), binning)
    survived_hist = fit(Histogram, ustrip.(e[survival_flag]), binning)
    cut_hist = fit(Histogram, ustrip.(e[.!survival_flag]), binning)
    result, report = get_peak_survival_fraction(peakhist, survived_hist, cut_hist; uncertainty, fit_func)
    return merge((peak = peak,), result), merge((peak = peak,), report)
end

function get_peak_survival_fraction(e::AbstractVector{<:Unitful.Energy{<:Real}}, survival_flag::AbstractVector{Bool};
    uncertainty::Bool=true, fit_func::Symbol=:gamma_def)
    @argcheck axes(e) == axes(survival_flag)
    bin_width = get_friedman_diaconis_bin_width(e)
    binning = ustrip(minimum(e)):ustrip(bin_width):ustrip(maximum(e) + bin_width)
    peakhist = fit(Histogram, ustrip.(e), binning)
    survived_hist = fit(Histogram, ustrip.(e[survival_flag]), binning)
    cut_hist = fit(Histogram, ustrip.(e[.!survival_flag]), binning)
    get_peak_survival_fraction(peakhist, survived_hist, cut_hist; uncertainty, fit_func)
end

function get_peak_survival_fraction(peakhist::Histogram, survived_hist::Histogram, cut_hist::Histogram;
    uncertainty::Bool=true, fit_func::Symbol=:gamma_def)
    peakstats = estimate_single_peak_stats(peakhist)
    result_before, report_before = fit_single_peak_th228(peakhist, peakstats; uncertainty, fit_func)
    result_after, report_after = fit_subpeaks_th228(survived_hist, cut_hist, result_before; uncertainty, fit_func)

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

function get_peak_survival_fraction(cut_parameter::AbstractVector{<:Unitful.RealOrRealQuantity}, e::AbstractVector{<:T}, peak::T, window::Union{AbstractVector{T}, Tuple{T, T}};
    low_cut::Unitful.RealOrRealQuantity=-Inf*unit(first(cut_parameter)), high_cut::Unitful.RealOrRealQuantity=Inf*unit(first(cut_parameter)), selection::AbstractVector{Bool}=trues(length(cut_parameter)), kwargs...) where T<:Unitful.Energy{<:Real}
    survival_flag = (low_cut .< cut_parameter .< high_cut) .&& selection
    get_peak_survival_fraction(e, peak, window, survival_flag; kwargs...)
end

Base.@deprecate get_peak_surrival_fraction(args...; kwargs...) get_peak_survival_fraction(args...; kwargs...)
export get_peak_survival_fraction, get_peak_surrival_fraction



"""
    get_continuum_survival_fraction(e, center, window, survival_flag; low_flag=survival_flag)
    get_continuum_survival_fraction(cut_parameter, e, center, window; low_cut=-Inf, high_cut=Inf, selection=trues(length(cut_parameter)))

Count events in the continuum window before and after a selection. The interval
overload forwards a Boolean flag to the counting method. The report retains the
existing `h_after_low` and `h_after_ds` fields: the former applies only the lower
bound, while the latter always represents the final selection, including `high_cut`
and `selection`.
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
