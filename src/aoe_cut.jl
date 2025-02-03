
"""
    get_sf_after_aoe_cut(aoe_cut::Unitful.RealOrRealQuantity, aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peak::T, window::Vector{T}, bin_width::T, result_before::NamedTuple; uncertainty::Bool=true, fit_func::Symbol=:gamma_def) where T<:Unitful.Energy{<:Real}

Get the survival fraction after a AoE cut value `aoe_cut` for a given `peak` and `window` size from a combined fit to the survived and cut histograms.

# Returns
- `sf`: Survival fraction after the cut
"""
function get_sf_after_aoe_cut(aoe_cut::Unitful.RealOrRealQuantity, aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peak::T, window::Vector{T}, bin_width::T, result_before::NamedTuple; uncertainty::Bool=true, fit_func::Symbol=:gamma_def) where T<:Unitful.Energy{<:Real}
    # get energy after cut and create histogram
    survived = fit(Histogram, ustrip.(e[aoe .>= aoe_cut]), ustrip(peak-first(window):bin_width:peak+last(window)))
    cut = fit(Histogram, ustrip.(e[aoe .< aoe_cut]), ustrip(peak-first(window):bin_width:peak+last(window)))
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
    # scale unit
    e_unit = u"keV"
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
    get_peaks_survival_fractions(psd_parameter::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peaks::Vector{<:T}, peak_names::Vector{Symbol}, windows::Vector{<:Tuple{T, T}}, psd_cut::Unitful.RealOrRealQuantity, cut_vector::BitVector,; 
    uncertainty::Bool=true, inverted_mode::Bool=false, bin_width_window::T=2.0u"keV", sigma_high_sided::Unitful.RealOrRealQuantity=Inf*unit(first(psd_parameter)), fit_funcs::Vector{Symbol}=fill(:gamma_def, length(peaks))) where T<:Unitful.Energy{<:Real}
 
Get the survival fraction of a peak after a PSD cut value `psd_cut` for a given `peak` and `window` size while performing a peak fit with fixed position.

# Arguments
- `psd_parameter`: Vector of PSD parameter values.
- `e`: Vector of energy values.
- `peaks`: Vector of peak positions in energy.
- `peak_names`: Vector of symbols representing peak names.
- `windows`: Vector of tuples specifying the lower and upper energy bounds for fitting each peak.
- `psd_cut`: The PSD threshold used for classification.
- `cut_vector`: BitVector indicating which events pass the PSD cut.

# Keyword Arguments
- `uncertainty`: Whether to compute uncertainties (default: `true`).
- `inverted_mode`: If `true`, inverts the PSD cut logic (default: `false`).
- `bin_width_window`: Bin width for histogramming within peak windows (default: `2 keV`).
- `sigma_high_sided`: Upper bound for peak fitting beyond `psd_cut` (default: `Inf`).
- `fit_funcs`: Vector specifying the peak fitting function for each peak (default: `:gamma_def`).

# Return 
- `result`: Dict of results for each peak
- `report`: Dict of reports for each peak
"""
function get_peaks_survival_fractions(psd_parameter::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peaks::Vector{<:T}, peak_names::Vector{Symbol}, windows::Vector{<:Tuple{T, T}}, psd_cut::Unitful.RealOrRealQuantity, cut_vector::BitVector,; 
    uncertainty::Bool=true, inverted_mode::Bool=false, bin_width_window::T=2.0u"keV", sigma_high_sided::Unitful.RealOrRealQuantity=Inf*unit(first(psd_parameter)), fit_funcs::Vector{Symbol}=fill(:gamma_def, length(peaks))) where T<:Unitful.Energy{<:Real}
    @assert length(peaks) == length(peak_names) == length(windows) "Length of peaks, peak_names and windows must be equal"
    # create return and result vectors
    v_result = Vector{NamedTuple}(undef, length(peak_names))
    v_report = Vector{NamedTuple}(undef, length(peak_names))
    
    # iterate throuh all peaks
    Threads.@threads for i in eachindex(peak_names)
        # extract peak, name and window
        peak, name, window, fit_func = peaks[i], peak_names[i], windows[i], fit_funcs[i]
        # fit peak
        result_peak, report_peak = get_peak_survival_fraction(psd_parameter, e, peak, collect(window), psd_cut, cut_vector; uncertainty=uncertainty, inverted_mode=inverted_mode, bin_width_window=bin_width_window, sigma_high_sided=sigma_high_sided, fit_func=fit_func)
        # save results
        v_result[i] = result_peak
        v_report[i] = report_peak
    end

    # create result and report dict
    result = Dict{Symbol, NamedTuple}(peak_names .=> v_result)
    report = Dict{Symbol, NamedTuple}(peak_names .=> v_report)
    
    return result, report
end
get_peaks_survival_fractions(psd_parameter, e, peaks, peak_names, left_window_sizes::Vector{<:Unitful.Energy{<:Real}}, right_window_sizes::Vector{<:Unitful.Energy{<:Real}}, psd_cut, cut_vector; kwargs...) = get_peaks_survival_fractions(psd_parameter, e, peaks, peak_names, [(l,r) for (l,r) in zip(left_window_sizes, right_window_sizes)], psd_cut, cut_vector; kwargs...)

function get_peaks_survival_fractions(psd_parameter::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peaks::Vector{<:T}, peak_names::Vector{Symbol}, windows::Vector{<:Tuple{T, T}}, psd_cut::Unitful.RealOrRealQuantity,; 
    uncertainty::Bool=true, inverted_mode::Bool=false, bin_width_window::T=2.0u"keV", sigma_high_sided::Unitful.RealOrRealQuantity=Inf*unit(first(psd_parameter)), fit_funcs::Vector{Symbol}=fill(:gamma_def, length(peaks))) where T<:Unitful.Energy{<:Real}
    @assert length(peaks) == length(peak_names) == length(windows) "Length of peaks, peak_names and windows must be equal"
    # create return and result vectors
    v_result = Vector{NamedTuple}(undef, length(peak_names))
    v_report = Vector{NamedTuple}(undef, length(peak_names))
    
    # iterate throuh all peaks
    Threads.@threads for i in eachindex(peak_names)
        # extract peak, name and window
        peak, name, window, fit_func = peaks[i], peak_names[i], windows[i], fit_funcs[i]
        # fit peak
        result_peak, report_peak = get_peak_survival_fraction(psd_parameter, e, peak, collect(window), psd_cut; uncertainty=uncertainty, inverted_mode=inverted_mode, bin_width_window=bin_width_window, sigma_high_sided=sigma_high_sided, fit_func=fit_func)
        # save results
        v_result[i] = result_peak
        v_report[i] = report_peak
    end

    # create result and report dict
    result = Dict{Symbol, NamedTuple}(peak_names .=> v_result)
    report = Dict{Symbol, NamedTuple}(peak_names .=> v_report)
    
    return result, report
end
get_peaks_survival_fractions(psd_parameter, e, peaks, peak_names, left_window_sizes::Vector{<:Unitful.Energy{<:Real}}, right_window_sizes::Vector{<:Unitful.Energy{<:Real}}, psd_cut; kwargs...) = get_peaks_survival_fractions(psd_parameter, e, peaks, peak_names, [(l,r) for (l,r) in zip(left_window_sizes, right_window_sizes)], psd_cut; kwargs...)

Base.@deprecate get_peaks_surrival_fractions(args...; kwargs...) get_peaks_survival_fractions(args...; kwargs...)
export get_peaks_survival_fractions, get_peaks_surrival_fractions

"""
    get_peak_survival_fraction(psd_parameter::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peak::T, window::Vector{T}, psd_cut::Unitful.RealOrRealQuantity; 
    uncertainty::Bool=true, inverted_mode::Bool=false, bin_width_window::T=2.0u"keV", sigma_high_sided::Unitful.RealOrRealQuantity=Inf*unit(first(psd_parameter)), fit_func::Symbol=:gamma_def) where T<:Unitful.Energy{<:Real}  

Get the survival fraction of a peak after a PSD cut value `psd_cut` for a given `peak` and `window` size while performing a peak fit with fixed position.
    
# Arguments
- `psd_parameter`: Vector of PSD parameters used for the cut.
- `e`: Vector of energy values.
- `peak`: The energy position of the peak.
- `window`: Energy window (lower and upper bounds) around the peak.
- `psd_cut`: Threshold value for the PSD cut.

# Keyword Arguments
- `uncertainty`: If `true`, includes uncertainty estimation in the fit.
- `inverted_mode`: If `true`, inverts the PSD cut logic.
- `bin_width_window`: Width of the histogram bins within the peak window.
- `sigma_high_sided`: Upper bound for PSD cut filtering.
- `fit_func`: Specifies the fitting function for peak estimation.

# Returns
A tuple containing:
- `result`: NamedTuple with peak position, counts before and after the cut, survival fraction, and goodness-of-fit metrics.
- `report`: NamedTuple with detailed fitting results for before and after the cut.
"""
function get_peak_survival_fraction(cut_vector::BitVector, e::Vector{<:T}, peak::T, window::Vector{T}; 
    uncertainty::Bool=true, bin_width_window::T=2.0u"keV", fit_func::Symbol=:gamma_def) where T<:Unitful.Energy{<:Real}  

    # Estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e[e .> peak - bin_width_window .&& e .< peak + bin_width_window])

    # Get energy before cut and create histogram
    peakhist = fit(Histogram, ustrip.(e), ustrip(peak-first(window):bin_width:peak+last(window)))
    
    # Estimate peak stats
    peakstats = estimate_single_peak_stats(peakhist)

    # Fit peak and return number of signal counts
    result_before, report_before = fit_single_peak_th228(peakhist, peakstats; uncertainty=uncertainty, fit_func=fit_func)

    # Apply cut using precomputed cut_vector
    e_survived, e_cut = e[cut_vector], e[.!cut_vector]

    # Get energy after cut and create histograms
    survived = fit(Histogram, ustrip(e_survived), ustrip(peak-first(window):bin_width:peak+last(window)))
    cut      = fit(Histogram, ustrip(e_cut),      ustrip(peak-first(window):bin_width:peak+last(window)))

    # Fit peak and return number of signal counts after cut
    result_after, report_after = fit_subpeaks_th228(survived, cut, result_before; uncertainty=uncertainty, fit_func=fit_func)

    # Calculate survival fraction
    sf = result_after.sf * 100u"percent"
    result = (
        peak = peak,
        fit_func=fit_func,
        n_before = result_before.n,
        n_after = result_before.n * result_after.sf,
        sf = sf, 
        gof = (after = result_after.gof, before = result_before.gof),
    )
    report = (
        peak = result.peak, 
        n_before = result.n_before,
        n_after = result.n_before * result_after.sf,
        sf = result.sf,
        before = report_before,
        after = report_after,
    )
    return result, report
end

# Overloaded function for psd_cut only
function get_peak_survival_fraction(psd_parameter::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peak::T, window::Vector{T}, psd_cut::Unitful.RealOrRealQuantity; 
    uncertainty::Bool=true, inverted_mode::Bool=false, bin_width_window::T=2.0u"keV", sigma_high_sided::Unitful.RealOrRealQuantity=Inf*unit(first(psd_parameter)), fit_func::Symbol=:gamma_def) where T<:Unitful.Energy{<:Real}

    cut_vector = if !inverted_mode
        psd_cut .< psd_parameter .< sigma_high_sided
    else
        psd_parameter .< psd_cut
    end

    get_peak_survival_fraction(cut_vector, e, peak, window; 
                               uncertainty=uncertainty, 
                               bin_width_window=bin_width_window, 
                               fit_func=fit_func)
end

# Overloaded function for psd_cut and existing cut_vector
function get_peak_survival_fraction(psd_parameter::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peak::T, window::Vector{T}, psd_cut::Unitful.RealOrRealQuantity, cut_vector::BitVector; 
    uncertainty::Bool=true, inverted_mode::Bool=false, bin_width_window::T=2.0u"keV", sigma_high_sided::Unitful.RealOrRealQuantity=Inf*unit(first(psd_parameter)), fit_func::Symbol=:gamma_def) where T<:Unitful.Energy{<:Real}

    combined_cut_vector = if !inverted_mode
        psd_cut .< psd_parameter .< sigma_high_sided .&& cut_vector
    else
        psd_parameter .< psd_cut .&& cut_vector
    end

    get_peak_survival_fraction(combined_cut_vector, e, peak, window; 
                               uncertainty=uncertainty, 
                               bin_width_window=bin_width_window, 
                               fit_func=fit_func)
end


Base.@deprecate get_peak_surrival_fraction(args...; kwargs...) get_peak_survival_fraction(args...; kwargs...)
export get_peak_survival_fraction, get_peak_surrival_fraction


"""
    get_continuum_survival_fraction(psd_parameter::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, center::T, window::T, psd_cut::Unitful.RealOrRealQuantity; 
    inverted_mode::Bool=false, sigma_high_sided::Unitful.RealOrRealQuantity=Inf*unit(first(psd_parameter))) where T<:Unitful.Energy{<:Real}
Get the survival fraction of a continuum after a psd_parameter cut value `psd_cut` for a given `center` and `window` size.


# Arguments
- `cut_vector`: BitVector indicating which events pass the PSD cut.
- `psd_parameter`: Vector of PSD parameter values.
- `e`: Vector of energy values.
- `center`: The center of the continuum in energy units.
- `window`: The window size around the `center` within which to evaluate the continuum.
- `psd_cut`: The PSD cut value used for thresholding the events.

# Keyword Arguments
- `sigma_high_sided`: The upper threshold for the PSD cut in case of a secondary cut (default: `Inf`).
- `inverted_mode`: If `true`, inverts the PSD cut logic (default: `false`).
- `uncertainty`: Whether to compute uncertainties for `n_before`, `n_after`, and `sf` (default: `true`).

# Return
- `result`: A tuple containing:
  - `window`: The window size measurement.
  - `n_before`: The number of events before applying the cut.
  - `n_after`: The number of events after applying the cut.
  - `sf`: The survival fraction of events after the cut (in percentage).
  
- `report`: A dictionary containing:
  - `h_before`: A histogram of energy values before the PSD cut.
  - `h_after_low`: A histogram of energy values after the PSD cut for values below `psd_cut`.
  - `h_after_ds`: A histogram of energy values after the PSD cut for values between `psd_cut` and `sigma_high_sided`.
  - `window`: The window size used.
  - `n_before`: The number of events before applying the cut.
  - `n_after`: The number of events after applying the cut.
  - `sf`: The survival fraction (in percentage).
  - `e_unit`: The energy unit used (e.g., keV).
  - `bin_width`: The calculated bin width for histograms.
"""
function get_continuum_survival_fraction(cut_vector::BitVector, psd_parameter::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, center::T, window::T, psd_cut::Unitful.RealOrRealQuantity,; 
    sigma_high_sided::Unitful.RealOrRealQuantity=Inf*unit(first(psd_parameter))) where T<:Unitful.Energy{<:Real}

    # Scale unit
    e_unit = u"keV"
    # Get energy around center
    e_wdw_idx = findall(center - window .< e .< center + window)
    psd_parameter = psd_parameter[e_wdw_idx]
    e = e[e_wdw_idx]
    cut_vector = cut_vector[e_wdw_idx]  # Ensure the cut vector is restricted to the window

    # Get bin width
    bin_width = get_friedman_diaconis_bin_width(e)
    # Number of events before cut
    n_before = length(e)
    # Apply cut using precomputed cut_vector
    e_survived, e_cut = e[cut_vector], e[.!cut_vector]
    n_after = length(e_survived)

    # Calculate survival fraction
    sf = n_after / n_before
    result = (
        window = measurement(center, window),
        n_before = measurement(n_before, sqrt(n_before)),
        n_after = measurement(n_after, sqrt(n_after)),
        sf = measurement(sf, sqrt(sf * (1-sf) / n_before)) * 100.0 * u"percent",
    )
    
    hist_binning = ustrip(e_unit, center - window):ustrip(e_unit, bin_width):ustrip(e_unit, center + window)
    report = (
        h_before = fit(Histogram, ustrip.(e_unit, e), hist_binning),
        h_after_low = fit(Histogram, ustrip.(e_unit, e[psd_cut .< psd_parameter]), hist_binning),
        h_after_ds = fit(Histogram, ustrip.(e_unit, e[psd_cut .< psd_parameter .< sigma_high_sided]), hist_binning),
        window = result.window,
        n_before = result.n_before,
        n_after = result.n_after,
        sf = result.sf,
        e_unit = e_unit,
        bin_width = bin_width,
    )
    return result, report
end

# Overloaded function for psd_cut only
function get_continuum_survival_fraction(psd_parameter::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, center::T, window::T, psd_cut::Unitful.RealOrRealQuantity; 
    inverted_mode::Bool=false, sigma_high_sided::Unitful.RealOrRealQuantity=Inf*unit(first(psd_parameter))) where T<:Unitful.Energy{<:Real}

    cut_vector = if !inverted_mode
        psd_cut .< psd_parameter .< sigma_high_sided
    else
        psd_parameter .< psd_cut
    end
    get_continuum_survival_fraction(cut_vector, psd_parameter, e, center, window, psd_cut; 
                                    sigma_high_sided=sigma_high_sided)
end

# Overloaded function for psd_cut and existing cut_vector
function get_continuum_survival_fraction(psd_parameter::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, center::T, window::T, psd_cut::Unitful.RealOrRealQuantity, cut_vector::BitVector; 
    inverted_mode::Bool=false, sigma_high_sided::Unitful.RealOrRealQuantity=Inf*unit(first(psd_parameter))) where T<:Unitful.Energy{<:Real}

    combined_cut_vector = if !inverted_mode
        psd_cut .< psd_parameter .< sigma_high_sided .&& cut_vector
    else
        psd_parameter .< psd_cut .&& cut_vector
    end

    get_continuum_survival_fraction(combined_cut_vector, psd_parameter, e, center, window, psd_cut; 
                                    sigma_high_sided=sigma_high_sided)
end

Base.@deprecate get_continuum_surrival_fraction(args...; kwargs...) get_continuum_survival_fraction(args...; kwargs...)
export get_continuum_survival_fraction, get_continuum_surrival_fraction
