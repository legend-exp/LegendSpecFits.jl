f_aoe_sigma(x, p) = p[1] .+ p[2]*exp.(-p[3]./x)


"""
    fit_aoe_corrections(e::Array{<:Real}, μ::Array{T}, σ::Array{T}) where T<:Real

Fit the corrections for the AoE value of the detector.
"""
function fit_aoe_corrections(e::Array{<:Real}, μ::Array{T}, σ::Array{T}) where T<:Real
    # fit compton band mus with linear function
    μ_cut = (mean(μ) - 2*std(μ) .< μ .&& μ .< mean(μ) + 2*std(μ))
    μ_scs = linregress(e[μ_cut], μ[μ_cut])
    μ_scs_slope, μ_scs_intercept = LinearRegression.slope(μ_scs)[1], LinearRegression.bias(μ_scs)[1]
    @debug "μ_scs_slope    : $μ_scs_slope"
    @debug "μ_scs_intercept: $μ_scs_intercept"

    # fit compton band sigmas with exponential function
    σ_scs = curve_fit(f_aoe_sigma, e, σ, [0.0, maximum(σ), 5.0])
    @debug "σ_scs offset: $(σ_scs.param[1])"
    @debug "σ_scs shift : $(σ_scs.param[2])"
    @debug "σ_scs phase : $(σ_scs.param[3])"
    
    (
        e = e[μ_cut],
        μ = μ[μ_cut],
        f_μ_scs = x -> μ_scs_slope * x + μ_scs_intercept,
        μ_scs_slope = μ_scs_slope,
        μ_scs_intercept = μ_scs_intercept,
        σ = σ[μ_cut],
        σ_scs = σ_scs.param,
        f_σ_scs = x -> Base.Fix2(f_aoe_sigma, σ_scs.param)(x)
    )
end
export fit_aoe_corrections

""" 
    correctAoE!(aoe::Array{T}, e::Array{T}, aoe_corrections::NamedTuple{(:e, :μ, :f_μ_scs, :μ_scs_slope, :μ_scs_intercept, :σ, :σ_scs, :f_σ_scs)}) where T<:Real

Correct the AoE values in the `aoe` array using the corrections in `aoe_corrections`.
"""
function correct_aoe!(aoe::Array{T}, e::Array{T}, aoe_corrections::NamedTuple{(:e, :μ, :f_μ_scs, :μ_scs_slope, :μ_scs_intercept, :σ, :σ_scs, :f_σ_scs)}) where T<:Real
    aoe ./= aoe_corrections.f_μ_scs.(e)
    aoe .-= 1.0
    aoe ./= aoe_corrections.f_σ_scs.(e)
end
export correct_aoe!


"""
    get_aoe_peakhists(aoe::Array{T}, e::Array{T}) where T<:Real

Get the histograms of the `DEP` in the AoE spectrum.
# Returns
- `hist`: Histogram of the `DEP` peak
- `stats`: Stats of the `DEP` peak
- `dep`: Position of the `DEP` peak
- `window`: Window size around the `DEP` peak
- `e`: Energy values in the window around the `DEP` peak
- `aoe`: AoE values in the window around the `DEP` peak
"""
function get_dep_peakhists(aoe::Array{T}, e::Array{T}) where T<:Real
    # DEP line and window
    dep, window = 1592.53, 25.0
    # create histogram
    dephist = fit(Histogram, e, dep-window:0.5:dep+window)
    depstats = estimate_single_peak_stats(dephist)
    # set peakstats to known peak position
    depstats = merge(depstats, (peak_pos = dep, ))
    result = (
        hist = dephist,
        stats = depstats,
        dep = dep,
        window = window,
        e = e[dep-window .< e .< dep+window],
        aoe = aoe[dep-window .< e .< dep+window]
        )
    return result
end
export get_dep_peakhists


"""
    get_n_after_psd_cut(psd_cut::T, aoe::Array{T}, e::Array{T}, peak::T, window::T,; uncertainty=true) where T<:Real

Get the number of counts after a cut value `psd_cut` for a given `peak` and `window` size whiile performing a peak fit with fixed position.

    # Returns
- `n`: Number of counts after the cut
- `n_err`: Uncertainty of the number of counts after the cut
"""
function get_n_after_psd_cut(psd_cut::T, aoe::Array{T}, e::Array{T}, peak::T, window::T,; uncertainty=true) where T<:Real
    # get energy after cut and create histogram
    peakhist = fit(Histogram, e[aoe .> psd_cut], peak-window:0.5:peak+window)
    # estimate peak stats
    peakstats = estimate_single_peak_stats(peakhist)
    # set peakstats to known peak position
    peakstats = merge(peakstats, (peak_pos = peak, ))
    # fit peak and return number of signal counts
    result, _ = fit_single_peak_th228(peakhist, peakstats, uncertainty=uncertainty, fixed_position=true)
    if uncertainty
        n, n_err = result.n, result.err.n
    else
        n, n_err = result.n, 0.0
    end
    return (n = result.n, n_err = n_err)
end
export get_n_after_psd_cut


"""
    get_psd_cut(aoe::Array{T}, e::Array{T},; dep_sf::Float64=0.9) where T<:Real

Get the PSD cut value for a given `DEP` surrival fraction `dep_sf` (Default: 90%). The cut value is determined by finding the cut value where the number of counts after the cut is `dep_sf` of the number of counts before the cut.
The algorithm is based on a root search function and expecting a Bisection.
# Returns
- `cut`: PSD cut value
- `n0`: Number of counts before the cut
- `n90`: Number of counts after the cut
"""
function get_psd_cut(aoe::Array{T}, e::Array{T},; dep_sf::Float64=0.9) where T<:Real
    # generate DEP peak histogram
    dep_peakhist = get_dep_peakhists(aoe, e)
    # fit before cut
    result_before, _ = fit_single_peak_th228(dep_peakhist.hist, dep_peakhist.stats, uncertainty=false, fixed_position=true)
    # get n0 before cut
    n90 = result_before.n * 0.9
    # get psd cut
    n_surrival_dep_f = cut -> get_n_after_psd_cut(cut, dep_peakhist.aoe, dep_peakhist.e, dep_peakhist.dep, dep_peakhist.window,; uncertainty=false).n - n90
    cut_search_interval = (-25.0, 0.0)
    psd_cut = find_zero(n_surrival_dep_f, cut_search_interval, Bisection())
    return (cut = psd_cut, n0 = result_before.n, n90 = n90)
end
export get_psd_cut


"""
    get_peak_surrival_fraction(aoe::Array{T}, e::Array{T}, peak::T, window::T, psd_cut::T,; uncertainty=true) where T<:Real

Get the surrival fraction of a peak after a PSD cut value `psd_cut` for a given `peak` and `window` size whiile performing a peak fit with fixed position.
    
# Returns
- `peak`: Peak position
- `n_before`: Number of counts before the cut
- `n_after`: Number of counts after the cut
- `sf`: Surrival fraction
- `err`: Uncertainties
"""
function get_peak_surrival_fraction(aoe::Array{T}, e::Array{T}, peak::T, window::T, psd_cut::T,; uncertainty=true) where T<:Real
    # get energy before cut and create histogram
    peakhist = fit(Histogram, e, peak-window:0.5:peak+window)
    # estimate peak stats
    peakstats = estimate_single_peak_stats(peakhist)
    # set peakstats to known peak position
    peakstats = merge(peakstats, (peak_pos = peak, ))
    # fit peak and return number of signal counts
    result_before, report_before = fit_single_peak_th228(peakhist, peakstats, uncertainty=uncertainty, fixed_position=true)
    # get energy after cut and create histogram
    peakhist = fit(Histogram, e[aoe .> psd_cut], peak-window:0.5:peak+window)
    # estimate peak stats
    peakstats = estimate_single_peak_stats(peakhist)
    # set peakstats to known peak position
    peakstats = merge(peakstats, (peak_pos = peak, ))
    # fit peak and return number of signal counts
    result_after, report_after = fit_single_peak_th228(peakhist, peakstats, uncertainty=uncertainty, fixed_position=true)
    # calculate surrival fraction
    sf = result_after.n / result_before.n
    result = (
        peak = peak, 
        n_before = result_before.n,
        n_after = result_after.n,
        sf = sf,
    )
    report = (
        peak = result.peak, 
        n_before = result.n_before,
        n_after = result.n_after,
        sf = result.sf,
        before = report_before,
        after = report_after,
    )
    if uncertainty
        sf_err = sf * sqrt((result_after.err.n/result_after.n)^2 + (result_before.err.n/result_before.n)^2)
        result = merge(result, (err = (
            n_before = result_before.err.n,
            n_after = result_after.err.n,
            sf = sf_err,
        ), ))
    end
    return result, report
end
export get_peak_surrival_fraction


"""
    get_peaks_surrival_fractions(aoe::Array{T}, e::Array{T}, peaks::Array{T}, peak_names::Array{Symbol}, windows::Array{T}, psd_cut::T,; uncertainty=true) where T<:Real

Get the surrival fraction of a peak after a PSD cut value `psd_cut` for a given `peak` and `window` size while performing a peak fit with fixed position.
# Return 
- `result`: Dict of results for each peak
- `report`: Dict of reports for each peak
"""
function get_peaks_surrival_fractions(aoe::Array{T}, e::Array{T}, peaks::Array{T}, peak_names::Array{Symbol}, windows::Array{T}, psd_cut::T,; uncertainty=true) where T<:Real
    # create return and result dicts
    result = Dict{Symbol, NamedTuple}()
    report = Dict{Symbol, NamedTuple}()
    # iterate throuh all peaks
    for (peak, name, window) in zip(peaks, peak_names, windows)
        # fit peak
        result_peak, report_peak = get_peak_surrival_fraction(aoe, e, peak, window, psd_cut; uncertainty=uncertainty)
        # save results
        result[name] = result_peak
        report[name] = report_peak
    end
    return result, report
end
export get_peaks_surrival_fractions

"""
    get_continuum_surrival_fraction(aoe::Array{T}, e::Array{T}, center::T, window::T, psd_cut::T,; uncertainty=true) where T<:Real

Get the surrival fraction of a continuum after a PSD cut value `psd_cut` for a given `center` and `window` size.

# Returns
- `center`: Center of the continuum
- `window`: Window size
- `n_before`: Number of counts before the cut
- `n_after`: Number of counts after the cut
- `sf`: Surrival fraction
- `err`: Uncertainties
"""
function get_continuum_surrival_fraction(aoe::Array{T}, e::Array{T}, center::T, window::T, psd_cut::T,; uncertainty=true) where T<:Real
    # get number of events in window before cut
    n_before = length(e[center-window .< e .< center+window])
    # get number of events after cut
    n_after = length(e[aoe .> psd_cut .&& center-window .< e .< center+window])
    # calculate surrival fraction
    sf = n_after / n_before
    result = (
        center = center, 
        window = window,
        n_before = n_before,
        n_after = n_after,
        sf = sf,
    )
    if uncertainty
        # calculate uncertainty
        sf_err = sqrt(sf * (1-sf) / n_before)
        result = merge(result, (err = (
            n_before = sqrt(n_before),
            n_after = sqrt(n_after),
            sf = sf_err,
        ), ))
    end
    return result
end
export get_continuum_surrival_fraction