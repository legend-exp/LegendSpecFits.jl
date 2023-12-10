# f_aoe_sigma(x, p) = p[1] .+ p[2]*exp.(-p[3]./x)
f_aoe_sigma(x, p) = sqrt(abs(p[1]) + abs(p[2])/x^2)
f_aoe_mu(x, p) = p[1] .+ p[2]*x


"""
    fit_aoe_corrections(e::Array{<:Real}, μ::Array{T}, σ::Array{T}) where T<:Real

Fit the corrections for the AoE value of the detector.
"""
function fit_aoe_corrections(e::Array{<:Real}, μ::Array{T}, σ::Array{T}) where T<:Real
    # fit compton band mus with linear function
    μ_cut = (mean(μ) - 2*std(μ) .< μ .&& μ .< mean(μ) + 2*std(μ))
    e, μ, σ = e[μ_cut], μ[μ_cut], σ[μ_cut]
    # μ_scs = linregress(e[μ_cut], μ[μ_cut])
    μ_scs = linregress(e, μ)
    μ_scs_slope, μ_scs_intercept = LinearRegression.slope(μ_scs)[1], LinearRegression.bias(μ_scs)[1]
    μ_scs = curve_fit(f_aoe_mu, e, μ, [μ_scs_intercept, μ_scs_slope])
    μ_scs_err = stderror(μ_scs)  
    @debug "μ_scs_slope    : $μ_scs_slope"
    @debug "μ_scs_intercept: $μ_scs_intercept"

    # fit compton band sigmas with exponential function
    σ_scs = curve_fit(f_aoe_sigma, e, σ, [median(σ), 5.0])
    σ_scs_err = stderror(σ_scs)
    @debug "σ_scs offset: $(σ_scs.param[1])"
    @debug "σ_scs shift : $(σ_scs.param[2])"
    
    (
        e = e,
        μ = μ,
        μ_scs = μ_scs.param,
        f_μ_scs = x -> μ_scs_slope * x + μ_scs_intercept,
        σ = σ,
        σ_scs = abs.(σ_scs.param),
        f_σ_scs = x -> Base.Fix2(f_aoe_sigma, σ_scs.param)(x),
        err = (σ_scs = σ_scs_err, 
        μ_scs = μ_scs_err
        )
    )
end
export fit_aoe_corrections

""" 
    correctAoE!(aoe::Array{T}, e::Array{T}, aoe_corrections::NamedTuple) where T<:Real

Correct the AoE values in the `aoe` array using the corrections in `aoe_corrections`.
"""
function correct_aoe!(aoe::Array{T}, e::Array{T}, aoe_corrections::NamedTuple) where T<:Real
    aoe ./= Base.Fix2(f_aoe_mu, aoe_corrections.μ_scs).(e)
    aoe .-= 1.0
    aoe ./= Base.Fix2(f_aoe_sigma, aoe_corrections.σ_scs).(e)
end
export correct_aoe!

function correct_aoe!(aoe::Array{T}, e::Array{T}, aoe_corrections::PropDict) where T<:Real
    correct_aoe!(aoe, e, NamedTuple(aoe_corrections))
end

"""
    prepare_dep_peakhist(e::Array{T}, dep::T,; relative_cut::T=0.5, n_bins_cut::Int=500) where T<:Real

Prepare an array of uncalibrated DEP energies for parameter extraction and calibration.
# Returns
- `result`: Result of the initial fit
- `report`: Report of the initial fit
"""
function prepare_dep_peakhist(e::Array{T}, dep::T,; relative_cut::T=0.5, n_bins_cut::Int=500) where T<:Real
    # get cut window around peak
    cuts = cut_single_peak(e, minimum(e), maximum(e); n_bins=n_bins_cut, relative_cut=relative_cut)
    # estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e[e .> cuts.low .&& e .< cuts.high])
    # create histogram
    dephist = fit(Histogram, e, minimum(e):bin_width:maximum(e))
    # get peakstats
    depstats = estimate_single_peak_stats(dephist)
    # initial fit for calibration and parameter extraction
    result, report = fit_single_peak_th228(dephist, depstats,; uncertainty=true, low_e_tail=false)
    # get calibration estimate from peak postion
    result = merge(result, (m_calib = dep / result.μ, ))
    return result, report
end
export prepare_dep_peakhist


"""
    get_n_after_psd_cut(psd_cut::T, aoe::Array{T}, e::Array{T}, peak::T, window::Array{T}, bin_width::T, result_before::NamedTuple, peakstats::NamedTuple; uncertainty=true) where T<:Real

Get the number of counts after a PSD cut value `psd_cut` for a given `peak` and `window` size whiile performing a peak fit with fixed position. The number of counts is determined by fitting the peak with a pseudo prior for the peak position.
# Returns
- `n`: Number of counts after the cut
- `n_err`: Uncertainty of the number of counts after the cut
"""
function get_n_after_psd_cut(psd_cut::T, aoe::Array{T}, e::Array{T}, peak::T, window::Array{T}, bin_width::T, result_before::NamedTuple, peakstats::NamedTuple; uncertainty::Bool=true, fixed_position::Bool=true) where T<:Real
    # get energy after cut and create histogram
    peakhist = fit(Histogram, e[aoe .> psd_cut], peak-first(window):bin_width:peak+last(window))
    # create pseudo_prior with known peak sigma in signal for more stable fit
    pseudo_prior = NamedTupleDist(σ = Normal(result_before.σ, 0.3), )
    # fit peak and return number of signal counts
    result, _ = fit_single_peak_th228(peakhist, peakstats,; uncertainty=uncertainty, fixed_position=fixed_position, low_e_tail=false, pseudo_prior=pseudo_prior)
    if uncertainty
        n, n_err = result.n, result.err.n
    else
        n, n_err = result.n, 0.0
    end
    return (n = result.n, n_err = n_err)
end
export get_n_after_psd_cut


"""
    get_psd_cut(aoe::Array{T}, e::Array{T},; dep::T=1592.53, window::Array{T}=[12.0, 10.0], dep_sf::Float64=0.9, cut_search_interval::Tuple{T,T}=(-25.0, 0.0), rtol::T=0.001) where T<:Real

Get the PSD cut value for a given `dep` and `window` size while performing a peak fit with fixed position. The PSD cut value is determined by finding the cut value for which the number of counts after the cut is equal to `dep_sf` times the number of counts before the cut.
The algorhithm utilizes a root search algorithm to find the cut value with a relative tolerance of `rtol`.
# Returns
- `cut`: PSD cut value
- `n0`: Number of counts before the cut
- `nsf`: Number of counts after the cut
- `err`: Uncertainties
"""
function get_psd_cut(aoe::Array{T}, e::Array{T},; dep::T=1592.53, window::Array{T}=[12.0, 10.0], dep_sf::Float64=0.9, cut_search_interval::Tuple{T,T}=(-25.0, 0.0), rtol::T=0.001, bin_width_window::T=3.0, fixed_position::Bool=true) where T<:Real
    # estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e[e .> dep - bin_width_window .&& e .< dep + bin_width_window])
    # create histogram
    dephist = fit(Histogram, e, dep-first(window):bin_width:dep+last(window))
    # get peakstats
    depstats = estimate_single_peak_stats(dephist)
    # cut window around peak
    aoe = aoe[dep-first(window) .< e .< dep+last(window)]
    e   =   e[dep-first(window) .< e .< dep+last(window)]
    # fit before cut
    result_before, _ = fit_single_peak_th228(dephist, depstats,; uncertainty=true, fixed_position=fixed_position, low_e_tail=false)
    # get n0 before cut
    nsf = result_before.n * dep_sf
    # get psd cut
    n_surrival_dep_f = cut -> get_n_after_psd_cut(cut, aoe, e, dep, window, bin_width, result_before, depstats; uncertainty=false, fixed_position=fixed_position).n - nsf
    psd_cut = find_zero(n_surrival_dep_f, cut_search_interval, Bisection(), rtol=rtol, maxiters=100)
    # return n_surrival_dep_f.(0.25:0.001:0.5)
    # get nsf after cut
    result_after = get_n_after_psd_cut(psd_cut, aoe, e, dep, window, bin_width, result_before, depstats; uncertainty=true, fixed_position=fixed_position)
    return (cut = psd_cut, n0 = result_before.n, nsf = result_after.n, err = (cut = psd_cut * rtol, n0 = result_before.err.n, nsf = result_after.n_err))
end
export get_psd_cut


"""
    get_peak_surrival_fraction(aoe::Array{T}, e::Array{T}, peak::T, window::Array{T}, psd_cut::T,; uncertainty::Bool=true, low_e_tail::Bool=true) where T<:Real

Get the surrival fraction of a peak after a PSD cut value `psd_cut` for a given `peak` and `window` size whiile performing a peak fit with fixed position.
    
# Returns
- `peak`: Peak position
- `n_before`: Number of counts before the cut
- `n_after`: Number of counts after the cut
- `sf`: Surrival fraction
- `err`: Uncertainties
"""
function get_peak_surrival_fraction(aoe::Array{T}, e::Array{T}, peak::T, window::Array{T}, psd_cut::T,; uncertainty::Bool=true, low_e_tail::Bool=true, bin_width_window::T=2.0) where T<:Real
    # estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e[e .> peak - bin_width_window .&& e .< peak + bin_width_window])
    # get energy before cut and create histogram
    peakhist = fit(Histogram, e, peak-first(window):bin_width:peak+last(window))
    # estimate peak stats
    peakstats = estimate_single_peak_stats(peakhist)
    # fit peak and return number of signal counts
    result_before, report_before = fit_single_peak_th228(peakhist, peakstats,; uncertainty=uncertainty, low_e_tail=low_e_tail)

    # get e after cut
    e = e[aoe .> psd_cut]
    # estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e[e .> peak - bin_width_window .&& e .< peak + bin_width_window])
    # get energy after cut and create histogram
    peakhist = fit(Histogram, e, peak-first(window):bin_width:peak+last(window))
    # create pseudo_prior with known peak sigma in signal for more stable fit
    pseudo_prior = NamedTupleDist(μ = ConstValueDist(result_before.μ), σ = Normal(result_before.σ, 0.1))
    pseudo_prior = NamedTupleDist(σ = Normal(result_before.σ, 0.1), )
    # estimate peak stats
    peakstats = estimate_single_peak_stats(peakhist)
    # fit peak and return number of signal counts
    result_after, report_after = fit_single_peak_th228(peakhist, peakstats,; uncertainty=uncertainty, low_e_tail=low_e_tail)
    # result_after, report_after = fit_single_peak_th228(peakhist, peakstats,; uncertainty=uncertainty, low_e_tail=low_e_tail, pseudo_prior=pseudo_prior)
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
function get_peaks_surrival_fractions(aoe::Array{T}, e::Array{T}, peaks::Array{T}, peak_names::Array{Symbol}, windows::Array{Tuple{T, T}}, psd_cut::T,; uncertainty=true, bin_width_window::T=2.0) where T<:Real
    # create return and result dicts
    result = Dict{Symbol, NamedTuple}()
    report = Dict{Symbol, NamedTuple}()
    # iterate throuh all peaks
    for (peak, name, window) in zip(peaks, peak_names, windows)
        # fit peak
        result_peak, report_peak = get_peak_surrival_fraction(aoe, e, peak, collect(window), psd_cut; uncertainty=uncertainty, bin_width_window=bin_width_window)
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