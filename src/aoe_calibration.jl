# f_aoe_sigma(x, p) = p[1] .+ p[2]*exp.(-p[3]./x)
@. f_aoe_sigma(x, p) = sqrt(abs(p[1]) + abs(p[2])/x^2)
f_aoe_mu(x, p) = p[1] .+ p[2].*x
"""
fit_aoe_corrections(e::Array{<:Unitful.Energy{<:Real}}, μ::Array{<:Real}, σ::Array{<:Real})

Fit the corrections for the AoE value of the detector.
# Returns
- `e`: Energy values
- `μ`: Mean values
- `σ`: Sigma values
- `μ_scs`: Fit result for the mean values
- `f_μ_scs`: Fit function for the mean values
- `σ_scs`: Fit result for the sigma values
- `f_σ_scs`: Fit function for the sigma values
"""
function fit_aoe_corrections(e::Array{<:Unitful.Energy{<:Real}}, μ::Array{<:Real}, σ::Array{<:Real}; e_expression::Union{String,Symbol}="e")
    # fit compton band mus with linear function
    μ_cut = (mean(μ) - 2*std(μ) .< μ .< mean(μ) + 2*std(μ))
    e, μ, σ = e[μ_cut], μ[μ_cut], σ[μ_cut]
    e_unit = unit(first(e))
    e = ustrip.(e_unit, e)
    
    # fit compton band µ with linear function
    result_µ, report_µ = chi2fit(1, e, µ; uncertainty=true)
    func_µ = "$(mvalue(result_µ.par[1])) + ($e_expression) * $(mvalue(result_µ.par[2]))$e_unit^-1"
    func_generic_µ = "p[1] + ($e_expression) * p[2]"
    par_µ = [result_µ.par[i] ./ e_unit^(i-1) for i=1:length(result_µ.par)] # add units
    result_µ = merge(result_µ, (par = par_µ, func = func_µ, func_generic = func_generic_µ, µ = µ)) 
    @debug "Compton band µ correction: $(result_µ.func)"

    # fit compton band σ with sqrt function 
    σ_cut = (mean(σ) - std(σ) .< σ .< mean(σ) + std(σ))
    f_fit_σ = f_aoe_sigma # fit function 
    result_σ, report_σ = chi2fit((x, p1, p2) -> f_fit_σ(x,[p1,p2]), e[σ_cut], µ[σ_cut]; uncertainty=true)
    par_σ = [result_σ.par[1], result_σ.par[2] * e_unit^2] # add unit 
    func_σ = nothing
    func_generic_σ = nothing
    if string(f_fit_σ) == "f_aoe_sigma"
        func_σ = "sqrt( abs($(mvalue(result_σ.par[1]))) + abs($(mvalue(result_σ.par[2]))$(e_unit)^2) / ($e_expression)^2)" # add units!! 
        func_generic_σ = "sqrt(abs(p[1]) + abs(p[2]) / ($e_expression)^2)" 
   end
   result_σ = merge(result_σ, (par = par_σ, func = func_σ, func_generic = func_generic_σ, σ = σ)) 
    @debug "Compton band σ normalization: $(result_σ.func)"

    # put everything together into A/E correction/normalization function 
    aoe_str = "(a / (($e_expression)$e_unit^-1))" # get aoe, but without unit. 
    func_aoe_corr = "($aoe_str - ($(result_µ.func))) / ($(result_σ.func))"
    func_generic_aoe_corr = "(aoe - $(result_µ.func_generic)) / $(result_σ.func_generic)"
    func_aoe_corr_ecal = replace(func_aoe_corr,e_expression => "e_cal") # function that can be used for already calibrated energies 

    result = (µ_compton = result_µ, σ_compton = result_σ, compton_bands = (e = e,), func = func_aoe_corr, func_generic = func_generic_aoe_corr, func_ecal = func_aoe_corr_ecal)
    report = (report_µ = report_µ, report_σ = report_σ)

    return result, report 
end
export fit_aoe_corrections

""" 
    correctaoe!(aoe::Array{T}, e::Array{T}, aoe_corrections::NamedTuple) where T<:Real

Correct the AoE values in the `aoe` array using the corrections in `aoe_corrections`.
"""
function correct_aoe!(aoe::Array{<:Unitful.RealOrRealQuantity}, e::Array{<:Unitful.Energy{<:Real}}, aoe_corrections::NamedTuple)
    aoe .-= Base.Fix2(f_aoe_mu, mvalue.(aoe_corrections.μ_scs)).(e)
    # aoe .-= 1.0*unit(aoe[1])
    aoe ./= Base.Fix2(f_aoe_sigma, mvalue.(aoe_corrections.σ_scs)).(e)
    ustrip.(aoe)
end
export correct_aoe!

correct_aoe!(aoe, e, aoe_corrections::PropDict) = correct_aoe!(aoe, e, NamedTuple(aoe_corrections))

"""
    prepare_dep_peakhist(e::Array{T}, dep::T,; relative_cut::T=0.5, n_bins_cut::Int=500) where T<:Real

Prepare an array of uncalibrated DEP energies for parameter extraction and calibration.
# Returns
- `result`: Result of the initial fit
- `report`: Report of the initial fit
"""
function prepare_dep_peakhist(e::Array{T}, dep::Quantity{T},; relative_cut::T=0.5, n_bins_cut::Int=500, uncertainty::Bool=true) where T<:Real
    # get cut window around peak
    cuts = cut_single_peak(e, minimum(e), maximum(e); n_bins=n_bins_cut, relative_cut=relative_cut)
    # estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e[e .> cuts.low .&& e .< cuts.high])
    # create histogram
    dephist = fit(Histogram, e, minimum(e):bin_width:maximum(e))
    # get peakstats
    depstats = estimate_single_peak_stats(dephist)
    # initial fit for calibration and parameter extraction
    result, report = fit_single_peak_th228(dephist, depstats,; uncertainty=uncertainty, low_e_tail=false)
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
function get_n_after_psd_cut(psd_cut::Unitful.RealOrRealQuantity, aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peak::T, window::Vector{T}, bin_width::T, result_before::NamedTuple, peakstats::NamedTuple; uncertainty::Bool=true, fixed_position::Bool=true) where T<:Unitful.Energy{<:Real}
    # get energy after cut and create histogram
    peakhist = fit(Histogram, ustrip.(e[aoe .> psd_cut]), ustrip(peak-first(window):bin_width:peak+last(window)))
    # create pseudo_prior with known peak sigma in signal for more stable fit
    pseudo_prior = NamedTupleDist(σ = Normal(result_before.σ, 0.3), )
    # fit peak and return number of signal counts
    result, _ = fit_single_peak_th228(peakhist, peakstats,; uncertainty=uncertainty, fixed_position=fixed_position, low_e_tail=false, pseudo_prior=pseudo_prior)
    return result.n
end
export get_n_after_psd_cut


"""
    get_psd_cut(aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T},; dep::T=1592.53u"keV", window::Vector{T}=[12.0, 10.0]u"keV", dep_sf::Float64=0.9, cut_search_interval::Tuple{Quantity{<:Real}, Quantity{<:Real}}=(-25.0u"keV^-1", 0.0u"keV^-1"), rtol::Float64=0.001, bin_width_window::T=3.0u"keV", fixed_position::Bool=true, sigma_high_sided::Float64=NaN, uncertainty::Bool=true, maxiters::Int=200) where T<:Unitful.Energy{<:Real}

Get the PSD cut value for a given `dep` and `window` size while performing a peak fit with fixed position. The PSD cut value is determined by finding the cut value for which the number of counts after the cut is equal to `dep_sf` times the number of counts before the cut.
The algorhithm utilizes a root search algorithm to find the cut value with a relative tolerance of `rtol`.
# Returns
- `cut`: PSD cut value
- `n0`: Number of counts before the cut
- `nsf`: Number of counts after the cut
"""
function get_psd_cut(aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T},; dep::T=1592.53u"keV", window::Vector{<:T}=[12.0, 10.0]u"keV", dep_sf::Float64=0.9, cut_search_interval::Tuple{<:Unitful.RealOrRealQuantity, <:Unitful.RealOrRealQuantity}=(-25.0u"keV^-1", 0.0u"keV^-1"), rtol::Float64=0.001, bin_width_window::T=3.0u"keV", fixed_position::Bool=true, sigma_high_sided::Float64=NaN, uncertainty::Bool=true, maxiters::Int=200) where T<:Unitful.Energy{<:Real}
    # check if a high sided AoE cut should be applied before the PSD cut is generated
    if !isnan(sigma_high_sided)
        e   =   e[aoe .< sigma_high_sided]
        aoe = aoe[aoe .< sigma_high_sided]
    end
    # estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e[e .> dep - bin_width_window .&& e .< dep + bin_width_window])
    # create histogram
    dephist = fit(Histogram, ustrip.(e), ustrip(dep-first(window):bin_width:dep+last(window)))
    # get peakstats
    depstats = estimate_single_peak_stats(dephist)
    # cut window around peak
    aoe = aoe[dep-first(window) .< e .< dep+last(window)]
    e   =   e[dep-first(window) .< e .< dep+last(window)]
    # fit before cut
    result_before, _ = fit_single_peak_th228(dephist, depstats,; uncertainty=uncertainty, fixed_position=fixed_position, low_e_tail=false)
    # get n0 before cut
    nsf = result_before.n * dep_sf
    # get psd cut
    n_surrival_dep_f = cut -> get_n_after_psd_cut(cut, aoe, e, dep, window, bin_width, mvalue(result_before), depstats; uncertainty=false, fixed_position=fixed_position) - nsf
    psd_cut = find_zero(n_surrival_dep_f, cut_search_interval, Bisection(), rtol=rtol, maxiters=maxiters)
    # return n_surrival_dep_f.(0.25:0.001:0.5)
    # get nsf after cut
    nsf = get_n_after_psd_cut(psd_cut, aoe, e, dep, window, bin_width, mvalue(result_before), depstats; uncertainty=uncertainty, fixed_position=fixed_position)
    return (lowcut = measurement(psd_cut, psd_cut * rtol), highcut = sigma_high_sided, n0 = result_before.n, nsf = nsf, sf = nsf / result_before.n * 100*u"percent")
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
function get_peak_surrival_fraction(aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peak::T, window::Vector{T}, psd_cut::Unitful.RealOrRealQuantity,; uncertainty::Bool=true, low_e_tail::Bool=true, bin_width_window::T=2.0u"keV", sigma_high_sided::Unitful.RealOrRealQuantity=NaN) where T<:Unitful.Energy{<:Real}
    # estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e[e .> peak - bin_width_window .&& e .< peak + bin_width_window])
    # get energy before cut and create histogram
    peakhist = fit(Histogram, ustrip.(e), ustrip(peak-first(window):bin_width:peak+last(window)))
    # estimate peak stats
    peakstats = estimate_single_peak_stats(peakhist)
    # fit peak and return number of signal counts
    result_before, report_before = fit_single_peak_th228(peakhist, peakstats,; uncertainty=uncertainty, low_e_tail=low_e_tail)

    # get e after cut
    if !isnan(sigma_high_sided)
        e = e[psd_cut .< aoe .< sigma_high_sided]
    else
        e = e[psd_cut .< aoe]
    end
    
    # estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e[e .> peak - bin_width_window .&& e .< peak + bin_width_window])
    # get energy after cut and create histogram
    peakhist = fit(Histogram, ustrip(e), ustrip(peak-first(window):bin_width:peak+last(window)))
    # create pseudo_prior with known peak sigma in signal for more stable fit
    pseudo_prior = NamedTupleDist(μ = ConstValueDist(result_before.μ), σ = Normal(result_before.σ, 0.1))
    pseudo_prior = NamedTupleDist(σ = Normal(result_before.σ, 0.1), )
    # estimate peak stats
    peakstats = estimate_single_peak_stats(peakhist)
    # fit peak and return number of signal counts
    result_after, report_after = fit_single_peak_th228(peakhist, peakstats,; uncertainty=uncertainty, low_e_tail=low_e_tail)
    # result_after, report_after = fit_single_peak_th228(peakhist, peakstats,; uncertainty=uncertainty, low_e_tail=low_e_tail, pseudo_prior=pseudo_prior)
    # calculate surrival fraction
    sf = result_after.n / result_before.n * 100.0 * u"percent"
    result = (
        peak = peak, 
        n_before = result_before.n,
        n_after = result_after.n,
        sf = sf
    )
    report = (
        peak = result.peak, 
        n_before = result.n_before,
        n_after = result.n_after,
        sf = result.sf,
        before = report_before,
        after = report_after,
    )
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
function get_peaks_surrival_fractions(aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peaks::Vector{<:T}, peak_names::Vector{Symbol}, windows::Vector{<:Tuple{T, T}}, psd_cut::Unitful.RealOrRealQuantity,; uncertainty::Bool=true, bin_width_window::T=2.0u"keV", low_e_tail::Bool=true, sigma_high_sided::Unitful.RealOrRealQuantity=NaN) where T<:Unitful.Energy{<:Real}
    # create return and result dicts
    result = Dict{Symbol, NamedTuple}()
    report = Dict{Symbol, NamedTuple}()
    # iterate throuh all peaks
    for (peak, name, window) in zip(peaks, peak_names, windows)
        # fit peak
        result_peak, report_peak = get_peak_surrival_fraction(aoe, e, peak, collect(window), psd_cut; uncertainty=uncertainty, bin_width_window=bin_width_window, low_e_tail=low_e_tail, sigma_high_sided=sigma_high_sided)
        # save results
        result[name] = result_peak
        report[name] = report_peak
    end
    return result, report
end
export get_peaks_surrival_fractions
get_peaks_surrival_fractions(aoe, e, peaks, peak_names, left_window_sizes::Vector{<:Unitful.Energy{<:Real}}, right_window_sizes::Vector{<:Unitful.Energy{<:Real}}, psd_cut; kwargs...) = get_peaks_surrival_fractions(aoe, e, peaks, peak_names, [(l,r) for (l,r) in zip(left_window_sizes, right_window_sizes)], psd_cut; kwargs...)

"""
    get_continuum_surrival_fraction(aoe:::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, center::T, window::T, psd_cut::Unitful.RealOrRealQuantity,; sigma_high_sided::Unitful.RealOrRealQuantity=NaN) where T<:Unitful.Energy{<:Real}

Get the surrival fraction of a continuum after a PSD cut value `psd_cut` for a given `center` and `window` size.

# Returns
- `center`: Center of the continuum
- `window`: Window size
- `n_before`: Number of counts before the cut
- `n_after`: Number of counts after the cut
- `sf`: Surrival fraction
"""
function get_continuum_surrival_fraction(aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, center::T, window::T, psd_cut::Unitful.RealOrRealQuantity,; sigma_high_sided::Unitful.RealOrRealQuantity=NaN) where T<:Unitful.Energy{<:Real}
    # get number of events in window before cut
    n_before = length(e[center - window .< e .< center + window])
    # get number of events after cut
    n_after = length(e[aoe .> psd_cut .&& center - window .< e .< center + window])
    if !isnan(sigma_high_sided)
        n_after = length(e[psd_cut .< aoe .< sigma_high_sided .&& center - window .< e .< center + window])
    end
    n_after = length(e[aoe .> psd_cut .&& center - window .< e .< center + window])
    # calculate surrival fraction
    sf = n_after / n_before
    result = (
        window = measurement(center, window),
        n_before = measurement(n_before, sqrt(n_before)),
        n_after = measurement(n_after, sqrt(n_after)),
        sf = measurement(sf, sqrt(sf * (1-sf) / n_before)) * 100.0 * u"percent",
    )
    return result
end
export get_continuum_surrival_fraction