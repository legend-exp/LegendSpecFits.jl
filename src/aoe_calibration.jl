# f_aoe_sigma(x, p) = p[1] .+ p[2]*exp.(-p[3]./x)
# @. f_aoe_sigma(x, p) = sqrt(abs(p[1]) + abs(p[2])/x^2)
@. f_aoe_sigma(x, p) = sqrt(p[1]^2 + p[2]^2/x^2)
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
    μ_cut = (mean(μ) - 2*std(μ) .< μ .< mean(μ) + 2*std(μ)) .&& muncert.(μ) .> 0.0
    e, μ, σ = e[μ_cut], μ[μ_cut], σ[μ_cut]
    e_unit = unit(first(e))
    e = ustrip.(e_unit, e)
    
    # fit compton band µ with linear function
    result_µ, report_µ = chi2fit(1, e, µ; uncertainty=true)
    func_µ = "$(mvalue(result_µ.par[1])) + ($e_expression) * $(mvalue(result_µ.par[2]))$e_unit^-1"
    func_generic_µ = "p1 + ($e_expression) * p2" #  "p[1] + ($e_expression) * p[2]"
    par_µ = [result_µ.par[i] ./ e_unit^(i-1) for i=1:length(result_µ.par)] # add units
    result_µ = merge(result_µ, (par = par_µ, func = func_µ, func_generic = func_generic_µ, µ = µ)) 
    report_µ = merge(report_µ, (e_unit = e_unit, label_y = "µ", label_fit = "Best Fit: $(mvalue(round(result_µ.par[1], digits=2))) + E * $(mvalue(round(ustrip(result_µ.par[2]) * 1e6, digits=2)))1e-6"))
    @debug "Compton band µ correction: $(result_µ.func)"

    # fit compton band σ with sqrt function 
    σ_cut = (mean(σ) - std(σ) .< σ .< mean(σ) + std(σ)) .&& muncert.(σ) .> 0.0
    f_fit_σ = f_aoe_sigma # fit function 
    result_σ, report_σ = chi2fit((x, p1, p2) -> f_fit_σ(x,[p1,p2]), e[σ_cut], σ[σ_cut]; uncertainty=true)
    par_σ = [result_σ.par[1], result_σ.par[2] * e_unit^2]
    func_σ = nothing
    func_generic_σ = nothing
    if string(f_fit_σ) == "f_aoe_sigma"
        # func_σ = "sqrt( abs($(mvalue(result_σ.par[1]))) + abs($(mvalue(result_σ.par[2]))$(e_unit)^2) / ($e_expression)^2)"
        func_σ = "sqrt( ($(mvalue(result_σ.par[1])))^2 + ($(mvalue(result_σ.par[2]))$(e_unit))^2 / ($e_expression)^2 )" 
        # func_generic_σ = "sqrt(abs(p[1]) + abs(p[2]) / ($e_expression)^2)"
        func_generic_σ = "sqrt( (p[1])^2 + (p[2])^2 / ($e_expression)^2 )"
    end
    result_σ = merge(result_σ, (par = par_σ, func = func_σ, func_generic = func_generic_σ, σ = σ))
    report_σ = merge(report_σ, (e_unit = e_unit, label_y = "σ", label_fit = "Best fit: sqrt($(round(mvalue(result_σ.par[1])*1e6, digits=1))e-6 + $(round(ustrip(mvalue(result_σ.par[2])), digits=2)) / E^2)"))
    @debug "Compton band σ normalization: $(result_σ.func)"

    # put everything together into A/E correction/normalization function 
    aoe_str = "(a / ( ($e_expression)$e_unit^-1) )" # get aoe, but without unit. 
    func_aoe_corr = "($aoe_str - ($(result_µ.func)) ) / ($(result_σ.func))"
    func_generic_aoe_corr = "(aoe - $(result_µ.func_generic)) / $(result_σ.func_generic)"
    func_aoe_corr_ecal = replace(func_aoe_corr, string(e_expression) => "e_cal") # function that can be used for already calibrated energies 

    result = (µ_compton = result_µ, σ_compton = result_σ, compton_bands = (e = e,), func = func_aoe_corr, func_generic = func_generic_aoe_corr, func_ecal = func_aoe_corr_ecal)
    report = (report_µ = report_µ, report_σ = report_σ)

    return result, report
end
export fit_aoe_corrections

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
    get_n_after_aoe_cut(aoe_cut::T, aoe::Array{T}, e::Array{T}, peak::T, window::Array{T}, bin_width::T, result_before::NamedTuple, peakstats::NamedTuple; uncertainty=true) where T<:Real

Get the number of counts after a AoE cut value `aoe_cut` for a given `peak` and `window` size while performing a peak fit with fixed position. The number of counts is determined by fitting the peak with a pseudo prior for the peak position.
# Returns
- `n`: Number of counts after the cut
"""
function get_n_after_aoe_cut(aoe_cut::Unitful.RealOrRealQuantity, aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peak::T, window::Vector{T}, bin_width::T, result_before::NamedTuple, peakstats::NamedTuple; uncertainty::Bool=true, fixed_position::Bool=true) where T<:Unitful.Energy{<:Real}
    # get energy after cut and create histogram
    peakhist = fit(Histogram, ustrip.(e[aoe .> aoe_cut]), ustrip(peak-first(window):bin_width:peak+last(window)))
    # create pseudo_prior with known peak sigma in signal for more stable fit
    pseudo_prior = if fixed_position
        NamedTupleDist(σ = Normal(result_before.σ, 0.1), μ = ConstValueDist(result_before.μ))
    else
        NamedTupleDist(σ = Normal(result_before.σ, 0.1), )
    end
    # fit peak and return number of signal counts
    result, _ = fit_single_peak_th228(peakhist, peakstats,; uncertainty=uncertainty, low_e_tail=false, pseudo_prior=pseudo_prior)
    return result.n
end
export get_n_after_aoe_cut


"""
    get_sf_after_aoe_cut(aoe_cut::T, aoe::Array{T}, e::Array{T}, peak::T, window::Array{T}, bin_width::T, result_before::NamedTuple; uncertainty=true) where T<:Real

Get the survival fraction after a AoE cut value `aoe_cut` for a given `peak` and `window` size from a combined fit to the survived and cut histograms.

# Returns
- `sf`: Survival fraction after the cut
"""
function get_sf_after_aoe_cut(aoe_cut::Unitful.RealOrRealQuantity, aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peak::T, window::Vector{T}, bin_width::T, result_before::NamedTuple; uncertainty::Bool=true) where T<:Unitful.Energy{<:Real}
    # get energy after cut and create histogram
    survived = fit(Histogram, ustrip.(e[aoe .>= aoe_cut]), ustrip(peak-first(window):bin_width:peak+last(window)))
    cut = fit(Histogram, ustrip.(e[aoe .< aoe_cut]), ustrip(peak-first(window):bin_width:peak+last(window)))
    # fit peak and return number of signal counts
    result, _ = fit_subpeaks_th228(survived, cut, result_before; uncertainty=uncertainty, low_e_tail=false)
    return result.sf
end
export get_sf_after_aoe_cut



"""
    get_aoe_cut(aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T},; dep::T=1592.53u"keV", window::Vector{T}=[12.0, 10.0]u"keV", dep_sf::Float64=0.9, cut_search_interval::Tuple{Quantity{<:Real}, Quantity{<:Real}}=(-25.0u"keV^-1", 0.0u"keV^-1"), rtol::Float64=0.001, bin_width_window::T=3.0u"keV", fixed_position::Bool=true, sigma_high_sided::Float64=NaN, uncertainty::Bool=true, maxiters::Int=200) where T<:Unitful.Energy{<:Real}

Get the AoE cut value for a given `dep` and `window` size while performing a peak fit with fixed position. The AoE cut value is determined by finding the cut value for which the number of counts after the cut is equal to `dep_sf` times the number of counts before the cut.
The algorhithm utilizes a root search algorithm to find the cut value with a relative tolerance of `rtol`.
# Returns
- `cut`: AoE cut value
- `n0`: Number of counts before the cut
- `nsf`: Number of counts after the cut
"""
function get_aoe_cut(aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T},; dep::T=1592.53u"keV", window::Vector{<:T}=[12.0, 10.0]u"keV", dep_sf::Float64=0.9, cut_search_interval::Tuple{<:Unitful.RealOrRealQuantity, <:Unitful.RealOrRealQuantity}=(-25.0*unit(first(aoe)), 1.0*unit(first(aoe))), rtol::Float64=0.001, bin_width_window::T=3.0u"keV", fixed_position::Bool=true, sigma_high_sided::Float64=NaN, uncertainty::Bool=true, maxiters::Int=200) where T<:Unitful.Energy{<:Real}
    # check if a high sided AoE cut should be applied before the AoE cut is generated
    if !isnan(sigma_high_sided)
        e   =   e[aoe .< sigma_high_sided]
        aoe = aoe[aoe .< sigma_high_sided]
    end
    # cut window around peak
    aoe = aoe[dep-first(window) .< e .< dep+last(window)]
    e   =   e[dep-first(window) .< e .< dep+last(window)]
    # estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e[dep - bin_width_window .< e .< dep + bin_width_window])
    # create histogram
    dephist = fit(Histogram, ustrip.(e), ustrip(dep-first(window):bin_width:dep+last(window)))
    # get peakstats
    depstats = estimate_single_peak_stats(dephist)
    # fit before cut
    result_before, _ = fit_single_peak_th228(dephist, depstats; uncertainty=uncertainty, fixed_position=fixed_position, low_e_tail=false)
    # get aoe cut
    sf_dep_f = cut -> get_sf_after_aoe_cut(cut, aoe, e, dep, window, bin_width, mvalue(result_before); uncertainty=false) - dep_sf
    aoe_cut = find_zero(sf_dep_f, cut_search_interval, Bisection(), rtol=rtol, maxiters=maxiters)
    # get sf after cut
    sf = get_sf_after_aoe_cut(aoe_cut, aoe, e, dep, window, bin_width, mvalue(result_before); uncertainty=uncertainty)
    return (lowcut = measurement(aoe_cut, aoe_cut * rtol), highcut = sigma_high_sided, n0 = result_before.n, nsf = result_before.n * sf, sf = sf * 100*u"percent")
end
export get_aoe_cut



"""
    get_peak_surrival_fraction(aoe::Array{T}, e::Array{T}, peak::T, window::Array{T}, aoe_cut::T,; uncertainty::Bool=true, low_e_tail::Bool=true) where T<:Real

Get the surrival fraction of a peak after a AoE cut value `aoe_cut` for a given `peak` and `window` size whiile performing a peak fit with fixed position.
    
# Returns
- `peak`: Peak position
- `n_before`: Number of counts before the cut
- `n_after`: Number of counts after the cut
- `sf`: Surrival fraction
- `err`: Uncertainties
"""
function get_peak_surrival_fraction(aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peak::T, window::Vector{T}, aoe_cut::Unitful.RealOrRealQuantity,; uncertainty::Bool=true, low_e_tail::Bool=true, bin_width_window::T=2.0u"keV", sigma_high_sided::Unitful.RealOrRealQuantity=NaN) where T<:Unitful.Energy{<:Real}
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
        # TODO: decide how to deal with the high sided cut!
        e = e[aoe .< sigma_high_sided]
        aoe = aoe[aoe .< sigma_high_sided]
    end
    e_survived = e[aoe_cut .<= aoe]
    e_cut = e[aoe_cut .> aoe]
    
    # estimate bin width
    bin_width = get_friedman_diaconis_bin_width(e[e .> peak - bin_width_window .&& e .< peak + bin_width_window])
    # get energy after cut and create histogram
    survived = fit(Histogram, ustrip(e_survived), ustrip(peak-first(window):bin_width:peak+last(window)))
    cut      = fit(Histogram, ustrip(e_cut),      ustrip(peak-first(window):bin_width:peak+last(window)))
    # fit peak and return number of signal counts
    result_after, report_after = fit_subpeaks_th228(survived, cut, result_before; uncertainty=uncertainty, low_e_tail=low_e_tail)
    # calculate surrival fraction
    sf = result_after.sf * 100u"percent"
    result = (
        peak = peak, 
        n_before = result_before.n,
        n_after = result_before.n * result_after.sf,
        sf = sf
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
export get_peak_surrival_fraction


"""
    get_peaks_surrival_fractions(aoe::Array{T}, e::Array{T}, peaks::Array{T}, peak_names::Array{Symbol}, windows::Array{T}, aoe_cut::T,; uncertainty=true) where T<:Real

Get the surrival fraction of a peak after a AoE cut value `aoe_cut` for a given `peak` and `window` size while performing a peak fit with fixed position.
# Return 
- `result`: Dict of results for each peak
- `report`: Dict of reports for each peak
"""
function get_peaks_surrival_fractions(aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, peaks::Vector{<:T}, peak_names::Vector{Symbol}, windows::Vector{<:Tuple{T, T}}, aoe_cut::Unitful.RealOrRealQuantity,; uncertainty::Bool=true, bin_width_window::T=2.0u"keV", low_e_tail::Bool=true, sigma_high_sided::Unitful.RealOrRealQuantity=NaN) where T<:Unitful.Energy{<:Real}
    # create return and result dicts
    result = Dict{Symbol, NamedTuple}()
    report = Dict{Symbol, NamedTuple}()
    # iterate throuh all peaks
    for (peak, name, window) in zip(peaks, peak_names, windows)
        # fit peak
        result_peak, report_peak = get_peak_surrival_fraction(aoe, e, peak, collect(window), aoe_cut; uncertainty=uncertainty, bin_width_window=bin_width_window, low_e_tail=low_e_tail, sigma_high_sided=sigma_high_sided)
        # save results
        result[name] = result_peak
        report[name] = report_peak
    end
    return result, report
end
export get_peaks_surrival_fractions
get_peaks_surrival_fractions(aoe, e, peaks, peak_names, left_window_sizes::Vector{<:Unitful.Energy{<:Real}}, right_window_sizes::Vector{<:Unitful.Energy{<:Real}}, aoe_cut; kwargs...) = get_peaks_surrival_fractions(aoe, e, peaks, peak_names, [(l,r) for (l,r) in zip(left_window_sizes, right_window_sizes)], aoe_cut; kwargs...)

"""
    get_continuum_surrival_fraction(aoe:::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, center::T, window::T, aoe_cut::Unitful.RealOrRealQuantity,; sigma_high_sided::Unitful.RealOrRealQuantity=NaN) where T<:Unitful.Energy{<:Real}

Get the surrival fraction of a continuum after a AoE cut value `aoe_cut` for a given `center` and `window` size.

# Returns
- `center`: Center of the continuum
- `window`: Window size
- `n_before`: Number of counts before the cut
- `n_after`: Number of counts after the cut
- `sf`: Surrival fraction
"""
function get_continuum_surrival_fraction(aoe::Vector{<:Unitful.RealOrRealQuantity}, e::Vector{<:T}, center::T, window::T, aoe_cut::Unitful.RealOrRealQuantity,; sigma_high_sided::Unitful.RealOrRealQuantity=NaN) where T<:Unitful.Energy{<:Real}
    # get number of events in window before cut
    n_before = length(e[center - window .< e .< center + window])
    # get number of events after cut
    n_after = length(e[aoe .> aoe_cut .&& center - window .< e .< center + window])
    if !isnan(sigma_high_sided)
        n_after = length(e[aoe_cut .< aoe .< sigma_high_sided .&& center - window .< e .< center + window])
    end
    n_after = length(e[aoe .> aoe_cut .&& center - window .< e .< center + window])
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