### check if the bin width should be fixed or calculated
### same holds for the histogram interval: I used something like [-20, 3] with bin width 0.05 most of the times and if these values get changed another fit function might be needed
### check if it works :-)

"""
## TO DO: Still needs to be updated once finalized

    ctc_aoe(aoe_all::Array{T}, ecal_all::Array{T}, qdrift_e_all::Array{T}, compton_bands::Array{T}, peak::T, window::T) where T<:Real

Correct for the drift time dependence of A/E parameter

# Returns (but may change):
    * `peak`: peak position
    * `window`: window size
    * `fct`: correction factor
    * `σ_before`: σ before correction
    * `σ_after`: σ after correction
    * `func`: function to correct aoe
    * `func_generic`: generic function to correct aoe
"""

function ctc_aoe(aoe_all::Vector{<:Real}, ecal_all::Vector{<:Unitful.RealOrRealQuantity}, qdrift_e_all::Vector{<:Real}, compton_bands::Vector{<:Unitful.RealOrRealQuantity}, peak::Real = 0.0, window::Tuple{<:Real, <:Real} = (50.0, 8.0), hist_start::Real = -20.0, hist_end::Real = 5.0, bin_width::Real = 0.05; aoe_expression::Union{Symbol, String}="aoe", qdrift_expression::Union{Symbol, String} = "qdrift / e", pseudo_prior::NamedTupleDist = NamedTupleDist(empty = true), pseudo_prior_all::NamedTupleDist = NamedTupleDist(empty = true), pol_order::Int=1) # deleted m_cal since no calibration

    # distribution compton band entries
    compton_start = ustrip.(compton_bands)
    compton_end = compton_start .+ 20
    # create mask (make sure everything is unitless)
    mask = falses(length(ustrip.(ecal_all)))
    for i in 1:length(compton_start)
        mask .|= compton_start[i] .<= ustrip.(ecal_all) .<= compton_end[i]
    end
    # Filter the energies, aoe and qdrift values based on the mask
    aoe = aoe_all[mask]
    qdrift_e = qdrift_e_all[mask]

    # create cut window around peak
    cut = peak - first(window) .< aoe .< peak + last(window)
    @debug "Cut window: $(peak - first(window)) < aoe < $(peak + last(window))"
    aoe_cut, qdrift_e_cut = aoe[cut], qdrift_e[cut]
    
    # calculate optimal bin width (if needed for other purposes)
    bin_width_window = 5.0 ### this parameter might be modified since it's copied from the energy case
    # bin_width        = get_friedman_diaconis_bin_width(aoe[peak - bin_width_window .< aoe .< peak + bin_width_window]) ### or use 0.05
    bin_width_qdrift = get_friedman_diaconis_bin_width(qdrift_e[peak - bin_width_window .< aoe .< peak + bin_width_window])


    # get σ before correction
    # fit peak (with the aoe_two_bck fit function which uses two EMGs)
    h_before = fit(Histogram, aoe_cut, hist_start:bin_width:hist_end) ### the current values: hist_start -20 and hist_end 3
    ps_before = estimate_single_peak_stats(h_before)
    result_before, report_before = fit_single_aoe_compton(h_before, ps_before, fit_func=:aoe_two_bck, pseudo_prior=pseudo_prior, uncertainty=true)
    @debug "Found Best σ before correction: $(round(result_before.σ, digits=2))"

    # create optimization function
    function f_optimize_ctc(fct, aoe, qdrift_e, bin_width)
        # calculate drift time corrected aoe
        aoe_ctc = aoe .+ PolCalFunc(0.0, fct...).(qdrift_e)
        # fit peak
        h = fit(Histogram, aoe_ctc, minimum(aoe_ctc):bin_width:maximum(aoe_ctc))
        ps = estimate_single_peak_stats(h)
        result_peak, report_peak = fit_single_aoe_compton(h, ps, fit_func=:aoe_two_bck, pseudo_prior=pseudo_prior, uncertainty=false) ### in ctc.jl uncertainty false is used (but I don't know why)
        # get σ and peak height
        μ = mvalue(result_peak.μ)
        σ = mvalue(result_peak.σ)
        p_height = maximum(report_peak.f_fit.(μ-0.2*σ:0.001:μ+0.2*σ)) # hardcoded 0.001 for now
        # use ratio of σ and peak height as optimization functional
        return log(σ/p_height)
    end

    # create function to minimize
    f_minimize = let f_optimize=f_optimize_ctc, aoe=aoe_cut, qdrift_e=qdrift_e_cut, bin_width=bin_width
        fct -> f_optimize(fct, aoe, qdrift_e, bin_width)
    end

    # minimize function
    qdrift_median = median(qdrift_e_cut)
    # upper bound
    fct_lb = [(1e-4 / qdrift_median)^(i) for i in 1:pol_order]
    @debug "Lower bound: $fct_lb"
    # lower bound
    fct_ub = [(50.0 / qdrift_median)^(i) for i in 1:pol_order]
    @debug "Upper bound: $fct_ub"
    # start value
    fct_start = [(1 / qdrift_median)^(i) for i in 1:pol_order]
    @debug "Start value: $fct_start"

    # optimization
    optf = OptimizationFunction((u, p) -> f_minimize(u), AutoForwardDiff())
    optpro = OptimizationProblem(optf, fct_start, (), lb=fct_lb, ub=fct_ub)
    res = solve(optpro, NLopt.LN_BOBYQA(), maxiters = 3000, maxtime=optim_time_limit)
    converged = (res.retcode == ReturnCode.Success)

    # get optimal correction factor
    fct = res.u
    @debug "Found Best FCT: $(fct .* 1e6)E-6"

    if !converged @warn "CTC did not converge" end

    # calculate drift time corrected aoe
    _aoe_ctc = aoe_cut .+ PolCalFunc(0.0, fct...).(qdrift_e_cut)
    
    # normalize once again to μ = 0 and σ = 1
    h_norm = fit(Histogram, _aoe_ctc, -20:bin_width:3) ### hard-coded values: should include some tolerance to higher values
    ps_norm = estimate_single_peak_stats(h_norm)
    result_norm, report_norm = fit_single_aoe_compton(h_norm, ps_norm, fit_func=:aoe_two_bck, pseudo_prior = pseudo_prior_all, uncertainty=true)
    μ_norm = mvalue(result_norm.μ)
    σ_norm = mvalue(result_norm.σ)
    aoe_ctc = (_aoe_ctc .- μ_norm) ./ σ_norm

    # get cal PropertyFunctions
    aoe_ctc_func = "( ( $(aoe_expression) ) + " * join(["$(fct[i]) * ( $(qdrift_expression) )^$(i)" for i in eachindex(fct)], " + ") * " - $(μ_norm) ) / $(σ_norm) "
    
    # create final histograms
    h_after = fit(Histogram, _aoe_ctc, hist_start:bin_width:hist_end) ### hard-coded values: should include some tolerance to higher values
    ps_after = estimate_single_peak_stats(h_after)
    result_after, report_after = fit_single_aoe_compton(h_after, ps_after, fit_func=:aoe_two_bck, pseudo_prior = pseudo_prior, uncertainty=true)


    ### maybe modify result and report (also depending on the recipe)
    result = (
        peak = peak,
        window = window,
        func = aoe_ctc_func,
        fct = fct,
        σ_before = result_before.σ,
        σ_after = result_after.σ,
        converged = converged
    )
    report = (
        peak = result.peak,
        window = result.window,
        fct = result.fct,
        bin_width = bin_width,
        bin_width_qdrift = bin_width_qdrift,
        aoe_peak = aoe_cut,
        aoe_ctc = _aoe_ctc, 
        qdrift_peak = qdrift_e_cut,
        h_before = h_before,
        h_after = h_after,
        σ_before = result.σ_before,
        σ_after = result.σ_after,
        report_before = report_before,
        report_after = report_after
    )
    return result, report
end
export ctc_aoe