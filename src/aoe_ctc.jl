
"""
    ctc_aoe(aoe_all::Array{T}, ecal_all::Array{T}, qdrift_all::Array{T}, compton_bands::Array{T}, peak::T, window::T) where T<:Real

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

function ctc_aoe(aoe_all::Vector{<:Real}, ecal_all::Vector{<:Real}, qdrift_all::Vector{<:Real}, compton_bands::Vector{<:Real}, peak::Real, window::Tuple{<:Real, <:Real}, hist_start::Real, hist_end::Real, bin_width::Real; aoe_expression::Union{Symbol, String}="aoe", pol_order::Int=1) # deleted m_cal since no calibration
    ### apply mask to get entries of Compton bands only without peaks (this can also happen outside of this function)
    # distribution compton band entries
    compton_start = ustrip.(compton_bands)
    compton_end = compton_start .+ 20
    # create mask (make sure everything is unitless)
    mask = falses(length(ustrip.(ecal_all)))
    for i in 1:length(compton_start)
        mask .|= compton_start[i] .<= ustrip.(ecal_all) .<= compton_end[i]
    end
    # Filter the energies, aoe and qdrift values based on the mask
    ecal = ecal_all[mask]
    aoe = aoe_all[mask]
    qdrift = qdrift_all[mask]

    ### Do we need the following part since our fit function is fitting the whole interval not just the peak (as it is in the energy case)?
    # create cut window around peak
    ### question: what is 'peak' (it's an argument of the function but where does it come from? Or is it just the guessed peak position, i.e. approximately 0 in our case?)
    cut = peak - first(window) .< aoe .< peak + last(window)
    @debug "Cut window: $(peak - first(window)) < aoe < $(peak + last(window))"
    aoe_cut, qdrift_cut = aoe[cut], qdrift[cut]
    
    ### check if we should use a fix bin width (as it is atm with 0.05) or use the following one ###
    # calculate optimal bin width (if needed for other purposes)
    bin_width_window = 5.0 ### this parameter might be modified since it's copied from the energy case
    bin_width        = get_friedman_diaconis_bin_width(aoe[peak - bin_width_window .< aoe .< peak + bin_width_window]) ### or use 0.05
    bin_width_qdrift = get_friedman_diaconis_bin_width(qdrift[peak - bin_width_window .< aoe .< peak + bin_width_window])


    # get σ before correction
    # fit peak (with the aoe_two_bck fit function which uses two EMGs)
    h_before = fit(Histogram, aoe_cut, hist_start:bin_width:hist_end) ### the current values: hist_start -20 and hist_end 3
    ps_before = estimate_single_peak_stats(h_before)
    result_before, report_before = LegendSpecFits.fit_single_aoe_compton(h_before, ps_before, fit_func=:aoe_two_bck, uncertainty=true)
    @debug "Found Best σ before correction: $(round(result_before.σ, digits=2))"

    # create optimization function
    function f_optimize_ctc(fct, aoe, ecal, qdrift, bin_width)
        # calculate drift time corrected aoe
        aoe_ctc = aoe .+ PolCalFunc(0.0, (fct ./ ecal)...).(qdrift) ### added division by energy
        # fit peak
        h = fit(Histogram, aoe_ctc, minimum(aoe_ctc):bin_width:maximum(aoe_ctc)) ### this is also just a part of the histogram so it might not work with the aoe_two_bck fit function (or use hist_start and hist_end instead)
        ps = estimate_single_peak_stats(h)
        result_peak, report_peak = LegendSpecFits.fit_single_aoe_compton(h, ps, fit_func=:aoe_two_bck, uncertainty=false) ### in ctc.jl uncertainty false is used (but I don't know why)
        # get σ and peak height
        σ = result_peak.σ
        p_height = maximum(report_peak.f_fit.(result_peak.μ-0.2*result_peak.σ:0.01:result_peak.μ+0.2*result_peak.σ)) ### f_fit should be part of the report, right?
        # use ratio of σ and peak height as optimization functional
        return log(σ/p_height)
    end

    # create function to minimize
    f_minimize = let f_optimize=f_optimize_ctc, aoe=aoe_cut, qdrift=qdrift_cut, bin_width=bin_width
        fct -> f_optimize(fct, aoe, ecal, qdrift, bin_width)
    end

    ### I haven't changed anything in the following optimization part up to the warning (should be independent of A/E or energy, I guess)
    # minimize function
    qdrift_median = median(qdrift_cut)
    # upper bound
    fct_lb = [(1e-4 / qdrift_median)^(i) for i in 1:pol_order]
    @debug "Lower bound: $fct_lb"
    # lower bound
    fct_ub = [(5.0 / qdrift_median)^(i) for i in 1:pol_order]
    @debug "Upper bound: $fct_ub"
    # start value
    fct_start = [(0.1 / qdrift_median)^(i) for i in 1:pol_order]
    @debug "Start value: $fct_start"

    # optimization
    optf = OptimizationFunction((u, p) -> f_minimize(u), AutoForwardDiff())
    optpro = OptimizationProblem(optf, fct_start, [], lb=fct_lb, ub=fct_ub)
    res = solve(optpro, NLopt.LN_BOBYQA(), maxiters = 3000, maxtime=optim_time_limit)
    converged = (res.retcode == ReturnCode.Success)

    # get optimal correction factor
    fct = res.u
    @debug "Found Best FCT: $(fct .* 1e6)E-6"

    if !converged @warn "CTC did not converge" end

    # calculate drift time corrected energy
    aoe_ctc = aoe_cut .+ PolCalFunc(0.0, (fct ./ ecal)...).(qdrift_cut)
    
    # get σ after correction
    # fit peak
    h_after = fit(Histogram, aoe_ctc, minimum(aoe_ctc):bin_width:maximum(aoe_ctc)) ### this is also just a part of the histogram so it might not work with the aoe_two_bck fit function (or use hist_start and hist_end instead)
    ps_after = estimate_single_peak_stats(h_after)
    result_after, report_after = LegendSpecFits.fit_single_aoe_compton(h_after, ps_after, fit_func=:aoe_two_bck, uncertainty=true)

    # get cal PropertyFunctions (added the division by energy)
    aoe_ctc_func = "$aoe_expression + " * join(["$(fct[i]) / ecal * qdrift^$(i)" for i in eachindex(fct)], " + ")

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
        aoe_ctc = aoe_ctc, 
        qdrift_peak = qdrift_cut,
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

### IMPORTANT:
### divide fct by energy for aoe correction (should be complete now)
### check if the 'peak only' part is needed -> if so, another fit function has to be used
### check if the bin width should be fixed or calculated
### same holds for the histogram interval: I used something like [-20, 3] with bin width 0.05 most of the times and if these values get changed another fit function might be needed!
### check if it works :-)
### also: the peak might be shifted after ctc so we might calibrate the classifier again to get a µ of 0