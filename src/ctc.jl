
"""
    ctc_energy(e::Array{T}, qdrift::Array{T}, peak::T, window::T) where T<:Real

Correct for the drift time dependence of the energy by minimizing the ratio of
the FWHM and the peak height of the peak around `peak` in `e` with a cut window
of `window`. The drift time dependence is given by `qdrift`.

# Returns 
    * `peak`: peak position
    * `window`: window size
    * `fct`: correction factor
    * `fwhm_before`: FWHM before correction
    * `fwhm_after`: FWHM after correction
    * `func`: function to correct energy
    * `func_generic`: generic function to correct energy
"""
function ctc_energy(e::Vector{<:Unitful.Energy{<:Real}}, qdrift::Vector{<:Real}, peak::Unitful.Energy{<:Real}, window::Tuple{<:Unitful.Energy{<:Real}, <:Unitful.Energy{<:Real}}, m_cal_simple::Unitful.Energy{<:Real}=1.0u"keV"; e_expression::Union{Symbol, String}="e", pol_order::Int=1)
    # create cut window around peak
    cut = peak - first(window) .< e .< peak + last(window)
    @debug "Cut window: $(peak - first(window)) < e < $(peak + last(window))"
    e_cut, qdrift_cut = e[cut], qdrift[cut]
    e_unit = u"keV"
    # calculate optimal bin width
    bin_width_window = 5.0u"keV"
    bin_width        = get_friedman_diaconis_bin_width(e[peak - bin_width_window .< e .< peak + bin_width_window])
    bin_width_qdrift = get_friedman_diaconis_bin_width(qdrift[peak - bin_width_window .< e .< peak + bin_width_window])

    # get FWHM before correction
    # fit peak
    h_before = fit(Histogram, ustrip.(e_unit, e_cut), ustrip(e_unit, minimum(e_cut)):ustrip(e_unit, bin_width):ustrip(e_unit, maximum(e_cut)))
    ps_before = estimate_single_peak_stats(h_before)
    result_before, report_before = fit_single_peak_th228(h_before, ps_before; uncertainty=true)
    @debug "Found Best FWHM before correction: $(round(result_before.fwhm, digits=2))"
    
    # create optimization function
    function f_optimize_ctc(fct, e, qdrift, bin_width)
        # calculate drift time corrected energy
        e_ctc = e .+ PolCalFunc(0.0, fct...).(qdrift)
        # fit peak
        h = fit(Histogram, e_ctc, minimum(e_ctc):bin_width:maximum(e_ctc))
        ps = estimate_single_peak_stats(h)
        @info ps.peak_fwhm
        result_peak, report_peak = fit_single_peak_th228(h, ps; uncertainty=false)
        # get fwhm and peak height
        fwhm = mvalue(result_peak.fwhm)
        @info "FWHM: $fwhm"
        p_height = maximum(report_peak.f_fit.(mvalue(result_peak.μ-0.2*result_peak.σ):0.01:mvalue(result_peak.μ+0.2*result_peak.σ)))
        # use ratio of fwhm and peak height as optimization functional
        return log(fwhm/p_height)
    end

    # create function to minimize
    f_minimize = let f_optimize=f_optimize_ctc, e=ustrip.(e_unit, e_cut), qdrift=qdrift_cut, bin_width=ustrip(e_unit, bin_width)
        fct -> f_optimize(fct, e, qdrift, bin_width)
    end

    # minimize function
    qdrift_median = median(qdrift_cut)
    # upper bound
    fct_lb = [(ustrip(e_unit, 1e-4u"keV") / qdrift_median)^(i) for i in 1:pol_order] .* e_unit
    @debug "Lower bound: $fct_lb"
    # lower bound
    fct_ub = [(ustrip(e_unit, 7u"keV") / qdrift_median)^(i) for i in 1:pol_order] .* e_unit
    @debug "Upper bound: $fct_ub"
    # start value
    fct_start = [(ustrip(e_unit, i*1.0u"keV") / qdrift_median)^(i) for i in 1:pol_order] .* e_unit
    @debug "Start value: $fct_start"

    # optimization
    optf = OptimizationFunction((u, p) -> f_minimize(u), AutoForwardDiff())
    optpro = OptimizationProblem(optf, ustrip.(e_unit, fct_start), (), lb=ustrip.(e_unit, fct_lb), ub=ustrip.(e_unit, fct_ub))
    res = solve(optpro, NLopt.LN_BOBYQA(), maxiters = 3000, maxtime=optim_time_limit)
    converged = (res.retcode == ReturnCode.Success)

    # get optimal correction factor
    fct = res.u .* e_unit
    @debug "Found Best FCT: $(fct .* 1e6)E-6"

    if !converged @warn "CTC did not converge" end

    # calculate drift time corrected energy
    e_ctc = e_cut .+ PolCalFunc(0.0, ustrip.(e_unit, fct)...).(qdrift_cut) .* e_unit
    
    # get FWHM after correction
    # fit peak
    h_after = fit(Histogram, ustrip.(e_unit, e_ctc), ustrip(e_unit, minimum(e_ctc)):ustrip(e_unit, bin_width):ustrip(e_unit, maximum(e_ctc)))
    ps_after = estimate_single_peak_stats(h_after)
    result_after, report_after = fit_single_peak_th228(h_after, ps_after; uncertainty=true)

    # get ADC fct factor and cal PropertyFunctions
    fct = fct ./ m_cal_simple
    e_ctc_func = "$e_expression + " * join(["$(mvalue(fct[i])) * qdrift^$(i)" for i in eachindex(fct)], " + ")

    result = (
        peak = peak,
        window = window,
        m_cal_simple = m_cal_simple,
        func = e_ctc_func,
        fct = fct,
        fwhm_before = result_before.fwhm*e_unit,
        fwhm_after = result_after.fwhm*e_unit,
        converged = converged
    )
    report = (
        peak = result.peak,
        window = result.window,
        fct = result.fct,
        bin_width = bin_width,
        bin_width_qdrift = bin_width_qdrift,
        e_peak = e_cut,
        e_ctc = e_ctc, 
        qdrift_peak = qdrift_cut,
        h_before = h_before,
        h_after = h_after,
        fwhm_before = result.fwhm_before,
        fwhm_after = result.fwhm_after,
        report_before = report_before,
        report_after = report_after
    )
    return result, report
end
export ctc_energy