
"""
    ctc_energy(
        e::Vector{<:Unitful.Energy{<:Real}}, qdrift::Vector{<:Real}, peak::Unitful.Energy{<:Real},
        window::Tuple{<:Unitful.Energy{<:Real}, <:Unitful.Energy{<:Real}},
        m_cal_simple::Unitful.Energy{<:Real} = 1.0u"keV";
        e_expression::Union{Symbol, String} = "e", pol_order::Int = 1
    )

Corrects the dependence of the energy on the drift time (charge-trapping
correction) in the peak at `peak`.

The events within `window = (lower, upper)` around `peak` are used. The corrected
energy is `e + Σ_i fct[i] * qdrift^i` for `i = 1:pol_order`. The coefficients
`fct` minimize the logarithm of the ratio of the FWHM to the height of the fitted
peak, found with a bounded derivative-free optimization (NLopt `LN_BOBYQA`) limited by the time
limit of [`set_timelimit`](@ref LegendSpecFits.set_timelimit). The peak is fitted with [`fit_single_peak_th228`](@ref).

## Arguments
* `e`: Calibrated energies.
* `qdrift`: Drift-time parameter of each event.
* `peak`: Peak energy.
* `window`: Widths `(lower, upper)` of the energy window around `peak`.
* `m_cal_simple`: Simple calibration constant (energy per uncalibrated unit). `fct` in the result is divided by it,
  so that the correction `func` applies to the uncalibrated energy.

## Keywords
* `e_expression`: Name of the energy variable in the returned `func`.
* `pol_order::Int = 1`: Order of the polynomial in `qdrift`.

## Returns
A tuple `(result, report)`. `result` is a `NamedTuple` with the fields
* `peak`, `window`, `m_cal_simple`: the input values.
* `func`: the correction as a Julia expression string, `e_expression + fct[1] * qdrift^1 + ...`.
* `fct`: the coefficients of the correction (divided by `m_cal_simple`).
* `fwhm_before`, `fwhm_after`: FWHM of the peak before and after the correction, in keV.
* `converged`: whether the optimization converged.

`report` contains `peak`, `window`, `fct`, `bin_width`, `bin_width_qdrift`, the
selected events (`e_peak`, `e_ctc`, `qdrift_peak`), the histograms `h_before`
and `h_after`, the FWHM values and the reports of both fits (`report_before`,
`report_after`), for plotting with `LegendMakie.lplot`.
"""
function ctc_energy(e::Vector{<:Unitful.Energy{<:Real}}, qdrift::Vector{<:Real}, peak::Unitful.Energy{<:Real}, window::Tuple{<:Unitful.Energy{<:Real}, <:Unitful.Energy{<:Real}}, m_cal_simple::Unitful.Energy{<:Real}=1.0u"keV"; e_expression::Union{Symbol, String}="e", pol_order::Int=1)
    # create cut window around peak
    cut = peak - first(window) .< e .< peak + last(window)
    @debug "Cut window: $(peak - first(window)) < e < $(peak + last(window))"
    e_cut, qdrift_cut = e[cut], qdrift[cut]
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
        result_peak, report_peak = fit_single_peak_th228(h, ps; uncertainty=false)
        # get fwhm and peak height
        fwhm = mvalue(result_peak.fwhm)
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