
"""
    f_optimize_ctc(fct, e, qdrift, bin_width)

Calculate the ratio of the FWHM and the peak height of the peak around `peak` in
`e` with a cut window of `window`. The drift time dependence is given by
`e_ctc` = `e` + `fct` * `qdrift`.

# Returns 
    * `fwhm / p_height`: FWHM of the peak divided by peak height
"""
function f_optimize_ctc(fct, e, qdrift, bin_width)
    # calculate drift time corrected energy
    e_ctc = e .+ fct .* qdrift
    # fit peak
    h = fit(Histogram, e_ctc, minimum(e_ctc):bin_width:maximum(e_ctc))
    ps = estimate_single_peak_stats(h)
    result_peak, report_peak = fit_single_peak_th228(h, ps; uncertainty=false)
    # get fwhm and peak height
    fwhm = result_peak.fwhm
    p_height = maximum(report_peak.f_fit.(result_peak.μ-0.2*result_peak.σ:0.01:result_peak.μ+0.2*result_peak.σ))
    # use ratio of fwhm and peak height as optimization functional
    return log(fwhm/p_height)
end

"""
    ctc_energy(e::Array{T}, qdrift::Array{T}, peak::T, window::T) where T<:Real

Correct for the drift time dependence of the energy by minimizing the ratio of
the FWHM and the peak height of the peak around `peak` in `e` with a cut window
of `window`. The drift time dependence is given by `qdrift`.

# Returns 
    * `peak`: peak position
    * `window`: window size
    * `fct`: correction factor
    * `bin_width`: optimal bin width
"""
function ctc_energy(e::Array{T}, qdrift::Array{T}, peak::T, window::T) where T<:Real
    # create cut window around peak
    cut = peak - window .< e .< peak + window
    e_cut, qdrift_cut = e[cut], qdrift[cut]
    # calculate optimal bin width
    bin_width        = 2 * (quantile(e_cut, 0.75) - quantile(e_cut, 0.25)) / ∛(length(e_cut))
    bin_width_qdrift = 2 * (quantile(qdrift_cut, 0.75) - quantile(qdrift_cut, 0.25)) / ∛(length(qdrift_cut))
    # create function to minimize
    f_minimize = let f_optimize=f_optimize_ctc, e=e_cut, qdrift=qdrift_cut, bin_width=bin_width
        fct -> f_optimize(fct, e, qdrift, bin_width)
    end

    # minimize function
    fct_range, fct_start = [0.0, 1e-3], [1e-7]
    opt_r = optimize(f_minimize, fct_range[1], fct_range[2], fct_start, Fminbox(GradientDescent()), Optim.Options(iterations=1000, show_trace=false, time_limit=600))
    # get optimal correction factor
    fct = Optim.minimizer(opt_r)[1]
    # calculate drift time corrected energy
    e_ctc = e_cut .+ fct .* qdrift_cut
    result = (
        peak = peak,
        window = window,
        fct = fct,
        bin_width = bin_width,
        bin_width_qdrift = bin_width_qdrift,
    )
    report = (
        peak = result.peak,
        window = result.window,
        fct = result.fct,
        bin_width = result.bin_width,
        bin_width_qdrift = result.bin_width_qdrift,
        e_peak = e_cut,
        e_ctc = e_ctc, 
        qdrift_peak = qdrift_cut,
        h_before = fit(Histogram, e_cut, minimum(e_cut):bin_width:maximum(e_cut)),
        h_after = fit(Histogram, e_ctc, minimum(e_ctc):bin_width:maximum(e_ctc))
    )
    return result, report
end
export ctc_energy