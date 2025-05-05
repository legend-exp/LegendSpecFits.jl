"""
    ctc_lq(lq_all::Array{T}, ecal_all::Array{T}, qdrift_e_all::Array{T}, compton_bands::Array{T}, peak::T, window::T) where T<:Real

Correct for the drift time dependence of the LQ parameter

# Returns (but may change):
    * `peak`: peak position
    * `window`: window size
    * `fct`: correction factor
    * `σ_before`: σ before correction
    * `σ_after`: σ after correction
    * `func`: function to correct lq
    * `func_generic`: generic function to correct lq
"""


function ctc_lq(lq::Vector{<:Real}, e::Vector{<:Unitful.RealOrRealQuantity}, qdrift::Vector{<:Real}, dep_µ::Unitful.AbstractQuantity, dep_σ::Unitful.AbstractQuantity, hist_start::Real = -0.5, hist_end::Real = 2.5, bin_width::Real = 0.01; 
    ctc_dep_edgesigma::Float64=3.0, lq_expression::Union{Symbol, String}="lq", qdrift_expression::Union{Symbol, String} = "qdrift / e", pol_order::Int=1) # deleted m_cal since no calibration

    # calculate DEP edges
    dep_left = dep_µ - ctc_dep_edgesigma * dep_σ
    dep_right = dep_µ + ctc_dep_edgesigma * dep_σ


    # cut lq values from dep
    cut = dep_left .< e .< dep_right .&& isfinite.(lq) .&& isfinite.(qdrift)
    @debug "Cut window: $(dep_left) < e < $(dep_right)"
    lq_cut, qdrift_cut = lq[cut], qdrift[cut]

    h_before = fit(Histogram, lq_cut, hist_start:bin_width:hist_end)

    # get σ before correction
    # fit peak
    cut_peak = cut_single_peak(lq_cut, -10, 20; n_bins=-1, relative_cut = 0.4)
    println("cut_peak: ", cut_peak)
    result_before, report_before = fit_single_trunc_gauss(lq_cut, cut_peak; uncertainty=false)
    @debug "Found Best σ before correction: $(round(result_before.σ, digits=2))"

    # create optimization function
    function f_optimize_ctc(fct, lq_cut, qdrift_cut)
        # calculate drift time corrected lq
        lq_ctc =  lq_cut .+ PolCalFunc(0.0, fct...).(qdrift_cut)
        # fit peak
        cuts_lq = cut_single_peak(lq_ctc, -10, 20; n_bins=-1, relative_cut = 0.4)
        result_after, _ = fit_single_trunc_gauss(lq_ctc, cuts_lq; uncertainty=false)
        return mvalue(result_after.σ)
    end

    # create function to minimize
    f_minimize = let f_optimize=f_optimize_ctc, lq_cut=lq_cut, qdrift_cut=qdrift_cut
        fct -> f_optimize(fct, lq_cut, qdrift_cut)
    end

    # lower bound
    fct_lb = [-0.1 for i in 1:pol_order]
    @debug "Lower bound: $fct_lb"
    # upper bound
    fct_ub = [0.1 for i in 1:pol_order]
    @debug "Upper bound: $fct_ub"
    # start value
    fct_start = [0.0 for i in 1:pol_order]
    @debug "Start value: $fct_start"


    println("fct_lb: ", fct_lb)
    println("fct_ub: ", fct_ub)
    println("fct_start: ", fct_start)

    # optimization
    optf = OptimizationFunction((u, p) -> f_minimize(u), AutoForwardDiff())
    optpro = OptimizationProblem(optf, fct_start, (), lb=fct_lb, ub=fct_ub)
    res = solve(optpro, NLopt.LN_BOBYQA(), maxiters = 3000, maxtime=optim_time_limit)
    converged = (res.retcode == ReturnCode.Success)

    # get optimal correction factor
    fct = res.u
    @debug "Found Best FCT: $(fct .* 1e6)E-6"

    if !converged @warn "CTC did not converge" end

    # calculate drift time corrected lq
    lq_ctc_corrected = lq_cut .+ PolCalFunc(0.0, fct...).(qdrift_cut)
    
    # normalize once again to μ = 0 and σ = 1
    h_after = fit(Histogram, lq_ctc_corrected, hist_start:bin_width:hist_end)
    
    _cuts_lq = cut_single_peak(lq_ctc_corrected, hist_start, eltype(hist_start)(10); n_bins=-1, relative_cut = 0.4)
    result_after, report_after = fit_single_trunc_gauss(lq_ctc_corrected, _cuts_lq, uncertainty=true)
    μ_norm = mvalue(result_after.μ)
    σ_norm = mvalue(result_after.σ)
    lq_ctc_normalized = (lq_ctc_corrected .- μ_norm) ./ σ_norm

    # get cal PropertyFunctions
    lq_ctc_func = "( ( $(lq_expression) ) + " * join(["$(fct[i]) * ( $(qdrift_expression) )^$(i)" for i in eachindex(fct)], " + ") * " - $(μ_norm) ) / $(σ_norm) "

    # create final histograms after normalization
    cuts_lq = cut_single_peak(lq_ctc_normalized, hist_start, eltype(hist_start)(10); n_bins=-1, relative_cut = 0.4)
    result_after_norm, report_after_norm = fit_single_trunc_gauss(lq_ctc_normalized, cuts_lq, uncertainty=true)

    h_after_norm = fit(Histogram, lq_ctc_normalized, -5:bin_width:10)

    ### maybe modify result and report (also depending on the recipe)
    result = (
        dep_left = dep_left,
        dep_right = dep_right,
        func = lq_ctc_func,
        fct = fct,
        σ_start = f_minimize(0.0),
        σ_optimal = f_minimize(fct),
        σ_before = result_before.σ,
        σ_after = result_after.σ,
        σ_after_norm = result_after_norm.σ,
        before = result_before,
        after = result_after,
        after_norm = result_after_norm,
        converged = converged
    )
    report = (
        dep_left = dep_left,
        dep_right = dep_right,
        fct = result.fct,
        bin_width = bin_width,
        lq_peak = lq_cut,
        lq_ctc_corrected = lq_ctc_corrected, 
        lq_ctc_normalized = lq_ctc_normalized, 
        qdrift_peak = qdrift_cut,
        h_before = h_before,
        h_after = h_after,
        h_after_norm = h_after_norm,
        σ_before = result.σ_before,
        σ_after = result.σ_after,
        σ_after_norm = result.σ_after_norm,
        report_before = report_before,
        report_after = report_after,
        report_after_norm = report_after_norm
    )
    return result, report
end
export ctc_lq