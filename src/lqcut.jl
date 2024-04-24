f_lq_DEP(x, v) = aoe_compton_signal_peakshape(x, v.μ, v.σ, v.n)


"""
    fit_lq_DEP_ampl(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background, :μ, :σ), NTuple{7, T}}; uncertainty::Bool=true) where T<:Real

Perform a fit of the peakshape to the data in `h` using the initial values in `ps` while using the `f_lq_DEP` function consisting of a gaussian peak.

# Returns
    * `result`: NamedTuple of the fit results containing values and errors
    * `report`: NamedTuple of the fit report which can be plotted
"""
function fit_lq_DEP(h::Histogram, ps::NamedTuple; uncertainty::Bool=true)
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
                μ = Uniform(ps.peak_pos-5*ps.peak_sigma, ps.peak_pos+5*ps.peak_sigma),
                σ = Uniform(0.5*ps.peak_sigma, 10*ps.peak_sigma),
                n = Uniform(0.001*ps.peak_counts, 50*ps.peak_counts),
            )
        
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = mean(pseudo_prior)

    # create loglikehood function
    f_loglike = let f_fit=f_lq_DEP, h=h
        v -> hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v) : 0, h)
    end

    # MLE
    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init))

    # best fit results
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

    if uncertainty
        f_loglike_array = let f_fit=aoe_compton_signal_peakshape, h=h
            v -> - hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v...) : 0, h)
        end

    
        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml))

        param_covariance = nothing
        if !all(isfinite.(H))
            @warn "Hessian matrix is not finite"
            param_covariance = zeros(length(v_ml), length(v_ml))
        else
            # Calculate the parameter covariance matrix
            param_covariance = inv(H)
        end
        if ~isposdef(param_covariance)
            param_covariance = nearestSPD(param_covariance)
        end
        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)

        # get p-value 
        pval, chi2, dof = p_value(f_lq_DEP, h, v_ml)
        # calculate normalized residuals
        residuals, residuals_norm, _, bin_centers = get_residuals(f_lq_DEP, h, v_ml)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"
        @debug "n: $(v_ml.n) ± $(v_ml_err.n)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml)]...),
                  (gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, 
                  residuals = residuals, residuals_norm = residuals_norm, bin_centers = bin_centers),))
    else
        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"
        @debug "n: $(v_ml.n)"

        result = merge(v_ml, )
    end
    report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(f_lq_DEP, v_ml)(x),
        )
    return result, report
end
export fit_lq_DEP

function lq_drift_time_correction(lq_DEP_dt, t_tcal)
    #lq box
    #sort array to exclude outliers
    sort_lq = sort(lq_DEP_dt)
    low_cut = Int(round(length(sort_lq) *0.02))
    high_cut = Int(round(length(sort_lq) *0.995))
    lq_prehist = fit(Histogram, lq_DEP_dt, range(sort_lq[low_cut], sort_lq[high_cut], length=100))
    lq_prestats = estimate_single_peak_stats(lq_prehist)
    lq_start = lq_prestats.peak_pos - 3*lq_prestats.peak_sigma
    lq_stop = lq_prestats.peak_pos + 3*lq_prestats.peak_sigma

    lq_edges = range(lq_start, stop=lq_stop, length=51) 
    # Create histograms with the same bin edges
    lq_hist_DEP = fit(Histogram, lq_DEP_dt, lq_edges)

    lq_DEP_stats = estimate_single_peak_stats(lq_hist_DEP)
    lq_result, lq_report = LegendSpecFits.fit_lq_DEP(lq_hist_DEP, lq_DEP_stats)
    µ_lq = mvalue(lq_result.μ)
    σ_lq = mvalue(lq_result.σ)

    #t_tcal box

    drift_prehist = fit(Histogram, t_tcal, range(minimum(t_tcal), stop=maximum(t_tcal), length=100))
    drift_prestats = estimate_single_peak_stats(drift_prehist)
    drift_start = drift_prestats.peak_pos - 3*drift_prestats.peak_sigma
    drift_stop = drift_prestats.peak_pos + 3*drift_prestats.peak_sigma
    
    drift_edges = range(drift_start, stop=drift_stop, length=71)
    drift_hist_DEP = fit(Histogram, t_tcal, drift_edges)
    
    drift_DEP_stats = estimate_single_peak_stats(drift_hist_DEP)
    drift_r1, drift_r2 = LegendSpecFits.fit_lq_DEP(drift_hist_DEP, drift_DEP_stats)
    µ_t = mvalue(drift_r1.μ)
    σ_t = mvalue(drift_r1.σ)

    #create 
    box_edges = (lq_lower = µ_lq - 2 * σ_lq, 
    lq_upper = µ_lq + 2 * σ_lq, 
    t_lower = µ_t - 2 * σ_t, 
    t_upper = µ_t + 2 * σ_t)
    return box_edges
end
export lq_drift_time_correction