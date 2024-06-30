# lq compton region peakshapes
f_lq_compton(x, v) = aoe_compton_peakshape(x, v.μ, v.σ, v.n, v.B, v.δ)
f_lq_sig(x, v)     = aoe_compton_signal_peakshape(x, v.μ, v.σ, v.n)
f_lq_bkg(x, v)     = aoe_compton_background_peakshape(x, v.μ, v.σ, v.B, v.δ)

# lq compton centroids energy depencence
f_lq_μ(x::T, v::Array{<:T}) where T<:Unitful.RealOrRealQuantity = -v[1] * x + v[2]
f_lq_μ(x::Array{<:T}, v::Array{<:T}) where T<:Unitful.RealOrRealQuantity = f_lq_μ.(x, v)
f_lq_μ(x, v::NamedTuple) = f_lq_μ(x, [v.μ_scs_slope, v.μ_scs_intercept])

# lq compton sigma energy depencence
f_lq_σ(x::T, v::Array{<:T}) where T<:Unitful.RealOrRealQuantity = exponential_decay(x, v[1], v[2], v[3])
f_lq_σ(x::Array{<:T}, v::Array{<:T}) where T<:Unitful.RealOrRealQuantity = exponential_decay.(x, v[1], v[2], v[3])
f_lq_σ(x, v::NamedTuple) = f_lq_σ(x, [v.σ_scs_amplitude, v.σ_scs_decay, v.σ_scs_offset])


"""
    fit_lq_compton(peakhists::Array, peakstats::StructArray, compton_bands::Array{T}) where T<:Real

Fit the A/E Compton bands using the `f_lq_compton` function consisting of a gaussian SSE peak and a step like background for MSE events.

# Returns
    * `result`: Dict of NamedTuples of the fit results containing values and errors for each compton band
    * `report`: Dict of NamedTuples of the fit report which can be plotted for each compton band
"""
function fit_lq_compton(peakhists::Vector{<:Histogram}, peakstats::StructArray, compton_bands::Array{T},; pars_lq::NamedTuple{(:μ, :μ_err, :σ, :σ_err)}=NamedTuple{(:μ, :μ_err, :σ, :σ_err)}(nothing, nothing, nothing, nothing), uncertainty::Bool=false) where T<:Unitful.Energy{<:Real}
    # create return and result dicts
    result = Dict{T, NamedTuple}()
    report = Dict{T, NamedTuple}()
    # iterate throuh all peaks
    for (i, band) in enumerate(compton_bands)
        # get histogram and peakstats
        h  = peakhists[i]
        ps = peakstats[i]
        if !isnothing(pars_lq.μ)
            ps = merge(peakstats[i], (μ = f_lq_μ(band, pars_lq.μ), σ = f_lq_σ(band, pars_lq.σ)))
        end
        # fit peak
        result_band, report_band = nothing, nothing
        try
            result_band, report_band = fit_single_lq_compton(h, ps, ; uncertainty=uncertainty)
        catch e
            @warn "Error fitting band $band: $e"
            continue
        end
        # save results
        result[band] = result_band
        report[band] = report_band
    end
    return result, report
end
export fit_lq_compton


"""
    fit_single_lq_compton(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background, :μ, :σ), NTuple{7, T}}; uncertainty::Bool=true) where T<:Real

Perform a fit of the peakshape to the data in `h` using the initial values in `ps` while using the `f_lq_compton` function consisting of a gaussian SSE peak and a step like background for MSE events.

# Returns
    * `result`: NamedTuple of the fit results containing values and errors
    * `report`: NamedTuple of the fit report which can be plotted
"""
function fit_single_lq_compton(h::Histogram, ps::NamedTuple; uncertainty::Bool=true)
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
                μ = Uniform(ps.peak_pos-0.5*ps.peak_sigma, ps.peak_pos+2*ps.peak_sigma),
                σ = Uniform(0.85*ps.peak_sigma, 1.2*ps.peak_sigma),
                n = LogUniform(0.01*ps.peak_counts, 5*ps.peak_counts),
                B = Uniform(0.1*ps.mean_background, 10*ps.mean_background),
                δ = Uniform(0.01, 1e2)
            )
    
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = mean(pseudo_prior)

    # create loglikehood function
    f_loglike = let f_fit=f_lq_compton, h=h
        v -> hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v) : 0, h)
    end

    # MLE
    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init))

    # best fit results
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

    f_loglike_array = let f_fit=aoe_compton_peakshape, h=h
        v -> - hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v...) : 0, h)
    end

    if uncertainty
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
        pval, chi2, dof = p_value(f_lq_compton, h, v_ml)
        # calculate normalized residuals
        residuals, residuals_norm, _, bin_centers = get_residuals(f_lq_compton, h, v_ml)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"
        @debug "n: $(v_ml.n) ± $(v_ml_err.n)"
        @debug "B: $(v_ml.B) ± $(v_ml_err.B)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml)]...),
                  (gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, 
                  residuals = residuals, residuals_norm = residuals_norm, bin_centers = bin_centers),))
    else
        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"
        @debug "n: $(v_ml.n)"
        @debug "B: $(v_ml.B)"

        result = merge(v_ml, )
    end
    report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(f_lq_compton, v_ml)(x),
            f_sig = x -> Base.Fix2(f_lq_sig, v_ml)(x),
            f_bck = x -> Base.Fix2(f_lq_bkg, v_ml)(x)
        )
    return result, report
end
