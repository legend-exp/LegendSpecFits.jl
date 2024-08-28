# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    fit_single_aoe_compton_with_fixed_μ_and_σ(h::Histogram, μ::Number, σ::Number, ps::NamedTuple; uncertainty::Bool=true)

Fit a single A/E Compton band using the `f_aoe_compton` function consisting of a gaussian SSE peak and a step like background for MSE events using fixed values for μ and σ.

# Returns
    * `neg_log_likelihood`: The negative log-likelihood of the likelihood fit
    * `report`: Dict of NamedTuples of the fit report which can be plotted for each compton band
"""
function fit_single_aoe_compton_with_fixed_μ_and_σ(h::Histogram, μ::Number, σ::Number, ps::NamedTuple; just_likelihood::Bool = false, fit_func::Symbol = :f_fit, background_center::Union{Real,Nothing} = μ, uncertainty::Bool=false)
    # create pseudo priors

    pseudo_prior = get_aoe_pseudo_prior(h, ps, fit_func;
        pseudo_prior = NamedTupleDist(μ = ConstValueShape(μ), σ = ConstValueShape(σ)))
        
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = mean(pseudo_prior)

    # get fit function with background center
    fit_function = get_aoe_fit_functions(; )[fit_func]

    # create loglikehood function
    f_loglike = let f_fit=fit_function, h=h
        v -> hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v) : 0, h)
    end

    # MLE
    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init))

    converged = Optim.converged(opt_r)
    !converged && @warn "Fit did not converge"

    if just_likelihood return Optim.minimum(opt_r) end

    # best fit results
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

    if uncertainty
            
        # only calculate errors for non-fixed parameters
        f_loglike_array(v) = - f_loglike(array_to_tuple(vcat([μ, σ], v), v_ml)) 

        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml)[3:end])

        param_covariance = nothing
        if !all(isfinite.(H))
            @warn "Hessian matrix is not finite"
            param_covariance = zeros(length(v_ml) - 2, length(v_ml) - 2)
        else
            # Calculate the parameter covariance matrix
            param_covariance = inv(H)
        end
        if ~isposdef(param_covariance)
            param_covariance = nearestSPD(param_covariance)
        end
        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(vcat([zero(typeof(μ)), zero(typeof(σ))], sqrt.(abs.(diag(param_covariance)))), v_ml)
        

        # get p-value 
        pval, chi2, dof = p_value(fit_function, h, v_ml)
        # calculate normalized residuals
        residuals, residuals_norm, _, bin_centers = get_residuals(fit_function, h, v_ml)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"
        @debug "n: $(v_ml.n) ± $(v_ml_err.n)"
        @debug "B: $(v_ml.B) ± $(v_ml_err.B)"
        @debug "p: $pval , chi2 = $(chi2) with $(dof) dof"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml)]...),
                (gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, converged = converged),))
        report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(fit_function, v_ml)(x),
            f_components = aoe_compton_peakshape_components(fit_func, v_ml),
            gof = merge(result.gof, (residuals = residuals, residuals_norm = residuals_norm,))        )
    else
        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"
        @debug "n: $(v_ml.n)"
        @debug "B: $(v_ml.B)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], NaN) for k in keys(v_ml)]...),
                    (gof = (converged = converged,) ,))
        report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(fit_function, v_ml)(x),
            f_components = aoe_compton_peakshape_components(fit_func, v_ml; background_center = background_center),
            gof = NamedTuple()
        )
    end
    
    return result, report
end

"""
    fit_aoe_compton_combined(peakhists::Vector{<:Histogram}, peakstats::StructArray, compton_bands::Array{T}, result_corrections::NamedTuple; pars_aoe::NamedTuple{(:μ, :μ_err, :σ, :σ_err)}=NamedTuple{(:μ, :μ_err, :σ, :σ_err)}(nothing, nothing, nothing, nothing), uncertainty::Bool=false) where T<:Unitful.Energy{<:Real}

Performed a combined fit over all A/E Compton band using the `f_aoe_compton` function consisting of a gaussian SSE peak and a step like background for MSE events,
assuming `f_aoe_mu` for μ and `f_aoe_sigma` for σ.

# Returns
    * `v_ml`: The fit result from the maximum-likelihood fit.
    * `report`: Dict of NamedTuples of the fit report which can be plotted for each compton band
"""
function fit_aoe_compton_combined(peakhists::Vector{<:Histogram}, peakstats::StructArray, compton_bands::Array{T}, result_corrections::NamedTuple; fit_func::Symbol = :f_fit, uncertainty::Bool=false) where T<:Unitful.Energy{<:Real}
    
    μA = ustrip(result_corrections.μ_compton.par[1])
    μB = ustrip(result_corrections.μ_compton.par[2])
    σA = ustrip(result_corrections.σ_compton.par[1])
    σB = ustrip(result_corrections.σ_compton.par[2])
    
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
        μA = Normal(mvalue(μA), let ΔμA = muncert(μA); ifelse(isnan(ΔμA), abs(0.1*mvalue(μA)), ΔμA) end),
        μB = Normal(mvalue(μB), let ΔμB = muncert(μB); ifelse(isnan(ΔμB), abs(0.1*mvalue(μB)), ΔμB) end),
        σA = Normal(mvalue(σA), let ΔσA = muncert(σA); ifelse(isnan(ΔσA), abs(0.1*mvalue(σA)), ΔσA) end),
        σB = Normal(mvalue(σB), let ΔσB = muncert(σB); ifelse(isnan(ΔσB), abs(0.1*mvalue(σB)), ΔσB) end),
    )
    
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)
    
    # start values for MLE
    v_init = mean(pseudo_prior)

    # create loglikehood function
    f_loglike = pars -> begin
        
        neg_log_likelihoods = zeros(typeof(pars.μA), length(compton_bands))

        # iterate through all peaks (multithreaded)
        Threads.@threads for i in eachindex(compton_bands)

            # get histogram and peakstats
            h  = peakhists[i]
            ps = peakstats[i]
            e = ustrip(compton_bands[i])
            μ = f_aoe_mu(e, (pars.μA, pars.μB))
            σ = f_aoe_sigma(e, (pars.σA, pars.σB))

            # fit peak
            try
                neg_log_likelihoods[i] = fit_single_aoe_compton_with_fixed_μ_and_σ(h, μ, σ, ps; just_likelihood = true, fit_func=fit_func)
            catch e
                @warn "Error fitting band $(compton_bands[i]): $e"
                continue
            end
        end
        return sum(neg_log_likelihoods)
    end
    
    # MLE
    opt_r = optimize(f_loglike ∘ inverse(f_trafo), f_trafo(v_init), NelderMead(), Optim.Options(time_limit = 120, show_trace=false, iterations = 1000))

    converged = Optim.converged(opt_r)
    !converged && @warn "Fit did not converge"

    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

    v_results = Vector{NamedTuple{(:μ, :σ, :n, :B, :δ, :gof)}}(undef, length(compton_bands))
    v_reports = Vector{NamedTuple{(:v, :h, :f_fit, :f_components, :gof)}}(undef, length(compton_bands))
    
    let pars = v_ml
        
        # iterate throuh all peaks (multithreaded)
        Threads.@threads for i in eachindex(compton_bands)

            # get histogram and peakstats
            h  = peakhists[i]
            ps = peakstats[i]
            e = ustrip(compton_bands[i])
            μ = f_aoe_mu(e, (pars.μA, pars.μB))
            σ = f_aoe_sigma(e, (pars.σA, pars.σB))

            # fit peak
            try
                v_results[i], v_reports[i] = fit_single_aoe_compton_with_fixed_μ_and_σ(h, μ, σ, ps; just_likelihood=false, fit_func=fit_func, uncertainty=uncertainty)
            catch e
                @warn "Error fitting band $(compton_bands[i]): $e"
                continue
            end
        end
    end

    band_results = Dict{T, NamedTuple}(compton_bands .=> v_results)
    band_reports = Dict{T, NamedTuple}(compton_bands .=> v_reports)

    if uncertainty && converged

        f_loglike_array = array -> begin
            pars = array_to_tuple(array, v_ml)
        
            neg_log_likelihoods = zeros(typeof(pars.μA), length(compton_bands))

            # iterate through all peaks (multithreaded)
            for i in eachindex(compton_bands)

                # get histogram and peakstats
                h  = peakhists[i]
                ps = peakstats[i]
                e = ustrip(compton_bands[i])
                μ = f_aoe_mu(e, (pars.μA, pars.μB))
                σ = f_aoe_sigma(e, (pars.σA, pars.σB))

                # fit peak
                try
                    neg_log_likelihoods[i] = fit_single_aoe_compton_with_fixed_μ_and_σ(h, μ, σ, ps; just_likelihood=true, fit_func=fit_func)
                catch e
                    @warn "Error fitting band $(compton_bands[i]): $e"
                    continue
                end
            end
            return sum(neg_log_likelihoods)
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

        # sum chi2 and dof of individual fits to compute the p-value for the combined fits
        chi2 = reduce((acc, x) -> acc + x.gof.chi2, values(band_results), init = 0.)
        dof  = reduce((acc, x) -> acc + x.gof.dof, values(band_results), init = 0.)
        pval = ccdf(Chisq(dof), chi2)

        # concatenate the normalized residuals of all individual fits
        residuals      = vcat(getproperty.(values(band_reports), :gof), :residuals)
        residuals_norm = vcat(getproperty.(values(band_reports), :gof), :residuals_norm)

        @debug "Best Fit values"
        @debug "μA: $(v_ml.μA) ± $(v_ml_err.μA)"
        @debug "μB: $(v_ml.μB) ± $(v_ml_err.μB)"
        @debug "σA: $(v_ml.σA) ± $(v_ml_err.σA)"
        @debug "σB: $(v_ml.σB) ± $(v_ml_err.σB)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml)]...),
                (gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, 
                residuals = residuals, residuals_norm = residuals_norm),) #, bin_centers = bin_centers),)
                )
    else
        @debug "Best Fit values"
        @debug "μA: $(v_ml.μA)"
        @debug "μB: $(v_ml.μB)"
        @debug "σA: $(v_ml.σA)"
        @debug "σB: $(v_ml.σB)"

        result = merge(v_ml, )
    end

    report = (
        v = v_ml,
        band_reports = band_reports
    )

    return result, report
end
export fit_aoe_compton_combined