# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    fit_single_aoe_compton_with_fixed_μ_and_σ(h::Histogram, μ::Number, σ::Number, ps::NamedTuple; uncertainty::Bool=true)

Fit a single A/E Compton band using the `f_aoe_compton` function consisting of a gaussian SSE peak and a step like background for MSE events using fixed values for μ and σ.

# Returns
    * `neg_log_likelihood`: The negative log-likelihood of the likelihood fit
    * `report`: Dict of NamedTuples of the fit report which can be plotted for each compton band
"""
function fit_single_aoe_compton_with_fixed_μ_and_σ(h::Histogram, μ::Number, σ::Number, ps::NamedTuple; fit_func::Symbol = :aoe_one_bck, background_center::Union{Real,Nothing} = μ, uncertainty::Bool=false)
    
    # create pseudo priors
    pseudo_prior = get_aoe_pseudo_prior(h, ps, fit_func;
        pseudo_prior = NamedTupleDist(μ = ConstValueShape(μ), σ = ConstValueShape(σ)))
        
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = Vector(mean(f_trafo.target_dist)) 

    # get fit function with background center
    fit_function = get_aoe_fit_functions(; background_center = background_center)[fit_func]

    # create loglikehood function
    f_loglike = let f_fit=fit_function, h=h
        v -> hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v) : 0, h)
    end

    # MLE
    optf = OptimizationFunction((u, p) -> ((-) ∘ f_loglike ∘ inverse(f_trafo))(u), AutoForwardDiff())
    optpro = OptimizationProblem(optf, v_init, ())
    res = solve(optpro, Optimization.LBFGS(), maxiters = 3000)#, maxtime=optim_time_limit)

    converged = (res.retcode == ReturnCode.Success)
    if !converged @warn "Fit did not converge" end

    # best fit results
    v_ml = inverse(f_trafo)(res.u)

    if uncertainty && converged
            
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

# This function calculates the same thing as fit_single_aoe_compton_with_fixed_μ_and_σ, but just returns the value of the negative log-likelihood
function neg_log_likelihood_single_aoe_compton_with_fixed_μ_and_σ(h::Histogram, μ::Real, σ::Real, ps::NamedTuple; fit_func::Symbol = :aoe_one_bck, background_center::Union{Real,Nothing} = μ, optimize::Bool=true)
    
    # create pseudo priors
    pseudo_prior = get_aoe_pseudo_prior(h, ps, fit_func;
        pseudo_prior = NamedTupleDist(μ = ConstValueShape(μ), σ = ConstValueShape(σ)))
        
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = Vector(mean(f_trafo.target_dist)) 

    # get fit function with background center
    fit_function = get_aoe_fit_functions(; background_center = background_center)[fit_func]

    # create loglikehood function
    f_loglike = let f_fit=fit_function, h=h
        v -> hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v) : 0, h)
    end

    # MLE
    if optimize
        optf = OptimizationFunction((u, p) -> ((-) ∘ f_loglike ∘ inverse(f_trafo))(u), AutoForwardDiff())
        optpro = OptimizationProblem(optf, v_init, ())
        res = solve(optpro, Optimization.LBFGS(), maxiters = 3000)#, maxtime=optim_time_limit)

        converged = (res.retcode == ReturnCode.Success)
        if !converged @warn "Fit did not converge" end

        return res.objective
    else
        return f_loglike(merge((μ = μ, σ = σ), ps))
    end
end

"""
    fit_aoe_compton_combined(peakhists::Vector{<:Histogram}, peakstats::StructArray, compton_bands::Array{T}, result_corrections::NamedTuple; pars_aoe::NamedTuple{(:μ, :μ_err, :σ, :σ_err)}=NamedTuple{(:μ, :μ_err, :σ, :σ_err)}(nothing, nothing, nothing, nothing), uncertainty::Bool=false) where T<:Unitful.Energy{<:Real}

Performed a combined fit over all A/E Compton band using the `f_aoe_compton` function consisting of a gaussian SSE peak and a step like background for MSE events,
assuming `f_aoe_mu` for `μ` and `f_aoe_sigma` for `σ`.

# Returns
    * `v_ml`: The fit result from the maximum-likelihood fit.
    * `report_μ`: Report to plot the combined fit result for the enery-dependence of `μ`.
    * `report_σ`: Report to plot the combined fit result for the enery-dependence of `σ`.
    * `report_bands`: Dict of NamedTuples of the fit report which can be plotted for each compton band
"""
function fit_aoe_compton_combined(peakhists::Vector{<:Histogram}, peakstats::StructArray, compton_bands::Array{T}, result_corrections::NamedTuple; 
    fit_func::Symbol = :aoe_one_bck, aoe_expression::Union{String,Symbol}="a / e", e_expression::Union{String,Symbol}="e", uncertainty::Bool=false) where T<:Unitful.Energy{<:Real}
    
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
    v_init = Vector(mean(f_trafo.target_dist))

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
                neg_log_likelihoods[i] = neg_log_likelihood_single_aoe_compton_with_fixed_μ_and_σ(h, μ, σ, ps; fit_func=fit_func)
            catch e
                @warn "Error fitting band $(compton_bands[i]): $e"
                continue
            end
        end
        return sum(neg_log_likelihoods)
    end
    
    # MLE
    optf = OptimizationFunction((u, p) -> (f_loglike ∘ inverse(f_trafo))(u), AutoForwardDiff())
    optpro = OptimizationProblem(optf, v_init, ())
    res = solve(optpro, NLopt.LN_BOBYQA(), maxiters = 5000, maxtime=5*optim_time_limit)
    converged = (res.retcode == ReturnCode.Success)
    if !converged @warn "Fit did not converge" end

    # best fit results
    v_ml = inverse(f_trafo)(res.u)

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
                v_results[i], v_reports[i] = fit_single_aoe_compton_with_fixed_μ_and_σ(h, μ, σ, ps; fit_func=fit_func, uncertainty=uncertainty)
            catch e
                @warn "Error fitting band $(compton_bands[i]): $e"
                continue
            end
        end
    end

    result_bands = OrderedDict{T, NamedTuple}(compton_bands .=> v_results)
    report_bands = OrderedDict{T, NamedTuple}(compton_bands .=> v_reports)

    if uncertainty && converged

        # create loglikehood function for single hist
        function _get_neg_log_likehood(e::Real, h::Histogram, vi::NamedTuple)
            # get fit function with background center
            fit_function = get_aoe_fit_functions(; background_center = f_aoe_mu(e, (v_ml.μA, v_ml.μB)))[fit_func]

            # create loglikehood function
            f_loglike = let f_fit=fit_function, h=h
                v -> hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v) : 0, h)
            end
            - f_loglike(vi)
        end

        # create full loglikehood function
        f_loglike_array = array -> begin
            pars = array_to_tuple(array, v_ml)
        
            neg_log_likelihoods = zeros(typeof(pars.μA), length(compton_bands))

            # iterate through all peaks (multithreaded)
            for i in eachindex(compton_bands)

                # get histogram and peakstats
                h  = peakhists[i]
                vi = v_results[i]
                e = ustrip(compton_bands[i])
                μ = f_aoe_mu(e, (pars.μA, pars.μB))
                σ = f_aoe_sigma(e, (pars.σA, pars.σB))
                
                neg_log_likelihoods[i] = _get_neg_log_likehood(e, h, merge(mvalue(vi), (μ = μ, σ = σ)))

                # fit peak
                # try
                # catch e
                #     @warn "Error fitting band $(compton_bands[i]): $e"
                #     continue
                # end
            end
            return sum(neg_log_likelihoods)
        end

        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml))

        param_covariance = if !all(isfinite.(H))
            @warn "Hessian matrix is not finite"
            zeros(length(v_ml), length(v_ml))
        else
            # Calculate the parameter covariance matrix
            inv(H)
        end
        if ~isposdef(param_covariance)
            param_covariance = nearestSPD(param_covariance)
        end
        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)

        # sum chi2 and dof of individual fits to compute the p-value for the combined fits
        chi2 = reduce((acc, x) -> if x.gof.converged acc + x.gof.chi2 else acc end, values(result_bands), init = 0.)
        dof  = reduce((acc, x) -> if x.gof.converged acc + x.gof.dof else acc end, values(result_bands), init = 0.)
        pval = ccdf(Chisq(dof), chi2)

        # concatenate the normalized residuals of all individual fits
        residuals      = vcat(get.(getproperty.(values(report_bands), :gof), :residuals, Ref([]))...)
        residuals_norm = vcat(get.(getproperty.(values(report_bands), :gof), :residuals_norm, Ref([]))...)

        @debug "Best Fit values"
        @debug "μA: $(v_ml.μA) ± $(v_ml_err.μA)"
        @debug "μB: $(v_ml.μB) ± $(v_ml_err.μB)"
        @debug "σA: $(v_ml.σA) ± $(v_ml_err.σA)"
        @debug "σB: $(v_ml.σB) ± $(v_ml_err.σB)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml)]...),
                (gof = (pvalue = pval,
                    chi2 = chi2, 
                    dof = dof, 
                    covmat = param_covariance, 
                    mean_residuals = mean(residuals_norm),
                    median_residuals = median(residuals_norm),
                    std_residuals = std(residuals_norm),
                    converged = converged),)
                )
    else
        @debug "Best Fit values"
        @debug "μA: $(v_ml.μA)"
        @debug "μB: $(v_ml.μB)"
        @debug "σA: $(v_ml.σA)"
        @debug "σB: $(v_ml.σB)"

        result = merge(v_ml, (gof = (converged = converged, ), ))
    end

    # Add same fields to result as fit_aoe_corrections
    e_unit = unit(first(compton_bands))
    es = ustrip.(e_unit, compton_bands)
    
    par_µ = [v_ml.µA, v_ml.µB / e_unit]
    func_µ = "$(mvalue(v_ml.µA)) + ($e_expression) * $(mvalue(v_ml.µB))$e_unit^-1"
    result_µ = (par = par_µ, func = func_µ)

    par_σ = [v_ml.σA, v_ml.σB * e_unit^2]
    func_σ = "sqrt( ($(mvalue(v_ml.σA)))^2 + ($(mvalue(v_ml.σB))$(e_unit))^2 / ($e_expression)^2 )"
    result_σ = (par = par_σ, func = func_σ)

    aoe_str = "($aoe_expression)"
    func_aoe_corr = "($aoe_str - ($(result_µ.func)) ) / ($(result_σ.func))"

    result = merge(result, (µ_compton = result_µ, σ_compton = result_σ, compton_bands = (e = es,), func = func_aoe_corr))

    # Create report for plotting the combined fit results
    label_fit_µ = latexstring("Combined Fit: \$ $(round(mvalue(v_ml.μA), digits=2)) $(mvalue(v_ml.μB) >= 0 ? "+" : "") $(round(mvalue(v_ml.μB)*1e6, digits=2))\\cdot 10^{-6} \\; E \$")
    label_fit_σ = latexstring("Combined Fit: \$\\sqrt{($(abs(round(mvalue(v_ml.σA)*1e3, digits=2)))\\cdot10^{-3})^2 + $(abs(round(ustrip(mvalue(v_ml.σB)), digits=2)))^2 / E^2}\$")
    #label_fit_µ = "Combined Fit:  + E * $(round(mvalue(v_ml.μB)*1e6, digits=2))e-6"
    #label_fit_σ = "Combined Fit: sqrt(($(round(mvalue(v_ml.σA)*1e6, digits=1))e-6)^2 + $(round(ustrip(mvalue(v_ml.σB)), digits=2))^2 / E^2)"
    #label_fit_σ = "Combined Fit: sqrt(($(round(mvalue(v_ml.σA)*1e3, digits=2))e-3)^2 + $(round(ustrip(mvalue(v_ml.σB)), digits=2))^2 / E^2)"
    
    μ_values = f_aoe_mu(es, (v_ml.μA, v_ml.μB))
    σ_values = f_aoe_sigma(es, (v_ml.σA, v_ml.σB))

    report_µ = (; values = µ_values, label_y = "µ", label_fit = label_fit_µ, energy = es)
    report_σ = (; values = σ_values, label_y = "σ", label_fit = label_fit_σ, energy = es)

    report = (
        v = v_ml,
        report_µ = report_µ,
        report_σ = report_σ,
        report_bands = report_bands
    )

    return result, report
end
export fit_aoe_compton_combined
