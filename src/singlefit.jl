"""
    fit_single_trunc_gauss(x::Array, cuts::NamedTuple{(:low, :high, :max), Tuple{Float64, Float64, Float64}})

Fit a single truncated Gaussian to the data `x` between `min_x` and `max_x`.
Returns `report` and `result`` with:
    * `f_fit`: fitted function
    * `μ`: mean of the Gaussian
    * `μ_err`: error of the mean
    * `σ`: standard deviation of the Gaussian
    * `σ_err`: error of the standard deviation
    * `n`: number of counts in the peak
"""
function fit_single_trunc_gauss(x::Vector{T}, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}) where T<:Unitful.RealOrRealQuantity
    @assert unit(cuts.low) == unit(cuts.high) == unit(cuts.max) == unit(x[1]) "Units of min_x, max_x and x must be the same"
    x_unit = unit(x[1])
    x, cut_low, cut_high, cut_max = ustrip.(x), ustrip(cuts.low), ustrip(cuts.high), ustrip(cuts.max)

    # cut peak out of data
    x = x[(x .> cut_low) .&& (x .< cut_high)]
    # create peak stats for start values
    ps = (peak_pos = cut_max, peak_sigma = cut_high - cut_low, peak_counts = length(x))
    @debug "Peak stats: $ps"
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
        μ = Uniform(ps.peak_pos-20, ps.peak_pos+20),
        σ = weibull_from_mx(ps.peak_sigma, 3*ps.peak_sigma),
        n = Uniform(ps.peak_counts-100, ps.peak_counts+100)
    )
    # create fit model
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)
    # f_trafo = LegendSpecFitsBATExt.get_distribution_transform(Normal, pseudo_prior)
    
    v_init  = mean(pseudo_prior)

    f_loglike = let cut_low = cut_low, cut_high = cut_high, x = x
        v -> (-1) * loglikelihood(truncated(Normal(v[1], v[2]), cut_low, cut_high), x)
    end

    # fit data
    opt_r = optimize(f_loglike ∘ inverse(f_trafo), f_trafo(v_init))
    μ, σ = inverse(f_trafo)(opt_r.minimizer)

    # Calculate the Hessian matrix using ForwardDiff
    H = ForwardDiff.hessian(f_loglike, [μ, σ])

    # Calculate the parameter covariance matrix
    param_covariance = inv(H)

    # Extract the parameter uncertainties
    μ_uncertainty = sqrt(param_covariance[1, 1])
    σ_uncertainty = sqrt(param_covariance[2, 2])

    @debug "μ: $μ ± $μ_uncertainty"
    @debug "σ: $σ ± $σ_uncertainty"

    result = (
        μ = measurement(μ, μ_uncertainty) * x_unit,
        σ = measurement(σ, σ_uncertainty) * x_unit,
        n = length(x)
    )
    report = (
        f_fit = t -> pdf(truncated(Normal(μ, σ), cut_low, cut_high), t),
        # f_fit = t -> length(x) ./ (cut_high - cut_low) .* pdf(Normal(μ, σ), t),
        μ = result.μ,
        σ = result.σ,
        n = result.n
    )
    return (result = result, report = report)
end
export fit_single_trunc_gauss

"""
    fit_half_centered_trunc_gauss(x::Array, cuts::NamedTuple{(:low, :high, :max), Tuple{Float64, Float64, Float64}})
Fit a single truncated Gaussian to the data `x` between `cut.low` and `cut.high`. The peak center is fixed at `μ` and the peak is cut in half either in the left or right half.
# Returns `report` and `result`` with:
    * `f_fit`: fitted function
    * `μ`: mean of the Gaussian
    * `μ_err`: error of the mean
    * `σ`: standard deviation of the Gaussian
    * `σ_err`: error of the standard deviation
    * `n`: number of counts in the peak
"""
function fit_half_centered_trunc_gauss(x::Vector{T}, μ::T, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}; left::Bool=false) where T<:Unitful.RealOrRealQuantity
    @assert unit(cuts.low) == unit(cuts.high) == unit(cuts.max) == unit(x[1]) == unit(μ) "Units of min_x, max_x and x must be the same"
    x_unit = unit(x[1])
    x, cut_low, cut_high, cut_max, μ = ustrip.(x), ustrip(cuts.low), ustrip(cuts.high), ustrip(cuts.max), ustrip(μ)

    # cut peak out of data
    x = ifelse(left, x[(x .> cut_low) .&& (x .< cut_high) .&& x .< μ], x[(x .> cut_low) .&& (x .< cut_high) .&& x .> μ])
    # create peak stats for start values
    ps = (peak_pos = cut_max, peak_sigma = cut_high - cut_low, peak_counts = length(x))
    @debug "Peak stats: $ps"
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
        μ = ConstValueDist(μ),
        σ = weibull_from_mx(ps.peak_sigma, 3*ps.peak_sigma),
        n = Uniform(ps.peak_counts-100, ps.peak_counts+100)
    )
    # create fit model
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)
    # f_trafo = LegendSpecFitsBATExt.get_distribution_transform(Normal, pseudo_prior)
    
    v_init  = mean(pseudo_prior)

    f_loglike = let cut_low = ifelse(left, cut_low, μ), cut_high = ifelse(left, μ, cut_high),  x = x
        v -> (-1) * loglikelihood(truncated(Normal(v[1], v[2]), cut_low, cut_high), x)
    end

    # fit data
    opt_r = optimize(f_loglike ∘ inverse(f_trafo), f_trafo(v_init))
    μ, σ = inverse(f_trafo)(opt_r.minimizer)

    # Calculate the Hessian matrix using ForwardDiff
    H = ForwardDiff.hessian(f_loglike, [μ, σ])

    # Calculate the parameter covariance matrix
    param_covariance = inv(H)

    # Extract the parameter uncertainties
    μ_uncertainty = sqrt(abs(param_covariance[1, 1]))
    σ_uncertainty = sqrt(abs(param_covariance[2, 2]))

    @debug "μ: $μ ± $μ_uncertainty"
    @debug "σ: $σ ± $σ_uncertainty"

    result = (
        μ = measurement(μ, μ_uncertainty) * x_unit,
        σ = measurement(σ, σ_uncertainty) * x_unit,
        n = length(x)
    )
    report = (
        # f_fit = t -> pdf(truncated(Normal(μ, σ), ifelse(left, cut_low, μ), ifelse(left, μ, cut_high)), t),
        f_fit = t -> pdf(Normal(μ, σ), t),
        μ = result.μ,
        σ = result.σ,
        n = result.n
    )
    return (result = result, report = report)
end
export fit_half_centered_trunc_gauss



"""
    fit_half_centered_trunc_gauss(x::Array, cuts::NamedTuple{(:low, :high, :max), Tuple{Float64, Float64, Float64}})
Fit a single truncated Gaussian to the data `x` between `cut.low` and `cut.high`. The peak center is fixed at `μ` and the peak is cut in half either in the left or right half.
# Returns `report` and `result`` with:
    * `f_fit`: fitted function
    * `μ`: mean of the Gaussian
    * `μ_err`: error of the mean
    * `σ`: standard deviation of the Gaussian
    * `σ_err`: error of the standard deviation
    * `n`: number of counts in the peak
"""
function fit_half_trunc_gauss(x::Vector{T}, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}; left::Bool=false) where T<:Unitful.RealOrRealQuantity
    @assert unit(cuts.low) == unit(cuts.high) == unit(cuts.max) == unit(x[1]) "Units of min_x, max_x and x must be the same"
    x_unit = unit(x[1])
    x, cut_low, cut_high, cut_max = ustrip.(x), ustrip(cuts.low), ustrip(cuts.high), ustrip(cuts.max)

    # cut peak out of data
    x = x[(x .> cut_low) .&& (x .< cut_high)]
    # create peak stats for start values
    ps = (peak_pos = cut_max, peak_sigma = std(x), peak_counts = length(x))
    @debug "Peak stats: $ps"
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
        μ = Uniform(ps.peak_pos-2*ps.peak_sigma, ps.peak_pos+2*ps.peak_sigma),
        σ = weibull_from_mx(ps.peak_sigma, 3*ps.peak_sigma),
        n = Uniform(ps.peak_counts-100, ps.peak_counts+100)
    )
    # create fit model
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)
    # f_trafo = LegendSpecFitsBATExt.get_distribution_transform(Normal, pseudo_prior)
    
    v_init  = mean(pseudo_prior)

    f_loglike = let cut_low = cut_low, cut_high = cut_high, cut_max = cut_max, left = left, x = x[ifelse(left, x .< cut_max, x .> cut_max)]
        v -> (-1) * loglikelihood(truncated(Normal(v[1], v[2]), ifelse(left, cut_low, cut_max), ifelse(left, cut_max, cut_high)), x)
    end

    # fit data
    opt_r = optimize(f_loglike ∘ inverse(f_trafo), f_trafo(v_init))
    μ, σ = inverse(f_trafo)(opt_r.minimizer)

    # Calculate the Hessian matrix using ForwardDiff
    H = ForwardDiff.hessian(f_loglike, [μ, σ])

    # Calculate the parameter covariance matrix
    param_covariance = inv(H)

    # Extract the parameter uncertainties
    μ_uncertainty = sqrt(abs(param_covariance[1, 1]))
    σ_uncertainty = sqrt(abs(param_covariance[2, 2]))

    @debug "μ: $μ ± $μ_uncertainty"
    @debug "σ: $σ ± $σ_uncertainty"

    result = (
        μ = measurement(μ, μ_uncertainty) * x_unit,
        σ = measurement(σ, σ_uncertainty) * x_unit,
        n = length(x)
    )
    report = (
        # f_fit = t -> pdf(truncated(Normal(μ, σ), ifelse(left, cut_low, μ), ifelse(left, μ, cut_high)), t),
        f_fit = t -> pdf(Normal(μ, σ), t),
        μ = result.μ,
        σ = result.σ,
        n = result.n
    )
    return (result = result, report = report)
end
export fit_half_trunc_gauss


#binned fits
f_gauss(x, v) = aoe_compton_signal_peakshape(x, v.μ, v.σ, v.n)

"""
    fit_binned_gauss(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background, :μ, :σ), NTuple{7, T}}; uncertainty::Bool=true) where T<:Real

Perform a binned fit of the peakshape to the data in `h` using the initial values in `ps` while using the `f_gauss` function consisting of a gaussian peak multiplied with an amplitude n.

# Returns
    * `result`: NamedTuple of the fit results containing values and errors
    * `report`: NamedTuple of the fit report which can be plotted
"""
function fit_binned_gauss(h::Histogram, ps::NamedTuple; uncertainty::Bool=true)
    # create pseudo priors
    pseudo_prior = NamedTupleDist(
                μ = Uniform(ps.peak_pos-5*ps.peak_sigma, ps.peak_pos+5*ps.peak_sigma),
                σ = Uniform(0.5*ps.peak_sigma, 5*ps.peak_sigma),
                n = Uniform(0.1*ps.peak_counts, 5*ps.peak_counts), 
            )
        
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = mean(pseudo_prior)

    # create loglikehood function
    f_loglike = let f_fit=f_gauss, h=h
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
        pval, chi2, dof = p_value(f_gauss, h, v_ml)
        # calculate normalized residuals
        residuals, residuals_norm, _, bin_centers = get_residuals(f_gauss, h, v_ml)

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
            f_fit = x -> Base.Fix2(f_gauss, v_ml)(x),
        )
    return result, report
end
export fit_binned_gauss