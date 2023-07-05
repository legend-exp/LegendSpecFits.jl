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
        μ = μ * x_unit,
        μ_err = μ_uncertainty * x_unit,
        σ = σ * x_unit,
        σ_err = σ_uncertainty * x_unit,
        n = length(x)
    )
    report = (
        f_fit = t -> pdf(truncated(Normal(μ, σ), cut_low, cut_high), t),
        # f_fit = t -> length(x) ./ (cut_high - cut_low) .* pdf(Normal(μ, σ), t),
        μ = result.μ,
        μ_err = result.μ_err,
        σ = result.σ,
        σ_err = result.σ_err,
        n = result.n
    )
    return (result = result, report = report)
end
export fit_single_trunc_gauss