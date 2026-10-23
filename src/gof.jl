# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    _prepare_data(h::Histogram{<:Real,1})

Returns the tuple `(counts, bin_widths, bin_centers)` of the one-dimensional histogram `h`.
"""
function _prepare_data(h::Histogram{<:Real,1})
    # get bin center, width and counts from histogrammed data
    bin_edges = first(h.edges)
    counts = h.weights
    bin_centers = (bin_edges[begin:end-1] .+ bin_edges[begin+1:end]) ./ 2
    bin_widths = bin_edges[begin+1:end] .- bin_edges[begin:end-1]
    return counts, bin_widths, bin_centers
end


"""
    _get_model_counts(f_fit::Base.Callable, v_ml, bin_centers, bin_widths)

Returns the expected counts in each bin: the bin width times the model
`f_fit(x, v_ml)` evaluated at the bin center.
"""
function _get_model_counts(f_fit::Base.Callable, v_ml::Union{NamedTuple, AbstractVector}, bin_centers::Union{StepRangeLen, Vector{<:Real}}, bin_widths::Union{StepRangeLen, Vector{<:Real}})
    model_func = Base.Fix2(f_fit, v_ml) # fix the fit parameters to ML best-estimate
    model_counts = bin_widths .* map(energy -> model_func(energy), bin_centers) # evaluate model at bin center (= binned measured energies)
    return model_counts
end



"""
    p_value(f_fit::Base.Callable, h::Histogram{<:Real,1}, v_ml::Union{NamedTuple, AbstractVector})

Calculates the p-value of a fit with a chi-squared test that assumes Gaussian
uncertainties, the model counts being the variance of each bin.

Only bins with at least one count enter the test. `f_fit(x, v_ml)` is the fit
function and `v_ml` the best-fit parameters. For bins with few counts the
Gaussian assumption is poor; see [`p_value_poissonll`](@ref) for a test that
stays valid there.

## Returns
A tuple `(pval, chi2, dof)` with the p-value, the chi-squared value and the
degrees of freedom (number of non-empty bins minus the number of parameters).
`pval` is `NaN` if `dof` is negative.
"""
function p_value(fit_func::Base.Callable, h::Histogram{<:Real,1}, v_ml::Union{NamedTuple, AbstractVector})
    # prepare data
    counts, bin_widths, bin_centers = _prepare_data(h)

    # get peakshape of best-fit 
    model_counts = _get_model_counts(fit_func, v_ml, bin_centers, bin_widths)

    # calculate chi2
    chi2 = sum((model_counts[counts .> 0] - counts[counts .> 0]) .^ 2 ./ model_counts[counts .> 0])
    npar = length(v_ml)
    dof = length(counts[counts .> 0]) - npar
    if dof<0
        pval = NaN # tbd 
    else
        pval = ccdf(Chisq(dof),chi2)
    end
    if any(counts .<= 5)
        @debug "Bin width <= $(round(minimum(counts), digits=0)) counts - Chi2 test might be not valid"
    else
        @debug "p-value = $(round(pval, digits=2))"
    end
    return pval, chi2, dof
end
export p_value


"""
    poisson_deviance(f_fit::Base.Callable, h::Histogram{<:Real,1}, v_ml::Union{NamedTuple, AbstractVector})

Calculates the Poisson deviance of the histogram `h` under the model
`f_fit(x, v_ml)`, `D = 2 Σ_i [λ_i − k_i + k_i log(k_i / λ_i)]`, with `k_i` the
counts and `λ_i` the bin width times the model at the bin center (the
likelihood-ratio statistic of Baker and Cousins). Also calculates its mean `E[D]`
and variance `Var[D]` under the model, summed from the exact Poisson moments of
every bin's term. A bin whose expectation vanishes contributes nothing when it is
empty and `Inf` otherwise.

## Returns
A tuple `(D, E, V)` of the deviance, its mean and its variance.
"""
function poisson_deviance(f_fit::Base.Callable, h::Histogram{<:Real,1}, v_ml::Union{NamedTuple, AbstractVector})
    counts, bin_widths, bin_centers = _prepare_data(h)
    λs = _get_model_counts(f_fit, v_ml, bin_centers, bin_widths)
    D = 0.0; E = 0.0; V = 0.0
    for (λ, k) in zip(λs, counts)
        if !(λ > 0)
            k > 0 && (D = Inf)
            continue
        end
        D += 2 * (λ - k + (k > 0 ? k * log(k / λ) : 0.0))
        # exact moments of the bin's term: the Poisson mass above k_max is below 1e-12
        d = Poisson(λ)
        k_max = ceil(Int, λ + 12 * sqrt(λ) + 30)
        m1 = 0.0; m2 = 0.0
        for j in 0:k_max
            t = 2 * (λ - j + (j > 0 ? j * log(j / λ) : 0.0))
            w = pdf(d, j)
            m1 += w * t
            m2 += w * t^2
        end
        E += m1
        V += m2 - m1^2
    end
    return D, E, V
end
export poisson_deviance

"""
    p_value_poissonll(f_fit::Base.Callable, h::Histogram{<:Real,1}, v_ml::Union{NamedTuple, AbstractVector})

Calculates the p-value of a maximum-likelihood fit with Poisson statistics with
the moment-corrected deviance test. The Poisson deviance `D` of
[`poisson_deviance`](@ref) is compared with its mean and variance under the
fitted model, the mean reduced by the number of fitted parameters, and the
p-value is the upper tail of the normal distribution at
`z = (D − (E[D] − n_par)) / sqrt(Var[D])`. Unlike the χ² approximation of the
deviance this stays valid for bins with few or no counts (Kaastra 2017,
A&A 605, A51, for the equivalent C statistic).

## Returns
A tuple `(pval, chi2, dof)` with `chi2 = D` and `dof` the number of bins
minus the number of parameters. `pval` is 0 if `D` is not finite.
"""
function p_value_poissonll(fit_func::Base.Callable, h::Histogram{<:Real,1}, v_ml::Union{NamedTuple, AbstractVector})
    chi2, expected, variance = poisson_deviance(fit_func, h, v_ml)
    npar = length(v_ml)
    dof = length(h.weights) - npar
    pval = isfinite(chi2) && variance > 0 ? ccdf(Normal(), (chi2 - (expected - npar)) / sqrt(variance)) : 0.0
    return pval, chi2, dof
end
export p_value_poissonll


"""
    likelihood_ratio(f_fit::Base.Callable, h::Histogram{<:Real,1})

Calculates the likelihood-ratio statistic `-2 (log L_model - log L_saturated)` of
the histogram `h`, where `L_model` is the Poisson likelihood for the expected
counts `bin_width * f_fit(bin_center)` and `L_saturated` is the likelihood with
the expected counts set equal to the observed counts.
"""
function likelihood_ratio(f_fit::Base.Callable, h::Histogram{<:Real,1})
    bin_edges = first(h.edges)
    counts = h.weights
    bin_centers = (bin_edges[begin:end-1] .+ bin_edges[begin+1:end]) ./ 2
    bin_widths = bin_edges[begin+1:end] .- bin_edges[begin:end-1]
    bin_ll(x, bw, k) = logpdf(Poisson(bw * f_fit(x)), k)
    loglikelihood_ml = sum(Base.Broadcast.broadcasted(bin_ll, bin_centers, bin_widths, counts))
    loglikelihood = sum(logpdf.(Poisson.(counts), counts))
    -2*(loglikelihood_ml - loglikelihood)
end
"""
    p_value_MC(
        f_fit::Base.Callable, h::Histogram{<:Real,1}, ps::NamedTuple, v_ml::NamedTuple;
        n_samples::Int64 = 1000
    )

Calculates a p-value with Monte Carlo sampling. This is computationally more
expensive than [`p_value`](@ref) and [`p_value_poissonll`](@ref).

`n_samples` histograms are drawn with Poisson-distributed bin counts around the
best-fit model `f_fit(x, v_ml)`. Each sample is fitted with
[`fit_single_peak_th228`](@ref), starting from the peak statistics `ps`, and the
negative log-likelihood of the sample fit is calculated. The p-value is the
fraction of samples whose negative log-likelihood is at most that of `h`.
"""
function p_value_MC(f_fit::Base.Callable, h::Histogram{<:Real,1}, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background, :mean_background_step, :mean_background_std), NTuple{7, T}}, v_ml::NamedTuple, ; n_samples::Int64=1000) where T<:Real
    counts, bin_widths, bin_centers = _prepare_data(h) # get data 
    # get peakshape of best-fit and maximum likelihood value
    model_func = Base.Fix2(f_fit, v_ml) # fix the fit parameters to ML best-estimate
    model_counts = bin_widths .* map(energy -> model_func(energy), bin_centers) # evaluate model at bin center (= binned measured energies)
    loglike_bf = -hist_loglike(model_func, h)

    # draw sample for each bin
    dists = Poisson.(model_counts) # create poisson distribution for each bin
    counts_mc_vec = rand.(dists, n_samples) # randomized histogram counts
    counts_mc = [[] for _ in 1:n_samples] #re-structure data_samples_vec to array of arrays, there is probably a better way to do this...
    for i = 1:n_samples
        counts_mc[i] = map(x -> x[i], counts_mc_vec)
    end

    # fit every sample histogram and calculate max. loglikelihood
    loglike_bf_mc = NaN .* ones(n_samples)
    h_mc = h # make copy of data histogram
    for i = 1:n_samples
        h_mc.weights = counts_mc[i] # overwrite counts with MC values
        result_fit_mc, _ = fit_single_peak_th228(h_mc, ps; uncertainty=false) # fit MC histogram
        fit_par_mc = mvalue(result_fit_mc[(:μ, :σ, :n, :step_amplitude, :skew_fraction, :skew_width, :background)])
        model_func_sample = Base.Fix2(f_fit, fit_par_mc) # fix the fit parameters to ML best-estimate
        loglike_bf_mc[i] = -hist_loglike(model_func_sample, h_mc) # loglikelihood for best-fit
    end

    # calculate p-value
    pval = sum(loglike_bf_mc .<= loglike_bf) ./ n_samples # preliminary. could be improved e.g. with interpolation
    return pval
end
export p_value_MC

"""
    get_residuals(f_fit::Base.Callable, h::Histogram{<:Real,1}, v_ml::Union{NamedTuple, AbstractVector})

Calculates bin-wise residuals and normalized residuals of the histogram `h`
with respect to the model `f_fit(x, v_ml)`, and a bin-wise p-value based on the
Poisson distribution of each bin. Only bins with positive expected counts are included.

## Returns
A tuple `(residuals, residuals_norm, p_value_binwise, bin_centers)` with
* `residuals`: model counts minus data counts in each bin.
* `residuals_norm`: `residuals` divided by the square root of the model counts.
* `p_value_binwise`: probability, for a Poisson distribution with the model counts as mean,
  of a deviation from the mean at least as large as observed.
* `bin_centers`: centers of the included bins.
"""
function get_residuals(f_fit::Base.Callable, h::Histogram{<:Real,1}, v_ml::Union{NamedTuple, AbstractVector})
    # prepare data
    counts, bin_widths, bin_centers = _prepare_data(h)

    # get peakshape of best-fit 
    model_counts = _get_model_counts(f_fit, v_ml, bin_centers, bin_widths)
    
    # calculate bin-wise residuals 
    residuals = model_counts[model_counts .> 0] - counts[model_counts .> 0]
    sigma = sqrt.(model_counts[model_counts .> 0])
    residuals_norm = residuals ./ sigma

    # calculate something like a bin-wise p-value (in case that makes sense)
    dist = Poisson.(model_counts[model_counts .> 0]) # each bin: poisson distributed

    cdf_value_low = cdf.(dist, model_counts[model_counts .> 0] .- abs.(residuals))
    cdf_value_up = 1 .- cdf.(dist, model_counts[model_counts .> 0] .+ abs.(residuals))
    p_value_binwise = cdf_value_low .+ cdf_value_up # significance of residuals -> ~proabability that residual (for a given bin) is as large as observed or larger
    return residuals, residuals_norm, p_value_binwise, bin_centers[model_counts .> 0]
end

