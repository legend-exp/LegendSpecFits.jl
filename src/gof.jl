# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    _prepare_data(h::Histogram{<:Real,1})
aux. function to convert histogram data into bin edges, bin width and bin counts
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
    _get_model_counts(f_fit::Base.Callable,v_ml::Union{NamedTuple, AbstractVector},bin_centers::StepRangeLen,bin_widths::StepRangeLen)
aux. function to get modelled peakshape based on  histogram binning and best-fit parameter
"""
function _get_model_counts(f_fit::Base.Callable, v_ml::Union{NamedTuple, AbstractVector}, bin_centers::Union{StepRangeLen, Vector{<:Real}}, bin_widths::Union{StepRangeLen, Vector{<:Real}})
    model_func = Base.Fix2(f_fit, v_ml) # fix the fit parameters to ML best-estimate
    model_counts = bin_widths .* map(energy -> model_func(energy), bin_centers) # evaluate model at bin center (= binned measured energies)
    return model_counts
end



""" 
    p_value(f_fit::Base.Callable, h::Histogram{<:Real,1},v_ml::Union{NamedTuple, AbstractVector}) 
calculate p-value based on least-squares, assuming gaussian uncertainty
baseline method to get goodness-of-fit (gof)
# input:
 * `f_fit`function handle of fit function (peakshape)
 * `h` histogram of data
 * `v_ml` best-fit parameters
# returns:
 * `pval` p-value of chi2 test
 * `chi2` chi2 value
 * `dof` degrees of freedom
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
    p_value_poissonll(f_fit::Base.Callable, h::Histogram{<:Real,1},v_ml::Union{NamedTuple, AbstractVector})
p-value via poisson likelihood ratio: baseline for ML fits using Poisson statistics and bins with low number of counts
"""
function p_value_poissonll(fit_func::Base.Callable, h::Histogram{<:Real,1}, v_ml::Union{NamedTuple, AbstractVector})
    counts, bin_widths, bin_centers = _prepare_data(h) # prepare data
    model_func = Base.Fix2(fit_func, v_ml) # fix the fit parameters to ML best-estimate

    bin_ll(x, bw, k) = logpdf(Poisson(bw * model_func(x)), k) # loglikelihood per bin, evaluated for best-fit parameters (v_ml)
    loglikelihood_ml = sum(Base.Broadcast.broadcasted(bin_ll, bin_centers, bin_widths, counts)) # joint loglikelihood, evaluated for best-fit parameters (v_ml)

    loglikelihood_null = sum(logpdf.(Poisson.(counts), counts))  # joint loglikelihood, evaluate for data only
    chi2 = -2*(loglikelihood_ml - loglikelihood_null) # likelihood ratio. this quantity should follow chi2 distribution. 

    npar = length(v_ml)
    dof = length(counts[counts .> 0]) - npar
    pval = if dof <= 0
        0.0
    else
        ccdf(Chisq(dof), chi2)
    end

    return pval, chi2, dof
end
export p_value_poissonll


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
export likelihood_ratio2
"""
    p_value_MC(f_fit::Base.Callable, h::Histogram{<:Real,1},ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background)},v_ml::NamedTuple,;n_samples::Int64=1000) 
alternative p-value calculation via Monte Carlo sampling. **Warning**: computational more expensive than p_vaule() and p_value_LogLikeRatio()
# Input:
 * `f_fit`function handle of fit function (peakshape)
 * `h` histogram of data
 * `ps` best-fit parameters
 * `v_ml` best-fit parameters
 * `n_samples` number of samples

# Performed Steps:
* Create n_samples randomized histograms. For each bin, samples are drawn from a Poisson distribution with λ = model peak shape (best-fit parameter)
* Each sample histogram is fit using the model function `f_fit`
* For each sample fit, the max. loglikelihood fit is calculated 

# Returns
* % p value --> comparison of sample max. loglikelihood and max. loglikelihood of best-fit
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
    residuals(f_fit::Base.Callable, h::Histogram{<:Real,1},v_ml::Union{NamedTuple, AbstractVector})
Calculate bin-wise residuals and normalized residuals. 
Calcualte bin-wise p-value based on poisson distribution for each bin.

# Input:
 * `f_fit`function handle of fit function (peakshape)
 * `h` histogram of data
 * `v_ml` best-fit parameters

# Returns:
 * `residuals` difference: model - data (histogram bin count)
 * `residuals_norm` normalized residuals: model - data / sqrt(model)
 * `p_value_binwise` p-value for each bin based on poisson distribution
 * `bin_centers` centers of the bins for which the `residuals` were determined
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

