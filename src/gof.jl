# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
"""
    `p_value(f_fit, h, v_ml)` : calculate p-value of chi2 test for goodness-of-fit
 input:
 * f_fit --> function handle of fit function (peakshape)
 * h --> histogram of data
 * v_ml --> best-fit parameters
 output:
 * pval --> p-value of chi2 test
 * chi2 --> chi2 value
 * dof --> degrees of freedom
"""

function prepare_data(h::Histogram{<:Real,1})
    # get bin center, width and counts from histogrammed data
    bin_edges = first(h.edges)
    counts = h.weights
    bin_centers = (bin_edges[begin:end-1] .+ bin_edges[begin+1:end]) ./ 2
    bin_widths = bin_edges[begin+1:end] .- bin_edges[begin:end-1]
    return counts, bin_widths, bin_centers
end

function p_value(f_fit::Base.Callable, h::Histogram{<:Real,1},v_ml::NamedTuple)
    # prepare data
    counts, bin_widths, bin_centers = prepare_data(h)

    # get peakshape of best-fit 
    model_func  = Base.Fix2(f_fit, v_ml) # fix the fit parameters to ML best-estimate
    model_counts = bin_widths.*map(energy->model_func(energy), bin_centers) # evaluate model at bin center (= binned measured energies)

    # calculate chi2
    chi2    = sum((model_counts[model_counts.>0]-counts[model_counts.>0]).^2 ./ model_counts[model_counts.>0])
    npar    = length(v_ml)
    dof    = length(counts[model_counts.>0])-npar
    pval    = ccdf(Chisq(dof),chi2)
    if any(model_counts.<=5)
              @warn "WARNING: bin with <=$(minimum(model_counts)) counts -  chi2 test might be not valid"
    else  
         @debug "p-value = $(round(pval,digits=2))"
    end
    return pval, chi2, dof
end
export p_value

""" alternative p-value via loglikelihood ratio"""
function p_value_LogLikeRatio(f_fit::Base.Callable, h::Histogram{<:Real,1},v_ml::NamedTuple)
    # prepare data
    counts, bin_widths, bin_centers = prepare_data(h)

    # get peakshape of best-fit 
    model_func  = Base.Fix2(f_fit, v_ml) # fix the fit parameters to ML best-estimate
    model_counts = bin_widths.*map(energy->model_func(energy), bin_centers) # evaluate model at bin center (= binned measured energies)

    # calculate chi2
    chi2    = sum((model_counts[model_counts.>0]-counts[model_counts.>0]).^2 ./ model_counts[model_counts.>0])
    npar    = length(v_ml)
    dof    = length(counts[model_counts.>0])-npar
    pval    = ccdf(Chisq(dof),chi2)
    if any(model_counts.<=5)
              @warn "WARNING: bin with <=$(minimum(model_counts)) counts -  chi2 test might be not valid"
    else  
         @debug "p-value = $(round(pval,digits=2))"
    end
    chi2   = 2*sum(model_counts.*log.(model_counts./counts)+model_counts-counts)
    pval   = ccdf(Chisq(dof),chi2)
return pval, chi2, dof
end
export p_value_LogLikeRatio