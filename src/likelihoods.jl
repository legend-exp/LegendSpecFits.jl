# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    hist_loglike(f_fit::Base.Callable, h::Histogram{<:Real,1})

Calculate the Poisson log-likelihood of a fit function `f_fit(x)` and a
histogram `h`. `f_fit` must accept all values `x` on the horizontal axis
of the histogram.

Currently uses a simple midpoint-rule integration of `f_fit` over the
bins of `h`.
"""
function hist_loglike(f_fit::Base.Callable, h::Histogram{<:Real,1})
    bin_edges = first(h.edges)
    counts = h.weights
    bin_centers = (bin_edges[begin:end-1] .+ bin_edges[begin+1:end]) ./ 2
    bin_widths = bin_edges[begin+1:end] .- bin_edges[begin:end-1]
    bin_ll(x, bw, k) = logpdf(Poisson(f_fit(x) < 0 ? 0 :  bw * f_fit(x)), k)
    sum(Base.Broadcast.broadcasted(bin_ll, bin_centers, bin_widths, counts))
end
export hist_loglike
