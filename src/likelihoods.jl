# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    hist_loglike(f_fit::Base.Callable, h::Histogram{<:Real,1})

Calculates the Poisson log-likelihood of the bin counts of the histogram `h`
given the density `f_fit(x)`. `f_fit` must accept all values `x` on the
horizontal axis of `h`.

The expected count in each bin is `f_fit` at the bin center times the bin
width (midpoint rule).
"""
function hist_loglike(f_fit::Base.Callable, h::Histogram{<:Real,1})
    bin_edges = first(h.edges)
    counts = h.weights
    bin_centers = midpoints(bin_edges)
    bin_widths = diff(bin_edges)
    # TODO: Prevent fit functions from returning negative PDF values 
    bin_ll(x, bw, k) = logpdf(Poisson(bw * f_fit(x)), k)
    sum(Base.Broadcast.broadcasted(bin_ll, bin_centers, bin_widths, counts))
end
export hist_loglike
