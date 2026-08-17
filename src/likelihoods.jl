# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    poisson_binll(λ::Real, k::Real)

Poisson log-likelihood of `k` counts for expectation `λ`; `-Inf` (instead of a `DomainError`)
for negative or non-finite `λ`, which the optimizer probes but must reject.
"""
poisson_binll(λ::Real, k::Real) = isfinite(λ) && λ >= zero(λ) ? logpdf(Poisson(λ), k) : oftype(float(λ), -Inf)


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
    bin_centers = midpoints(bin_edges)
    bin_widths = diff(bin_edges)
    # TODO: prevent fit functions from returning negative PDF values in the first place
    bin_ll(x, bw, k) = poisson_binll(bw * f_fit(x), k)
    sum(Base.Broadcast.broadcasted(bin_ll, bin_centers, bin_widths, counts))
end
export hist_loglike
