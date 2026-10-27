# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    HistLogLike(h::Histogram{<:Real,1})

Binned Poisson log-likelihood of the histogram `h`, with the bin centers,
bin widths, counts and the parameter-independent terms `loggamma(k + 1)`
computed once. Calling `HistLogLike(h)(f_fit)` returns the same value as
`hist_loglike(f_fit, h)`, so a `HistLogLike` should be constructed once and
reused when the likelihood is evaluated repeatedly, e.g. inside an optimizer.

The expected count in each bin is `bin_width * f_fit(bin_center)`. A negative
expected count throws a `DomainError`, like `Poisson(λ)`. Counts that are
negative or not integer have a log-likelihood of `-Inf`.
"""
struct HistLogLike{C<:AbstractVector,W<:AbstractVector,K<:AbstractVector,L<:AbstractVector}
    bin_centers::C
    bin_widths::W
    counts::K
    loggamma_counts::L
end

function HistLogLike(h::Histogram{<:Real,1})
    bin_edges = first(h.edges)
    counts = h.weights
    # Infinity makes the log-likelihood -Inf, as for `logpdf(Poisson(λ), k)` with an invalid count k
    loggamma_counts = map(k -> k >= 0 && isinteger(k) ? loggamma(k + 1) : oftype(float(k), Inf), counts)
    HistLogLike(midpoints(bin_edges), diff(bin_edges), counts, loggamma_counts)
end

function (ll::HistLogLike)(f_fit::Base.Callable)
    function bin_ll(x, bw, k, lgk)
        λ = bw * f_fit(x)
        # TODO: Prevent fit functions from returning negative PDF values
        λ >= zero(λ) || throw(DomainError(λ, "Poisson: the condition λ >= zero(λ) is not satisfied."))
        xlogy(k, λ) - λ - lgk
    end
    sum(Base.Broadcast.broadcasted(bin_ll, ll.bin_centers, ll.bin_widths, ll.counts, ll.loggamma_counts))
end


"""
    hist_loglike(f_fit::Base.Callable, h::Histogram{<:Real,1})

Calculate the Poisson log-likelihood of a fit function `f_fit(x)` and a
histogram `h`. `f_fit` must accept all values `x` on the horizontal axis
of the histogram.

Currently uses a simple midpoint-rule integration of `f_fit` over the
bins of `h`.

When the same histogram is evaluated many times, construct a
[`HistLogLike`](@ref) once instead.
"""
hist_loglike(f_fit::Base.Callable, h::Histogram{<:Real,1}) = HistLogLike(h)(f_fit)
export HistLogLike
export hist_loglike
