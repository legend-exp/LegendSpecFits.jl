# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    HistLogLike(h::Histogram{<:Real,1})
    (ll::HistLogLike)(f_fit)

Calculates the binned Poisson log-likelihood of the histogram `h`, with all
terms that do not depend on the model computed once.

The constructor stores the bin centers, bin widths, counts and the terms
`loggamma(k + 1)` of the counts `k`. Calling `ll(f_fit)` returns the same value
as `hist_loglike(f_fit, h)`. A `HistLogLike` is constructed once and reused when
the likelihood is evaluated repeatedly, for example inside an optimizer.

The expected count in each bin is `bin_width * f_fit(bin_center)`. `f_fit` must
accept all bin centers of `h`.

## Arguments
* `h::Histogram{<:Real,1}`: Histogram whose counts are compared with the model.
* `f_fit`: Callable that returns the density at a position `x`.

## Returns
The log-likelihood as a number. It is `-Inf` if a bin count is negative or not
an integer. A negative expected count throws a `DomainError`, as `Poisson(λ)`
does.
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

Calculates the Poisson log-likelihood of the bin counts of the histogram `h`
given the density `f_fit(x)`. `f_fit` must accept all values `x` on the
horizontal axis of `h`.

The expected count in each bin is `f_fit` at the bin center times the bin
width (midpoint rule).

When the same histogram is evaluated many times, construct a
[`HistLogLike`](@ref) once instead.
"""
hist_loglike(f_fit::Base.Callable, h::Histogram{<:Real,1}) = HistLogLike(h)(f_fit)
export HistLogLike
export hist_loglike
