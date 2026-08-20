# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    BinnedPoissonProcess(intensity::AbstractMeasure, edges::AbstractVector{<:Real}) <: MeasureBase.AbstractMeasure

Binned view of a Poisson point process with the given intensity measure:
its variates are vectors of non-negative integer bin counts for the given
bin edges (`logdensityof` assumes valid counts, [`binned_poisson_likelihood`](@ref)
validates them). The count in each bin is Poisson-distributed, with the
intensity mass in the bin as expectation, computed via analytic
`MeasureBase.smf` differences where available (all spectral shapes,
distributed over weighted measures and superpositions) and via Simpson's
rule otherwise.
`Distribution` intensities are converted to measures automatically.
"""
struct BinnedPoissonProcess{M<:AbstractMeasure,E<:AbstractVector{<:Real}} <: AbstractMeasure
    intensity::M
    edges::E
end
export BinnedPoissonProcess

BinnedPoissonProcess(intensity::Distribution, edges::AbstractVector{<:Real}) = BinnedPoissonProcess(batmeasure(intensity), edges)

function DensityInterface.logdensityof(m::BinnedPoissonProcess, counts::AbstractVector{<:Real})
    λ = _bin_masses(m.intensity, m.edges)
    sum(Base.Broadcast.broadcasted(_poisson_logpdf, λ, counts))
end


"""
    binned_poisson_likelihood(model, h::Histogram{<:Real,1})

Binned Poisson likelihood of the observed histogram `h`: the likelihood of
`BinnedPoissonProcess(model(p), h.edges)` given the observed bin counts
`h.weights`, `model` being a Markov kernel that maps parameters to an
intensity measure (e.g. a `@pf` property function combining shape measures,
see [`GaussPeak`](@ref) etc.).

A fixed intensity measure instead of `model` yields a parameter-independent
likelihood.

The bin counts of `h` must be non-negative integers: a weighted histogram
is not Poisson-distributed data and requires a different statistical
treatment.
"""
binned_poisson_likelihood(model, h::Histogram{<:Real,1}) =
    likelihoodof(ffcomp(Base.Fix2(BinnedPoissonProcess, only(h.edges)), model), _checked_counts(h.weights))

binned_poisson_likelihood(intensity::AbstractMeasure, h::Histogram{<:Real,1}) =
    likelihoodof(Returns(BinnedPoissonProcess(intensity, only(h.edges))), _checked_counts(h.weights))

export binned_poisson_likelihood

function _checked_counts(counts::AbstractVector{<:Real})
    all(k -> k >= 0 && isinteger(k), counts) || throw(ArgumentError(
        "Poisson bin counts must be non-negative integers, a weighted histogram requires a different statistical treatment"
    ))
    return counts
end


# Expected mass of `m` in each bin. Weighted measures and superpositions
# are decomposed, leaves with a Stieltjes measure function (all spectral
# shapes) are integrated via analytic smf differences (exact up to
# floating-point cancellation for bins deep in a peak tail), others fall
# back to Simpson's rule on the density (which can miss narrow, unresolved
# peaks entirely):
function _bin_masses(m, edges::AbstractVector{<:Real})
    _bin_masses_impl(m, edges, smf(m, first(edges)))
end

_bin_masses_impl(m, edges::AbstractVector{<:Real}, ::Real) = diff(map(Base.Fix1(smf, m), edges))
_bin_masses_impl(m, edges::AbstractVector{<:Real}, ::MeasureBase.NoSMF) = _simpson_bin_masses(m, edges)

function _simpson_bin_masses(m, edges::AbstractVector{<:Real})
    lo = edges[begin:end-1]
    hi = edges[begin+1:end]
    f = Base.Fix1(densityof, m)
    @. (hi - lo) / 6 * (f(lo) + 4 * f((lo + hi) / 2) + f(hi))
end

_bin_masses(m::BAT.BATWeightedMeasure, edges::AbstractVector{<:Real}) = exp(m.logweight) * _bin_masses(m.base, edges)
_bin_masses(m::BAT.BATSuperpositionMeasure, edges::AbstractVector{<:Real}) = sum(Base.Fix2(_bin_masses, edges), m.components)
_bin_masses(m::MeasureBase.WeightedMeasure, edges::AbstractVector{<:Real}) = exp(m.logweight) * _bin_masses(m.base, edges)
_bin_masses(m::MeasureBase.SuperpositionMeasure, edges::AbstractVector{<:Real}) = sum(Base.Fix2(_bin_masses, edges), m.components)


# Poisson log-pmf with the k = 0 limit at λ = 0. Any negative rate is
# invalid and must yield -Inf, so that optimizers reject intensities that
# turn negative (e.g. via polynomial shape components) instead of being
# rewarded for them in empty bins:
_poisson_logpdf(λ::Real, k::Real) =
    λ > 0 ? k * log(λ) - λ - loggamma(k + 1) :
    (iszero(λ) && iszero(k) ? -float(λ) : oftype(float(λ), -Inf))
