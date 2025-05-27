# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    LegendSpecFits.stabilize_dist(d::Distribution)

Return a numerically stable version of the distribution `d`.

Tries to ensure that the (log-)PDF of the result is finite over its whole
support.
"""
stabilize_dist(d::Distribution) = d

function stabilize_dist(d::Weibull)
    α = d.α
    ϵ = oftype(α, max(eps(typeof(α)), eps(Float32)))
    lthresh = isinf(logpdf(d, 0)) ? ϵ : zero(α)
    return truncated(d, lthresh, Inf)
end


"""
    weibull_from_mx(m::Real, x::Real, p_x::Real = 0.6827)::Weibull

Construct a Weibull distribution with a given median `m` and a given
`p_x`-quantile `x`.

Useful to construct priors for positive quantities.
"""
function weibull_from_mx(m::Real, x::Real, p_x::Real = 0.6827)
    α = log(-log(1-p_x) / log(2)) / log(x/m)
    θ = m / log(2)^(1/α)
    return stabilize_dist(Weibull(α, θ))
end
export weibull_from_mx
