# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    LegendSpecFits.stabilize_dist(d::Distribution)

Returns a numerically stable version of the distribution `d`.

For a `Weibull` distribution with infinite log-density at zero, the result is
truncated to start at a small positive value. Other distributions are returned
unchanged.
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

Constructs a `Weibull` distribution with median `m` whose `p_x`-quantile is
`x`. The default `p_x = 0.6827` corresponds to a one-sigma quantile.

The result is suited as a prior for positive quantities.
"""
function weibull_from_mx(m::Real, x::Real, p_x::Real = 0.6827)
    α = log(-log(1-p_x) / log(2)) / log(x/m)
    θ = m / log(2)^(1/α)
    return stabilize_dist(Weibull(α, θ))
end
export weibull_from_mx
