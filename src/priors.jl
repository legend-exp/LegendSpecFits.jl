# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).


"""
    weibull_from_mx(m::Real, x::Real, p_x::Real = 0.6827)::Weibull

Construct a Weibull distribution with a given median `m` and a given
`p_x`-quantile `x`.

Useful to construct priors for positive quantities.
"""
function weibull_from_mx(m::Real, x::Real, p_x::Real = 0.6827)
    α = log(-log(1-p_x) / log(2)) / log(x/m)
    θ = m / log(2)^(1/α)
    Weibull(α, θ)
end
export weibull_from_mx
