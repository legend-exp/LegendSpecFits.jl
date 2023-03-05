# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    LegendSpecFits.gauss_pdf(x::Real, μ::Real, σ::Real)

Equivalent to `pdf(Normal(μ, σ), x)`
"""
gauss_pdf(x::Real, μ::Real, σ::Real) = inv(σ * sqrt2π) * exp(-((x - μ) / σ)^2  / 2)


"""
    ex_gauss_pdf(x::Real, μ::Real, σ::Real, θ::Real)

The PDF of an
[Exponentially modified Gaussian distribution](https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution)
with Gaussian parameters `μ`, `σ` and exponential scale `θ` at `x`.

It is the PDF of the distribution that descibes the random process
`rand(Normal(μ, σ)) + rand(Exponential(θ))`.
"""
function ex_gauss_pdf(x::Real, μ::Real, σ::Real, θ::Real)
    R = float(promote_type(typeof(x), typeof(σ), typeof(θ)))
    x_μ = x - μ
    gauss_pdf_value = inv(σ * sqrt2π) * exp(-(x_μ/σ)^2 / 2)

    y = if θ < σ * R(10^-6)
        # Use asymptotic form for very small θ - necessary?
        R(gauss_pdf_value / (1 + x_μ * θ / σ^2))
    elseif σ/θ - x_μ/σ < 0
        # Original:
        R(inv(2*θ) * exp((σ/θ)^2/2 - x_μ/θ) * erfc(invsqrt2 * (σ/θ - x_μ/σ)))
    else
        # More stable, numerically, for small values of θ:
        R(gauss_pdf_value * σ/θ * sqrthalfπ * erfcx(invsqrt2 * (σ/θ - x_μ/σ)))
    end
    @assert !isnan(y) && !isinf(y)
    return y
end


"""
    step_gauss(x::Real, μ::Real, σ::Real)

Evaluates the convulution of a Heaviside step function and the
PDF of `Normal(μ, σ)` at `x`.

The result does not correspond to a PDF as it is not normalizable.
"""
step_gauss(x::Real, μ::Real, σ::Real) = erfc( (μ-x) / (sqrt2 * σ) ) / 2


"""
    gamma_peakshape(
        x::Real, μ::Real, σ::Real, n::Real,
        step_amplitude::Real, skew_fraction::Real, skew_width::Real
    )
    
Describes the shape of a typical gamma peak in a detector.
"""
function gamma_peakshape(
    x::Real, μ::Real, σ::Real, n::Real,
    step_amplitude::Real, skew_fraction::Real, skew_width::Real
)
    skew = skew_width * μ
    return n * (
            (1 - skew_fraction) * gauss_pdf(x, μ, σ) +
            skew_fraction * ex_gauss_pdf(-x, -μ, σ, skew)
        ) + step_amplitude * step_gauss(-x, -μ, σ);
end
export gamma_peakshape
