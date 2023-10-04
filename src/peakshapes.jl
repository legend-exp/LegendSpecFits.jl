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
    linear_function(x::Real, slope::Real, intercept::Real)

Evaluates a linear function at `x` with parameters `slope` and `intercept`.
"""
function linear_function(
    x::Real, slope::Real, intercept::Real
)
    return slope * x + intercept
end
export linear_function

"""
    exponential_decay(x::Real, amplitude::Real, decay::Real, offset::Real)

Evaluates an exponential decay function at `x` with parameters `amplitude`, `decay` and `offset`.
"""
function exponential_decay(
    x::Real, amplitude::Real, decay::Real, offset::Real 
)
    return amplitude * exp(-decay * x) + offset
end
export exponential_decay

"""
    gamma_peakshape(
        x::Real, μ::Real, σ::Real, n::Real,
        step_amplitude::Real, skew_fraction::Real, skew_width::Real,
        background::Real
    )
    
Describes the shape of a typical gamma peak in a detector with a flat background.
"""
function gamma_peakshape(
    x::Real, μ::Real, σ::Real, n::Real,
    step_amplitude::Real, skew_fraction::Real, skew_width::Real,
    background::Real
)
    skew = skew_width * μ
    return n * (
            (1 - skew_fraction) * gauss_pdf(x, μ, σ) +
            skew_fraction * ex_gauss_pdf(-x, -μ, σ, skew)
        ) + step_amplitude * step_gauss(-x, -μ, σ) + background;
end
export gamma_peakshape

"""
    signal_peakshape(
        x::Real, μ::Real, σ::Real, n::Real,
        skew_fraction::Real, skew_width::Real,
    )
    
Describes the signal part of the shape of a typical gamma peak in a detector.
"""
function signal_peakshape(
    x::Real, μ::Real, σ::Real, n::Real,
    skew_fraction::Real
)
    return n * (1 - skew_fraction) * gauss_pdf(x, μ, σ)
end
export signal_peakshape

"""
    background_peakshape(
        x::Real, μ::Real, σ::Real, n::Real,
        skew_fraction::Real, skew_width::Real,
    )
    
Describes the background part of the shape of a typical gamma peak in a detector.
"""
function background_peakshape(
    x::Real, μ::Real, σ::Real, 
    step_amplitude::Real, background::Real
)
    return gamma_peakshape(x, μ, σ, 0.0, step_amplitude, 0.0, 0.0, background)
end
export background_peakshape

"""
    lowEtail_peakshape(
        x::Real, μ::Real, σ::Real, n::Real,
        skew_fraction::Real, skew_width::Real,
    )
    
Describes the low-E signal tail part of the shape of a typical gamma peak in a detector.
"""
function lowEtail_peakshape(
    x::Real, μ::Real, σ::Real, n::Real,
    skew_fraction::Real, skew_width::Real
)
    skew = skew_width * μ
    return n * skew_fraction * ex_gauss_pdf(-x, -μ, σ, skew)
end
export lowEtail_peakshape


"""
    ex_step_gauss(x::Real, l::Real, k::Real, t::Real, d::Real)

Evaluates an extended step gauss model at `x` with parameters `l`, `k`, `t` and `d`.

"""
function ex_step_gauss(
    x::Real, l::Real, k::Real, 
    t::Real, d::Real
)
    return (exp(k*(x-l)) + d) / (exp((x-l)/t) + l)
end
export ex_step_gauss

"""
    aoe_compton_peakshape(
        x::Real, μ::Real, σ::Real, n::Real,
        background::Real, δ::Real
    )

Describes the shape of a typical A/E Compton peak in a detector as a gaussian SSE peak and a step like background for MSE events.
"""
function aoe_compton_peakshape(
    x::Real, μ::Real, σ::Real, n::Real,
    background::Real, δ::Real
)
    return n * gauss_pdf(x, μ, σ) +
        background * ex_gauss_pdf(-x, -μ, σ, δ)
end
export aoe_compton_peakshape

"""
    aoe_compton_signal_peakshape(
        x::Real, μ::Real, σ::Real, n::Real
    )

Describes the signal shape of a typical A/E Compton peak in a detector as a gaussian SSE peak.
"""
function aoe_compton_signal_peakshape(
    x::Real, μ::Real, σ::Real, n::Real
)
    return n * gauss_pdf(x, μ, σ)
end
export aoe_compton_signal_peakshape

"""
    aoe_compton_background_peakshape(
        x::Real, μ::Real, σ::Real,
        background::Real, δ::Real
    )

Describes the background shape of a typical A/E Compton peak in a detector as a step like background for MSE events.
"""
function aoe_compton_background_peakshape(
    x::Real, μ::Real, σ::Real,
    background::Real, δ::Real
)
    return background * ex_gauss_pdf(-x, -μ, σ, δ)
end
export aoe_compton_background_peakshape


