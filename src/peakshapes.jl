# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    LegendSpecFits.gauss_pdf(x::Real, μ::Real, σ::Real)

Evaluates the probability density function of a normal distribution with mean
`μ` and standard deviation `σ` at `x`. Equivalent to `pdf(Normal(μ, σ), x)`.
"""
gauss_pdf(x::Real, μ::Real, σ::Real) = inv(σ * sqrt2π) * exp(-((x - μ) / σ)^2  / 2)

"""
    ex_gauss_pdf(x::Real, μ::Real, σ::Real, θ::Real)

Evaluates the probability density function of an
[exponentially modified Gaussian distribution](https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution)
with Gaussian mean `μ` and standard deviation `σ` and exponential scale `θ` at `x`.

It is the density of the random variable
`rand(Normal(μ, σ)) + rand(Exponential(θ))`. The function throws an
`AssertionError` if the result is not finite.
"""
function ex_gauss_pdf(x::Real, μ::Real, σ::Real, θ::Real)
    R = float(promote_type(typeof(x), typeof(σ), typeof(θ)))
    x_μ = x - μ

    y = if iszero(θ) && iszero(σ)
        iszero(x_μ) ? one(R) : zero(R)
    elseif θ < σ * R(10^-6)
        # Use asymptotic form for very small θ - necessary?
        R(inv(sqrt2π) * exp(-(x_μ/σ)^2 / 2) / (σ + x_μ * θ / σ))
    elseif σ/θ - x_μ/σ < 0
        # Original:
        R(inv(2*θ) * exp((σ/θ)^2/2 - x_μ/θ) * erfc(invsqrt2 * (σ/θ - x_μ/σ)))
    else
        # More stable, numerically, for small values of θ:
        R(inv(sqrt2π) * exp(-(x_μ/σ)^2 / 2)/θ * sqrthalfπ * erfcx(invsqrt2 * (σ/θ - x_μ/σ)))
    end
    @assert isfinite(y)
    return y
end

"""
    step_gauss(x::Real, μ::Real, σ::Real)

Evaluates the convolution of a Heaviside step function with the
PDF of `Normal(μ, σ)` at `x`, i.e. the cumulative distribution function of
`Normal(μ, σ)`. It rises from 0 to 1 around `μ`.

The result is not a probability density because it is not normalizable.
"""
step_gauss(x::Real, μ::Real, σ::Real) = erfc( (μ-x) / (sqrt2 * σ) ) / 2

"""
    linear_function(x::Real, slope::Real, intercept::Real)

Evaluates the linear function `slope * x + intercept` at `x`.
"""
function linear_function(
    x::Real, slope::Real, intercept::Real
)
    return slope * x + intercept
end
export linear_function

"""
    exponential_decay(x::Real, amplitude::Real, decay::Real, offset::Real)

Evaluates `amplitude * exp(-decay * x) + offset` at `x`.
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
        background::Real;
        skew_fraction_highE::Real = 0.0, skew_width_highE::Real = 0.0,
        background_kwargs...
    )

Evaluates the shape of a gamma peak in an energy spectrum at `x`. The shape is
the sum of [`signal_peakshape`](@ref), [`lowEtail_peakshape`](@ref),
[`highEtail_peakshape`](@ref) and [`background_peakshape`](@ref).

## Arguments
* `x`: Energy at which the shape is evaluated.
* `μ`, `σ`: Mean and standard deviation of the Gaussian signal peak.
* `n`: Total number of counts in the peak (Gaussian part plus tails).
* `step_amplitude`: Amplitude of the step-like background from Compton-scattered gammas.
* `skew_fraction`, `skew_width`: Fraction of `n` in the low-energy tail and its
  exponential scale relative to `μ` (the scale is `skew_width * μ`).
* `background`: Energy-independent background level.

## Keywords
* `skew_fraction_highE = 0.0`, `skew_width_highE = 0.0`: Fraction and relative
  exponential scale of the high-energy tail. The default disables the tail.
* `background_kwargs...`: Passed to [`background_peakshape`](@ref)
  (`background_slope`, `background_exp`, `background_center`).
"""
function gamma_peakshape(
    x::Real, μ::Real, σ::Real, n::Real,
    step_amplitude::Real, skew_fraction::Real, skew_width::Real,
    background::Real;  
    skew_fraction_highE::Real = 0.0, skew_width_highE::Real= 0.0, 
    background_kwargs... 
)
    return  signal_peakshape(x, μ, σ, n, skew_fraction; skew_fraction_highE = skew_fraction_highE) +
            lowEtail_peakshape(x, μ, σ, n, skew_fraction, skew_width) +
            highEtail_peakshape(x, μ, σ, n, skew_fraction_highE, skew_width_highE) +
            background_peakshape(x, µ, σ, step_amplitude, background; background_kwargs...)
end
export gamma_peakshape

"""
    signal_peakshape(
        x::Real, μ::Real, σ::Real, n::Real,
        skew_fraction::Real; skew_fraction_highE::Real = 0.0
    )

Evaluates the Gaussian signal part of a gamma peak at `x`:
`n * (1 - skew_fraction - skew_fraction_highE)` times the density of
`Normal(μ, σ)`. Returns zero if `σ` is zero.
"""
function signal_peakshape(x::Real, μ::Real, σ::Real, n::Real, skew_fraction::Real;  skew_fraction_highE::Real = 0.0)
    return iszero(σ) ? zero(x) : n * (1 - skew_fraction - skew_fraction_highE) * gauss_pdf(x, μ, σ)
end
export signal_peakshape

"""
    background_peakshape(
        x::Real, μ::Real, σ::Real,
        step_amplitude::Real, background::Real;
        background_slope::Real = 0.0, background_exp = 0.0, background_center::Real = μ
    )

Evaluates the background part of a gamma peak at `x`:

```math
A\\,\\mathrm{S}(x; \\mu, \\sigma) + b\\,e^{-d\\,(x - x_c)} + s\\,(x - x_c)\\,,
```

where ``A`` is `step_amplitude`, ``b`` is `background`, ``d`` is
`background_exp`, ``s`` is `background_slope` and ``x_c`` is
`background_center`. The step ``\\mathrm{S}`` is a Gaussian-smoothed step with
width `σ` that falls from 1 to 0 across `μ` with increasing `x` (more
background below the peak). With the default keywords the exponential term
reduces to the constant `background`.
"""
function background_peakshape(
    x::Real, μ::Real, σ::Real, 
    step_amplitude::Real, background::Real; 
    background_slope::Real = 0.0, background_exp = 0.0, background_center::Real = μ
)
    step_amplitude * step_gauss(-x, -μ, σ) + background * exp(-background_exp * (x - background_center)) + background_slope * (x - background_center)
end
export background_peakshape

"""
    lowEtail_peakshape(
        x::Real, μ::Real, σ::Real, n::Real,
        skew_fraction::Real, skew_width::Real
    )

Evaluates the low-energy tail of a gamma peak at `x`: `n * skew_fraction` times
an exponentially modified Gaussian that extends toward lower energies, with
exponential scale `skew_width * μ`. Returns zero if `σ` is zero.
"""
function lowEtail_peakshape(
    x::Real, μ::Real, σ::Real, n::Real,
    skew_fraction::Real, skew_width::Real
)
    skew = skew_width * μ
    return iszero(σ) ? zero(x) : n * skew_fraction * ex_gauss_pdf(-x, -μ, σ, skew)
end
export lowEtail_peakshape

"""
    highEtail_peakshape(
        x::Real, μ::Real, σ::Real, n::Real,
        skew_fraction_h::Real, skew_width_h::Real
    )

Evaluates the high-energy tail of a gamma peak at `x`: `n * skew_fraction_h`
times an exponentially modified Gaussian that extends toward higher energies,
with exponential scale `skew_width_h * μ`. Returns zero if `σ` is zero.
"""
function highEtail_peakshape(
    x::Real, μ::Real, σ::Real, n::Real,
    skew_fraction_h::Real, skew_width_h::Real
)
    skew = skew_width_h * μ
    return return iszero(σ) ? zero(x) : n * skew_fraction_h * ex_gauss_pdf(x, μ, σ, skew)
end
export highEtail_peakshape

"""
    ex_step_gauss(x::Real, l::Real, k::Real, t::Real, d::Real)

Evaluates `(exp(k * (x - l)) + d) / (exp((x - l) / t) + l)` at `x`.

This is the energy dependence of the A/E resolution model of GERDA
(Eq. (1) in EPJC 73 (2013) 2583).
"""
function ex_step_gauss(
    x::Real, l::Real, k::Real, 
    t::Real, d::Real
)
    # `+ l` as published in GERDA Eq. (1), EPJC 73 (2013) 2583 (l ≈ 1 in
    # normalized A/E, so nearly identical to a logistic-style `+ 1`):
    return (exp(k*(x-l)) + d) / (exp((x-l)/t) + l)
end
export ex_step_gauss

"""
    aoe_compton_peakshape(
        x::Real, μ::Real, σ::Real, n::Real,
        background::Real, δ::Real
    )

Evaluates the A/E distribution of events in a Compton region at `x`: a Gaussian
single-site-event (SSE) peak with mean `μ`, standard deviation `σ` and
amplitude `n`, plus a multi-site-event (MSE) background `background` modeled as
an exponentially modified Gaussian with scale `δ` that extends toward lower A/E.
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

Evaluates the signal part of [`aoe_compton_peakshape`](@ref) at `x`: a Gaussian
single-site-event peak with mean `μ`, standard deviation `σ` and amplitude `n`.
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

Evaluates the background part of [`aoe_compton_peakshape`](@ref) at `x`: an
exponentially modified Gaussian (mirrored, tail toward lower A/E) with amplitude
`background`, mean `μ`, standard deviation `σ` and scale `δ`.
"""
function aoe_compton_background_peakshape(
    x::Real, μ::Real, σ::Real,
    background::Real, δ::Real
)
    return background * ex_gauss_pdf(-x, -μ, σ, δ)
end
export aoe_compton_background_peakshape

"""
    double_gaussian(
        x::Real, μ1::Real, σ1::Real, n1::Real, 
        μ2::Real, σ2::Real, n2::Real
    )

Evaluates the sum of two Gaussian densities at `x`, with means `μ1`, `μ2`,
standard deviations `σ1`, `σ2` and amplitudes `n1`, `n2`.
"""
function double_gaussian(
    x::Real, μ1::Real, σ1::Real, n1::Real, 
    μ2::Real, σ2::Real, n2::Real
)
    return n1 * gauss_pdf(x, μ1, σ1) + n2 * gauss_pdf(x, μ2, σ2)  
end
export double_gaussian


###############################################################
# additional peakshape functions for two EMG background
###############################################################

"""
    two_emg_aoe_compton_peakshape(
        x::Real,
        μ::Real, σ::Real, n::Real, background::Real, δ::Real,
        μ2::Real, σ2::Real, background2_fraction::Real, δ2::Real
    )

Evaluates the A/E distribution of a Compton region with two background
components at `x`: a Gaussian signal peak (`μ`, `σ`, `n`) plus `background`
times a mixture of two mirrored exponentially modified Gaussians. The first has
parameters `μ`, `σ`, `δ` and weight `1 - background2_fraction`, the second has
`μ2`, `σ2`, `δ2` and weight `background2_fraction`. Returns zero if `σ` or
`σ2` is zero.
"""
function two_emg_aoe_compton_peakshape( # total fit function with two EMGs
    x::Real, 
    μ::Real, σ::Real, n::Real, background::Real, δ::Real, 
    μ2::Real, σ2::Real, background2_fraction::Real, δ2::Real
)
    return iszero(σ) || iszero(σ2) ? zero(x) : n * gauss_pdf(x, μ, σ) + background * ( (1 - background2_fraction) * ex_gauss_pdf(-x, -μ, σ, δ) + background2_fraction * ex_gauss_pdf(-x, -μ2, σ2, δ2))
end
export two_emg_aoe_compton_peakshape

"""
    two_emg_aoe_compton_background_peakshape(
        x::Real,
        μ::Real, σ::Real, background::Real, δ::Real,
        μ2::Real, σ2::Real, background2_fraction::Real, δ2::Real
    )

Evaluates the background part of [`two_emg_aoe_compton_peakshape`](@ref) at `x`.
"""
function two_emg_aoe_compton_background_peakshape( # background function with two EMGs
    x::Real, 
    μ::Real, σ::Real, background::Real, δ::Real,
    μ2::Real, σ2::Real, background2_fraction::Real, δ2::Real
)
    return background * ( (1 - background2_fraction) * ex_gauss_pdf(-x, -μ, σ, δ) + background2_fraction * ex_gauss_pdf(-x, -μ2, σ2, δ2))
end
