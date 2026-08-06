# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

# Shape measures are intensity components of spectral models. They are
# measures over the real line with Lebesgue base measure. Peak shapes have
# unit total mass, background shapes have infinite mass over ℝ, both are
# combined into model measures via `+` and `*` (BAT measure combinators).


"""
    abstract type SpectralShape <: BAT.BATMeasure

Supertype of the spectral shape measures.
"""
abstract type SpectralShape <: BAT.BATMeasure end

ValueShapes.varshape(::SpectralShape) = ScalarShape{Real}()

DensityInterface.logdensityof(m::SpectralShape, x::Real) = log(densityof(m, x))

MeasureBase.massof(::SpectralShape) = MeasureBase.UnknownMass()

# All spectral shapes implement `MeasureBase.smf` (the Stieltjes measure
# function, a CDF generalization determined only up to an additive
# constant), so that interval and bin masses can be computed exactly.

_std_normal_cdf(z::Real) = erfc(-z * invsqrt2) / 2

LinearAlgebra.normalize(m::SpectralShape) = _normalize_shape(m, massof(m))
_normalize_shape(m::SpectralShape, mass::Real) = mass ≈ 1 ? m : weightedmeasure(-log(mass), m)
_normalize_shape(m::SpectralShape, mass) = throw(ArgumentError("Cannot normalize $(nameof(typeof(m))), its total mass is not finite"))


"""
    GaussPeak(μ, σ) <: SpectralShape

Gaussian peak shape with unit total mass.
"""
struct GaussPeak{T<:Real} <: SpectralShape
    μ::T
    σ::T

    function GaussPeak(μ::T, σ::T) where {T<:Real}
        isfinite(σ) && σ > 0 || throw(ArgumentError("GaussPeak requires finite σ > 0"))
        new{T}(μ, σ)
    end
end
export GaussPeak

GaussPeak(μ::Real, σ::Real) = GaussPeak(promote(μ, σ)...)

DensityInterface.densityof(m::GaussPeak, x::Real) = gauss_pdf(x, m.μ, m.σ)

function DensityInterface.logdensityof(m::GaussPeak, x::Real)
    z = (x - m.μ) / m.σ
    -z^2 / 2 - log(m.σ) - log2π / 2
end

MeasureBase.smf(m::GaussPeak, x::Real) = _std_normal_cdf((x - m.μ) / m.σ)

# Mass `true` is the non-promoting multiplicative one (like LinearAlgebra.I):
MeasureBase.massof(::GaussPeak) = true


# Log-density of an exponentially modified Gaussian, Δ being the distance
# from μ along the tail direction. Same numerically stabilized branches as
# `ex_gauss_pdf`, but assert-free and safe from overflow in log space:
function _emg_logdensityof(Δ::Real, σ::Real, θ::Real)
    iszero(θ) && return -(Δ / σ)^2 / 2 - log(σ) - log2π / 2
    z = σ / θ - Δ / σ
    if z > 0
        -(Δ / σ)^2 / 2 - log(2 * θ) + log(erfcx(invsqrt2 * z))
    else
        (σ / θ)^2 / 2 - Δ / θ - log(2 * θ) + log(erfc(invsqrt2 * z))
    end
end


"""
    LowETailPeak(μ, σ, θ) <: SpectralShape

Exponentially modified Gaussian peak shape with unit total mass and
low-energy tail of scale `θ` (the LEGEND low-energy tail orientation).
The shape underlying [`lowEtail_peakshape`](@ref).
"""
struct LowETailPeak{T<:Real} <: SpectralShape
    μ::T
    σ::T
    θ::T

    function LowETailPeak(μ::T, σ::T, θ::T) where {T<:Real}
        isfinite(σ) && σ > 0 || throw(ArgumentError("LowETailPeak requires finite σ > 0"))
        isfinite(θ) && θ >= 0 || throw(ArgumentError("LowETailPeak requires finite θ ≥ 0"))
        new{T}(μ, σ, θ)
    end
end
export LowETailPeak

LowETailPeak(μ::Real, σ::Real, θ::Real) = LowETailPeak(promote(μ, σ, θ)...)

DensityInterface.densityof(m::LowETailPeak, x::Real) = exp(logdensityof(m, x))

DensityInterface.logdensityof(m::LowETailPeak, x::Real) = _emg_logdensityof(m.μ - x, m.σ, m.θ)

# EMG CDF via F(x) = Φ((x - μ)/σ) ∓ θ f(x), mirrored for the low-E tail:
function MeasureBase.smf(m::LowETailPeak, x::Real)
    Δ = m.μ - x
    _std_normal_cdf(-Δ / m.σ) + m.θ * exp(_emg_logdensityof(Δ, m.σ, m.θ))
end

MeasureBase.massof(::LowETailPeak) = true


"""
    HighETailPeak(μ, σ, θ) <: SpectralShape

Exponentially modified Gaussian peak shape with unit total mass and
high-energy tail of scale `θ`, the mirror image of [`LowETailPeak`](@ref).
The shape underlying [`highEtail_peakshape`](@ref).
"""
struct HighETailPeak{T<:Real} <: SpectralShape
    μ::T
    σ::T
    θ::T

    function HighETailPeak(μ::T, σ::T, θ::T) where {T<:Real}
        isfinite(σ) && σ > 0 || throw(ArgumentError("HighETailPeak requires finite σ > 0"))
        isfinite(θ) && θ >= 0 || throw(ArgumentError("HighETailPeak requires finite θ ≥ 0"))
        new{T}(μ, σ, θ)
    end
end
export HighETailPeak

HighETailPeak(μ::Real, σ::Real, θ::Real) = HighETailPeak(promote(μ, σ, θ)...)

DensityInterface.densityof(m::HighETailPeak, x::Real) = exp(logdensityof(m, x))

DensityInterface.logdensityof(m::HighETailPeak, x::Real) = _emg_logdensityof(x - m.μ, m.σ, m.θ)

function MeasureBase.smf(m::HighETailPeak, x::Real)
    Δ = x - m.μ
    _std_normal_cdf(Δ / m.σ) - m.θ * exp(_emg_logdensityof(Δ, m.σ, m.θ))
end

MeasureBase.massof(::HighETailPeak) = true


"""
    GaussianStep(μ, σ) <: SpectralShape

Gaussian-smoothed step shape with unit plateau height below `μ`,
`erfc((x - μ) / (√2 σ)) / 2`. Infinite total mass over ℝ.
"""
struct GaussianStep{T<:Real} <: SpectralShape
    μ::T
    σ::T

    function GaussianStep(μ::T, σ::T) where {T<:Real}
        isfinite(σ) && σ > 0 || throw(ArgumentError("GaussianStep requires finite σ > 0"))
        new{T}(μ, σ)
    end
end
export GaussianStep

GaussianStep(μ::Real, σ::Real) = GaussianStep(promote(μ, σ)...)

DensityInterface.densityof(m::GaussianStep, x::Real) = step_gauss(-x, -m.μ, m.σ)

function DensityInterface.logdensityof(m::GaussianStep, x::Real)
    z = (x - m.μ) / (sqrt2 * m.σ)
    z > 0 ? log(erfcx(z) / 2) - z^2 : log(erfc(z) / 2)
end

# Antiderivative of Φ(-t): t Φ(-t) - φ(t), the additive constant is free:
function MeasureBase.smf(m::GaussianStep, x::Real)
    t = (x - m.μ) / m.σ
    m.σ * (t * _std_normal_cdf(-t) - exp(-t^2 / 2) / sqrt2π)
end


"""
    PolynomialShape(p0, p1, ...; x0 = 0) <: SpectralShape

Polynomial shape with density `p0 + p1 (x - x0) + p2 (x - x0)^2 + ...`.
The coefficients must keep the density non-negative on the fit region. An
expansion point `x0` near the region of interest improves the numerical
conditioning of the coefficients.
"""
struct PolynomialShape{T<:Real,N} <: SpectralShape
    coeffs::NTuple{N,T}
    x0::T
end
export PolynomialShape

function PolynomialShape(coeffs::Real...; x0::Real = false)
    p = promote(coeffs..., x0)
    PolynomialShape(Base.front(p), last(p))
end

DensityInterface.densityof(m::PolynomialShape, x::Real) = evalpoly(x - m.x0, m.coeffs)

function MeasureBase.smf(m::PolynomialShape{T,N}, x::Real) where {T,N}
    u = x - m.x0
    u * evalpoly(u, ntuple(i -> m.coeffs[i] / i, Val(N)))
end


"""
    ExpDecay(μ, τ) <: SpectralShape

Exponential shape with density `exp(-(x - μ) / τ)`, unit height at `μ` and
decaying towards larger values. Infinite total mass over ℝ.
"""
struct ExpDecay{T<:Real} <: SpectralShape
    μ::T
    τ::T

    function ExpDecay(μ::T, τ::T) where {T<:Real}
        τ > 0 || throw(ArgumentError("ExpDecay requires τ > 0 (τ = Inf yields a constant density)"))
        new{T}(μ, τ)
    end
end
export ExpDecay

ExpDecay(μ::Real, τ::Real) = ExpDecay(promote(μ, τ)...)

DensityInterface.densityof(m::ExpDecay, x::Real) = exp(-(x - m.μ) / m.τ)

DensityInterface.logdensityof(m::ExpDecay, x::Real) = -(x - m.μ) / m.τ

MeasureBase.smf(m::ExpDecay, x::Real) = isinf(m.τ) ? x - m.μ : -m.τ * exp(-(x - m.μ) / m.τ)
