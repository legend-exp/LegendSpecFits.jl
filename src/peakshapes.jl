# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    LegendSpecFits.gauss_pdf(x::Real, μ::Real, σ::Real)

Equivalent to `pdf(Normal(μ, σ), x)`

# Arguments
    * `x`: Data
    * `μ`: Mean values
    * `σ`: Sigma values
"""
gauss_pdf(x::Real, μ::Real, σ::Real) = inv(σ * sqrt2π) * exp(-((x - μ) / σ)^2  / 2)


"""
    ex_gauss_pdf(x::Real, μ::Real, σ::Real, θ::Real)

The PDF of an
[Exponentially modified Gaussian distribution](https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution)
with Gaussian parameters `μ`, `σ` and exponential scale `θ` at `x`.

It is the PDF of the distribution that descibes the random process
`rand(Normal(μ, σ)) + rand(Exponential(θ))`.

# Arguments
    * `x`: x values
    * `μ`: Mean values
    * `σ`: Sigma values (standard deviation)
    * `θ`: Angle values

# Returns
    * Exponentially modified gaussian probability distribution function

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

Evaluates the convulution of a Heaviside step function and the
PDF of `Normal(μ, σ)` at `x`.

The result does not correspond to a PDF as it is not normalizable.

# Arguments
    * `x`: x values
    * `μ`: Mean values
    * `σ`: Standard deviations

"""
step_gauss(x::Real, μ::Real, σ::Real) = erfc( (μ-x) / (sqrt2 * σ) ) / 2


"""
    linear_function(x::Real, slope::Real, intercept::Real)

Evaluates a linear function at `x` with parameters `slope` and `intercept`.

# Arguments
    * `x`: x value
    * `slope`: slope of function
    * `intercept`: y-intercept of the linear function

# Returns
    * returns the corresponding y-value of the linear function at a given x-value.

# Example
    linear_function(2, 4, 3)

    will return:
    11

    from the calculation: 4 * 2 + 3.
"""

function linear_function(x::Real, slope::Real, intercept::Real)

    return slope * x + intercept
end
export linear_function


"""
    exponential_decay(x::Real, amplitude::Real, decay::Real, offset::Real)

Evaluates an exponential decay function at `x` with parameters `amplitude`, `decay` and `offset`.

# Arguments
    * `x`: x values
    * `amplitude`: amplitude of decay
    * `decay`: Rate of decay
    * `offset`: Offset 

# Returns
    * Evaluated exponential decay function 

"""

function exponential_decay(x::Real, amplitude::Real, decay::Real, offset::Real )
    return amplitude * exp(-decay * x) + offset
end
export exponential_decay

"""
    gamma_peakshape(
    x::Real, μ::Real, σ::Real, n::Real,
    step_amplitude::Real, skew_fraction::Real, skew_width::Real,
    background::Real;  
    skew_fraction_highE::Real = 0.0, skew_width_highE::Real= 0.0, 
    background_kwargs... 
)
    
Standard gamma peakshape: Describes the shape of a typical gamma peak in a detector.
Components: 
- Gaussian signal peak with `μ`, `σ`, `n - skew_fraction - skew_fraction_highE`
- low-energy tail: `skew_fraction`, `skew_width`
- high-energy tail: `skew_fraction_highE`, `skew_width_highE` (optional, default off)
- background:
    - energy-independent `background`
    - step-function scaled with `step_amplitude` from Compton scattered gammas
    - linear slope: `background_slope` (optional, default off)
    - exponential decay: `background_exp` (optional, default off)

# Arguments
    * `x`: x values
    * `μ`: Mean values
    * `σ`: Standard deviation
    * `n`: Counts
    * `step_amplitude`: Step amplitude
    * `skew_fraction`: Skew fraction
    * `skew_width`: Width of skew
    * `background`: Energy-independent background

# Keywords  
    * `skew_fraction_highE`: High energy skew fraction
    * `skew_width_highE`: High energy skew width

# Returns
    * `signal_peakshape`: Signal peakshape
    * `lowEtail_peakshape`: Low energy tail peakshape
    * `highEtail_peakshape`: high energy tail peakshape
    * `background_peakshape`: Background peakshape

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
    signal_peakshape(x::Real, μ::Real, σ::Real, n::Real, skew_fraction::Real;  skew_fraction_highE::Real = 0.0)
    
Describes the signal part of the shape of a typical gamma peak in a detector.

# Arguments
    * `x`: x values
    * `μ`: Mean values
    * `σ`: Sigma values
    * `n`: Counts
    * `skew_fraction`: Fraction of skewed values

# Keywords
    * `skew_fraction_highE`: High energy skew fraction

# Returns
    * Calculated signal peakshape

"""
function signal_peakshape(x::Real, μ::Real, σ::Real, n::Real, skew_fraction::Real;  skew_fraction_highE::Real = 0.0)
    return iszero(σ) ? zero(x) : n * (1 - skew_fraction - skew_fraction_highE) * gauss_pdf(x, μ, σ)
end
export signal_peakshape

"""
    background_peakshape(
    x::Real, μ::Real, σ::Real, 
    step_amplitude::Real, background::Real; 
    background_slope::Real = 0.0, background_exp = 0.0, background_center::Real = µ
)

Describes the background part of the shape of a typical gamma peak in a detector.
  components: 
- step-function scaled with `step_amplitude`
- energy-independent background: `background`
- linear slope: `background_slope` (optional)
- exponential decay:  `background_exp` (optional)

# Arguments
    * `x`: x values
    * `μ`: Mean values
    * `σ`: Sigma values
    * `step_amplitude`: Scales the step-function
    * `background`: Energy-independent background

# Keywords
    * `backround_slope`: Linear slope
    * `background_exp`: Exponential decay
    * `background_center`: Center of background fit curve

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
        skew_fraction::Real, skew_width::Real,
    )
    
Describes the low-E signal tail part of the shape of a typical gamma peak in a detector.

# Arguments
    * `x`: x values
    * `μ`: Mean values
    * `σ`: Sigma values
    * `n`: Counts
    * `skew_fraction`: Skew fraction
    * `skew_width`: Width of the skewed section of plot

# Returns
    * Low energy signal tail calculation
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
    
Describes the high-E signal tail part of the shape. 

# Arguments
    * `x`: x values
    * `μ`: Mean values
    * `σ`: Sigma values
    * `n`:
    * `skew_fraction_h`: 
    * `skew_width_h`: width of skew tail

# Returns
    n * skew_fraction_h * ex_gauss_pdf(x, μ, σ, skew)

TO DO: argument descriptions
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

Evaluates an extended step gauss model at `x` with parameters `l`, `k`, `t` and `d`.

# Arguments
    * `x`: x value
    * `l`:
    * `k`:
    * `t`:
    * `d`:

# Returns
    (exp(k*(x-l)) + d) / (exp((x-l)/t) + l)

TO DO: argument descriptions
"""
function ex_step_gauss(x::Real, l::Real, k::Real, t::Real, d::Real)

    return (exp(k*(x-l)) + d) / (exp((x-l)/t) + l)
end
export ex_step_gauss

"""
    aoe_compton_peakshape(
        x::Real, μ::Real, σ::Real, n::Real,
        background::Real, δ::Real
    )

Describes the shape of a typical A/E Compton peak in a detector as a gaussian SSE peak and a step like background for MSE events.

# Arguments
    * `x`: x-values
    * `μ`: Mean values
    * `σ`: Sigma values
    * `n`:
    * `background`: detector background
    * `δ`: 

# Returns
    n * gauss_pdf(x, μ, σ) + background * ex_gauss_pdf(-x, -μ, σ, δ)

TO DO: argument descriptions
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

# Arguments
    * `x`: x-values
    * `μ`: Mean values
    * `σ`: Sigma values
    * `n`: counts

# Returns
    n * gauss_pdf(x, μ, σ)

TO DO: argument description check. 
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

# Arguments
    * `x`: x-values
    * `μ`: Mean values
    * `σ`: Sigma values
    * `background`: Background
    * `δ`:

# Returns
    background * ex_gauss_pdf(-x, -μ, σ, δ)

TO DO: argument descriptions
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

Evaluates the sum of two gaussians at `x` with parameters `μ1`, `σ1`, `n1`, `μ2`, `σ2`, `n2`.

# Arguments
    * `x`: x values
    * `μ1`: Mean values of gaussian 1
    * `σ1`: Sigma values of gaussian 1
    * `n1`: counts of gaussian 1
    * `μ2`: Mean values of gaussian 2
    * `σ2`: Sigma values of gaussian 2
    * `n3`: counts of gaussian 2

# Returns 
    n1 * gauss_pdf(x, μ1, σ1) + n2 * gauss_pdf(x, μ2, σ2) 

TO DO: check argument descriptions.

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
    two_emg_aoe_compton_peakshape( # total fit function with two EMGs
        x::Real, 
        μ::Real, σ::Real, n::Real, background::Real, δ::Real, 
        μ2::Real, σ2::Real, background2::Real, δ2::Real
    )

# Arguments
    * `x`: Data
    * `μ`: Mean
    * `σ`: Standard deviation
    * `n`: 
    * `background`: 
    * `δ`:
    * `μ2`:
    * `σ2`:
    * `background2`: 
    * `δ2`: 

# Returns
    * Fit function of A/E compton peakshape wtih 2 EMGs

TO DO: arguments, description
"""

function two_emg_aoe_compton_peakshape( # total fit function with two EMGs
    x::Real, 
    μ::Real, σ::Real, n::Real, background::Real, δ::Real, 
    μ2::Real, σ2::Real, background2::Real, δ2::Real
)
    return iszero(σ) || iszero(σ2) ? zero(x) : n * gauss_pdf(x, μ, σ) + background * ex_gauss_pdf(-x, -μ, σ, δ) + background2 * ex_gauss_pdf(-x, -μ2, σ2, δ2)
end
export two_emg_aoe_compton_peakshape

"""
    two_emg_aoe_compton_background_peakshape( # background function with two EMGs
    x::Real, 
    μ::Real, σ::Real, background::Real, δ::Real,
    μ2::Real, σ2::Real, background2::Real, δ2::Real
)

# Arguments
    * `x`: Data
    * `μ`: Mean
    * `σ`: Standard deviation
    * `background`: 
    * `δ`:
    * `μ2`:
    * `σ2`:
    * `background2`: 
    * `δ2`: 

# Returns
    * Background function of A/E compton peakshape wtih 2 EMGs

TO DO: arguments, description
"""

function two_emg_aoe_compton_background_peakshape( # background function with two EMGs
    x::Real, 
    μ::Real, σ::Real, background::Real, δ::Real,
    μ2::Real, σ2::Real, background2::Real, δ2::Real
)
    return background * ex_gauss_pdf(-x, -μ, σ, δ) + background2 * ex_gauss_pdf(-x, -μ2, σ2, δ2)
end
export two_aoe_compton_background_peakshape