# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

# Objectives accepted by `objective_value`, `fit_fwhm_ft` and `optimize_rtft`.
const energy_objectives = (:w90, :fwhm, :asym, :fwhm_asym)

"""
    peak_width_metrics(v::NamedTuple; containment::Real=0.9)

Width and shape metrics of the signal part of a gamma peak: the Gaussian plus low-energy tail
`gamma_sigWithTail` of `get_th228_fit_functions`, without step and background, for the
parameters `v.μ, v.σ, v.n, v.skew_fraction, v.skew_width`. The parameters may be plain numbers or
`Measurement`s; uncertainties are ignored. All widths are in the units of `v`.

# Returns
- `fwhm`: full width at half maximum, as `estimate_fwhm`
- `w90`: length of the shortest interval that contains the fraction `containment` of the shape
- `asym`: integral of the shape below `μ` divided by the integral above `μ`; `(1 + f) / (1 - f)`
  for a tail fraction `f` whose tail scale is large compared to `σ`
- `tail_fraction`: `skew_fraction`
- `tail_scale`: decay length of the tail, `skew_width * μ`
- `centroid`: mean of the shape, as `peak_centroid`

`w90` and `asym` use the trapezoid-rule cumulative of the shape on 4001 uniform points from
`μ - max(10σ, 8 tail_scale)` to `μ + 6σ`, normalized to one, with linear interpolation at the
interval ends and at `μ`.

Throws an `ArgumentError` naming the parameter if any of `μ, σ, n, skew_fraction, skew_width` is
not finite, if `μ <= 0`, `σ <= 0`, `n <= 0`, `skew_width < 0`, or `skew_fraction` is outside
`[0, 1)`, and if `containment` is outside `(0, 1)`. Throws an `ErrorException` if the FWHM of the
shape cannot be determined.
"""
function peak_width_metrics(v::NamedTuple; containment::Real=0.9)
    p = map(x -> float(mvalue(x)), (μ = v.μ, σ = v.σ, n = v.n, skew_fraction = v.skew_fraction, skew_width = v.skew_width))
    for (name, x) in pairs(p)
        isfinite(x) || throw(ArgumentError("peak_width_metrics: parameter $name is not finite, got $x"))
    end
    p.μ > 0 || throw(ArgumentError("peak_width_metrics: parameter μ must be positive, got $(p.μ)"))
    p.σ > 0 || throw(ArgumentError("peak_width_metrics: parameter σ must be positive, got $(p.σ)"))
    p.n > 0 || throw(ArgumentError("peak_width_metrics: parameter n must be positive, got $(p.n)"))
    0 <= p.skew_fraction < 1 || throw(ArgumentError("peak_width_metrics: parameter skew_fraction must lie in [0, 1), got $(p.skew_fraction)"))
    p.skew_width >= 0 || throw(ArgumentError("peak_width_metrics: parameter skew_width must be non-negative, got $(p.skew_width)"))
    0 < containment < 1 || throw(ArgumentError("peak_width_metrics: containment must lie in (0, 1), got $containment"))

    fwhm = estimate_fwhm(p)
    isfinite(fwhm) || error("peak_width_metrics: FWHM of the signal shape could not be determined for $p")

    tail_scale = p.skew_width * p.μ
    x = range(p.μ - max(10 * p.σ, 8 * tail_scale), p.μ + 6 * p.σ; length=4001)
    f_sig = get_th228_fit_functions().gamma_sigWithTail
    y = [f_sig(xi, p) for xi in x]
    cdf = cumsum([zero(eltype(y)); (y[begin+1:end] .+ y[begin:end-1]) ./ 2])
    cdf ./= last(cdf)

    k = searchsortedlast(x, p.μ)
    cdf_μ = cdf[k] + (cdf[k+1] - cdf[k]) * (p.μ - x[k]) / step(x)
    asym = cdf_μ / (1 - cdf_μ)

    # Two-pointer sweep: `j` is the first grid point whose cumulative reaches the target of the
    # current left end `i`; both only move forward because the cumulative is non-decreasing.
    w90 = Inf
    j = firstindex(cdf)
    for i in eachindex(cdf)
        target = cdf[i] + containment
        target > last(cdf) && break
        while cdf[j] < target
            j += 1
        end
        x_right = x[j-1] + (target - cdf[j-1]) / (cdf[j] - cdf[j-1]) * step(x)
        w90 = min(w90, x_right - x[i])
    end

    return (fwhm = fwhm, w90 = w90, asym = asym, tail_fraction = p.skew_fraction, tail_scale = tail_scale, centroid = peak_centroid(p))
end
export peak_width_metrics

"""
    objective_value(metrics::NamedTuple, objective::Symbol)

Value of `objective` for a row of metrics as returned by `peak_width_metrics` or
`evaluate_energy_point`: `metrics.w90`, `metrics.fwhm`, `metrics.asym`, or
`metrics.fwhm * metrics.asym` for `:fwhm_asym`. Throws an `ArgumentError` for any other symbol.
"""
function objective_value(metrics::NamedTuple, objective::Symbol)
    objective == :w90 && return metrics.w90
    objective == :fwhm && return metrics.fwhm
    objective == :asym && return metrics.asym
    objective == :fwhm_asym && return metrics.fwhm * metrics.asym
    throw(ArgumentError("objective must be one of $energy_objectives, got :$objective"))
end
export objective_value
