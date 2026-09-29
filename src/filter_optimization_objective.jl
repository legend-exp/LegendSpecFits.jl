# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

# Objectives accepted by `objective_value`, `fit_fwhm_ft` and `optimize_rtft`.
const energy_objectives = (:w90, :fwhm, :asym, :fwhm_asym)

# Row recorded by the scan and search drivers for an evaluation that threw.
# Field names and types equal those returned by `evaluate_energy_point`.
const nan_energy_point = (
    fwhm_before = NaN * e_unit, fwhm = NaN * e_unit, w90 = NaN * e_unit, asym = NaN,
    tail_fraction = NaN, tail_scale = NaN * e_unit, centroid = NaN * e_unit,
    fct = NaN, m_cal_simple = NaN * e_unit, ctc_converged = false, n_events = 0
)

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

"""
    select_optimum(objective_values::AbstractVector{<:Real}, params::AbstractVector{<:Tuple}; rel_tol::Real)

Index of the selected optimum. Candidates whose objective lies within `min * (1 + rel_tol)` of the
smallest objective tie; among ties the lexicographically smallest parameter tuple wins, i.e. the
smallest first element, then the smallest second. `NaN` objectives are never selected. The
returned index belongs to `eachindex(objective_values, params)`.

Throws an `ArgumentError` if all objectives are `NaN`, if the smallest objective is not positive
and finite, or if `rel_tol < 0`; throws a `DimensionMismatch` if the two vectors have different
indices.
"""
function select_optimum(objective_values::AbstractVector{<:Real}, params::AbstractVector{<:Tuple}; rel_tol::Real)
    rel_tol >= 0 || throw(ArgumentError("select_optimum: rel_tol must be non-negative, got $rel_tol"))
    valid = filter(i -> !isnan(objective_values[i]), eachindex(objective_values, params))
    isempty(valid) && throw(ArgumentError("select_optimum: all objective values are NaN"))
    best = minimum(objective_values[i] for i in valid)
    0 < best < Inf || throw(ArgumentError("select_optimum: smallest objective must be positive and finite, got $best"))
    ties = filter(i -> objective_values[i] <= best * (1 + rel_tol), valid)
    return argmin(i -> params[i], ties)
end
export select_optimum

"""
    evaluate_energy_point(e::AbstractVector{<:Real}, qdrift::AbstractVector{<:Real}, peak, window, min_e, max_e, rel_cut_fit; n_bins::Int=-1, apply_ctc::Bool=true, containment::Real=0.9)

Evaluate the peak shape of the uncalibrated energies `e` (ADC) around `peak` for one filter
setting.

Events with finite, positive `e` and `qdrift` are kept. `cut_single_peak` on `(min_e, max_e)`
locates the peak mode; events within 15 half-widths of the mode enter the evaluation, and
`m_cal_simple = peak / mode` converts them to keV. With `apply_ctc=true`, `ctc_energy`
fits the linear charge-trapping correction and the metrics of `peak_width_metrics` are
computed from the peak fit after the correction. With `apply_ctc=false`, the uncorrected peak is
fitted with the same histogram as the uncorrected fit inside `ctc_energy`.

# Returns
A NamedTuple row `(fwhm_before, fwhm, w90, asym, tail_fraction, tail_scale, centroid, fct,
m_cal_simple, ctc_converged, n_events)`:
- `fwhm_before`, `fwhm`, `w90`, `tail_scale`, `centroid`: `Unitful` energies in keV, calibrated
  with this point's `m_cal_simple`
- `asym`, `tail_fraction`: unitless
- `fct`: first-order element of `ctc_energy`'s `result.fct` in ADC per `qdrift` unit, such that
  `e + fct * qdrift` is the corrected energy in ADC
- `m_cal_simple`: keV per ADC
- `ctc_converged`: convergence flag of the correction
- `n_events`: number of events in the 15-half-width window

With `apply_ctc=false`, `fct` is `0.0`, `ctc_converged` is `true` and `fwhm_before == fwhm`.

Throws an `ErrorException` naming the cause if fewer than 100 selected events lie in
`(min_e, max_e)`, if the peak-statistics guess has a non-positive entry, or if the peak fit
returns a non-finite or non-positive `σ`. Errors of the fit or the correction propagate.
"""
function evaluate_energy_point(e::AbstractVector{<:Real}, qdrift::AbstractVector{<:Real}, peak::Unitful.Energy{<:Real}, window::Tuple{<:Unitful.Energy{<:Real}, <:Unitful.Energy{<:Real}}, min_e::Real, max_e::Real, rel_cut_fit::Real; n_bins::Int=-1, apply_ctc::Bool=true, containment::Real=0.9)
    keep = filter(i -> isfinite(e[i]) && isfinite(qdrift[i]) && e[i] > 0 && qdrift[i] > 0, eachindex(e, qdrift))
    e_sel = Float64[e[i] for i in keep]
    qdrift_sel = Float64[qdrift[i] for i in keep]
    n_range = count(x -> min_e < x < max_e, e_sel)
    n_range >= 100 || error("evaluate_energy_point: $n_range events in ($min_e, $max_e), at least 100 required")

    fit_cut = cut_single_peak(e_sel, Float64(min_e), Float64(max_e); n_bins, relative_cut=Float64(rel_cut_fit))
    half_window = 15 * (fit_cut.max - fit_cut.low)
    in_window = findall(x -> fit_cut.max - half_window < x < fit_cut.max + half_window, e_sel)
    e_win, qdrift_win = e_sel[in_window], qdrift_sel[in_window]

    bin_width = get_friedman_diaconis_bin_width(e_win)
    h = fit(Histogram, e_win, minimum(e_win):bin_width:maximum(e_win))
    ps = estimate_single_peak_stats_th228(h)
    all(>(0), tuple_to_array(ps)) || error("evaluate_energy_point: invalid peak-statistics guess $ps")

    m_cal_simple = uconvert(e_unit, peak) / fit_cut.max
    e_cal = e_win .* m_cal_simple
    if apply_ctc
        ctc_result, ctc_report = ctc_energy(e_cal, qdrift_win, peak, window, m_cal_simple)
        v = ctc_report.report_after.v
        fwhm_before = mvalue(ustrip(e_unit, ctc_result.fwhm_before)) * e_unit
        fct = first(ctc_result.fct)
        ctc_converged = ctc_result.converged
    else
        e_peak = ustrip.(e_unit, filter(x -> peak - first(window) < x < peak + last(window), e_cal))
        bin_width_peak = ustrip(e_unit, get_friedman_diaconis_bin_width(filter(x -> peak - 5.0u"keV" < x < peak + 5.0u"keV", e_cal)))
        h_peak = fit(Histogram, e_peak, minimum(e_peak):bin_width_peak:maximum(e_peak))
        _, fit_report = fit_single_peak_th228(h_peak, estimate_single_peak_stats(h_peak); uncertainty=false)
        v = fit_report.v
        fct = 0.0
        ctc_converged = true
    end
    isfinite(v.σ) && v.σ > 0 || error("evaluate_energy_point: peak fit returned σ = $(v.σ)")

    metrics = peak_width_metrics(v; containment)
    fwhm = metrics.fwhm * e_unit
    return (
        fwhm_before = apply_ctc ? fwhm_before : fwhm, fwhm = fwhm, w90 = metrics.w90 * e_unit,
        asym = metrics.asym, tail_fraction = metrics.tail_fraction, tail_scale = metrics.tail_scale * e_unit,
        centroid = metrics.centroid * e_unit, fct = fct, m_cal_simple = m_cal_simple,
        ctc_converged = ctc_converged, n_events = length(e_win)
    )
end
export evaluate_energy_point
