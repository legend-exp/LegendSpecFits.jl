# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

# Objectives accepted by `objective_value`, `fit_fwhm_ft` and `optimize_rtft`.
const energy_objectives = (:w90, :fwhm, :asym, :fwhm_asym, :fwhm_ct)

# Row recorded by the scan and search drivers for an evaluation that threw.
# Field names and types equal those returned by `evaluate_energy_point`.
const nan_energy_point = (
    fwhm_before = NaN * e_unit, fwhm = NaN * e_unit, w90 = NaN * e_unit, asym = NaN,
    tail_fraction = NaN, tail_scale = NaN * e_unit, centroid = NaN * e_unit,
    fct = NaN, ct_spread = NaN * e_unit, m_cal_simple = NaN * e_unit, ctc_converged = false, n_events = 0
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

`w90` and `asym` use the trapezoid-rule cumulative of the shape on a uniform grid from
`μ - max(10σ, 8 tail_scale)` to `μ + 6σ`, normalized to one, with linear interpolation at the
interval ends and at `μ`. The grid has at least 4001 points and a step of at most `σ / 50`.

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
    x_low = p.μ - max(10 * p.σ, 8 * tail_scale)
    x_high = p.μ + 6 * p.σ
    x = range(x_low, x_high; length=max(4001, ceil(Int, (x_high - x_low) / (p.σ / 50)) + 1))
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
    objective_value(metrics::NamedTuple, objective::Symbol; ctc_rel_uncertainty=nothing)

Value of `objective` for a row of metrics as returned by `peak_width_metrics` or
`evaluate_energy_point`: `metrics.w90`, `metrics.fwhm`, `metrics.asym`,
`metrics.fwhm * metrics.asym` for `:fwhm_asym`, or `sqrt(fwhm^2 + (k * ct_spread)^2)` with
`k = ctc_rel_uncertainty` for `:fwhm_ct`. `:fwhm_ct` combines the FWHM after the charge-trapping
correction in quadrature with the fraction `k` of the drift-time broadening that the correction
removes (`metrics.ct_spread`); `k` is the assumed relative error of the correction. The keyword is
ignored by the other objectives.

Throws an `ArgumentError` for any other symbol, and for `:fwhm_ct` if `ctc_rel_uncertainty` is
`nothing`, negative or not finite.
"""
function objective_value(metrics::NamedTuple, objective::Symbol; ctc_rel_uncertainty=nothing)
    objective == :w90 && return metrics.w90
    objective == :fwhm && return metrics.fwhm
    objective == :asym && return metrics.asym
    objective == :fwhm_asym && return metrics.fwhm * metrics.asym
    if objective == :fwhm_ct
        k = ctc_rel_uncertainty
        (k isa Real && isfinite(k) && k >= 0) || throw(ArgumentError("objective :fwhm_ct requires ctc_rel_uncertainty to be a finite number >= 0, got $k"))
        return sqrt(metrics.fwhm^2 + (k * metrics.ct_spread)^2)
    end
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
and finite, or if `rel_tol` is not in `[0, Inf)`; throws a `DimensionMismatch` if the two vectors have different
indices.
"""
function select_optimum(objective_values::AbstractVector{<:Real}, params::AbstractVector{<:Tuple}; rel_tol::Real)
    0 <= rel_tol < Inf || throw(ArgumentError("select_optimum: rel_tol must satisfy 0 <= rel_tol < Inf, got $rel_tol"))
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
`m_cal_simple = peak / mode` converts them to keV for the fits. With `apply_ctc=true`, `ctc_energy`
fits the linear charge-trapping correction and the metrics of `peak_width_metrics` are
computed from the peak fit after the correction. With `apply_ctc=false`, the uncorrected peak is
fitted with the same histogram as the uncorrected fit inside `ctc_energy`.

# Returns
A NamedTuple row `(fwhm_before, fwhm, w90, asym, tail_fraction, tail_scale, centroid, fct,
ct_spread, m_cal_simple, ctc_converged, n_events)`:
- `fwhm_before`, `fwhm`, `w90`, `tail_scale`, `centroid`: `Unitful` energies in keV. Each is the
  value from the fit in `m_cal_simple`-calibrated keV multiplied by `peak / μ`, where `μ` is the
  fitted peak position of that same fit (the uncorrected fit for `fwhm_before`, the fit after the
  correction for the others). The fitted peak therefore sits at `peak`, and `centroid == peak`
  for a symmetric peak.
- `ct_spread`: `Unitful` energy in keV, the FWHM-equivalent spread of the linear drift-time term
  removed by the correction, `2 * sqrt(2 * log(2)) * |fct| * m_cal_simple * std(qdrift)` times
  `peak / μ` of the fit after the correction, with `std(qdrift)` taken over the events inside the
  correction window `(peak - first(window), peak + last(window))` of the calibrated energies
- `asym`, `tail_fraction`: unitless
- `fct`: first-order element of `ctc_energy`'s `result.fct` in ADC per `qdrift` unit, such that
  `e + fct * qdrift` is the corrected energy in ADC
- `m_cal_simple`: keV per ADC
- `ctc_converged`: convergence flag of the correction
- `n_events`: number of events in the 15-half-width window

With `apply_ctc=false`, `fct` and `ct_spread` are zero, `ctc_converged` is `true` and `fwhm_before == fwhm`.

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
        fwhm_before = mvalue(ustrip(e_unit, ctc_result.fwhm_before)) * e_unit * (ustrip(e_unit, peak) / mvalue(ctc_report.report_before.v.μ))
        fct = first(ctc_result.fct)
        in_ctc_window = peak - first(window) .< e_cal .< peak + last(window)
        qdrift_std = std(qdrift_win[in_ctc_window])
        ctc_converged = ctc_result.converged
    else
        e_peak = ustrip.(e_unit, filter(x -> peak - first(window) < x < peak + last(window), e_cal))
        bin_width_peak = ustrip(e_unit, get_friedman_diaconis_bin_width(filter(x -> peak - 5.0u"keV" < x < peak + 5.0u"keV", e_cal)))
        h_peak = fit(Histogram, e_peak, minimum(e_peak):bin_width_peak:maximum(e_peak))
        _, fit_report = fit_single_peak_th228(h_peak, estimate_single_peak_stats(h_peak); uncertainty=false)
        v = fit_report.v
        fct = 0.0
        qdrift_std = 0.0
        ctc_converged = true
    end
    isfinite(v.σ) && v.σ > 0 || error("evaluate_energy_point: peak fit returned σ = $(v.σ)")

    metrics = peak_width_metrics(v; containment)
    scale = ustrip(e_unit, peak) / mvalue(v.μ) * e_unit
    fwhm = metrics.fwhm * scale
    ct_spread = 2 * sqrt(2 * log(2)) * abs(fct) * ustrip(e_unit, m_cal_simple) * qdrift_std * scale
    return (
        fwhm_before = apply_ctc ? fwhm_before : fwhm, fwhm = fwhm, w90 = metrics.w90 * scale,
        asym = metrics.asym, tail_fraction = metrics.tail_fraction, tail_scale = metrics.tail_scale * scale,
        centroid = metrics.centroid * scale, fct = fct, ct_spread = ct_spread, m_cal_simple = m_cal_simple,
        ctc_converged = ctc_converged, n_events = length(e_win)
    )
end
export evaluate_energy_point

"""
    optimize_rtft(f_energy, qdrift::AbstractVector{<:Real}, rt0, ft0; rt_bounds::Tuple{Any,Any}, ft_bounds::Tuple{Any,Any}, feasible=(rt, ft) -> true, step0::Tuple{Any,Any}, lattice, objective::Symbol=:fwhm, ctc_rel_uncertainty=nothing, rel_tol::Real=0.005, max_evals::Int=60, eval_kwargs...)

Joint search of rise time `rt` and flat-top time `ft` minimizing `objective` (see
`objective_value`) of `evaluate_energy_point`. The objective `:fwhm_ct` requires
`ctc_rel_uncertainty`, the assumed relative error of the correction, and `apply_ctc = true`.

`f_energy(rt, ft)` returns the energies (ADC) of the waveform set at one filter setting, aligned
with `qdrift`; it is called concurrently from several threads. `rt0`, `ft0`, the bounds, `step0` and
`lattice` are `Unitful` times. Every candidate lies on the lattice `k * lattice` and is evaluated at
most once. `eval_kwargs` must contain `peak`, `window`, `min_e`, `max_e`, `rel_cut_fit` and may
contain `n_bins`, `apply_ctc`, `containment`; they are forwarded to `evaluate_energy_point`.

Compass pattern search with diagonal moves: starting from the seed `(rt0, ft0)` rounded to the
lattice and the steps `step0` rounded to multiples of the lattice, the eight neighbors at the
current steps that lie inside the bounds (snapped to the lattice with a tolerance of `1e-6`
lattice steps), satisfy `feasible(rt, ft)` and are not yet evaluated are evaluated in parallel
(`Threads.@threads`). The search moves to the best neighbor if its objective is below
`(1 - rel_tol)` times the current one; otherwise both steps are halved, never below one lattice
step. It stops when both steps equal the lattice and no neighbor improves, with `converged = true`
if no admissible neighbor exists at the final point or at least one admissible neighbor was
evaluated successfully, and with `converged = false` if all admissible neighbors failed. It also
stops with `converged = false` when `max_evals` evaluations are reached. The optimum is chosen with
`select_optimum` among the final point and its evaluated neighbors at the final steps, so ties
within `rel_tol` go to the shortest `rt`, then the shortest `ft`.

An evaluation that throws is recorded with its error text in `status` and `NaN` metrics, and the
search continues.

# Returns
- `result`: `(rt, ft, fwhm, w90, asym, fct, ct_spread, min_objective, objective, n_evals, converged)`
- `report`: `(rt, ft, objective, table, path, objective_values)`, where `table` is a `TypedTables.Table` with one
  row per evaluation and columns `rt, ft, stage, step_rt, step_ft, status, fwhm_before, fwhm, w90,
  asym, tail_fraction, tail_scale, centroid, fct, ct_spread, m_cal_simple, ctc_converged, n_events`
  (`stage` is `:seed` or `:neighbor`), `path` is the vector of accepted `(rt, ft)`
  positions starting with the seed, and `objective_values` are the values of `objective` for the
  rows of `table` (`NaN` for failed rows).

Throws an `ArgumentError` for an unknown `objective`, for `objective == :fwhm_ct` with a missing,
negative or non-finite `ctc_rel_uncertainty` or with `apply_ctc = false`, `lattice <= 0`, `max_evals < 1`,
non-positive `step0`, `rt0` or `ft0` outside their bounds, a seed that rounds to an infeasible or
out-of-bounds lattice point, or missing or unsupported `eval_kwargs`. Throws an `ErrorException`
if the seed evaluation fails.
"""
function optimize_rtft(f_energy, qdrift::AbstractVector{<:Real}, rt0::Unitful.Time{<:Real}, ft0::Unitful.Time{<:Real};
        rt_bounds::Tuple{Any,Any}, ft_bounds::Tuple{Any,Any}, feasible=(rt, ft) -> true, step0::Tuple{Any,Any}, lattice::Unitful.Time{<:Real},
        objective::Symbol=:fwhm, ctc_rel_uncertainty::Union{Nothing, Real}=nothing, rel_tol::Real=0.005, max_evals::Int=60, eval_kwargs...)
    objective in energy_objectives || throw(ArgumentError("objective must be one of $energy_objectives, got :$objective"))
    lattice > zero(lattice) || throw(ArgumentError("optimize_rtft: lattice must be positive, got $lattice"))
    0 <= rel_tol < Inf || throw(ArgumentError("optimize_rtft: rel_tol must satisfy 0 <= rel_tol < Inf, got $rel_tol"))
    max_evals >= 1 || throw(ArgumentError("optimize_rtft: max_evals must be at least 1, got $max_evals"))
    all(>(zero(lattice)), step0) || throw(ArgumentError("optimize_rtft: step0 must be positive, got $step0"))
    first(rt_bounds) <= rt0 <= last(rt_bounds) || throw(ArgumentError("optimize_rtft: rt0 = $rt0 outside rt_bounds = $rt_bounds"))
    first(ft_bounds) <= ft0 <= last(ft_bounds) || throw(ArgumentError("optimize_rtft: ft0 = $ft0 outside ft_bounds = $ft_bounds"))
    required = (:peak, :window, :min_e, :max_e, :rel_cut_fit)
    kw = values(eval_kwargs)
    absent = filter(k -> !haskey(kw, k), required)
    isempty(absent) || throw(ArgumentError("optimize_rtft: eval_kwargs must contain $absent"))
    unsupported = filter(k -> !(k in (required..., :n_bins, :apply_ctc, :containment)), keys(kw))
    isempty(unsupported) || throw(ArgumentError("optimize_rtft: unsupported eval_kwargs $unsupported"))
    point_kwargs = Base.structdiff(kw, NamedTuple{required})
    if objective == :fwhm_ct
        get(kw, :apply_ctc, true) || throw(ArgumentError("optimize_rtft: objective :fwhm_ct requires apply_ctc = true"))
        (ctc_rel_uncertainty isa Real && isfinite(ctc_rel_uncertainty) && ctc_rel_uncertainty >= 0) ||
            throw(ArgumentError("optimize_rtft: objective :fwhm_ct requires ctc_rel_uncertainty to be a finite number >= 0, got $ctc_rel_uncertainty"))
    end

    to_lattice(t) = uconvert(NoUnits, t / lattice)
    rt_range = ceil(Int, to_lattice(first(rt_bounds)) - 1e-6):floor(Int, to_lattice(last(rt_bounds)) + 1e-6)
    ft_range = ceil(Int, to_lattice(first(ft_bounds)) - 1e-6):floor(Int, to_lattice(last(ft_bounds)) + 1e-6)
    admissible(k) = k[1] in rt_range && k[2] in ft_range && feasible(k[1] * lattice, k[2] * lattice)
    neighbors(c, s) = [(c[1] + di * s[1], c[2] + dj * s[2]) for di in -1:1 for dj in -1:1 if (di, dj) != (0, 0)]
    seed = (round(Int, to_lattice(rt0)), round(Int, to_lattice(ft0)))
    admissible(seed) || throw(ArgumentError("optimize_rtft: seed ($rt0, $ft0) rounds to an infeasible or out-of-bounds lattice point"))
    steps = (max(1, round(Int, to_lattice(first(step0)))), max(1, round(Int, to_lattice(last(step0)))))

    head = (rt = 0 * lattice, ft = 0 * lattice, stage = :seed, step_rt = 0 * lattice, step_ft = 0 * lattice, status = "")
    record_type = typeof(merge(head, nan_energy_point))
    records = Vector{record_type}()
    cache = Dict{Tuple{Int, Int}, Int}()
    cache_lock = ReentrantLock()
    function evaluate!(todo, stage, s)
        batch = Dict{Tuple{Int, Int}, record_type}()
        Threads.@threads for k in todo
            rt, ft = k[1] * lattice, k[2] * lattice
            row, status = try
                convert(typeof(nan_energy_point), evaluate_energy_point(f_energy(rt, ft), qdrift, kw.peak, kw.window, kw.min_e, kw.max_e, kw.rel_cut_fit; point_kwargs...)), "ok"
            catch err
                err isa InterruptException && rethrow()
                @warn "optimize_rtft: evaluation failed at rt = $rt, ft = $ft" exception = err
                nan_energy_point, sprint(showerror, err)
            end
            record = merge((rt = rt, ft = ft, stage = stage, step_rt = s[1] * lattice, step_ft = s[2] * lattice, status = status), row)
            lock(cache_lock) do
                batch[k] = record
            end
        end
        for k in sort!(collect(keys(batch)))
            push!(records, batch[k])
            cache[k] = lastindex(records)
        end
    end
    is_ok(k) = haskey(cache, k) && records[cache[k]].status == "ok"
    obj(k) = ustrip(objective_value(records[cache[k]], objective; ctc_rel_uncertainty))

    evaluate!([seed], :seed, steps)
    is_ok(seed) || error("optimize_rtft: seed evaluation failed: $(records[cache[seed]].status)")
    current = seed
    path = [(seed[1] * lattice, seed[2] * lattice)]
    converged = false
    while length(records) < max_evals
        hood = filter(admissible, neighbors(current, steps))
        todo = filter(k -> !haskey(cache, k), hood)
        budget = max_evals - length(records)
        evaluate!(first(todo, budget), :neighbor, steps)
        length(todo) > budget && break
        evaluated = filter(is_ok, hood)
        if !isempty(evaluated)
            best = argmin(k -> (obj(k), k), evaluated)
            if obj(best) < obj(current) * (1 - rel_tol)
                current = best
                push!(path, (best[1] * lattice, best[2] * lattice))
                continue
            end
        end
        if steps == (1, 1)
            converged = isempty(hood) || !isempty(evaluated)
            break
        end
        steps = (max(1, first(steps) ÷ 2), max(1, last(steps) ÷ 2))
    end

    final = [current; filter(k -> admissible(k) && is_ok(k), neighbors(current, steps))]
    best_row = records[cache[final[select_optimum([obj(k) for k in final], final; rel_tol)]]]
    result = (
        rt = best_row.rt, ft = best_row.ft, fwhm = best_row.fwhm, w90 = best_row.w90, asym = best_row.asym, fct = best_row.fct, ct_spread = best_row.ct_spread,
        min_objective = objective_value(best_row, objective; ctc_rel_uncertainty), objective = objective,
        n_evals = length(records), converged = converged
    )
    report = (rt = result.rt, ft = result.ft, objective = objective, table = Table(records), path = path,
        objective_values = [objective_value(r, objective; ctc_rel_uncertainty) for r in records])
    return result, report
end
export optimize_rtft
