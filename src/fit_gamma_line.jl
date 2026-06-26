# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
# =============================================================================
# fit_gamma_line.jl — Bayesian or MLE γ-line fit on an arbitrary histogram
# =============================================================================
# Generic, production-spectrum γ-line fitter (Gaussian peak + polynomial
# background), modelled on `fit_single_peak_th228` but stripped of the Th228
# calibration-specific tail/step components and extended with a BAT MCMC mode
# and a quadratic background term for sub-500 keV lines.
#
# Public API:
#   fit_gamma_line(h, ps; fit_func, method, uncertainty, pseudo_prior, ...) → (result, report)
#   get_gamma_line_fit_functions(; background_center)         registry
#   gamma_line_peakshape_components(fit_func; background_center)
#   gamma_line_peakshape_components(fit_func, v; background_center)
#   get_gamma_line_pseudo_prior(h, ps, fit_func; pseudo_prior, ...)
#
# `report` always has the `(:v, :h, :f_fit, :f_components, :gof)` shape so the
# existing LegendMakie recipe (`LegendMakie.lplot!(report::NamedTuple{(...)})`)
# works without changes. BAT-only extras (`S_samples`, `ul_90`, etc.) live in
# `result`.
# =============================================================================

"""
    get_gamma_line_fit_functions(; background_center::Union{Real,Nothing} = nothing)

Peak-fit-function registry for `fit_gamma_line`. Each entry maps a `Symbol`
preset → `(x, v) -> rate(x; v)`:

| Symbol            | Parameters                                          | Shape |
|-------------------|-----------------------------------------------------|---|
| `:gauss_flat`     | `μ, σ, n, background`                               | `n·𝒩(x|μ,σ) + b0` |
| `:gauss_lin`      | `..., background_slope`                             | `+ b1·(x − E_c)` |
| `:gauss_quad`     | `..., background_curv`                              | `+ b2·(x − E_c)²` |
| `:gauss_lin_tail` | `..., skew_fraction, skew_width, background_slope`  | low-E exponential tail on the Gaussian |

`background_center` (defaults to `v.μ`) anchors the polynomial. For
`:gauss_lin_tail` the tail follows the LSF `lowEtail_peakshape` parameterisation.
"""
function get_gamma_line_fit_functions(; background_center::Union{Real,Nothing} = nothing)
    bc(v) = background_center === nothing ? v.μ : background_center
    return (
        gauss_flat = (x, v) ->
            signal_peakshape(x, v.μ, v.σ, v.n, 0.0) + v.background,
        gauss_lin = (x, v) ->
            signal_peakshape(x, v.μ, v.σ, v.n, 0.0) +
            v.background + v.background_slope * (x - bc(v)),
        gauss_quad = (x, v) -> begin
            δ = x - bc(v)
            signal_peakshape(x, v.μ, v.σ, v.n, 0.0) +
                v.background + v.background_slope * δ + v.background_curv * δ * δ
        end,
        gauss_lin_tail = (x, v) ->
            signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction) +
            lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width) +
            v.background + v.background_slope * (x - bc(v)),
    )
end
export get_gamma_line_fit_functions

"""
    gamma_line_peakshape_components(fit_func::Symbol; background_center) -> (funcs, labels, colors, linestyles)

Decompose `fit_func` into (signal, [low-E tail], background) parts so the
plot recipe can stack them. Returns a NamedTuple of NamedTuples — same layout
as `peakshape_components` in `specfit_functions.jl`.
"""
function gamma_line_peakshape_components(fit_func::Symbol;
                                          background_center::Union{Real,Nothing} = nothing)
    bc(v) = background_center === nothing ? v.μ : background_center
    if fit_func === :gauss_flat
        funcs = (
            f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, 0.0),
            f_bck = (x, v) -> v.background,
        )
    elseif fit_func === :gauss_lin
        funcs = (
            f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, 0.0),
            f_bck = (x, v) -> v.background + v.background_slope * (x - bc(v)),
        )
    elseif fit_func === :gauss_quad
        funcs = (
            f_sig = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, 0.0),
            f_bck = (x, v) -> begin
                δ = x - bc(v)
                v.background + v.background_slope * δ + v.background_curv * δ * δ
            end,
        )
    elseif fit_func === :gauss_lin_tail
        funcs = (
            f_sig      = (x, v) -> signal_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction),
            f_lowEtail = (x, v) -> lowEtail_peakshape(x, v.μ, v.σ, v.n, v.skew_fraction, v.skew_width),
            f_bck      = (x, v) -> v.background + v.background_slope * (x - bc(v)),
        )
    else
        throw(ArgumentError("Unknown gamma-line fit function: $fit_func"))
    end
    labels     = (f_sig = "Signal",     f_lowEtail = "Low-energy tail", f_bck = "Background")
    colors     = (f_sig = :orangered1,  f_lowEtail = :orange,           f_bck = :dodgerblue2)
    linestyles = (f_sig = :solid,       f_lowEtail = :dashdot,          f_bck = :dash)
    return (funcs = funcs, labels = labels, colors = colors, linestyles = linestyles)
end

"""
    gamma_line_peakshape_components(fit_func, v; background_center = v.μ)

Convenience overload — same as the no-`v` form but the returned `funcs` are
already partial-evaluated at `v`, so `funcs.f_sig(x)` works directly. Mirrors
the LSF `peakshape_components(fit_func, v; ...)` pattern.
"""
function gamma_line_peakshape_components(fit_func::Symbol, v::NamedTuple;
                                          background_center::Union{Real,Nothing} = nothing)
    bc = background_center === nothing ? v.μ : background_center
    comps = gamma_line_peakshape_components(fit_func; background_center = bc)
    fixed = NamedTuple{keys(comps.funcs)}(
        Tuple(x -> Base.Fix2(getproperty(comps.funcs, k), v)(x) for k in keys(comps.funcs))
    )
    return (; comps..., funcs = fixed)
end
export gamma_line_peakshape_components

"""
    get_gamma_line_pseudo_prior(h, ps, fit_func; pseudo_prior, fixed_position,
                                  σ_μ_default = 0.6, σ_σ_default = 0.6)
        -> NamedTupleDist

Build the standard `NamedTupleDist` of priors for `fit_gamma_line`, with
optional user overrides via `pseudo_prior`. Priors are derived from the
seed `ps` (see `estimate_single_peak_stats`):

  * `μ`  ~ `Normal(peak_pos, σ_μ_default)`              (`ConstValueDist` if `fixed_position`)
  * `σ`  ~ `truncated(Normal(peak_sigma, σ_σ_default); lower = 0.1)`
  * `n`  ~ `Uniform(0, max(10·peak_counts, 200))`
  * `background`       ~ `truncated(Normal(mean_background, 5·mean_background_std); lower = 0.0)`
  * `background_slope` ~ `Normal(0, 1.0)`                                (only `:gauss_lin*`, `:gauss_quad`)
  * `background_curv`  ~ `Normal(0, max(mean_background, 1) / (h_width)²)` (only `:gauss_quad`)
  * `skew_fraction`    ~ `truncated(weibull_from_mx(0.002, 0.008), 1e-7, 0.5)` (only `:gauss_lin_tail`)
  * `skew_width`       ~ `weibull_from_mx(σ/μ, 1.2·σ/μ)`                  (only `:gauss_lin_tail`)
"""
function get_gamma_line_pseudo_prior(h::Histogram, ps::NamedTuple, fit_func::Symbol;
                                       pseudo_prior::NamedTupleDist = NamedTupleDist(empty = true),
                                       fixed_position::Bool = false,
                                       σ_μ_default::Real = 0.6,
                                       σ_σ_default::Real = 0.6)
    edges      = first(h.edges)
    win_keV    = Float64(maximum(edges) - minimum(edges))
    bkg_mean   = max(Float64(ps.mean_background), 1.0)
    bkg_std    = max(Float64(ps.mean_background_std), sqrt(bkg_mean))
    n_max      = max(10 * Float64(ps.peak_counts), 200.0)

    base = NamedTupleDist(
        μ = fixed_position ? ConstValueDist(Float64(ps.peak_pos)) :
                              Normal(Float64(ps.peak_pos), σ_μ_default),
        σ = truncated(Normal(Float64(ps.peak_sigma), σ_σ_default); lower = 0.1),
        n = Uniform(0.0, n_max),
        background       = truncated(Normal(bkg_mean, 5 * bkg_std); lower = 0.0),
        background_slope = Normal(0.0, 1.0),
        background_curv  = Normal(0.0, bkg_mean / max(win_keV * win_keV, 1.0)),
        skew_fraction    = truncated(weibull_from_mx(0.002, 0.008), 1e-7, 0.5),
        skew_width       = weibull_from_mx(Float64(ps.peak_sigma) / Float64(ps.peak_pos),
                                            1.2 * Float64(ps.peak_sigma) / Float64(ps.peak_pos)),
    )
    (; μ, σ, n, background, background_slope, background_curv, skew_fraction, skew_width) = base

    standard = if fit_func === :gauss_flat
        NamedTupleDist(; μ, σ, n, background)
    elseif fit_func === :gauss_lin
        NamedTupleDist(; μ, σ, n, background, background_slope)
    elseif fit_func === :gauss_quad
        NamedTupleDist(; μ, σ, n, background, background_slope, background_curv)
    elseif fit_func === :gauss_lin_tail
        NamedTupleDist(; μ, σ, n, background, background_slope, skew_fraction, skew_width)
    else
        throw(ArgumentError("Unknown gamma-line fit function: $fit_func"))
    end

    if :empty in keys(pseudo_prior)
        return standard
    else
        @assert all(f -> f in keys(standard), keys(pseudo_prior)) "Pseudo priors can only have $(keys(standard)) as fields; got $(keys(pseudo_prior))."
        return merge(standard, pseudo_prior)
    end
end
export get_gamma_line_pseudo_prior

# =============================================================================
# fit_gamma_line — main entry point
# =============================================================================

"""
    fit_gamma_line(h::Histogram, ps::NamedTuple;
                    fit_func::Symbol      = :gauss_flat,
                    method::Symbol        = :mle,             # :mle | :bat
                    uncertainty::Bool     = true,
                    pseudo_prior::NamedTupleDist = NamedTupleDist(empty = true),
                    background_center::Union{Real,Nothing} = nothing,
                    fixed_position::Bool  = false,
                    σ_μ_default::Real = 0.6,
                    σ_σ_default::Real = 0.6,
                    # :bat-only kwargs (ignored otherwise)
                    nsamples::Int  = 10^5,
                    nchains::Int   = 4,
                    n_thin::Int    = 2_000,
                    n_sigma_detect::Real = 3.0)
        -> (result::NamedTuple, report::NamedTuple)

Bayesian (`method = :bat`) or maximum-likelihood (`method = :mle`)
binned-Poisson fit of a Gaussian γ-line on a polynomial background.

Returns the standard LSF `(result, report)` pair:

  * `result` — `Measurement`-valued NT of the fit parameters (`μ, σ, n,
    background[, background_slope[, background_curv]]`) plus `fwhm`, `fit_func`
    and a `gof = (pvalue, chi2, dof, covmat, ..., converged)` sub-NT. For
    `method = :bat`, also `S_samples::Vector{Float64}` (thinned signal-count
    posterior), `S_weights`, `ul_90` (90 % Bayesian UL on n).
  * `report` — `(v, h, f_fit, f_components, gof)` so the existing
    `LegendMakie.lplot!(report::NamedTuple{(:v, :h, :f_fit, :f_components, :gof)})`
    recipe just works.
"""
function fit_gamma_line(h::Histogram, ps::NamedTuple;
                         fit_func::Symbol      = :gauss_flat,
                         method::Symbol        = :mle,
                         uncertainty::Bool     = true,
                         pseudo_prior::NamedTupleDist = NamedTupleDist(empty = true),
                         background_center::Union{Real,Nothing} = nothing,
                         fixed_position::Bool  = false,
                         σ_μ_default::Real = 0.6,
                         σ_σ_default::Real = 0.6,
                         nsamples::Int  = 10^5,
                         nchains::Int   = 4,
                         n_thin::Int    = 2_000,
                         n_sigma_detect::Real = 3.0)
    pp = get_gamma_line_pseudo_prior(h, ps, fit_func;
                                       pseudo_prior = pseudo_prior,
                                       fixed_position = fixed_position,
                                       σ_μ_default = σ_μ_default,
                                       σ_σ_default = σ_σ_default)
    bc = background_center === nothing ? Float64(ps.peak_pos) : Float64(background_center)
    fit_function = get_gamma_line_fit_functions(; background_center = bc)[fit_func]

    if method === :mle
        return _fit_gamma_line_mle(h, fit_func, fit_function, pp;
                                    background_center = bc, uncertainty = uncertainty)
    elseif method === :bat
        return _fit_gamma_line_bat(h, fit_func, fit_function, pp;
                                    background_center = bc,
                                    nsamples = nsamples, nchains = nchains,
                                    n_thin = n_thin, n_sigma_detect = n_sigma_detect)
    else
        throw(ArgumentError("fit_gamma_line: method must be :mle or :bat, got $method"))
    end
end
export fit_gamma_line

# ── MLE path (clone of fit_single_peak_th228's MLE block, simplified) ────────

function _fit_gamma_line_mle(h::Histogram, fit_func::Symbol,
                              fit_function::Function, pp::NamedTupleDist;
                              background_center::Real, uncertainty::Bool)
    f_trafo = BAT.DistributionTransform(Normal, pp)
    v_init  = Vector(mean(f_trafo.target_dist))

    edges_lo, edges_hi = extrema(h.edges[1])
    f_loglike = let f_fit = fit_function, h = h
        v -> hist_loglike(x -> x in Interval(edges_lo, edges_hi) ? f_fit(x, v) : zero(typeof(x)), h)
    end

    optf   = OptimizationFunction((u, _) -> ((-) ∘ f_loglike ∘ inverse(f_trafo))(u), AutoForwardDiff())
    optpro = OptimizationProblem(optf, v_init, ())
    res    = solve(optpro, Optimization.LBFGS(), maxiters = 3000)
    converged = (res.retcode == ReturnCode.Success)
    converged || @warn "fit_gamma_line MLE did not converge" fit_func ps.peak_pos
    v_ml = inverse(f_trafo)(res.u)

    v_keys = keys(pp)
    f_loglike_array(v) = -hist_loglike(x -> fit_function(x, NamedTuple{v_keys}(v)), h)

    if uncertainty && converged
        H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml))
        cov = nearestSPD(inv(H))
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(cov))), v_ml)
        pval, chi2, dof = p_value_poissonll(fit_function, h, v_ml)
        residuals, residuals_norm, _, _ = get_residuals(fit_function, h, v_ml)

        fwhm     = 2.354820045030949 * v_ml.σ
        fwhm_err = 2.354820045030949 * v_ml_err.σ

        meas = NamedTuple{keys(v_ml)}(
            Tuple(measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml))
        )
        gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = cov,
               mean_residuals   = mean(residuals_norm),
               median_residuals = median(residuals_norm),
               std_residuals    = std(residuals_norm),
               converged        = converged)
        result = merge(meas, (fwhm = measurement(fwhm, fwhm_err), fit_func = fit_func, gof = gof))
        report = (
            v             = v_ml,
            h             = h,
            f_fit         = x -> fit_function(x, v_ml),
            f_components  = gamma_line_peakshape_components(fit_func, v_ml; background_center = background_center),
            gof           = merge(gof, (residuals = residuals, residuals_norm = residuals_norm)),
        )
        return result, report
    else
        fwhm = 2.354820045030949 * v_ml.σ
        meas = NamedTuple{keys(v_ml)}(
            Tuple(measurement(v_ml[k], NaN) for k in keys(v_ml))
        )
        gof    = (converged = converged,)
        result = merge(meas, (fwhm = measurement(fwhm, NaN), fit_func = fit_func, gof = gof))
        report = (
            v            = v_ml,
            h            = h,
            f_fit        = x -> fit_function(x, v_ml),
            f_components = gamma_line_peakshape_components(fit_func, v_ml; background_center = background_center),
            gof          = NamedTuple(),
        )
        return result, report
    end
end

# ── BAT MCMC path ────────────────────────────────────────────────────────────

function _fit_gamma_line_bat(h::Histogram, fit_func::Symbol,
                              fit_function::Function, pp::NamedTupleDist;
                              background_center::Real,
                              nsamples::Int, nchains::Int,
                              n_thin::Int, n_sigma_detect::Real)
    edges_lo, edges_hi = extrema(h.edges[1])
    f_loglike(v) = hist_loglike(x -> x in Interval(edges_lo, edges_hi) ? fit_function(x, v) : zero(typeof(x)), h)
    posterior = BAT.PosteriorMeasure(DensityInterface.logfuncdensity(f_loglike), pp)

    samples = BAT.bat_sample(posterior,
                              BAT.MCMCSampling(mcalg = BAT.MetropolisHastings(),
                                                nsteps  = nsamples,
                                                nchains = nchains)).result

    # Marginal stats per parameter — paired thinning for cross-parameter use
    n = length(samples)
    idx = n <= n_thin ? collect(1:n) : unique(round.(Int, range(1, n; length = n_thin)))
    v_keys = keys(pp)
    wts    = StatsBase.weights(Float64.([Float64(s.weight) for s in samples]))
    mode_v = BAT.mode(samples)

    function _qsigma(key)
        vals = Float64[getproperty(s.v, key) for s in samples]
        lo, hi = quantile(vals, wts, 0.16), quantile(vals, wts, 0.84)
        return Float64((hi - lo) / 2)
    end
    meas = NamedTuple{v_keys}(
        Tuple(measurement(Float64(getproperty(mode_v, k)), _qsigma(k)) for k in v_keys)
    )

    # Posterior signal-count tail → UL_90 + thinned samples for downstream paired
    S_samples = Float64[s.v.n for s in samples[idx]]
    S_weights = Float64[Float64(s.weight) for s in samples[idx]]
    full_S    = Float64[s.v.n for s in samples]
    ul_90     = Float64(quantile(full_S, wts, 0.9))
    n_mode, n_sig = value(meas.n), uncertainty(meas.n)
    detection = (isfinite(n_mode) && n_sig > 0 && n_mode / n_sig >= n_sigma_detect) ? :detected : :upper_limit

    # GoF at the mode (uses the same chi-squared helper as the MLE path so
    # plotting/recipes have a meaningful `gof.residuals` to draw).
    pval, chi2, dof = p_value_poissonll(fit_function, h, mode_v)
    residuals, residuals_norm, _, _ = get_residuals(fit_function, h, mode_v)
    gof = (pvalue = pval, chi2 = chi2, dof = dof,
           covmat = nothing,
           mean_residuals   = mean(residuals_norm),
           median_residuals = median(residuals_norm),
           std_residuals    = std(residuals_norm),
           converged        = true)

    fwhm     = 2.354820045030949 * meas.σ
    result = merge(meas,
        (fwhm = fwhm, fit_func = fit_func, gof = gof,
         S_samples = S_samples, S_weights = S_weights, ul_90 = ul_90, detection = detection))
    report = (
        v            = mode_v,
        h            = h,
        f_fit        = x -> fit_function(x, mode_v),
        f_components = gamma_line_peakshape_components(fit_func, mode_v; background_center = background_center),
        gof          = merge(gof, (residuals = residuals, residuals_norm = residuals_norm)),
    )
    return result, report
end
