# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    fit_pmt_spe(x::AbstractVector{<:Real}; kwargs...)

Fits the single-photoelectron (1 p.e.) pulse-height spectrum of a photomultiplier
tube (PMT) with an exponential and the 1 p.e. and 2 p.e. peaks.

The model, in counts per unit of `x`, is

    f(x) = n_exp / τ * exp(-(x - x_min) / τ) + n_1 * N(x; μ1, σ1) + r2 * n_1 * N(x; 2 μ1, √2 σ1)

where `N` is a normalized Gaussian. `n_exp` and `n_1` are the numbers of entries of the
exponential (above `x_min`) and of the 1 p.e. peak, and `r2` is the ratio of 2 p.e. to 1 p.e.
entries, bounded by 1 so that the fit cannot place the 1 p.e. peak at half its position.

The pedestal is a sharp, non-Gaussian peak at the low end and is not fitted. The fit starts
at the valley between the pedestal and the 1 p.e. peak. The histogram of `x` in `hist_range`
with bins of `bin_width` is smoothed with a 3-bin moving average. Starting at
`first(peak_search_range)`, the valley is the first bin after which the smoothed spectrum rises
for two bins. The start value of `μ1` is the maximum of the smoothed spectrum between the valley
and `last(peak_search_range)`. The fit range starts at `x_min = valley - x_min_offset` (at least
`first(hist_range)`), and the binned Poisson likelihood of the histogram above `x_min` is
maximized, with L-BFGS-B and, if that does not converge, with BOBYQA. The bounds of the
parameters are `μ1` in `peak_search_range`, `σ1` between `bin_width / 2` and the width of
`peak_search_range`, `τ` between `bin_width / 2` and the fit range, `r2` in `[0, 1]`, and
positive `n_exp` and `n_1`.

## Arguments
* `x::AbstractVector{<:Real}`: Pulse heights. Non-finite values are ignored.

## Keywords
* `hist_range = (0.0, 100.0)`: Range of the histogram, two numbers (a tuple or a vector) in the unit of `x`. It must be a whole number of bins.
* `bin_width::Real = 1.0`: Bin width.
* `peak_search_range = (6.0, 60.0)`: Range in which the valley and the 1 p.e. peak are searched, two numbers within `hist_range`.
* `x_min_offset::Real = 2.0`: Distance of the fit start below the valley.
* `σ1_rel_init::Real = 0.33`: Start value of `σ1` relative to the start value of `μ1`.
* `τ_init::Real = 5.0`: Start value of `τ`.
* `r2_init::Real = 0.1`: Start value of `r2`.
* `min_counts::Real = 50`: Minimum number of entries above `x_min`.
* `uncertainty::Bool = true`: Compute the parameter uncertainties and the goodness of fit.

The defaults are for pulse heights in ADC units with the 1 p.e. peak at about 10 to 20 units.

An `ArgumentError` is thrown if a range does not have two elements or `peak_search_range`
does not lie within `hist_range`, if `hist_range` is not a whole number of bins, if
`peak_search_range` covers fewer than three bins, if the smoothed spectrum does not rise again in
`peak_search_range` (no 1 p.e. peak above the pedestal), or if there are fewer than `min_counts`
entries above `x_min`. A warning is emitted if the fit does not converge.

## Returns
A tuple `(result, report)`. `result` is a `NamedTuple` with the fields
* `μ1`, `σ1`: position and width of the 1 p.e. peak.
* `τ`: decay constant of the exponential.
* `n_exp`, `n_1`: numbers of entries of the exponential and of the 1 p.e. peak.
* `r2`: ratio of 2 p.e. to 1 p.e. entries.
* `x_min`: start of the fit range.
* `gof`: goodness of fit. With `uncertainty`, it has `converged`, `pvalue`, `chi2`, `dof`,
  `covmat`, `residuals`, `residuals_norm` and `bin_centers`; without, only `converged`.

With `uncertainty`, the parameters are `Measurement`s. Their uncertainties are `NaN` if the fit
did not converge or the Hessian matrix is not finite.

`report` contains the histogram `h` of all of `hist_range`, the model `f_fit` (`NaN` below
`x_min`), `f_components` (a `NamedTuple` with the functions `exponential`, `pe1` and `pe2`),
`x_min`, `μ1` and `gof`, for plotting with `LegendMakie.lplot`.
"""
function fit_pmt_spe(x::AbstractVector{<:Real}; hist_range = (0.0, 100.0), bin_width::Real = 1.0, peak_search_range = (6.0, 60.0),
                     x_min_offset::Real = 2.0, σ1_rel_init::Real = 0.33,
                     τ_init::Real = 5.0, r2_init::Real = 0.1, min_counts::Real = 50, uncertainty::Bool = true)
    length(hist_range) == 2 || throw(ArgumentError("fit_pmt_spe: hist_range must have two elements, got $hist_range"))
    length(peak_search_range) == 2 || throw(ArgumentError("fit_pmt_spe: peak_search_range must have two elements, got $peak_search_range"))
    lo, hi = float(first(peak_search_range)), float(last(peak_search_range))
    first(hist_range) <= lo < hi <= last(hist_range) ||
        throw(ArgumentError("fit_pmt_spe: peak_search_range $peak_search_range must lie within hist_range $hist_range"))
    edges = float(first(hist_range)):float(bin_width):float(last(hist_range))
    last(edges) ≈ last(hist_range) ||
        throw(ArgumentError("fit_pmt_spe: hist_range $hist_range is not a whole number of bins of width $bin_width"))
    h = fit(Histogram, filter(isfinite, x), edges)
    centers = midpoints(edges)

    # 3-bin moving average, zero-padded at the ends
    counts = float.(h.weights)
    padded = vcat(0.0, counts, 0.0)
    smoothed = [(padded[i] + padded[i+1] + padded[i+2]) / 3 for i in eachindex(counts)]

    # valley: where the falling pedestal tail turns into the rising 1 p.e. peak
    in_search = findall(c -> lo <= c <= hi, centers)
    length(in_search) >= 3 ||
        throw(ArgumentError("fit_pmt_spe: peak_search_range $peak_search_range covers fewer than 3 bins of width $bin_width"))
    i_valley = findfirst(i -> smoothed[i+1] > smoothed[i] && smoothed[i+2] > smoothed[i], in_search[1:end-2])
    isnothing(i_valley) &&
        throw(ArgumentError("fit_pmt_spe: the spectrum does not rise again in $peak_search_range: no 1 p.e. peak above the pedestal"))
    i_valley = in_search[i_valley]
    above_valley = filter(>(i_valley), in_search)
    i_peak = above_valley[argmax(view(smoothed, above_valley))]
    μ1_seed = centers[i_peak]
    valley = centers[i_valley]

    # histogram of the fit range
    i_min = findfirst(>=(max(valley - x_min_offset, first(hist_range))), edges)
    x_fit_min = edges[i_min]
    h_fit = Histogram(edges[i_min:end], h.weights[i_min:end])
    n_fit = sum(h_fit.weights)
    n_fit >= min_counts ||
        throw(ArgumentError("fit_pmt_spe: $n_fit entries above x_min = $x_fit_min, fewer than min_counts = $min_counts"))

    # start values from the data and the keywords, inside the bounds
    σ1_lo, σ1_hi = bin_width / 2, hi - lo
    τ_lo, τ_hi = bin_width / 2, last(edges) - x_fit_min
    σ1_seed = clamp(σ1_rel_init * μ1_seed, σ1_lo, σ1_hi)
    n1_seed = max(sum(counts[findall(c -> abs(c - μ1_seed) <= σ1_seed, centers)]) / 0.68, 1.0)
    nexp_seed = max(n_fit - n1_seed, 1.0)
    inside(v, a, b) = clamp(v, a + 1e-3 * (b - a), b - 1e-3 * (b - a))
    v_seed = (μ1 = inside(μ1_seed, lo, hi), σ1 = inside(σ1_seed, σ1_lo, σ1_hi), τ = inside(τ_init, τ_lo, τ_hi), n_exp = nexp_seed,
              n_1 = n1_seed, r2 = inside(r2_init, 0.0, 1.0))

    f_fit(x, v) = v.n_exp / v.τ * exp(-(x - x_fit_min) / v.τ) + v.n_1 * gauss_pdf(x, v.μ1, v.σ1) +
                  v.r2 * v.n_1 * gauss_pdf(x, 2 * v.μ1, sqrt(2) * v.σ1)

    # pseudo priors: their support gives the bounds; the fit starts at v_seed
    pseudo_prior = NamedTupleDist(μ1 = truncated(Normal(μ1_seed, (hi - lo) / 4), lo, hi),
                                  σ1 = truncated(Normal(σ1_seed, σ1_seed), σ1_lo, σ1_hi), τ = truncated(Normal(τ_init, τ_init), τ_lo, τ_hi),
                                  n_exp = weibull_from_mx(nexp_seed, 2 * nexp_seed), n_1 = weibull_from_mx(n1_seed, 2 * n1_seed),
                                  r2 = truncated(Normal(r2_init, 0.5), 0.0, 1.0))
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)
    v_init = Vector(f_trafo(v_seed))

    f_loglike = let f_fit = f_fit, ll = HistLogLike(h_fit)
        v -> ll(x -> f_fit(x, v))
    end

    # MLE
    optf = OptimizationFunction((u, p) -> ((-) ∘ f_loglike ∘ inverse(f_trafo))(u), AutoForwardDiff())
    res = solve(OptimizationProblem(optf, v_init, ()), LBFGSB(); maxiters = 3000)
    # A weakly constrained exponential (few entries below the 1 p.e. peak) leaves a flat
    # direction in which the line search of LBFGSB can stop early: continue from there with
    # a derivative-free method.
    if res.retcode != ReturnCode.Success
        res = solve(OptimizationProblem(optf, res.u, ()), NLopt.LN_BOBYQA(); maxiters = 5000)
    end
    converged = (res.retcode == ReturnCode.Success)
    converged || @warn "fit_pmt_spe: fit did not converge (retcode $(res.retcode))"
    v_ml = inverse(f_trafo)(res.u)

    if uncertainty
        n_par = length(v_ml)
        param_covariance = fill(NaN, n_par, n_par)
        if converged
            f_loglike_array(v) = -f_loglike(array_to_tuple(v, v_ml))
            H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml))
            if all(isfinite, H) && !all(iszero, H)
                param_covariance = inv(H)
                isposdef(param_covariance) || (param_covariance = nearestSPD(param_covariance))
            else
                @warn "fit_pmt_spe: Hessian matrix is not finite, no uncertainties"
            end
        end
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)

        pval, chi2, dof = p_value_poissonll(f_fit, h_fit, v_ml)
        residuals, residuals_norm, _, bin_centers = get_residuals(f_fit, h_fit, v_ml)
        gof = (converged = converged, pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, residuals = residuals,
               residuals_norm = residuals_norm, bin_centers = bin_centers)
        pars = NamedTuple{keys(v_ml)}(measurement.(values(v_ml), values(v_ml_err)))
    else
        gof = (converged = converged,)
        pars = v_ml
    end

    result = merge(pars, (x_min = x_fit_min, gof = gof))
    report = (h = h, f_fit = x -> x >= x_fit_min ? f_fit(x, v_ml) : NaN,
              f_components = (exponential = x -> x >= x_fit_min ? v_ml.n_exp / v_ml.τ * exp(-(x - x_fit_min) / v_ml.τ) : NaN,
                              pe1 = x -> v_ml.n_1 * gauss_pdf(x, v_ml.μ1, v_ml.σ1),
                              pe2 = x -> v_ml.r2 * v_ml.n_1 * gauss_pdf(x, 2 * v_ml.μ1, sqrt(2) * v_ml.σ1)),
              x_min = x_fit_min, μ1 = result.μ1, gof = gof)
    return result, report
end
export fit_pmt_spe
