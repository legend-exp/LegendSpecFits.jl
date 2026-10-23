"""
    pmts_calibration(pulse_heights::AbstractVector{<:Real}; kwargs...)

Single-photoelectron (1 p.e.) fit for one muon-veto PMT, LED calibration data.
Port of `fit_spe` / `classify` from `mvcal.py` (L. Steinhart, muon-veto operations).

Model (fitted ABOVE the pedestal, no pedestal term):

    f(x) = b·exp(-x/τ)                                   exponential (pedestal tail)
         + a1·exp(-(x-μ1)²/2σ1²)                          1 p.e.
         + r2·a1·exp(-(x-2μ1)²/2(√2σ1)²)                  2 p.e., r2 ∈ [0, 1]

The pedestal of the smoothed pulse height is a sharp, non-Gaussian peak at 0-2 LSB,
so the fit starts at the valley between pedestal and 1 p.e. peak (`x_min`).
`r2 ≤ 1`: at LED occupancies the 2 p.e. peak is never higher than the 1 p.e. peak;
without this the fit can put the 1 p.e. peak at half its position.

Procedure: histogram 0…`hist_max` in `bin_width` bins, 3-bin moving average;
1 p.e. seed = maximum of the smoothed histogram in [`peak_search_low`, `peak_search_high`];
valley = minimum of the smoothed histogram in [`valley_low`, seed); `x_min = max(valley - 2, x_min_floor)`.

Status:
  * `:fit_failed` — fewer than `min_counts_fit` entries above `x_min`, or the fit threw
  * `:bad_fit`    — `μ1_err > max_mu1_err` or `χ²/ndf > max_chi2ndf` or μ1 ∉ (`mu1_low`, `mu1_high`)
                    or fewer than `min_n_spe` entries above `x_min`
  * `:ok`         — otherwise

kwargs (defaults = mvcal.py): `hist_max = 100`, `bin_width = 1`, `peak_search_low = 6`,
`peak_search_high = 60`, `valley_low = 2`, `x_min_floor = 3`, `min_counts_fit = 50`,
`max_mu1_err = 0.5`, `max_chi2ndf = 5`, `mu1_low = 5`, `mu1_high = 55`, `min_n_spe = 500`,
`energy_var = "pulse_height"` (column name used in `cal_func`). Other kwargs are ignored.

Returns `(result, report)`:
    result : (μ, σ, chi2ndf, n_spe, x_min, status, fit_failed, cal_func, error)
             μ is a `Measurement` (μ1 ± fit error); `cal_func = "1/μ1 * <energy_var> * 1e"`
             (photoelectrons, unit `e` as in Juleana production pars and the evt muon cut) for
             status `:ok`, otherwise `"NaN * <energy_var> * 1e"` (the caller decides on a fallback)
    report : (edges, counts, popt, x_min) — `popt = [b, τ, μ1, σ1, a1, r2]` or empty
"""
function pmts_calibration end
export pmts_calibration

_spe_gauss(x, a, μ, σ) = a * exp(-0.5 * ((x - μ) / σ)^2)

"""
    pmts_spe_components(x, p)

Components `(exponential, pe1, pe2)` of the 1 p.e. model for parameters
`p = [b, τ, μ1, σ1, a1, r2]` (see [`pmts_calibration`](@ref)).
"""
function pmts_spe_components(x, p)
    b, τ, μ1, σ1, a1, r2 = p
    (exponential = @.(b * exp(-x / τ)),
     pe1 = _spe_gauss.(x, a1, μ1, σ1),
     pe2 = _spe_gauss.(x, r2 * a1, 2μ1, sqrt(2) * σ1))
end
export pmts_spe_components

_pmt_nan_cal(energy_var) = "NaN * $energy_var * 1e"

pmts_spe_model(x, p) = (c = pmts_spe_components(x, p); c.exponential .+ c.pe1 .+ c.pe2)
export pmts_spe_model

function pmts_calibration(pulse_heights::AbstractVector{<:Real};
    hist_max::Real             = 100.0,
    bin_width::Real            = 1.0,
    peak_search_low::Real      = 6.0,
    peak_search_high::Real     = 60.0,
    valley_low::Real           = 2.0,
    x_min_floor::Real          = 3.0,
    min_counts_fit::Real       = 50,
    max_mu1_err::Real          = 0.5,
    max_chi2ndf::Real          = 5.0,
    mu1_low::Real              = 5.0,
    mu1_high::Real             = 55.0,
    min_n_spe::Real            = 500,
    energy_var::AbstractString = "pulse_height",
    kwargs...
)
    v = filter(isfinite, pulse_heights)
    edges   = 0.0:float(bin_width):float(hist_max)
    centers = collect(edges[1:end-1] .+ step(edges) / 2)
    cnt = float.(fit(Histogram, v, edges).weights)
    # 3-bin moving average, zero-padded (= np.convolve(cnt, ones(3)/3, "same"))
    pad = vcat(0.0, cnt, 0.0)
    sm  = [(pad[i] + pad[i+1] + pad[i+2]) / 3 for i in eachindex(cnt)]

    m  = (centers .>= peak_search_low) .& (centers .<= peak_search_high)
    i1 = findall(m)[argmax(sm[m])]
    μ10, a10 = centers[i1], sm[i1]
    mv = (centers .>= valley_low) .& (centers .< μ10)
    iv = any(mv) ? findall(mv)[argmin(sm[mv])] : 3
    x_min = max(centers[iv] - 2, float(x_min_floor))

    sel = centers .>= x_min
    x, y = centers[sel], cnt[sel]
    n_spe = Int(sum(y))
    report_empty = (edges = collect(edges), counts = cnt, popt = Float64[], x_min = x_min)
    _failed(msg) = ((μ = measurement(NaN, NaN), σ = NaN, chi2ndf = NaN, n_spe = n_spe, x_min = x_min,
                     status = :fit_failed, fit_failed = true, cal_func = _pmt_nan_cal(energy_var), error = msg),
                    report_empty)
    n_spe < min_counts_fit && return _failed("no signal above the pedestal")

    err = sqrt.(max.(y, 1.0))
    p0 = [max(sm[iv], 1.0) * exp(centers[iv] / 5), 5.0, μ10, 0.33 * μ10, max(a10, 1.0), 0.1]
    lo = [0.0, 0.5, 4.0, 1.0, 0.0, 0.0]
    hi = [Inf, 50.0, 60.0, 30.0, Inf, 1.0]
    p0 = clamp.(p0, lo .+ 1e-6, hi .- 1e-6)
    f = try
        curve_fit(pmts_spe_model, x, y, 1.0 ./ err .^ 2, p0; lower = lo, upper = hi, maxIter = 100_000)
    catch e
        return _failed(first(sprint(showerror, e), 80))
    end
    popt = coef(f)
    # weights 1/err² -> covariance inv(JᵀWJ), i.e. absolute errors (scipy absolute_sigma=True)
    perr = try stderror(f) catch; fill(NaN, length(popt)) end
    r = (y .- pmts_spe_model(x, popt)) ./ err
    chi2ndf = sum(abs2, r) / (length(x) - length(popt))
    μ1, μ1_err, σ1 = popt[3], perr[3], popt[4]

    # NaN fit errors count as bad (mvcal.py would let them pass)
    status = !(μ1_err <= max_mu1_err) || chi2ndf > max_chi2ndf || !(mu1_low < μ1 < mu1_high) || n_spe < min_n_spe ?
             :bad_fit : :ok
    cal_func = status == :ok ? "1/$μ1 * $energy_var * 1e" : _pmt_nan_cal(energy_var)
    return (μ = measurement(μ1, μ1_err), σ = σ1, chi2ndf = chi2ndf, n_spe = n_spe, x_min = x_min,
            status = status, fit_failed = status != :ok, cal_func = cal_func, error = ""),
           (edges = collect(edges), counts = cnt, popt = popt, x_min = x_min)
end
