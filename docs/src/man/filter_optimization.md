# Filter Optimization

The energy filter of a high-purity germanium (HPGe) detector is a trapezoidal filter with a rise time (RT) and a flat-top time (FT). The filter optimization scans these times and selects the values that give the best energy resolution. Three functions evaluate such scans: [`fit_enc_sigmas`](@ref) selects the rise time from the electronic noise, [`fit_fwhm_ft`](@ref) selects the flat-top time from the resolution of the 2614.5 keV line, and [`fit_sf_wl`](@ref) selects the window length of the A/E filter from the survival fraction of the single escape peak. The optimization of the SiPM filter is described in [SiPM Calibration](@ref).

All functions follow the `(result, report)` convention. `result` holds the optimum as a `Measurement` whose uncertainty is the step of the scan grid, and `report` holds the figure of merit for every grid point for plotting with `lplot` from LegendMakie. They throw an error if no grid point gives a valid fit.

## Grid Layout

The functions receive the output of the processing for every grid point. The input has one row per grid point (rise time, flat-top time or window length) and one column per event: row `i` contains the values obtained with `grid[i]`. A `Matrix` has this layout; an array of events where each event holds one value per grid point works as well, because the code reads it through its flat view. The grid itself is a range with a step, for example `range(0.5u"μs", step = 0.5u"μs", length = 20)`.

Grid points without data (all zero or all non-finite) are skipped. The optimum is a grid point, so the resolution of the result is the grid step.

## Rise Time from the Noise

The electronic noise of the filter output depends on the rise time. [`fit_enc_sigmas`](@ref) takes a matrix of values with one row per rise time, where each row follows a Gaussian distribution whose width is the equivalent noise charge (ENC) at that rise time. The arguments `min_enc` and `max_enc` limit the range of values considered, `nbins` is the number of histogram bins, and `rel_cut_fit` is the `relative_cut` of [`cut_single_peak`](@ref).

For each rise time, the function

1. finds the peak region of the row with `cut_single_peak`,
2. fits a truncated Gaussian to it with [`fit_single_trunc_gauss`](@ref), and
3. takes the standard deviation of the fit as the ENC of this rise time.

The result is `rt`, the rise time with the smallest ENC, and `min_enc`, that ENC. The report contains the rise times and ENC values of all successful fits.

The example scans 21 rise times of synthetic data whose noise has a minimum at the center of the grid:

```@example filteropt
using LegendSpecFits, Unitful, Random, Measurements

Random.seed!(1)
rts = range(0.5u"μs", step = 0.5u"μs", length = 21)
width = 0.2 .+ 0.6 .* ((-1:0.1:1) .^ 2)          # noise width per rise time
enc_grid = width .* randn(5_000)' .+ 2            # rows: rise times, columns: events
result_rt, report_rt = fit_enc_sigmas(enc_grid, rts, 0.0, 4.0, 100, 0.1)
(rt = result_rt.rt, min_enc = result_rt.min_enc)
```

The true minimum is at 5.5 μs with a width of 0.2.

## Flat-Top Time from the Resolution

After the rise time is fixed, the flat-top time is scanned. [`fit_fwhm_ft`](@ref) takes a matrix of energies with one row per flat-top time. The energies are in arbitrary units (for example uncalibrated filter output); `min_e` and `max_e` bracket the 2614.5 keV peak in these units and `rel_cut_fit` is the `relative_cut` of [`cut_single_peak`](@ref). The method has two forms, with and without the drift-time parameter `qdrift` of every event.

### Scan Without Charge-Trapping Correction

Without `qdrift`, the function, for each flat-top time,

1. skips the row if fewer than 100 energies lie between `min_e` and `max_e`,
2. restricts the energies to a window around the peak, found with `cut_single_peak`,
3. fits the gamma peak shape with [`fit_single_peak_th228`](@ref) (without uncertainties), and
4. stores the FWHM and the position of the highest histogram bin.

The FWHM values are converted to keV with a single factor that places the mean peak position of all flat-top times at `peak` (keyword, 2614.5 keV by default). Among the flat-top times with an FWHM within `ft_fwhm_tol` (default 0.1 keV) of the minimum FWHM, the shortest one is selected. A longer flat top is therefore selected only if it improves the FWHM by more than `ft_fwhm_tol`. The result contains `ft`, the optimal flat-top time, and `min_fwhm`, its FWHM in keV. The argument `rt` records the rise time of the scan and does not enter the computation.

The example scans 10 flat-top times of synthetic energies:

```@example filteropt
Random.seed!(2)
fts = range(0.5u"μs", step = 0.5u"μs", length = 10)
sigmas = 2.0 .+ 0.15 .* abs.((1:10) .- 4)         # best resolution at the fourth flat-top time
e_grid = reduce(hcat, [2614.5 .+ s .* randn(6_000) for s in sigmas])'
result_ft, report_ft = fit_fwhm_ft(Matrix(e_grid), fts, 1.0u"μs", 2500.0, 2700.0, 0.5)
(ft = result_ft.ft, min_fwhm = result_ft.min_fwhm)
```

The resolution of the fourth flat-top time, $2\sqrt{2\ln 2} \cdot 2.0 = 4.71$ keV, is the smallest. The FWHM values have no uncertainty (`NaN`) because the fits run without the uncertainty calculation.

### Evaluation of One Filter Setting

With `qdrift`, every filter setting is evaluated by [`evaluate_energy_point`](@ref). It

1. keeps the events with finite, positive energy and `qdrift`, finds the peak mode with `cut_single_peak` on `(min_e, max_e)`, and calibrates the energies to keV with `m_cal_simple = peak / mode`,
2. keeps the events within 15 half-widths of the mode,
3. with `apply_ctc = true` (the default), fits the linear charge-trapping correction with [`ctc_energy`](@ref) in the `window` around `peak`, and fits the peak after the correction; with `apply_ctc = false` it fits the uncorrected peak, and
4. computes the width and shape metrics of the fitted peak with [`peak_width_metrics`](@ref).

[`peak_width_metrics`](@ref) uses only the signal part of the fitted shape, the Gaussian plus the low-energy tail, without step and background. Because it includes the tail, it detects a setting that narrows the Gaussian core but moves events into the tail. It returns

* `fwhm`: full width at half maximum of the signal shape,
* `w90`: length of the shortest interval that contains the fraction `containment` (default 0.9) of the signal shape,
* `asym`: integral of the shape below the peak position divided by the integral above it,
* `tail_fraction` and `tail_scale`: the fraction of events in the tail and its decay length, and
* `centroid`: mean of the shape.

[`evaluate_energy_point`](@ref) scales these widths to keV with the ratio of `peak` to the fitted peak position and adds `fwhm_before` (FWHM of the uncorrected fit), `fct` (the fitted correction coefficient), `ct_spread` (the FWHM-equivalent spread in keV of the drift-time term that the correction removes), `m_cal_simple`, `ctc_converged` and `n_events`.

### Objectives and Tie Rule

The objective that the optimization minimizes is chosen with the keyword `objective` of [`objective_value`](@ref):

| `objective`  | Value                                          |
|:-------------|:-----------------------------------------------|
| `:fwhm`      | `fwhm` (default)                               |
| `:w90`       | `w90`                                          |
| `:asym`      | `asym`                                         |
| `:fwhm_asym` | `fwhm * asym`                                  |
| `:fwhm_ct`   | $\sqrt{\mathrm{fwhm}^2 + (k\,\mathrm{ct\_spread})^2}$ |

For `:fwhm_ct`, $k$ is the keyword `ctc_rel_uncertainty`, the relative uncertainty of the charge-trapping correction. The objective penalizes settings that rely on a large correction: the correction removes a spread `ct_spread` from the peak, and a fraction $k$ of it is treated as unknown. It requires `apply_ctc = true`. The other objectives ignore `ctc_rel_uncertainty`.

Settings whose objectives differ by less than the fit can resolve are tied. [`select_optimum`](@ref) takes the smallest objective and treats all candidates with an objective of at most `min * (1 + rel_tol)` as tied. Among them, the candidate with the lexicographically smallest key tuple wins. [`fit_fwhm_ft`](@ref) uses the key `(-ft,)`, so a tie is resolved toward the longest flat-top time, with `rel_tol = 0.005` by default.

### Scan With Charge-Trapping Correction

With `qdrift`, [`fit_fwhm_ft`](@ref) evaluates the rows in parallel and selects the flat-top time with the objective and tie rule above. The keywords `peak`, `window`, `n_bins`, `containment`, `objective`, `ctc_rel_uncertainty` and `rel_tol` and the positional `apply_ctc` set the evaluation. A row whose evaluation throws (too few events, an invalid fit) is kept in the report with the error text in `status` and `NaN` metrics, and a warning is logged. The call throws an error only if no row succeeds.

The result contains `ft` (with the step of the grid as uncertainty), `min_fwhm` (the FWHM in keV of the selected row), `min_objective` and `objective`. The report contains these fields, the `table` with one row per flat-top time (the columns `ft` and `status` followed by the fields of `evaluate_energy_point`) and `objective_values`.

The example evaluates the same synthetic energies with random drift-time parameters that the energies do not depend on, so the fitted correction is small and `:fwhm_ct` selects the same flat-top time as `:fwhm`:

```@example filteropt
Random.seed!(3)
qdrift = 100 .+ 20 .* randn(6_000)
e_matrix = Matrix(e_grid)
result_c, report_c = fit_fwhm_ft(e_matrix, fts, qdrift, 1.0u"μs", 2500.0, 2700.0, 0.5)
result_ct, _ = fit_fwhm_ft(e_matrix, fts, qdrift, 1.0u"μs", 2500.0, 2700.0, 0.5;
    objective = :fwhm_ct, ctc_rel_uncertainty = 0.1)
(ft = result_c.ft, min_fwhm = result_c.min_fwhm, ft_ct = result_ct.ft,
    status = unique(report_c.table.status))
```

The columns of `report_c.table` hold the metrics for every flat-top time, for example `report_c.table.fwhm`, `report_c.table.w90` and `report_c.table.fct`.

### Joint Rise-Time and Flat-Top Search

The rise time and the flat-top time are not independent. [`optimize_rtft`](@ref) searches both together on a lattice of multiples of `lattice`. Its input `f_energy(rt, ft)` returns the energies of the waveforms filtered with one setting, aligned with `qdrift`; the search calls it only for settings it evaluates.

The search starts at the seed `(rt0, ft0)` rounded to the lattice and evaluates all admissible neighbors at the current steps `step0` in parallel: the settings inside `rt_bounds` and `ft_bounds`, accepted by `feasible(rt, ft)`, and not yet evaluated. It moves to the best neighbor if its objective is smaller than `(1 - rel_tol)` times the current one. Otherwise it halves both steps, down to one lattice step. It stops when both steps equal the lattice and no neighbor improves (`converged = true`), or after `max_evals` evaluations (`converged = false`). The optimum is selected by `select_optimum` among the final point and its evaluated neighbors with the key `(rt, -ft)`: within the tolerance, the shortest rise time and then the longest flat-top time win.

The result contains `rt`, `ft`, the metrics `fwhm`, `w90`, `asym`, `fct` and `ct_spread` at the optimum, `min_objective`, `objective`, `n_evals` and `converged`. The report contains `table` (one row per evaluation with the columns `rt`, `ft`, `stage`, `step_rt`, `step_ft` and `status`, followed by the fields of `evaluate_energy_point`), the accepted positions `path` and `objective_values`. A call looks as follows:

```julia
result, report = optimize_rtft(f_energy, qdrift, 4.0u"μs", 2.0u"μs";
    rt_bounds = (1.0u"μs", 8.0u"μs"), ft_bounds = (0.5u"μs", 4.0u"μs"),
    step0 = (1.0u"μs", 0.5u"μs"), lattice = 0.25u"μs",
    peak = 2614.5u"keV", window = (35.0u"keV", 25.0u"keV"),
    min_e = 2500.0, max_e = 2700.0, rel_cut_fit = 0.5)
result.rt, result.ft, result.converged
```

## Window Length of the A/E Filter

The A/E parameter (see [A/E Pulse-Shape Discrimination](@ref)) depends on the window length of the Savitzky-Golay smoothing filter that is applied to the waveform to obtain the current amplitude. [`fit_sf_wl`](@ref) selects the window length that rejects the most multi-site events at a fixed acceptance of single-site events. It uses the double escape peak (DEP, 1592.5 keV), which contains single-site events, to set the cut, and the single escape peak (SEP, 2103.5 keV), which contains mostly multi-site events, to measure the rejection.

The inputs are the uncalibrated energies and the A/E values of the DEP and SEP events, `e_dep`, `aoe_dep`, `e_sep` and `aoe_sep`, where `aoe_dep` and `aoe_sep` hold one A/E value per window length of the grid `a_grid_wl_sg` for every event. For every window length, the function

1. calibrates the energies with a simple scale factor from a fit of the SEP peak ([`LegendSpecFits.prepare_sep_peakhist`](@ref)),
2. sets the search interval for the cut: it starts at the `min_aoe_quantile` quantile (default 0.1) of the DEP A/E values plus `min_aoe_offset` and ends at the upper edge of the bin after the maximum of the A/E histogram, which is limited by the `max_aoe_quantile` quantile (default 0.99) plus `max_aoe_offset`,
3. determines the lower A/E cut that keeps a survival fraction of 90 % of the DEP peak ([`get_low_aoe_cut`](@ref)),
4. applies this cut to the SEP events and determines the survival fraction of the SEP peak with [`get_peak_survival_fraction`](@ref).

Window lengths for which a step fails are skipped with a warning. The selected window length has the smallest SEP survival fraction, restricted to values between 1 % and 100 %. The keywords `dep`, `dep_window`, `sep`, `sep_window` and `sep_rel_cut` set the peak energies and fit windows, and `dep_cut_search_fit_func` and `sep_cut_search_fit_func` select the peak shape of the fits (see [Peak Shapes and Likelihoods](@ref)).

The result contains `wl` (the window length with the step of the grid as uncertainty), `sf` (the SEP survival fraction), and the numbers of DEP and SEP events `n_dep` and `n_sep`. The report contains the survival fraction for every window length that was processed. A call looks as follows:

```julia
result, report = fit_sf_wl(e_dep, aoe_dep, e_sep, aoe_sep, a_grid_wl_sg;
    min_aoe_quantile = 0.1, max_aoe_quantile = 0.99)
result.wl, result.sf
lplot(report)
```

## Combining the Scans

The rise-time scan runs first with a fixed flat-top time. `fit_enc_sigmas` selects the rise time, which is then fixed for the flat-top scan with `fit_fwhm_ft`. [`optimize_rtft`](@ref) can then refine both times together, starting from these values. The A/E window length is optimized separately with `fit_sf_wl`.
