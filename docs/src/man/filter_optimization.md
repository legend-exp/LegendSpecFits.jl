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

After the rise time is fixed, the flat-top time is scanned. [`fit_fwhm_ft`](@ref) takes a matrix of energies with one row per flat-top time and fits the 2614.5 keV peak for each of them. The energies are in arbitrary units (for example uncalibrated filter output); `min_e` and `max_e` bracket the peak in these units and `rel_cut_fit` is the `relative_cut` of `cut_single_peak`.

For each flat-top time, the function

1. skips the row if fewer than 100 energies lie between `min_e` and `max_e`,
2. restricts the energies to a window around the peak, found with `cut_single_peak`,
3. fits the gamma peak shape with [`fit_single_peak_th228`](@ref) (without uncertainties), and
4. stores the FWHM and the position of the highest histogram bin.

The FWHM values are converted to keV with a single factor that places the mean peak position of all flat-top times at `peak` (keyword, 2614.5 keV by default). Among the flat-top times with an FWHM within `ft_fwhm_tol` (default 0.1 keV) of the minimum FWHM, the shortest one is selected. A longer flat top is therefore selected only if it improves the FWHM by more than `ft_fwhm_tol`.

A second method takes an additional vector `qdrift` with the drift-time parameter of every event, used for the charge-trapping correction of [`ctc_energy`](@ref). With `apply_ctc = true` (the default of this method), the energies of each flat-top time are corrected before the FWHM is determined, so that the optimum is selected for the corrected resolution. The `window` keyword gives the fit window around `peak` for the correction. The `rt` argument records the rise time of the scan and does not enter the computation.

The result contains `ft`, the optimal flat-top time, and `min_fwhm`, its FWHM in keV.
The example scans 10 flat-top times of synthetic energies:

```@example filteropt
Random.seed!(2)
fts = range(0.5u"μs", step = 0.5u"μs", length = 10)
sigmas = 2.0 .+ 0.15 .* abs.((1:10) .- 4)         # best resolution at the fourth flat-top time
e_grid = reduce(hcat, [2614.5 .+ s .* randn(6_000) for s in sigmas])'
result_ft, report_ft = fit_fwhm_ft(Matrix(e_grid), fts, 1.0u"μs", 2500.0, 2700.0, 0.5)
(ft = result_ft.ft, min_fwhm = result_ft.min_fwhm)
```

The resolution of the fourth flat-top time, $2\sqrt{2\ln 2} \cdot 2.0 = 4.71$ keV, is the smallest. The FWHM values have no uncertainty (`NaN`) because the fits inside `fit_fwhm_ft` run without the uncertainty calculation.

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

In the production dataflow, the rise-time scan runs first with a fixed flat-top time. `fit_enc_sigmas` selects the rise time, which is then fixed for the flat-top scan that `fit_fwhm_ft` evaluates. The results of both are merged into one parameter set. The A/E window length is optimized separately.
