# LQ Cut

LQ (late charge) is a pulse-shape parameter that measures the amount of charge collected late in a detector pulse.
Events with an increased LQ, for example interactions close to the detector surface, are rejected with a one-sided cut.
LegendSpecFits.jl corrects the drift-time dependence of LQ with the double escape peak (DEP) of the 2614.5 keV Th-228 line, normalizes LQ to the width of the DEP distribution, and measures the survival fractions of the cut.
The LQ value itself is computed in the signal-processing stage and is an input here.

## Inputs and Pre-Normalization

The functions use the energy-corrected LQ (`lq / e` in the dataflow), an effective drift time `dt_eff` (`qdrift / e`), the calibrated energies `e_cal`, and the mean `dep_μ` and standard deviation `dep_σ` of the DEP peak in energy, taken from the energy calibration.
Before the correction, the dataflow scales LQ so that the main population is near 1, with [`cut_single_peak`](@ref) and its field `max`:

```julia
cuts = cut_single_peak(lq_e_corr, 0.0, quantile(filter(isfinite, lq_e_corr), 0.99); n_bins = -1)
lq_e_corr ./= cuts.max
```

## Drift-Time Correction

LQ of single-site events depends on the drift time.
[`lq_ctc_correction`](@ref) fits this dependence with DEP events and subtracts it:

1. Select events with energy within `ctc_dep_edgesigma` (default 3.0) standard deviations of the DEP mean.
2. Fit a truncated Gaussian to their LQ distribution (`fit_single_trunc_gauss`); the fit window is found with `cut_single_peak` and `ctc_lq_precut_relative_cut` (default 0.25). Keep events with LQ within `lq_outlier_sigma` (default 2.0) standard deviations of the fitted mean.
3. Restrict the drift time to a range selected by `ctc_driftime_cutoff_method`: `:percentile` (default) uses the quantiles `ctc_dt_eff_low_quantile` (0.15) and `ctc_dt_eff_high_quantile` (0.95); `:gaussian` uses `dt_eff_outlier_sigma` (2.0) standard deviations around a Gaussian fit and fails for detectors with two drift-time populations; `:double_gaussian` uses the range in which a double-Gaussian fit exceeds 10% of its maximum. Other values throw an `ArgumentError`.
4. Fit a polynomial of order `pol_fit_order` (default 1) of LQ as a function of drift time to the remaining events.

The result contains `func`, an expression string that subtracts the polynomial from LQ, the fit result `fit_result`, and the selection limits `box_constraints`.
The keywords `lq_e_corr_expression` and `dt_eff_expression` set the names that appear in `func`:

```
( lq ) - p0 * (dt)^0 - p1 * (dt)^1
```

The report contains the fits and the selection limits for plotting with `lplot` from LegendMakie.
The keyword `uncertainty` controls whether the fits compute parameter uncertainties (default `false`).

## LQ Normalization

[`lq_norm`](@ref) scales the corrected LQ so that the DEP distribution has mean 0 and standard deviation 1.
It builds the LQ histogram of events within `dep_sideband_sigma` (default 4.5) standard deviations of the DEP mean and subtracts the histograms of two sidebands, which together have the same energy width.
If `dep_σ` is below 2 keV, the sidebands lie on both sides of the peak; otherwise both lie below the peak, to stay clear of the neighboring 1620.5 keV Bi-212 line.
Negative bins after the subtraction are set to zero.
A binned truncated Gaussian fit (`fit_binned_trunc_gauss`) to the subtracted histogram, truncated at `cut_truncation_sigma` (default 3.5) standard deviations around the estimated peak position, gives the mean $\mu_\mathrm{LQ}$ and standard deviation $\sigma_\mathrm{LQ}$.
The result field `func` is the expression

```math
\mathrm{LQ}_\mathrm{norm} = \frac{\mathrm{LQ}_\mathrm{ctc} - \mu_\mathrm{LQ}}{\sigma_\mathrm{LQ}}\,,
```

where $\mathrm{LQ}_\mathrm{ctc}$ is the drift-time-corrected LQ given by `lq_class_expression`.
The report contains the histograms of the DEP and the sidebands, the fit report, and the sideband energy limits.

The following example applies both steps to synthetic data with a drift-time dependence and a flat energy background:

```@example lq
using LegendSpecFits, Random, Unitful

Random.seed!(3)
N, Nb = 20_000, 6_000
dep_μ, dep_σ = 1592.5u"keV", 1.5u"keV"
e_cal = [dep_μ .+ dep_σ .* randn(N); (1550 .+ 90 .* rand(Nb))u"keV"]
dt_eff = 0.5 .+ 0.1 .* randn(N + Nb)
lq = [0.02 .* randn(N); 0.2 .* rand(Nb) .- 0.05] .+ 0.01 .* dt_eff .+ 0.05

result_ctc, report_ctc = lq_ctc_correction(lq, dt_eff, e_cal, dep_μ, dep_σ;
    lq_e_corr_expression = "lq", dt_eff_expression = "dt")
lq_ctc = lq .- report_ctc.drift_time_func.(dt_eff)

result_norm, report_norm = lq_norm(dep_μ, dep_σ, e_cal, lq_ctc; lq_class_expression = result_ctc.func)
println(result_norm.func)
result_norm.fit_result.σ
```

## Cut and Survival Fractions

The LQ cut is a configured upper limit on the normalized LQ, in units of the DEP standard deviation (`high_cut_sigma` in the dataflow); it is not fitted.
The survival fractions use the same functions as the A/E cut, with the normalized LQ as the cut parameter and only an upper bound:

```julia
result_peaks, report_peaks = get_peaks_survival_fractions(
    lq_norm_values, e_cal, peaks, peak_names, windows_left, windows_right;
    high_cut = high_cut_sigma, fit_funcs = fit_funcs,
)
result_qbb, report_qbb = get_continuum_survival_fraction(
    lq_norm_values, e_cal, qbb, qbb_window; high_cut = high_cut_sigma,
)
```

[`get_peaks_survival_fractions`](@ref) fits each peak with and without the cut and returns the signal survival fraction `sf` in percent; [`get_continuum_survival_fraction`](@ref) counts events around $Q_{\beta\beta}$.
Both are described in [Survival Fractions](@ref).
The A/E and LQ cuts are combined in the PSD efficiency step, which passes a Boolean selection built from both cuts as `survival_flag` to the same functions.

## Order of the LQ Steps

For each LQ definition, the dataflow runs:

1. Pre-normalization with `cut_single_peak`.
2. [`lq_ctc_correction`](@ref) with the DEP parameters from the energy calibration; its `func` becomes the LQ expression.
3. [`lq_norm`](@ref) on the corrected LQ.
4. Survival fractions with the upper cut.

## Compton-Band Fits

[`fit_lq_compton`](@ref) fits the LQ distribution in Compton bands with the same peak shape as the A/E bands ([`aoe_compton_peakshape`](@ref)).
It takes the band histograms `peakhists`, the start values `peakstats`, and the band energies, in the format returned by [`generate_aoe_compton_bands`](@ref), and returns an ordered dictionary of results and reports keyed by the band energy.
The dataflow does not use it.
