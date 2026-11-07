# SiPM Calibration

A silicon photomultiplier (SiPM) signal is a sum of discrete photoelectron (PE) pulses. The amplitude spectrum of the triggered pulses shows a noise peak followed by peaks at 1 PE, 2 PE and higher. The calibration converts amplitudes to PE units from these peak positions. Two functions do this in sequence: [`sipm_simple_calibration`](@ref) determines a linear calibration from the peak positions, and [`fit_sipm_spectrum`](@ref) fits the calibrated spectrum to obtain position and resolution of every PE peak. [`fit_sipm_wl`](@ref) and [`fit_sipm_threshold`](@ref) use them to optimize the trigger filter.

All functions return `(result, report)`. The report is plotted with `lplot` from LegendMakie.

## SiPM Simple Calibration

[`sipm_simple_calibration`](@ref) takes a vector of uncalibrated trigger amplitudes. It performs these steps:

1. It locates the noise peak between `initial_min_amp` and `initial_max_amp` with [`cut_single_peak`](@ref) (`relative_cut_noise_cut`).
2. It sets a threshold between the noise and the 1 PE peak at the center of the first valley after the noise peak. `n_fwhm_noise_cut` limits the valley search to that many noise-peak half-widths above the noise peak center. If no valley is found, the threshold falls back to a position three half-widths above the noise peak or to the end of the search range. With `n_fwhm_noise_cut = 0`, `initial_min_amp` is the threshold, and a negative value places the threshold at the noise peak maximum plus that many half-widths.
3. It finds the peak positions above the threshold on a Savitzky-Golay smoothed histogram with the peak finder of RadiationSpectra.jl. The search is repeated with finer bins if it fails and falls back to local maxima. At least two peaks are needed; otherwise an error is thrown. Neighboring peaks closer than 60 % of the largest peak distance are merged.
4. It assigns the first two peaks to the PE values `min_pe_peak` and `min_pe_peak + 1`, which defines the linear calibration

```math
\mathrm{PE} = c\,a + o\,, \qquad c = \frac{1}{a_2 - a_1}\,, \qquad o = n_1 - c\,a_1\,,
```

where $a$ is the amplitude, $a_1$ and $a_2$ are the positions of the first two peaks and $n_1$ is `min_pe_peak`. The distance $a_2 - a_1$ is the gain in amplitude units.

The trigger threshold is a second threshold, placed after the calibration. It is the lowest minimum of the Savitzky-Golay smoothed calibrated spectrum between the noise threshold and `min_pe_peak` (smoothing width 0.15 PE). If the spectrum has no minimum there that is lower than the spectrum at the noise threshold, for example because it only rises from the noise threshold to the 1 PE peak, or if the noise threshold is less than one smoothing window away from the 1 PE peak, the trigger threshold equals the noise threshold. If the spectrum has several minima, the deepest one is used. Where the single-trigger spectrum is empty between the noise tail and the 1 PE peak, the trigger threshold lies in the middle of the empty region. The dataflow converts `trigger_threshold` with the final calibration curve to PE.

The `peakfinder_*` keywords control the peak finder; `peakfinder_σ` is the expected peak width in bins and must be an integer. Its default is an estimate from the noise peak.

If the input is a `VectorOfVectors` with the trigger amplitudes of each waveform, the function determines the threshold from the pool of all triggers, optionally restricted to waveforms with at most `cut_pool_max_mult` triggers (`single_trigger_only = true`), and calibrates the triggers above the threshold of waveforms with exactly one such trigger.

The result contains `peakpos`, the coefficients `c` and `offset`, the noise-peak properties `noisepeakpos` and `noisepeakwidth`, and the thresholds `noise_threshold` and `trigger_threshold` (uncalibrated) with `noise_threshold_cal` and `trigger_threshold_cal` (calibrated). The report contains the histograms, the thresholds, the calibrated amplitudes `pe_simple_cal` and the calibration function `f_simple_calib` with its inverse `f_simple_uncal`, which are used by the following functions.

The example generates amplitudes with a gain of 12, an offset of 3 and Poisson-distributed numbers of photoelectrons:

```@example sipm
using LegendSpecFits, Random, Distributions, Measurements

Random.seed!(3)
gain, offset = 12.0, 3.0
npe = rand(Poisson(0.9), 60_000)
amplitudes = [rand(Normal(offset + k * gain, sqrt(1.0 + k))) for k in npe]
result_simple, report_simple = sipm_simple_calibration(amplitudes; initial_max_amp = 80.0)
(peakpos = result_simple.peakpos, c = result_simple.c, noise_threshold = result_simple.noise_threshold,
    trigger_threshold = result_simple.trigger_threshold)
```

The positions of the first peaks are at about 15 and 27, which is `offset + gain` and `offset + 2 gain`. The inverse of the calibration slope, `1 / result_simple.c`, is the gain. The spectrum has no valley between the noise cut and the 1 PE peak, so the trigger threshold equals the noise threshold.

## Spectrum Fit

[`fit_sipm_spectrum`](@ref) fits a Gaussian mixture model with `n_mixtures` components to the calibrated amplitudes between `min_pe` and `max_pe`. By default, there are four components per PE interval, `ceil(Int, (max_pe - min_pe) * 4)`, because a peak with a non-Gaussian shape needs more than one Gaussian. The fit uses the expectation-maximization algorithm of GaussianMixtures.jl with `nInit` initializations (default `method = :kmeans`) and `nIter` iterations, and a diagonal covariance (`kind = :diag`).

The result is evaluated for each integer PE value `pe` in the range:

* The position is the weighted mean of the centers of all Gaussians within `pe ± Δpe_peak_assignment` (default 0.3 PE), with their weights.
* The resolution is the FWHM of the sum of these Gaussians, found numerically.
* Both are also given in uncalibrated units, using the function `f_uncal` (pass `report_simple.f_simple_uncal`). The resolution in uncalibrated units is `f_uncal(res) - f_uncal(0)`.

With `uncertainty = true`, the parameter uncertainties follow from the Hessian matrix of the binned log-likelihood of the mixture, and `gof` contains the p-value from [`p_value_poissonll`](@ref) (see [Cuts and Goodness of Fit](@ref)). The fields of the result are `μ`, `σ`, `w` (Gaussian parameters), `peaks`, `positions_cal`, `resolutions_cal`, `positions`, `resolutions`, `n_mixtures`, `n_pos_mixtures` (number of Gaussians per peak) and `gof`.

A typical sequence continues the example above. It is not executed in this page because the mixture fit takes several seconds:

```julia
result_fit, report_fit = fit_sipm_spectrum(report_simple.pe_simple_cal, 0.5, 3.5;
    f_uncal = report_simple.f_simple_uncal)
result_fit.peaks        # 1:1:3
result_fit.positions    # positions of the PE peaks, uncalibrated amplitude
result_fit.resolutions  # FWHM of the PE peaks, uncalibrated amplitude
lplot(report_fit)
```

## SiPM Calibration Curve

The positions found by the spectrum fit can be fitted with a polynomial calibration curve with [`fit_calibration`](@ref). This accounts for deviations from linearity at higher amplitudes and propagates the uncertainties of the positions:

```julia
using Unitful, UnitfulAtomic

finite = isfinite.(Measurements.value.(result_fit.positions))
result_cal, report_cal = fit_calibration(1, result_fit.positions[finite],
    collect(result_fit.peaks)[finite] .* u"e_au"; e_expression = "e")
```

`fit_calibration` requires the peak values with a unit. The dataflow uses the atomic unit of charge `u"e_au"` from UnitfulAtomic.jl as a stand-in for PE counts. `result_cal.func` is the calibration as an expression string.

## Trigger Threshold

[`fit_sipm_threshold`](@ref) estimates the typical trigger threshold from a vector of threshold values, for example one value per event or per channel. It keeps the values between `min_cut` and `max_cut`, builds a histogram (`n_bins`, or the Freedman-Diaconis bin width if `n_bins` is not positive), and, with `fit_thresholds = true`, fits a truncated Gaussian to the peak of the histogram with [`fit_binned_trunc_gauss`](@ref). The window is determined by [`cut_single_peak`](@ref) with `relative_cut`. The result is the fit result (`μ`, `σ`, `n`, `gof`) extended by `μ_simple` and `σ_simple`, the mean and standard deviation of the values in the range. With `fit_thresholds = false`, only the simple estimates are returned.

!!! warning
    `fit_sipm_threshold` removes the values outside `min_cut` and `max_cut` from the input vector. Pass a copy if the original is needed.

The example estimates the threshold from synthetic values:

```@example sipm
thresholds = 20 .+ 0.5 .* randn(5_000)
result_thr, report_thr = fit_sipm_threshold(copy(thresholds), 17.0, 23.0)
(μ = result_thr.μ, σ = result_thr.σ, μ_simple = result_thr.μ_simple)
```

## Window Length Scan

The SiPM trigger filter has a window length that trades noise against the sharpness of the PE peaks. [`fit_sipm_wl`](@ref) evaluates a scan of the window length. Its input is `trig_max_grid`, a `VectorOfVectors` whose element `i` contains the trigger amplitudes obtained with window length `e_grid_wl[i]`, and the range `e_grid_wl`. An optional vector `thresholds` gives the trigger threshold for each window length.

For each window length, the function runs `sipm_simple_calibration` (with the threshold as `initial_min_amp`) and `fit_sipm_spectrum` (between `min_pe_fit` and `max_pe_fit`) and determines

* the gain, the distance between the first PE peak and the noise peak in uncalibrated units,
* the position and the FWHM resolution `res_1pe` of the 1 PE peak.

The selected window length minimizes the objective

```math
\sqrt{\mathrm{res}_{1\,\mathrm{PE}}}\;\frac{\sqrt{t}}{g}\,,
```

where $\mathrm{res}_{1\,\mathrm{PE}}$ is the resolution of the 1 PE peak, $g$ is the gain and $t$ is the threshold. If all thresholds are zero, $t = 1$ is used for all window lengths. Window lengths for which a step fails are skipped with a warning, and an error is thrown if none succeeds.

The result contains the optimal `wl` (with the step of the grid as uncertainty) and the `obj`, `res_1pe`, `gain`, `pos_1pe` and `threshold` at that point. The report contains these values for all window lengths and the reports `report_simple` and `report_fit` of the optimal one, so that `lplot(report)` and `lplot(report.report_simple, cal = true)` show the scan and the calibration of the optimum. After the window length is chosen, `fit_sipm_threshold` determines the threshold for the chosen filter:

```julia
result_wl, report_wl = fit_sipm_wl(trig_max_grid, e_grid_wl, thresholds_grid)
result_thr, report_thr = fit_sipm_threshold(thresholds, min_cut, max_cut; n_bins = 100)
```

## PMT Single-Photoelectron Fit

[`fit_pmt_spe`](@ref) fits the pulse-height spectrum of a photomultiplier tube (PMT) at a light level where most pulses contain at most one photoelectron. The spectrum has a sharp pedestal at low amplitudes, a falling tail of the pedestal and a broad 1 PE peak, with a weaker 2 PE peak at twice its position. The function does not fit the pedestal. It fits the model

```math
f(x) = \frac{n_\mathrm{exp}}{\tau}\, e^{-(x - x_\mathrm{min})/\tau} + n_1\, \mathcal{N}(x;\, \mu_1, \sigma_1) + r_2\, n_1\, \mathcal{N}(x;\, 2\mu_1, \sqrt{2}\,\sigma_1)\,,
```

in counts per unit of the pulse height $x$ above the start of the fit range $x_\mathrm{min}$. Here $\mathcal{N}(x;\, \mu, \sigma)$ is a normalized Gaussian, $n_\mathrm{exp}$ is the number of entries of the exponential above $x_\mathrm{min}$ and $\tau$ its decay constant, $n_1$ is the number of entries of the 1 PE peak with position $\mu_1$ and width $\sigma_1$, and $r_2 \le 1$ is the ratio of 2 PE to 1 PE entries. The bound on $r_2$ keeps the fit from placing the 1 PE peak at half its position.

The fit proceeds in these steps:

1. It histograms the pulse heights in `hist_range` with bins of `bin_width` and smooths the histogram with a 3-bin moving average.
2. It searches the valley between the pedestal and the 1 PE peak: the first bin in `peak_search_range` after which the smoothed spectrum rises for two bins. The start value of $\mu_1$ is the maximum of the smoothed spectrum above the valley. An [`ArgumentError`](https://docs.julialang.org/en/v1/base/base/#Core.ArgumentError) is thrown if there is no such rise.
3. It sets $x_\mathrm{min}$ to `x_min_offset` below the valley and maximizes the binned Poisson likelihood of the histogram above $x_\mathrm{min}$, with the parameter bounds given in [`fit_pmt_spe`](@ref).

The defaults of `hist_range = (0, 100)`, `bin_width = 1` and `peak_search_range = (6, 60)` are for pulse heights in ADC units with the 1 PE peak at about 10 to 20 units. For other scales, change these keywords together. `uncertainty = false` skips the Hessian matrix and the goodness-of-fit test, and the parameters are then plain numbers instead of `Measurement`s.

The result contains `μ1`, `σ1`, `τ`, `n_exp`, `n_1`, `r2`, `x_min` and `gof` (with `pvalue`, `chi2` and `dof` if `uncertainty = true`). The report contains the histogram `h`, the model `f_fit`, its components `f_components` (`exponential`, `pe1`, `pe2`) and can be plotted with `lplot`. The example fits synthetic pulse heights with a pedestal, its tail, and 1 PE and 2 PE peaks:

```@example sipm
Random.seed!(5)
μ1, σ1 = 17.0, 6.0
pulse_heights = vcat(abs.(rand(Normal(0.8, 0.3), 100_000)),       # pedestal
                     rand(Exponential(3.0), 15_000),               # pedestal tail
                     rand(Normal(μ1, σ1), 10_000),                 # 1 PE
                     rand(Normal(2μ1, sqrt(2) * σ1), 1_000))       # 2 PE
result_pmt, report_pmt = fit_pmt_spe(pulse_heights)
(μ1 = result_pmt.μ1, σ1 = result_pmt.σ1, r2 = result_pmt.r2, x_min = result_pmt.x_min,
    pvalue = result_pmt.gof.pvalue)
```

The fitted $\mu_1$ and $\sigma_1$ agree with the generated values of 17 and 6 within their uncertainties.
