# A/E Pulse-Shape Discrimination

The ratio A/E of the maximum current amplitude A of a detector pulse to its energy E separates single-site events (SSE) from multi-site events (MSE) and from some surface events.
LegendSpecFits.jl fits the A/E distribution in the Compton continuum to normalize A/E as a function of energy, finds the A/E cut that keeps a chosen fraction of single-site events, and measures how many events of several Th-228 calibration lines and of the continuum survive that cut.
A runnable end-to-end chain is given in the [A/E tutorial](../tutorials/aoe.md); this page describes the concepts and the functions.

## Event Classes and Reference Peaks

A neutrinoless double-beta decay deposits its energy mostly within a few millimeters, which produces a single current pulse with a large A/E.
A photon that scatters several times in the detector deposits energy at separated positions, so the current pulses of the individual deposits are spread in time and A/E is smaller.
Events close to the detector surface can have an A/E that is smaller or larger than the single-site value.

The Th-228 calibration spectrum provides samples of both classes:

| Feature | Energy (keV) | Dominant event class |
|---|---|---|
| Double escape peak (DEP) of the 2614.5 keV line | 1592.5 | single-site; used as a proxy for signal events |
| Single escape peak (SEP) | 2103.5 | multi-site |
| Full energy peak (FEP) | 2614.5 | multi-site |
| Compton continuum | any, e.g. around $Q_{\beta\beta}$ = 2039.06 keV | mixture of both classes |

In the DEP both 511 keV annihilation photons escape the detector, so the remaining energy is deposited locally.
The fraction of DEP events that pass an A/E cut therefore estimates the signal efficiency of the cut.
The survival fractions of the SEP, FEP, and the continuum around $Q_{\beta\beta}$ quantify how strongly the cut rejects background-like events.

## A/E Normalization

The raw A/E value depends on the detector and on the pulse-processing settings.
A first normalization divides A/E by the position of the single-site peak, so that this peak is located near 1.
[`cut_single_peak`](@ref) returns this position as the field `max`, together with the bounds `low` and `high` of a window around the peak:

```julia
cuts = cut_single_peak(aoe, 0.0, quantile(filter(isfinite, aoe), 0.99); n_bins = -1)
aoe_normalized = aoe ./ cuts.max
aoe_expression = "<property function of A/E> / $(cuts.max)"
```

The fits described below determine the remaining dependence of the peak position and width on energy.
After that correction, the A/E value is `(A/E - μ(E)) / σ(E)`: the single-site peak is at 0 and has a width of 1.

## Compton Bands

The fits use events from several narrow energy intervals ("Compton bands") of the continuum.
[`generate_aoe_compton_bands`](@ref) takes the A/E values, the calibrated energies, the lower band edges `compton_bands`, and the common band width `compton_window`.
For every band, it selects the events with energy between the band edge and the edge plus the width and with positive A/E, builds an A/E histogram, and estimates start values for the fit:

```julia
bands = generate_aoe_compton_bands(aoe, e_cal, compton_bands, compton_window)
(; peakhists, peakstats) = bands
```

The histogram range is restricted to the region around the single-site peak, and the bin width follows the Freedman-Diaconis rule.
The returned `NamedTuple` also contains the histogram limits and simple estimates of the energy dependence of the peak position and width.
`aoe` and `e_cal` must have the same length.
The estimate of the peak width uses all bands from the 20th on, so at least 20 bands are required; with fewer bands the function throws an `ArgumentError`.

## A/E Peak Shape

In one Compton band, the A/E distribution is the sum of a Gaussian single-site peak and a multi-site background that extends toward lower A/E.
[`aoe_compton_peakshape`](@ref) implements the model

```math
f(x) = n \, \mathcal{N}(x;\, \mu, \sigma) + B \, \mathrm{EMG}(-x;\, -\mu, \sigma, \delta)\,,
```

where $x$ is the A/E value, $\mathcal{N}(x; \mu, \sigma) = \exp\!\left(-(x-\mu)^2 / (2\sigma^2)\right) / (\sigma\sqrt{2\pi})$ is the Gaussian density with mean $\mu$ and standard deviation $\sigma$, and $n$ is the number of events in the single-site peak.
The background term is an exponentially modified Gaussian (EMG), the density of the sum of a Gaussian random variable with mean $\mu_g$ and standard deviation $\sigma$ and an exponentially distributed variable with scale $\theta$:

```math
\mathrm{EMG}(x;\, \mu_g, \sigma, \theta) = \frac{1}{2\theta}\,
\exp\!\left(\frac{\sigma^2}{2\theta^2} - \frac{x-\mu_g}{\theta}\right)
\mathrm{erfc}\!\left(\frac{1}{\sqrt{2}}\left(\frac{\sigma}{\theta} - \frac{x-\mu_g}{\sigma}\right)\right)\,,
```

where $\mathrm{erfc}$ is the complementary error function.
The argument $-x$ and the mean $-\mu$ mirror the EMG, so the background tail extends toward lower A/E.
$B$ is the number of events in the background component and $\delta$ is the scale of its tail, in units of A/E.
The background shares $\mu$ and $\sigma$ with the signal peak.
The signal and background parts can be evaluated separately with [`aoe_compton_signal_peakshape`](@ref) and [`aoe_compton_background_peakshape`](@ref).

The fit function `:aoe_two_bck` adds a second EMG with its own mean $\mu_2$, width $\sigma_2$, and scale $\delta_2$ ([`two_emg_aoe_compton_peakshape`](@ref)):

```math
f_2(x) = n \, \mathcal{N}(x;\, \mu, \sigma) + B \left[(1 - f_B)\, \mathrm{EMG}(-x;\, -\mu, \sigma, \delta) + f_B\, \mathrm{EMG}(-x;\, -\mu_2, \sigma_2, \delta_2)\right]\,,
```

where $f_B$ is the fraction of the background in the second component (`B2f` in the fit results).
The default fit function is `:aoe_one_bck` with the parameters `μ, σ, n, B, δ`; `:aoe_two_bck` has the additional parameters `μ2, σ2, B2f, δ2`.
Other peak shapes are described in [Peak Shapes](peakshapes.md).

## Fitting Single Bands

[`fit_aoe_compton`](@ref) fits every band histogram with a binned maximum-likelihood fit (Poisson likelihood per bin).
The keyword `fit_func` selects the peak shape and `uncertainty = true` computes parameter uncertainties from the Hessian of the likelihood.
Without uncertainties, the parameter errors are `NaN`, and the later correction fit cannot use the bands.
The bands are fitted via

```julia
result_fit, report_fit = fit_aoe_compton(peakhists, peakstats, compton_bands; uncertainty = true)
```

Both return values are ordered dictionaries keyed by the lower band edge.
Each entry of `result_fit` contains the fitted parameters as `Measurement`s (`μ`, `σ`, `n`, `B`, `δ`, and the second-background parameters for `:aoe_two_bck`), the full width at half maximum (FWHM) of the peak, the name `fit_func`, and a `gof` field with the p-value, $\chi^2$, degrees of freedom, covariance matrix, normalized residuals, and a `converged` flag.
The entries of `report_fit` hold the fit values, the histogram, the fit function and its components, and the residuals for plotting with `lplot` from LegendMakie.
If the fit of a band throws an error, the function logs a warning and continues; building the result dictionary then fails because that band has no result.

The dataflow keeps the bands with `gof.converged` and a p-value above a configured threshold and uses their $\mu$ and $\sigma$ in the next step:

```julia
good = [result_fit[b].gof.converged && result_fit[b].gof.pvalue >= p_value_cut for b in compton_bands]
compton_bands_qc = compton_bands[good]
μ = [result_fit[b].μ for b in compton_bands_qc]
σ = [result_fit[b].σ for b in compton_bands_qc]
```

## Energy Dependence of the Peak

The position and width of the single-site peak depend on energy.
[`fit_aoe_corrections`](@ref) fits the band results $\mu_i$ and $\sigma_i$ at the band energies $E_i$ with the models

```math
\mu(E) = \mu_A + \mu_B\, E\,, \qquad \sigma(E) = \sqrt{\sigma_A^2 + \frac{\sigma_B^2}{E^2}}\,,
```

where $\mu_A$ and $\sigma_A$ are dimensionless offsets, $\mu_B$ is a slope in units of 1/keV, and $\sigma_B$ is a term in units of keV that dominates at low energy.
The function uses a weighted least-squares fit (`chi2fit`) and skips bands whose fit parameter uncertainties are zero or `NaN`.

It returns a `(result, report)` pair.
`result.µ_compton` and `result.σ_compton` contain the fit parameters `par` (with units), the goodness of fit, and the expression strings `func`; `result.compton_bands.e` contains the band energies used.
`result.func` is the normalization as an expression string that can be evaluated with a property function on a table of hits, with the placeholders replaced by the keywords `aoe_expression` and `e_expression`:

```@example aoe_corrections
using LegendSpecFits, Random, Unitful, Measurements

Random.seed!(2)
bands = collect((1000:100:1900)u"keV")
E = ustrip.(bands)
μ = measurement.(0.99 .+ 1e-5 .* E .+ 2e-4 .* randn(length(E)), 3e-4)
σ = measurement.(sqrt.(0.012^2 .+ 11.0^2 ./ E .^ 2) .+ 1e-4 .* randn(length(E)), 3e-4)

result, report = fit_aoe_corrections(bands, μ, σ; aoe_expression = "aoe", e_expression = "e")
println(result.func)
```

The value of the model at the energy of an event is subtracted from its A/E and the difference is divided by $\sigma(E)$, so the normalized A/E has mean 0 and standard deviation 1 for single-site events at any energy.
The reports `report.report_μ` and `report.report_σ` hold the data and the fit curves for plotting.

### Combined Fit

[`fit_aoe_compton_combined`](@ref) replaces the independent fits of the bands by a single fit of all bands in which the peak position and width follow the models above.
The four parameters $\mu_A$, $\mu_B$, $\sigma_A$, $\sigma_B$ are free in the likelihood, with Gaussian priors centered on the result of `fit_aoe_corrections` (passed as `result_corrections`); the other parameters of each band are fitted for every trial value.
The result has the same fields as the result of `fit_aoe_corrections` (`µ_compton`, `σ_compton`, `compton_bands`, `func`), so `result.func` can be used as a replacement, and additionally contains the parameters `μA`, `μB`, `σA`, `σB` and a `gof` field.
The combined fit is called via

```julia
result_combined, report_combined = fit_aoe_compton_combined(
    peakhists, peakstats, compton_bands_qc, result_correction;
    aoe_expression, e_expression, uncertainty = true,
)
```

The dataflow runs this fit when the `use_combined_fit` option is set.

## A/E Charge-Trapping Correction

The A/E of single-site events can also depend on the drift time of the charges, for example because of charge trapping.
[`ctc_aoe`](@ref) removes a polynomial dependence on a drift-time parameter, by default `qdrift / e` (the digital signal processing (DSP) parameter `qdrift` divided by the energy).
It uses the A/E that is already normalized with `result_correction.func` and is called via

```julia
result_ctc, report_ctc = LegendSpecFits.ctc_aoe(
    aoe_corrected, e_cal, qdrift_e, compton_bands_qc;
    aoe_expression = result_correction.func, qdrift_expression = "qdrift / e",
)
```

Only events in the Compton bands (each band extends 20 keV above its lower edge) and with an A/E in `[peak - window[1], peak + window[2]]` enter the fit.
The defaults are `peak = 0` (the normalized single-site position) and `window = (50, 8)`.
With $q$ the drift-time parameter and $c_i$ the coefficients, the corrected value is

```math
a_\mathrm{ctc} = a + \sum_{i=1}^{p} c_i\, q^i\,,
```

where $a$ is the normalized A/E and $p$ is the polynomial order (`pol_order`, default 1).
The coefficients are found with a bounded derivative-free optimizer (BOBYQA from NLopt) that minimizes the standard deviation of a right-sided truncated Gaussian fit to $a_\mathrm{ctc}$ around the single-site peak.
A final truncated Gaussian fit then re-centers the corrected A/E to 0 and scales its width to 1.
The result contains the expression string `func` of the full correction, the coefficients `fct`, the widths before and after the correction, the fit results `before`, `after`, and `after_norm`, and the flag `converged`; a warning is logged if the optimizer does not converge.
The Compton-band peak shape is fitted with `fit_func = :aoe_two_bck` by default.
The correction is optional in the dataflow (option `apply_ctc`), and `result_ctc.func` then replaces `result_correction.func` as the final A/E expression.

## Low and High Cuts

[`get_low_aoe_cut`](@ref) determines the lower A/E threshold such that a chosen fraction `dep_sf` (default 0.9) of the DEP events survives.
It selects the events around the DEP (`window = [12.0, 10.0]u"keV"` below and above `dep`), fits the DEP peak without any cut, and searches for the threshold with the bisection method on `cut_search_interval` until the survival fraction obtained from a combined fit of the surviving and the rejected events equals `dep_sf` within the relative tolerance `rtol`.
With `fixed_position = true`, the peak centroid is fixed to the value `dep` in these fits.
The cut is determined via

```julia
result_cut, report_cut = get_low_aoe_cut(aoe, e_cal; dep = 1592.53u"keV", dep_sf = 0.9)
result_cut.lowcut, result_cut.highcut
```

The result is a `NamedTuple` with the fields `lowcut` (a `Measurement` whose uncertainty is set by `rtol`), `highcut`, `n0`, `nsf` (the number of DEP events before and after the cut), and `sf` (survival fraction in percent).
The upper cut is not determined by a fit: `highcut` is the value of the keyword `sigma_high_sided` in the unit of the A/E input and is `Inf` by default, i.e., no upper cut.
For normalized A/E, the value is in units of the single-site width $\sigma$.
`report_cut` contains the energy histograms before the cut, after the low cut, and after both cuts for plotting with `lplot`.
An event passes the cuts if `lowcut < aoe < highcut`.

## Survival Fractions

The survival fraction of an energy peak is the fraction of the peak events that remain after an event selection.
[`get_peaks_survival_fractions`](@ref) evaluates it for several peaks.
For each peak, it fits the peak before the cut, and then fits the energy histograms of surviving and rejected events simultaneously with a shared peak shape, so the survival fraction refers to the peak signal without the underlying background.
The survival fractions of the Th-228 lines are obtained via

```julia
peaks = [583.19, 1592.53, 1620.5, 2103.53, 2614.51]u"keV"   # Th-228 lines
names = [:Tl208FEP_583, :Tl208DEP, :Bi212FEP, :Tl208SEP, :Tl208FEP]
windows_left  = [15.0, 12.0, 10.0, 25.0, 20.0]u"keV"                   # window below each peak
windows_right = [15.0, 10.0, 10.0, 25.0, 20.0]u"keV"                   # window above each peak

result_peaks, report_peaks = get_peaks_survival_fractions(
    aoe, e_cal, peaks, names, windows_left, windows_right;
    low_cut = result_cut.lowcut, high_cut = result_cut.highcut,
    uncertainty = true, bin_width_window = 2.0u"keV",
)
result_peaks[:Tl208SEP].sf   # survival fraction in percent, with uncertainty
```

The function accepts either an event selection (a `Vector{Bool}` `survival_flag`, passed after the windows) or a cut parameter (here A/E) with the keywords `low_cut`, `high_cut`, and `selection`, an additional Boolean mask.
The cut bounds are exclusive.
The energies, windows, and bin width must share one floating-point element type, e.g., `Float64` keV.
The windows can be given as a vector of `(left, right)` tuples or as two vectors of widths.
The keyword `fit_funcs` selects the peak shape of each peak (default `:gamma_def`, see [Peak Shapes](peakshapes.md)).
The result is an ordered dictionary keyed by the peak names; each entry contains `peak`, `n_before`, `n_after`, `sf` (percent), `fit_func`, and `gof`.
The peaks are fitted in parallel threads.

[`get_peak_survival_fraction`](@ref) performs the same computation for one peak and also accepts histograms of all, surviving, and rejected events.
[`get_continuum_survival_fraction`](@ref) counts the events in a window `center ± window` before and after the selection; it is used for the continuum around $Q_{\beta\beta}$, where no peak is fitted.
Its uncertainty is the binomial uncertainty of the ratio.

```@example aoe_continuum
using LegendSpecFits, Random, Distributions, Unitful

Random.seed!(1)
n = 60_000
e = (2039.06 .+ 200 .* (rand(n) .- 0.5))u"keV"
aoe = randn(n)                                  # normalized A/E: single-site peak at 0, width 1
tail = rand(n) .< 0.5
aoe[tail] .-= rand(Exponential(8.0), count(tail))   # half of the events have a multi-site tail

result, report = get_continuum_survival_fraction(aoe, e, 2039.06u"keV", 25.0u"keV"; low_cut = -1.5)
result.sf
```

## Order of the A/E Steps

The production dataflow chains the functions in this order for every detector and A/E definition:

1. Normalize A/E to the single-site peak with `cut_single_peak` and `cuts.max`.
2. Build the bands with [`generate_aoe_compton_bands`](@ref) and fit them with [`fit_aoe_compton`](@ref) (`uncertainty = true`).
3. Keep the bands that converged and have a sufficient p-value.
4. Determine the energy dependence with [`fit_aoe_corrections`](@ref), or with [`fit_aoe_compton_combined`](@ref) if the combined fit is enabled. The `func` string of the result is the normalization.
5. Optionally apply [`ctc_aoe`](@ref), whose `func` becomes the final A/E expression.
6. Compute the cut with [`get_low_aoe_cut`](@ref) on the final A/E.
7. Evaluate the survival fractions of the calibration peaks with [`get_peaks_survival_fractions`](@ref) for the low cut alone and for the low and high cut together, and of the $Q_{\beta\beta}$ continuum with [`get_continuum_survival_fraction`](@ref).

An earlier optimization step chooses the pulse-processing window length that minimizes the SEP survival fraction at fixed DEP survival fraction with [`fit_sf_wl`](@ref); see [Filter Optimization](filter_optimization.md).
The survival fractions of the A/E cut are combined with those of the LQ cut ([LQ Cut](lq.md)) in the PSD efficiency step.
