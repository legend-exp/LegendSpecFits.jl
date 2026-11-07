# Energy Calibration

The energy calibration of a germanium detector converts the uncalibrated energy estimator of a waveform (for example the output of an energy filter, in analog-to-digital converter (ADC) units) to keV. It uses the gamma lines of a Th-228 calibration source and proceeds in four steps: a first linear calibration from the most prominent line, fits of all calibration lines, a calibration curve through the fitted line positions, and a curve for the energy resolution.

## Calibration Chain

The functions are applied in this order, and each step takes the output of the previous one:

1. [`simple_calibration`](@ref) finds the 2614.5 keV line in the uncalibrated spectrum, derives a linear calibration constant `c` from it, and creates a histogram and start values around every calibration line.
2. [`fit_peaks`](@ref) fits a peak shape to each line.
3. [`fit_calibration`](@ref) fits a polynomial through the fitted line positions and the known line energies.
4. [`fit_fwhm`](@ref) fits the energy resolution (FWHM versus energy) through the fitted line widths.

The dataflow applies the chain separately for each energy estimator, for example for the energy before and after the charge-trapping correction ([Charge-Trapping Correction](@ref)). In a second pass over a data partition, the calibration function from step 3 is applied to the energies first, and steps 2 to 4 run on calibrated keV values, with [`peakhists_gamma`](@ref) creating the histograms in place of [`simple_calibration`](@ref).

## Calibration Lines

The default line set is the Th-228 spectrum, which contains these lines (in keV):

| Line | Energy |
|:--|--:|
| Tl-208 gamma | 583.2 |
| Bi-212 gamma | 727.3 |
| Tl-208 gamma | 860.6 |
| Tl-208 double escape peak (DEP) | 1592.5 |
| Bi-212 gamma | 1620.5 |
| Tl-208 single escape peak (SEP) | 2103.5 |
| Tl-208 full-energy peak (FEP) | 2614.5 |

The lines are passed as a vector of energies with units. The line at 2614.5 keV must be the one used for the first calibration. Individual lines can be left out of the calibration and resolution fits by selecting the entries before calling [`fit_calibration`](@ref) and [`fit_fwhm`](@ref). The resolution at Qββ = 2039.06 keV, the Q value of the double-beta decay of Ge-76, is read from the fitted resolution curve, because there is no line at this energy.

## Automatic Calibration

[`autocal_energy`](@ref) determines a linear calibration constant when no estimate is available, for example for the energy that the data acquisition system (DAQ) records online. It histograms the spectrum, finds peaks with the peak finder of RadiationSpectra.jl, and matches them to the known line energies. With `mode = :ratio`, the match uses the ratios of the peak positions (`determine_calibration_constant_through_peak_ratios`); with `mode = :fit`, it uses `calibrate_spectrum`. It returns `c` (keV per uncalibrated unit) and a function `f_calib` that multiplies by `c`. The result and report are the same `NamedTuple`.

```julia
result, report = autocal_energy(daqenergy, [583.191u"keV", 2614.51u"keV"]; mode = :ratio)
result.c
```

## Simple Calibration

[`simple_calibration`](@ref) needs the uncalibrated energies, the line energies, and the window below and above each line that is histogrammed. The windows are given either as a vector of tuples `(left, right)` or as two vectors. The keyword `calib_type` is required:

* `calib_type = :th228` searches for the 2614.5 keV peak with fixed peak-finder settings.
* `calib_type = :gamma` is the general form for other sources. The keyword `peaksearch_type` selects either the most prominent peak (`:most_prominent`) or one specific line (`:single`, with the line index in `peakfinder_gamma`). A Co-60 source with the lines at 1173.2 keV and 1332.5 keV is calibrated with this variant.

If the spectrum is noisy, the keyword `quantile_perc` skips the peak finder and sets the peak position to the given quantile of the uncalibrated energies. The keyword `binning_peak_window` sets the window around each line that determines the bin width of the peak histograms.

The result contains the calibration constant `c`, the histograms `peakhists` around the lines, and `peakstats`, a `StructArray` with a start estimate of position, width, counts and background of each peak (see [`estimate_single_peak_stats`](@ref)). Histograms and start values are the input of [`fit_peaks`](@ref).

## Peak Fits

[`fit_peaks`](@ref) fits all lines in parallel on the available threads. Only `calib_type = :th228` is supported; the function fits each line with [`fit_single_peak_th228`](@ref). A single histogram can be fitted with [`fit_single_peak_th228`](@ref) directly, for example for a line of another source, with start values from [`estimate_single_peak_stats`](@ref).

The peak shape is a Gaussian on a background, with a tail towards low energies from incomplete charge collection. The keyword `fit_func` (a vector with one symbol per line in [`fit_peaks`](@ref)) selects the variant:

| `fit_func` | Components |
|:--|:--|
| `:gamma_def` | Gaussian, low-energy tail, flat background, step background (default) |
| `:gamma_tails` | `:gamma_def` plus a high-energy tail |
| `:gamma_bckSlope` | `:gamma_def` plus a linear background slope |
| `:gamma_bckExp` | `:gamma_def` plus an exponential background |
| `:gamma_bckFlat` | `:gamma_def` without the step |
| `:gamma_tails_bckFlat` | `:gamma_tails` without the step |
| `:gamma_minimal` | Gaussian and flat background only |

The functions are built from [`gamma_peakshape`](@ref); the page [Peak Shapes and Likelihoods](@ref) describes the components. A simpler model is more stable for lines with few counts, and a model with more components can describe lines with a visible background structure. The fit maximizes the binned Poisson likelihood, so that bins with few counts are treated correctly.

The result is an `OrderedDict` keyed by the line energies. The entry of each line holds the model parameters as `Measurement` values, the `fwhm` with its uncertainty, the `centroid` (the peak position corrected for the mean offset of the tails), `fit_func`, and a goodness-of-fit entry `gof`. The report of each line has the histogram, the fitted function, its components, and the residuals.

The fitted positions and widths are on the scale of the input histograms. With the keywords `e_unit` and `m_cal_simple`, set to the `unit` and `c` of [`simple_calibration`](@ref), they are converted back to the uncalibrated scale, which is the scale [`fit_calibration`](@ref) needs. For histograms that are already in keV, `e_unit = u"keV"` is sufficient.

The dataflow checks every fitted line before it enters the calibration: a lower limit on the FWHM, an upper limit on the FWHM per window, and a condition on the fit result that is written as an expression string and evaluated on the entry of each line. Lines that fail are excluded from the next two steps.

## Calibration Curve

[`fit_calibration`](@ref) takes the polynomial order, the fitted positions (`Measurement` values, uncalibrated), and the known line energies. It minimizes a χ² that includes the uncertainties of the positions, with [`chi2fit`](@ref). The result contains the parameters `par` (the coefficient of `e^i` has the unit keV per uncalibrated unit to the power `i`), the goodness of fit, and the expression strings `func` and `func_err` (see [Conventions](@ref)). The keyword `e_expression` sets the name of the energy column in the strings.

## Resolution Curve

[`fit_fwhm`](@ref) fits the resolution of the lines in calibrated keV with

```math
\mathrm{FWHM}(E)^2 = \mathrm{enc}^2 + F \cdot E + c_t \cdot E^2\,,
```

where $\mathrm{enc}$ is the FWHM at zero energy, which describes the electronic noise, $F$ is the coefficient of the linear term, from which the Fano factor of germanium is derived (with a pair creation energy of 2.96 eV), and $c_t$ is the coefficient of the quadratic term. The term in $E^2$ is only included for `pol_order = 2`. The result gives the FWHM at Qββ in the field `qbb`, as well as `func` for the uncalibrated energy and `func_cal` for the calibrated energy (keyword `e_type_cal` names its column). For `pol_order = 2`, `pull_fano = true` ties the linear coefficient to the literature Fano factor of germanium.

## Calibration Example

The following example creates a synthetic Th-228 spectrum in ADC units, with the lines of the table above on a falling continuum, and runs the chain. The simulated calibration constant is 0.0125 keV per ADC unit and the resolution follows a noise term plus a term proportional to the energy.

```@example energy
using LegendSpecFits, Unitful, Random, Measurements, Statistics
using Unitful: keV
Random.seed!(1)

lines = [583.191, 727.330, 860.564, 1592.53, 1620.50, 2103.53, 2614.51]keV
n_counts = [4000, 1300, 700, 400, 350, 600, 6000]
fwhm_true(E) = sqrt(1.2^2 + 0.0016 * E)       # keV, for E in keV
c_true = 0.0125                               # keV per ADC unit

peaks = [ustrip(l) .+ fwhm_true(ustrip(l)) / 2.355 .* randn(n) for (l, n) in zip(lines, n_counts)]
continuum = 200 .- 400 .* log.(rand(150_000))
e_keV = filter(<(2800), vcat(peaks..., continuum))
e_uncal = e_keV ./ c_true                     # uncalibrated energies in ADC units
length(e_uncal)
```

The first calibration uses a window of 20 keV on each side of every line:

```@example energy
window_sizes = [(20.0keV, 20.0keV) for _ in lines]
result_simple, report_simple = simple_calibration(
    e_uncal, lines, window_sizes; calib_type = :th228
)
result_simple.c
```

The peak fits return positions and widths on the uncalibrated scale because `e_unit` and `m_cal_simple` are set:

```@example energy
result_fit, report_fit = fit_peaks(
    result_simple.peakhists, result_simple.peakstats, lines;
    calib_type = :th228, e_unit = result_simple.unit, m_cal_simple = result_simple.c
)
result_fit[2614.51keV].fwhm
```

The calibration curve is fitted through the centroids:

```@example energy
centroids = [result_fit[l].centroid for l in lines]
result_calib, report_calib = fit_calibration(1, centroids, lines)
result_calib.func
```

The expression string is a function of the uncalibrated energy `e`. Evaluating it on a value shows the calibrated energy and, with `func_err`, its uncertainty. In the dataflow, `ljl_propfunc` evaluates the string on a table column (see [Conventions](@ref)); here the string is turned into a function directly:

```@example energy
calibrate = Core.eval(@__MODULE__, Meta.parse("e -> " * result_calib.func))
calibrate_err = Core.eval(@__MODULE__, Meta.parse("e -> " * result_calib.func_err))
Base.invokelatest(calibrate, 209160.0), Base.invokelatest(calibrate_err, 209160.0)
```

The widths of the peaks are converted from ADC units to keV with the slope of the calibration curve, and the resolution curve is fitted:

```@example energy
fwhm = [result_fit[l].fwhm * result_calib.par[2] for l in lines]
result_fwhm, report_fwhm = fit_fwhm(1, lines, fwhm)
result_fwhm.qbb
```

With LegendMakie.jl and a Makie backend, `lplot(report_calib)` and `lplot(report_fwhm)` draw the curves, and `lplot(report_fit[2614.51keV])` draws the fit of one line:

```@example energy
using CairoMakie, LegendMakie
lplot(report_fit[2614.51keV])
```
