# Charge-Trapping Correction

Charge carriers that drift through a germanium detector can be trapped before they reach the electrodes, so that the measured energy of an event with a given deposited energy depends on the drift path. This widens the peaks in the energy spectrum. [`ctc_energy`](@ref) estimates a correction of the energy that is a polynomial in a drift-time proxy and that minimizes the width of a chosen peak.

## Drift-Time Proxy

The correction uses the quantity `qdrift`, an estimator from the waveform analysis that is correlated with the drift time of the charge carriers. It enters only as a regressor: the function does not need to know how `qdrift` was computed, only that the energy of a peak depends smoothly on it. The values of `qdrift` and of the energy must come from the same events and have the same length.

## Method

The function selects the events whose energy lies in a window around the chosen `peak`, from `peak - window[1]` to `peak + window[2]`. Typically this is the 2614.5 keV full-energy peak of Th-228, which has high statistics. For trial coefficients $f_i$ it builds the corrected energy

```math
E_\mathrm{ctc} = E + \sum_{i=1}^{n} f_i \, q_\mathrm{drift}^i\,,
```

where $E$ is the input energy, $q_\mathrm{drift}$ the drift-time proxy, and $n$ the polynomial order (keyword `pol_order`, default 1). For each trial it histograms the corrected energies, fits the peak with [`fit_single_peak_th228`](@ref), and evaluates the logarithm of the ratio of the FWHM to the peak height of the fit. A narrow, high peak minimizes this ratio. The coefficients are found with a bounded derivative-free optimization (NLopt), starting from a value that shifts the energy by 1 keV at the median `qdrift`, and are limited to a range that corresponds to shifts between 0.0001 keV and 7 keV at the median `qdrift`. The time of each optimization step is limited by `LegendSpecFits.set_timelimit`; the field `converged` of the result is `false` if the optimizer stops before it finds a minimum.

The FWHM of the peak is fitted before and after the correction, so that the improvement is part of the result (`fwhm_before`, `fwhm_after`).

## Arguments and Result

The function takes the energies with units, `qdrift`, the line energy `peak` and the window as a tuple `(left, right)` of energies:

```julia
result, report = ctc_energy(
    e, qdrift, 2614.5u"keV", (25.0u"keV", 25.0u"keV"), c;
    e_expression = "e", pol_order = 1,
)
```

The optional fifth positional argument is the calibration constant `c` of [`simple_calibration`](@ref) (default `1.0u"keV"`). The energies passed to the function are in keV, and `c` converts the coefficients to the uncalibrated scale.

The result is a `NamedTuple` with

* `peak`, `window`, `m_cal_simple`: the inputs,
* `fct`: the fitted coefficients $f_i$, divided by `c`, so that they apply to `qdrift` and the uncalibrated energy,
* `func`: the correction as an expression string,
* `fwhm_before`, `fwhm_after`: the fitted FWHM of the peak (in keV) without and with the correction,
* `converged`: whether the optimizer converged.

The report holds the selected events, the corrected energies, the histograms before and after, and the peak fits of both. `lplot(report)` draws the peak before and after the correction.

## Expression String

The expression string has the form

```julia
result.func
# "e + 0.0148 * qdrift^1"
```

where `e` is the name set by `e_expression` and `qdrift` is the name of the drift-time column in the data. Passing the name of an earlier correction in `e_expression` extends that correction. The string is evaluated with `ljl_propfunc` of LegendDataManagement.jl on the table of the events (see [Conventions](@ref)), which gives the corrected energy for all events, not only those in the peak window:

```julia
e_ctc = ljl_propfunc(result.func).(table)
```

## Use in the Calibration Chain

The dataflow runs the correction for each energy estimator before the energy calibration, as follows:

1. [`simple_calibration`](@ref) on the uncorrected energy gives the constant `c`.
2. [`ctc_energy`](@ref) is called with the energies multiplied by `c`, the peak and window from the configuration, `c` as `m_cal_simple`, and `e_expression` set to the name of the energy column. The coefficients are therefore valid for the uncalibrated energy.
3. The `func` string defines a new energy column, the corrected energy.
4. A second [`simple_calibration`](@ref), the peak fits, [`fit_calibration`](@ref) and [`fit_fwhm`](@ref) are then run on the corrected energy as described in [Energy Calibration](@ref). The final calibration function is expressed in the corrected energy.

The correction is fitted on one peak and applied to all events. A related correction for the A/E parameter is [`ctc_aoe`](@ref), and for the LQ parameter [`lq_ctc_correction`](@ref).

## Correction Example

The following example applies the correction to a synthetic 2614.5 keV peak whose energy decreases linearly with `qdrift`:

```@example ctc
using LegendSpecFits, Unitful, Random
using Unitful: keV
Random.seed!(2)

n = 20_000
qdrift = 100 .+ 200 .* rand(n)                  # arbitrary units
e = (2614.5 .+ randn(n) .- 0.015 .* qdrift) .* keV

result, report = ctc_energy(e, qdrift, 2614.5keV, (25keV, 25keV))
result.fct, result.func
```

The fitted coefficient is close to the simulated slope of 0.015 keV per unit of `qdrift`, and the FWHM improves:

```@example ctc
result.fwhm_before, result.fwhm_after
```

If the energies are in keV and the calibration constant `c` of a preceding simple calibration is passed, the coefficient is divided by `c` and the string applies to the uncalibrated energy:

```@example ctc
c = 0.0125keV
result_adc, _ = ctc_energy(e, qdrift, 2614.5keV, (25keV, 25keV), c; e_expression = "e_uncal")
result_adc.func
```
