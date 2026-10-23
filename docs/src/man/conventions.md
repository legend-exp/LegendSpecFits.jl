# Conventions

The fit and calibration functions of LegendSpecFits.jl share a common interface: they return a `(result, report)` pair, carry units and uncertainties in their results, and describe fitted functions as expression strings that other packages evaluate on data tables.

## Result and Report

Most functions return a tuple `(result, report)`:

* `result` is a `NamedTuple` (or an `OrderedDict` of `NamedTuple`s, one per peak) with the numbers that are stored as parameters: fit parameters, goodness of fit, and expression strings. It contains no large arrays.
* `report` holds what is needed to draw the fit: histograms, the fitted function as a Julia closure, residuals, and the input points. It is passed to plot recipes and is not stored in parameter files.

For example, a calibration fit returns:

```julia
result, report = fit_calibration(1, positions, lines)
result.func     # expression string for the calibration curve
lplot(report)   # figure of the fit
```

Some helper functions return plain tuples instead, for example [`peakhists_gamma`](@ref), which returns `(peakhists, peakstats, h, bin_widths)`. The docstring of each function lists its return fields.

## Units

Energies carry [Unitful.jl](https://github.com/JuliaPhysics/Unitful.jl) units. Calibrated energies and calibration lines are given in keV, for example `2614.5u"keV"`. Uncalibrated values, such as ADC units or raw filter outputs, are plain numbers. The calibration constant `c` returned by [`simple_calibration`](@ref) therefore has the unit keV and means keV per uncalibrated unit.

The package uses `keV` as the internal energy unit (`LegendSpecFits.e_unit`). Fit functions strip units before they optimize and attach them again to the results. Functions that accept both calibrated and uncalibrated input take the energy unit as a keyword (`e_unit`) or derive it from the unit of the input.

!!! note
    Methods are specific about unitful and unitless input. Convert explicitly with `ustrip` or by multiplying with `u"keV"` if a method rejects the input.

## Uncertainties

Fitted parameters are `Measurement` values of [Measurements.jl](https://github.com/JuliaPhysics/Measurements.jl), `value ± uncertainty`. The uncertainty comes from the inverse of the Hessian of the negative log-likelihood (or of the χ² function) at the optimum. Parameters from one fit are correlated: the covariance matrix is stored in the goodness-of-fit entry `gof.covmat`, and the functions that build derived quantities (calibration curve, resolution curve, FWHM) propagate it.

Values and uncertainties are extracted with [`LegendSpecFits.mvalue`](@ref) and [`LegendSpecFits.muncert`](@ref), which also work on arrays and `NamedTuple`s.

Most fit functions take a keyword `uncertainty::Bool`. With `uncertainty = false`, the function skips the Hessian and the goodness of fit, and all uncertainties are `NaN`. This shortens the run time of fits that are called many times, such as the inner steps of an optimization. If a fit does not converge, the field `converged` in the result (or in `gof`) is `false`.

## Goodness of Fit

Peak fits and curve fits report a `gof` `NamedTuple`. Binned likelihood fits provide a p-value, a χ² value with its degrees of freedom, the covariance matrix `covmat`, residuals, and `converged`. The χ² fits of [`chi2fit`](@ref) report `pvalue`, `chi2min`, `dof`, `covmat`, normalized residuals, and `converged`. The page [Cuts and Goodness of Fit](@ref) describes the test statistics.

## Expression Strings

Results that define a function of data, such as a calibration, carry the function as a string in the fields `func`, `func_err` and, for resolution curves, `func_cal` and `func_cal_err`. The strings are Julia expressions that refer to columns of a data table by name:

```julia
result.func
# "-0.0336keV .* (e).^0 .+ 0.0125keV .* (e).^1"
```

The variable name is set by the keyword `e_expression` (default `"e"`). A caller that has already applied a correction passes the expression of that correction as `e_expression`, so that the new expression is built on top of the previous one. The charge-trapping correction, for example, returns `"e + <fct> * qdrift^1"`; this string is the `e_expression` of the following [`fit_calibration`](@ref), whose `func` then defines the calibrated energy. The strings are stored in the parameter files of the data production.

Downstream, the strings are evaluated by `ljl_propfunc` of [LegendDataManagement.jl](https://github.com/legend-exp/LegendDataManagement.jl), which turns a string into a function and applies it to the named columns of a table:

```julia
using LegendDataManagement
e_cal = ljl_propfunc(result.func).(table)   # uses the column `e`
```

The `keV` in the string is the Unitful unit. The `func_err` variant writes the uncertainty of the curve with `±`; evaluating it on a column gives `Measurement` values that include the correlation between the fit parameters. Because the strings contain only column names, numbers and units, they can be evaluated without LegendSpecFits.jl being loaded.

## Histograms

Spectra are binned into `Histogram` objects of [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl). One-dimensional histograms use the bin edges as the only axis, and counts are integers. Peak fits take the histogram around the peak and use the bin edges and counts directly, so the bin width must be small compared to the peak width. The functions that create peak histograms ([`simple_calibration`](@ref), [`peakhists_gamma`](@ref)) choose the bin width with the Freedman-Diaconis rule on the data within a window around each line (keyword `binning_peak_window`).

## Plotting Reports

The reports are drawn by plot recipes of [LegendMakie.jl](https://github.com/legend-exp/LegendMakie.jl). With a Makie backend loaded, `lplot` draws a report according to its content, and keywords such as `figsize` adjust the figure:

```julia
using CairoMakie, LegendMakie
fig = lplot(report)
save("fit.pdf", fig)
```

The recipe is selected by the fields of the report; for example, a report with `type = :cal` is drawn as a calibration curve and `type = :fwhm` as a resolution curve. LegendSpecFits.jl itself does not depend on a plotting package.

## Fit Keywords

A few keywords have the same meaning in all fit functions that offer them:

* `uncertainty::Bool`: Calculate the uncertainties and the goodness of fit (see above).
* `fit_func::Symbol`: Selects the peak model by name. The Th-228 line fits accept the models listed in [`fit_single_peak_th228`](@ref); the A/E fits use their own set of names.
* `pseudo_prior`: A `NamedTupleDist` from Distributions.jl and BAT.jl whose entries replace the default prior of the same parameter. The priors are centered on estimates from the histogram, such as the peak position and the FWHM, and keep the optimizer in a plausible range. They are not a Bayesian prior in the sense of a posterior sampling: the fits report the maximum of the likelihood with the Hessian uncertainty.
* `e_expression`: Name or expression of the energy variable in the returned strings.
* `m_cal_simple`: The constant of a preceding [`simple_calibration`](@ref). Functions that work on the uncalibrated scale use it to convert results to ADC units or to choose start values.

The optimizers work on parameters that are transformed to an unbounded range with distribution transforms of BAT.jl and use L-BFGS-B from Optimization.jl; the nonlinear fits of the charge-trapping correction use NLopt. The time limit of the NLopt steps is set with `LegendSpecFits.set_timelimit`.
