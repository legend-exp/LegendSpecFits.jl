# LegendSpecFits.jl

LegendSpecFits.jl is a Julia package for fitting spectra and calibrating the energy and pulse-shape parameters of high-purity germanium (HPGe) and silicon photomultiplier (SiPM) detectors of the [LEGEND](https://legend-exp.org) experiment. It is part of the LEGEND Julia analysis software and is called by the production dataflow ([Juleana.jl](https://github.com/legend-exp/Juleana.jl)) as well as in interactive analyses.

The package provides binned maximum-likelihood fits of gamma lines and other distributions, calibration curves and energy resolution curves, charge-trapping corrections, A/E and late-charge (LQ) pulse-shape cuts together with their survival fractions, energy filter optimization, and SiPM calibration. Most fit functions return their numerical results together with a report that LegendMakie.jl turns into figures.

## Installation

LegendSpecFits.jl is registered in the [LEGEND Julia registry](https://github.com/legend-exp/LegendJuliaRegistry). After adding the registry once, the package can be installed with the Julia package manager:

```julia
using Pkg
pkg"registry add General https://github.com/legend-exp/LegendJuliaRegistry.git"
Pkg.add("LegendSpecFits")
```

To work on the package itself, use `Pkg.develop("LegendSpecFits")`.

## Overview

The manual pages describe the concepts and the workflow of each part of the package:

* [Conventions](@ref) describes what all fit functions have in common: the `(result, report)` return pair, units, uncertainties, expression strings and plotting.
* [Energy Calibration](@ref) covers the HPGe calibration chain from the first rough calibration to the calibration curve and the energy resolution curve.
* [Charge-Trapping Correction](@ref) corrects the energy for incomplete charge collection using the drift-time proxy `qdrift`.
* [Peak Shapes and Likelihoods](@ref) lists the peak models and the likelihood and prior functions used by the fits.
* [A/E Pulse-Shape Discrimination](@ref), [LQ Cut](@ref) and [Cuts and Goodness of Fit](@ref) cover pulse-shape cuts, survival fractions, simple quality cuts and goodness-of-fit measures.
* [Filter Optimization](@ref) covers the choice of energy filter parameters from the electronic noise and the resolution.
* [SiPM Calibration](@ref) covers the calibration of SiPM spectra.

The tutorials run complete analyses on synthetic data. The [API](@ref) page lists all documented functions and types.

## Related Packages

* [LegendMakie.jl](https://github.com/legend-exp/LegendMakie.jl) provides the plot recipes (`lplot`) that draw the reports returned by the fit functions.
* [LegendDataManagement.jl](https://github.com/legend-exp/LegendDataManagement.jl) reads LEGEND data and metadata and evaluates the expression strings that the fit functions return (`ljl_propfunc`).
* [RadiationSpectra.jl](https://github.com/JuliaPhysics/RadiationSpectra.jl) provides the peak finder used for the first calibration.
* [BAT.jl](https://github.com/bat/BAT.jl) provides the distribution transforms and the optimization interface behind the likelihood fits.
* [Juleana.jl](https://github.com/legend-exp/Juleana.jl) is the production dataflow that applies the functions of this package to LEGEND data.
