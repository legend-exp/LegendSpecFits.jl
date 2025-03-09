# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    LegendSpecFits

Template for Julia packages.
"""
module LegendSpecFits

using LinearAlgebra
using Statistics
using Random

using ArgCheck
using ArraysOfArrays
using BAT
using DensityInterface
using Distributions
using FillArrays
using Format
using ForwardDiff
using GaussianMixtures
using IntervalSets
using InverseFunctions
using IrrationalConstants
using LaTeXStrings
using LogExpFunctions
using LsqFit
using Measurements
using Measurements: value
using Measurements: uncertainty
using OrderedCollections
using Optimization
using OptimizationBBO
using OptimizationNLopt
using OptimizationOptimJL
using PropDicts
using RadiationSpectra
using RadiationSpectra: peakfinder
using Roots
using SavitzkyGolay
using SpecialFunctions
using StatsBase
using StructArrays
using Tables
using TypedTables
using Unitful
using ValueShapes

MaybeWithEnergyUnits = Union{Real, Unitful.Energy{<:Real}}

include("utils.jl")
include("memory_utils.jl")
include("peakshapes.jl")
include("likelihoods.jl")
include("priors.jl")
include("peakstats.jl")
include("simple_cuts.jl")
include("filter_optimization.jl")
include("singlefit.jl")
include("specfit.jl")
include("chi2fit.jl")
include("fit_calibration.jl")
include("fit_fwhm.jl")
include("simple_calibration.jl")
include("auto_calibration.jl")
include("aoefit_functions.jl")
include("aoe_pseudo_prior.jl")
include("aoefit.jl")
include("aoe_fit_calibration.jl")
include("aoefit_combined.jl")
include("aoe_cut.jl")
include("aoe_ctc.jl")
include("aoe_filter_optimization.jl")
include("specfit_combined.jl")
include("ctc.jl")
include("qc.jl")
include("gof.jl")
include("precompile.jl")
include("lqfit.jl")
include("lqcut.jl")
include("pseudo_prior.jl")
include("specfit_functions.jl")
include("calfunc.jl")
include("sipm_simple_calibration.jl")
include("sipmfit.jl")
include("sipm_filter_optimization.jl")
abstract type UncertTag end
ForwardDiff.:(≺)(::Type{<:ForwardDiff.Tag}, ::Type{UncertTag}) = true
ForwardDiff.:(≺)(::Type{UncertTag}, ::Type{<:ForwardDiff.Tag}) = false
ForwardDiff.:(≺)(::Type{UncertTag}, ::Type{UncertTag}) = false

end # module
