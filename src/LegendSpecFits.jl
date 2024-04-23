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
using Distributions
using FillArrays
using ForwardDiff
using IntervalSets
using InverseFunctions
using IrrationalConstants
using LegendDataManagement
using LinearRegression
using LsqFit
using Measurements
using Measurements: value as mvalue
using Measurements: uncertainty as muncert
using Optim
using PropDicts
using RadiationSpectra
using Roots
using SpecialFunctions
using StatsBase
using StructArrays
using Tables
using TypedTables
using Unitful
using ValueShapes

include("utils.jl")
include("peakshapes.jl")
include("likelihoods.jl")
include("priors.jl")
include("cut.jl")
include("aoefit.jl")
include("filter_optimization.jl")
include("singlefit.jl")
include("specfit.jl")
include("chi2fit.jl")
include("fit_calibration.jl")
include("fwhm.jl")
include("simple_calibration.jl")
include("auto_calibration.jl")
include("aoe_calibration.jl")
include("specfit_combined.jl")
include("ctc.jl")
include("qc.jl")
include("gof.jl")
include("precompile.jl")
include("lqfit.jl")
include("lqcut.jl")

abstract type UncertTag end
ForwardDiff.:(≺)(::Type{<:ForwardDiff.Tag}, ::Type{UncertTag}) = true
ForwardDiff.:(≺)(::Type{UncertTag}, ::Type{<:ForwardDiff.Tag}) = false
ForwardDiff.:(≺)(::Type{UncertTag}, ::Type{UncertTag}) = false

end # module
