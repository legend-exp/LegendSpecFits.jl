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
using Distributions
using FillArrays
using InverseFunctions
using IrrationalConstants
using RadiationSpectra
using SpecialFunctions
using StatsBase
using StructArrays
using Tables
using TypedTables
using Unitful
using ValueShapes
using IntervalSets
using Roots
using BAT
using LsqFit
using Optim
using ForwardDiff
using LinearRegression
using PropDicts

include("utils.jl")
include("peakshapes.jl")
include("likelihoods.jl")
include("priors.jl")
include("cut.jl")
include("aoefit.jl")
include("filter_optimization.jl")
include("singlefit.jl")
include("specfit.jl")
include("fwhm.jl")
include("simple_calibration.jl")
include("auto_calibration.jl")
include("aoe_calibration.jl")
include("specfit_combined.jl")
include("ctc.jl")
include("qc.jl")

include("precompile.jl")

end # module
