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
using Unitful
using ValueShapes
using LsqFit
using IntervalSets
using Roots
using Plots
using Optim
using LinearRegression

include("utils.jl")
include("peakshapes.jl")
include("likelihoods.jl")
include("priors.jl")
include("specfit.jl")
include("fwhm.jl")
include("simple_calibration.jl")
include("auto_calibration.jl")
include("fit_peaks.jl")
include("plot_recipes.jl")

@static if !isdefined(Base, :get_extension)
    using Requires
    include("../ext/LegendSpecFitsRecipesBaseExt.jl")
end

include("precompile.jl")

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require BAT = "c0cd4b16-88b7-57fa-983b-ab80aecada7e" include("../ext/LegendSpecFitsBATExt.jl")
    end
end

end # module
