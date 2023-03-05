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

include("specfit.jl")

@static if !isdefined(Base, :get_extension)
    using Requires
    include("../ext/LegendSpecFitsRecipesBaseExt.jl")
end

include("precompile.jl")

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require BAT = "c0cd4b16-88b7-57fa-983b-ab80aecada7e" include("../ext/LegendSpecFitsBATExt.jl")
        @require Optim = "429524aa-4258-5aef-a3af-852621145aeb" include("../ext/LegendSpecFitsOptimExt.jl")
    end
end

end # module
