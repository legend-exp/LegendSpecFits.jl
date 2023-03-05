# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

module LegendSpecFitsOptimExt

isdefined(Base, :get_extension) ? (using Optim) : (using ..Optim)


end # module LegendSpecFitsOptimExt
