# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

module LegendSpecFitsBATExt

isdefined(Base, :get_extension) ? (using BAT) : (using ..BAT)


end # module LegendSpecFitsBATExt
