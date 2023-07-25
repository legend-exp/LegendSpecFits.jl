# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

module LegendSpecFitsBATExt

isdefined(Base, :get_extension) ? (using BAT) : (using ..BAT)

using Distributions

#TODO get this running, doesn't work at the moment
LegendSpecFits.get_distribution_transform(d::Distribution, pprior::NamedTupleDist) = BAT.DistributionTransform(d, pprior)

end # module LegendSpecFitsBATExt
