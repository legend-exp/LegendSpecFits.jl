# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using Test
using LegendSpecFits
import Documenter

Documenter.DocMeta.setdocmeta!(
    LegendSpecFits,
    :DocTestSetup,
    :(using LegendSpecFits);
    recursive=true,
)
Documenter.doctest(LegendSpecFits)
