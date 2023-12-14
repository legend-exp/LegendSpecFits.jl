# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using LegendSpecFits

# Doctest setup
DocMeta.setdocmeta!(
    LegendSpecFits,
    :DocTestSetup,
    :(using LegendSpecFits);
    recursive=true,
)

makedocs(
    sitename = "LegendSpecFits",
    modules = [LegendSpecFits],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://legend-exp.github.io/LegendSpecFits.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    warnonly = ("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/legend-exp/LegendSpecFits.jl.git",
    forcepush = true,
    push_preview = true,
)
