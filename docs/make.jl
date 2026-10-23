# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using Literate
using LegendSpecFits

function fix_literate_output(content)
    content = replace(content, "EditURL = \"@__REPO_ROOT_URL__/\"" => "")
    return content
end

gen_content_dir = joinpath(@__DIR__, "src", "tutorials")
for tut_lit_fn in filter(fn -> endswith(fn, "_lit.jl"), readdir(gen_content_dir))
    lit_src_fn = joinpath(gen_content_dir, tut_lit_fn)
    tut_basename = tut_lit_fn[1:end-7] # remove "_lit.jl"
    Literate.markdown(lit_src_fn, gen_content_dir, name = tut_basename, documenter = true, credit = true, postprocess = fix_literate_output)
    Literate.notebook(lit_src_fn, gen_content_dir, execute = false, name = tut_basename, documenter = true, credit = true)
    Literate.script(lit_src_fn, gen_content_dir, keep_comments = false, name = tut_basename, documenter = true, credit = false)
end

# Doctest setup
DocMeta.setdocmeta!(
    LegendSpecFits,
    :DocTestSetup,
    :(using LegendSpecFits);
    recursive=true,
)

makedocs(
    sitename = "LegendSpecFits.jl",
    modules = [LegendSpecFits],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://legend-exp.github.io/LegendSpecFits.jl/stable/",
        size_threshold = nothing,
        size_threshold_warn = nothing,
        example_size_threshold = nothing,
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "Conventions" => "man/conventions.md",
            "Energy Calibration" => "man/energy_calibration.md",
            "Charge-Trapping Correction" => "man/ctc.md",
            "Peak Shapes and Likelihoods" => "man/peakshapes.md",
            "A/E Pulse-Shape Discrimination" => "man/aoe.md",
            "LQ Cut" => "man/lq.md",
            "Filter Optimization" => "man/filter_optimization.md",
            "Cuts and Goodness of Fit" => "man/cuts_gof.md",
            "SiPM Calibration" => "man/sipm.md",
        ],
        "Tutorials" => [
            "tutorials/energy_calibration.md",
            "tutorials/aoe.md",
        ],
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
