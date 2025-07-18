# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package LegendSpecFits" begin
    #include("test_aqua.jl")
    include("test_filter_optimization.jl")
    include("test_specfit.jl")
    include("test_fit_chisq.jl")
    include("test_singlefit.jl")
    include("test_cuts.jl")
    include("test_calibration.jl")
    include("test_docs.jl")
    include("test_lq.jl")
    include("test_aoe.jl")
    include("test_simplecal.jl")
    isempty(Test.detect_ambiguities(LegendSpecFits))
end # testset
