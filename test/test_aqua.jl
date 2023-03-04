# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import LegendSpecFits

Test.@testset "Aqua tests" begin
    Aqua.test_all(LegendSpecFits)
end # testset
