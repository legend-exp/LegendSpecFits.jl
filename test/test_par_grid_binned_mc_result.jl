# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits
using Test

using ArraysOfArrays

using LegendSpecFits: ParGridBinnedMCResult, interpolate_counts


@testset "par_grid_binned_mc_result" begin
    data = ParGridBinnedMCResult(
        (a = 1:0.5:5, b = -3:1:3),
        (0:0.1:100, 0:0.25:200),
        ArrayOfSimilarArrays{Int,2,2}(rand(0:999, 1001, 801, 9, 7))
    )

    p = (a = 2.3, b = 0.7)
    @test @inferred(interpolate_counts(data)) isa Function
    @test @inferred(interpolate_counts(data, p)) == @inferred(interpolate_counts(data)(p))

    # ToDo: Test correctness of interpolation!
end
