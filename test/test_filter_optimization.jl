# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits
using Test
using Unitful

@testset "Test filter optimization" begin
    tmp = -1:0.1:1
    min_enc0 = 0.2
    enc_grid = ((min_enc0 .+ 3*min_enc0 .* tmp.^2).*randn(20_000)' .+ 2)
    enc_grid_rt = range(0.5u"μs", step = 0.5u"μs", length = length(tmp))
    min_enc = 0.0
    max_enc = 4.0
    nbins = 100
    rel_cut_fit = 0.1

    result, _ = LegendSpecFits.fit_enc_sigmas(enc_grid, enc_grid_rt, min_enc, max_enc, nbins, rel_cut_fit)
    @test isapprox(result.rt, enc_grid_rt[argmin(tmp.^2)], atol = step(enc_grid_rt))
    @test isapprox(result.min_enc, min_enc0, rtol = 0.05)

    # Set one row to zero (should be ignored)
    enc_grid[4,:] .*= 0
    result, _ = LegendSpecFits.fit_enc_sigmas(enc_grid, enc_grid_rt, min_enc, max_enc, nbins, rel_cut_fit)
    @test isapprox(result.rt, enc_grid_rt[argmin(tmp.^2)], atol = step(enc_grid_rt))
    @test isapprox(result.min_enc, min_enc0, rtol = 0.05)

    # Set one row to NaN (should be ignored)
    enc_grid[6,:] .= NaN
    result, _ = LegendSpecFits.fit_enc_sigmas(enc_grid, enc_grid_rt, min_enc, max_enc, nbins, rel_cut_fit)
    @test isapprox(result.rt, enc_grid_rt[argmin(tmp.^2)], atol = step(enc_grid_rt))
    @test isapprox(result.min_enc, min_enc0, rtol = 0.05)
end


