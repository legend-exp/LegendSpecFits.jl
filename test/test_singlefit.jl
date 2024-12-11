# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
using LegendSpecFits
using Test
using LegendDataTypes: fast_flatten
using Measurements: value as mvalue
using Distributions
using StatsBase
using Interpolations
using Unitful 
include("test_utils.jl")


@testset "singlefit" begin
    # compose dummy data 
    dist = Normal(0.0, 1.0)
    x = rand(dist, 1000000)
    x_cut = cut_single_peak(x, -3.0, 3.0)
    
    # fit the data
    res, _ = fit_single_trunc_gauss(x; uncertainty=true)
    res_cut, _ = fit_single_trunc_gauss(x, x_cut; uncertainty=true)
    # check the results
    @testset "fit_single_trunc_gauss" begin
        @test isapprox(mvalue(res.μ), 0.0, atol = 0.1)
        @test isapprox(mvalue(res.σ), 1.0, atol = 0.1)
        @test isapprox(mvalue(res_cut.μ), 0.0, atol = 0.1)
        @test isapprox(mvalue(res_cut.σ), 1.0, atol = 0.1)
    end

    res_right, _ = fit_half_centered_trunc_gauss(x, 0.0, x_cut; left=false, uncertainty=true)
    res_left, _ = fit_half_centered_trunc_gauss(x, 0.0, x_cut; left=true, uncertainty=true)
    # check the results
    @testset "fit_half_centered_trunc_gauss" begin
        @test isapprox(mvalue(res_right.μ), 0.0, atol = 0.1)
        @test isapprox(mvalue(res_right.σ), 1.0, atol = 0.1)
        @test isapprox(mvalue(res_left.μ), 0.0, atol = 0.1)
        @test isapprox(mvalue(res_left.σ), 1.0, atol = 0.1)
    end

    res_right, _ = fit_half_trunc_gauss(x, x_cut; left=false, uncertainty=true)
    res_left, _ = fit_half_trunc_gauss(x, x_cut; left=true, uncertainty=true)
    # check the results
    @testset "fit_half_trunc_gauss" begin
        @test isapprox(mvalue(res_right.μ), 0.0, atol = 0.1)
        @test isapprox(mvalue(res_right.σ), 1.0, atol = 0.1)
        @test isapprox(mvalue(res_left.μ), 0.0, atol = 0.1)
        @test isapprox(mvalue(res_left.σ), 1.0, atol = 0.1)
    end

    hist = fit(Histogram, x, -3.0:0.05:3.0)
    res, _ = fit_binned_trunc_gauss(hist; uncertainty=true)
    res_cut, _ = fit_binned_trunc_gauss(hist, x_cut; uncertainty=true)
    # check the results
    @testset "fit_binned_trunc_gauss" begin
        @test isapprox(mvalue(res.μ), 0.0, atol = 0.1)
        @test isapprox(mvalue(res.σ), 1.0, atol = 0.1)
        @test isapprox(mvalue(res_cut.μ), 0.0, atol = 0.1)
        @test isapprox(mvalue(res_cut.σ), 1.0, atol = 0.1)
    end
end