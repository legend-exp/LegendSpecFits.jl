# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
using LegendSpecFits
using Test
using Logging
using LegendDataTypes: fast_flatten
using Measurements: value as mvalue
using Distributions
using StatsBase
using Interpolations
using Unitful
using TypedTables
using PropDicts
include("test_utils.jl")


logger = ConsoleLogger(stderr, Logging.Info)

with_logger(logger) do
    @testset "cuts" begin
        # compose dummy data 
        dist = Normal(0.0, 1.0)
        x = rand(dist, 1000000)
        @testset "cut_single_peak" begin
            x_cut = @test_nowarn cut_single_peak(x, -10.0, 10.0; relative_cut = 0.5, n_bins = -1)
            @test isapprox(x_cut.max, 0.0; atol=0.1)
            @test isapprox(x_cut.high - x_cut.low, 2.355; atol=0.1)
            
            x_cut_unit = @test_nowarn cut_single_peak(x .*u"s", -10.0u"s", 10.0u"s"; relative_cut = 0.5, n_bins = -1)
            @test unit(x_cut_unit.max) == u"s"
            @test isapprox(x_cut_unit.max, 0.0u"s"; atol=0.1u"s")
            @test isapprox(x_cut_unit.high - x_cut_unit.low, 2.355u"s"; atol=0.1u"s")
        end

        @testset "get_centered_gaussian_window_cut" begin
            result, report = @test_nowarn get_centered_gaussian_window_cut(x, -10.0, 10.0, 2.0; relative_cut = 0.01, n_bins = -1, left=false, fixed_center=false)
            @test isapprox(mvalue(result.low_cut), -2.0; atol=0.05)
            @test isapprox(mvalue(result.high_cut), 2.0; atol=0.05)
            @test isapprox(mvalue(result.μ), 0.0; atol=0.05)
            @test isapprox(mvalue(result.σ), 1.0; atol=0.05)

            result, report = @test_nowarn get_centered_gaussian_window_cut(x, -10.0, 10.0, 2.0; relative_cut = 0.01, n_bins = -1, left=true, fixed_center=true)
            @test isapprox(mvalue(result.low_cut), -2.0; atol=0.05)
            @test isapprox(mvalue(result.high_cut), 2.0; atol=0.05)
            @test isapprox(mvalue(result.μ), 0.0; atol=0.05)
            @test isapprox(mvalue(result.σ), 1.0; atol=0.05)

            result, report = @test_nowarn get_centered_gaussian_window_cut(x .* u"s", -10.0u"s", 10.0u"s", 2.0; relative_cut = 0.01, n_bins = -1, left=true, fixed_center=false)
            @test unit(result.low_cut) == unit(result.high_cut) == u"s"
            @test isapprox(mvalue(result.low_cut), -2.0u"s"; atol=0.05u"s")
            @test isapprox(mvalue(result.high_cut), 2.0u"s"; atol=0.05u"s")
            @test isapprox(mvalue(result.μ), 0.0u"s"; atol=0.05u"s")
            @test isapprox(mvalue(result.σ), 1.0u"s"; atol=0.05u"s")
        end
    end

    @testset "qc" begin
        # compose dummy data
        dist1 = Normal(0.0, 1.0)
        
        result, report = @test_nowarn qc_window_cut(rand(dist1, 1000000), -10.0, 10.0, 2.0; NamedTuple(PropDict(:relative_cut => 0.01, :n_bins => -1, :fixed_center => false, :left => false))...)

        @test isapprox(mvalue(result.low_cut), -2.0; atol=0.05)
        @test isapprox(mvalue(result.high_cut), 2.0; atol=0.05)
        @test isapprox(mvalue(result.μ), 0.0; atol=0.05)
        @test isapprox(mvalue(result.σ), 1.0; atol=0.05)
        
        dist2 = Normal(0.0, 2.0)
        t = Table(x1 = rand(dist1, 1000000), x2 = rand(dist2, 1000000))

        config = PropDict(:x1 => PropDict(
            :min => -10.0,
            :max => 10.0,
            :sigma => 2.0,
            :kwargs => PropDict(:relative_cut => 0.01, :n_bins => -1, :fixed_center => false, :left => false)
        ),
        :x2 => PropDict(
            :min => -15.0,
            :max => 15.0,
            :sigma => 1.0,
            :kwargs => PropDict(:relative_cut => 0.01, :n_bins => -1, :fixed_center => false, :left => true)
        ))
        
        result, report = @test_nowarn qc_window_cut(t, config, (:x1, :x2))

        @test isapprox(mvalue(result.x2.low_cut), -2.0; atol= 0.05)
        @test isapprox(mvalue(result.x2.high_cut), 2.0; atol= 0.05)
        @test isapprox(mvalue(result.x2.μ), 0.0; atol= 0.05)
        @test isapprox(mvalue(result.x2.σ), 2.0; atol= 0.05)

        t = Table(x1 = rand(dist1, 1000000) .* u"s", x2 = rand(dist2, 1000000) .* u"s")
        config.x1 = PropDict(
            :min => -10.0u"s",
            :max => 10.0u"s",
            :sigma => 2.0,
            :kwargs => PropDict(:relative_cut => 0.01, :n_bins => -1, :fixed_center => false, :left => false)
        )
        config.x2 = PropDict(
            :min => -15.0u"s",
            :max => 15.0u"s",
            :sigma => 1.0,
            :kwargs => PropDict(:relative_cut => 0.01, :n_bins => -1, :fixed_center => false, :left => true)
        )
        result, report = @test_nowarn qc_window_cut(t, config, (:x1, :x2,))

        @test unit(result.x1.low_cut) == unit(result.x1.high_cut) == u"s"
        @test unit(result.x2.low_cut) == unit(result.x2.high_cut) == u"s"
        
        @test isapprox(mvalue(result.x1.low_cut), -2.0u"s"; atol=0.05u"s")
        @test isapprox(mvalue(result.x2.low_cut), -2.0u"s"; atol=0.05u"s")
    end
end