# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits
using Test
using Measurements: value as mvalue
using Distributions
using Unitful 

include("test_utils.jl")


@testset "A/E energy correction" begin

    # generate example A/E distribution with E-dependence and low A/E tail
    e_cal = rand(Distributions.Exponential(300), 5_000_000) .+ 300
    μA, μB, σA, σB = 1.01, -4e-6, 5e-3, 12.0
    myμ(E) = μA + μB * E
    myσ(E) = sqrt(σA^2 + σB^2/E^2)
    aoe = [let _μ = myμ(E), _σ = myσ(E); (rand() < 0.2 ? -rand(Distributions.Exponential(5*_σ)) : 0) + _σ*randn() + _μ; end for E in e_cal]


    # fit the A/E vs. E distribution
    compton_bands = collect((550:50:2350)u"keV")
    compton_window = 20u"keV"
    compton_band_peakhists = LegendSpecFits.generate_aoe_compton_bands(aoe, e_cal*u"keV", compton_bands, compton_window)
    result_fit, report_fit = LegendSpecFits.fit_aoe_compton(compton_band_peakhists.peakhists, compton_band_peakhists.peakstats, compton_bands, uncertainty=true)
    μs = [result_fit[band].μ for band in compton_bands]
    σs = [result_fit[band].σ for band in compton_bands]
    result_fit_single, report_fit_single = LegendSpecFits.fit_aoe_corrections(compton_bands, μs, σs)
    result_fit_combined, report_fit_combined = LegendSpecFits.fit_aoe_compton_combined(compton_band_peakhists.peakhists, compton_band_peakhists.peakstats, compton_bands, result_fit_single, uncertainty=true)

    # check that the measured results agree within 5% with the original values
    @testset "Individual A/E fits" begin
        @test isapprox(mvalue(result_fit_single.µ_compton.par[1]),      μA,             rtol = 0.1)
        @test isapprox(mvalue(result_fit_single.µ_compton.par[2]),      μB * u"keV^-1", rtol = 0.1)
        @test isapprox(abs(mvalue(result_fit_single.σ_compton.par[1])), σA,             rtol = 0.1)
        @test isapprox(abs(mvalue(result_fit_single.σ_compton.par[2])), σB * u"keV^2",  rtol = 0.1)
    end

    @testset "Combined A/E fits" begin
        @test isapprox(mvalue(result_fit_combined.µA),      μA, rtol = 0.1)
        @test isapprox(mvalue(result_fit_combined.μB),      μB, rtol = 0.1)
        @test isapprox(abs(mvalue(result_fit_combined.σA)), σA, rtol = 0.1)
        @test isapprox(abs(mvalue(result_fit_combined.σB)), σB, rtol = 0.1)
    end
end