# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
using LegendSpecFits
using Test
using Random
using StatsBase

@testset "SiPM simple calibration" begin
    rng = Random.MersenneTwister(1234)
    # amplitude spectrum with 5 P.E. peaks of 1.4 ADC spacing and some background
    peaks(rng) = vcat([randn(rng, round(Int, exp10(6 - 0.5 * i))) .* 0.3 .+ i * 1.4 for i in Base.OneTo(5)]..., randn(rng, 10_000) .* 1.5 .+ 6)

    @testset "no noise tail: trigger threshold at the noise cut" begin
        result, report = sipm_simple_calibration(peaks(rng); n_fwhm_noise_cut = 0.0, initial_min_amp = 0.3)
        @test result.c ≈ 1 / 1.4 rtol = 0.1
        @test result.noise_threshold == 0.3
        @test result.trigger_threshold_cal == result.noise_threshold_cal
        @test result.trigger_threshold ≈ result.f_simple_uncal(result.trigger_threshold_cal)
        @test haskey(report, :trigger_threshold) && haskey(report, :trigger_threshold_cal)
    end

    @testset "noise tail: trigger threshold in the valley" begin
        e_uncal = vcat(peaks(rng), randexp(rng, 1_000_000) .* 0.15)
        result, report = sipm_simple_calibration(e_uncal; n_fwhm_noise_cut = 0.0, initial_min_amp = 0.3)
        @test result.c ≈ 1 / 1.4 rtol = 0.1
        # the valley lies between the noise tail (ends around 0.6 PE) and the 1 PE peak (σ = 0.2 PE)
        @test result.noise_threshold_cal < result.trigger_threshold_cal < 1 - 2 * 0.3 / 1.4
        @test result.trigger_threshold ≈ result.f_simple_uncal(result.trigger_threshold_cal)
    end

    @testset "wide empty valley: trigger threshold in its middle" begin
        e_uncal = vcat(peaks(rng), randexp(rng, 1_000_000) .* 0.04)
        result, report = sipm_simple_calibration(e_uncal; n_fwhm_noise_cut = 0.0, initial_min_amp = 0.2)
        # the single-trigger spectrum is empty from ~0.25 to ~0.6 PE: the threshold sits well inside
        @test 0.3 < result.trigger_threshold_cal < 0.6
    end
end
