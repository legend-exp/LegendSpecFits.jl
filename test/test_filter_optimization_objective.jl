# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits
using Test
using Measurements: measurement, value as mvalue
using Random
using TypedTables
using Unitful

@testset "Filter optimization objective" begin
    μ = 2614.5
    peak = 2614.5u"keV"
    window = (35.0u"keV", 25.0u"keV")

    @testset "peak_width_metrics" begin
        v_gauss = (μ = μ, σ = 1.2, n = 1000.0, skew_fraction = 0.0, skew_width = 0.005)
        m = peak_width_metrics(v_gauss)
        @test keys(m) == (:fwhm, :w90, :asym, :tail_fraction, :tail_scale, :centroid)
        @test m.w90 ≈ 3.289707 * 1.2 rtol = 0.01
        @test m.asym ≈ 1.0 rtol = 0.01
        @test m.fwhm ≈ 2.354820 * 1.2 rtol = 0.01
        @test m.tail_fraction == 0.0
        @test m.tail_scale ≈ 0.005 * μ
        @test m.centroid ≈ μ
        @test peak_width_metrics(v_gauss; containment = 0.6827).w90 ≈ 2 * 1.2 rtol = 0.01
        @test peak_width_metrics(map(x -> measurement(x, 0.1 * x + 0.01), v_gauss)) == m

        # a tail scale of 50σ puts nearly all tail events below μ
        sw = 50 / μ
        v_tail(f, s) = (μ = μ, σ = 1.0, n = 1000.0, skew_fraction = f, skew_width = s)
        for f in (0.05, 0.1, 0.2)
            m_tail = peak_width_metrics(v_tail(f, sw))
            @test m_tail.asym ≈ (1 + f) / (1 - f) rtol = 0.01
            @test m_tail.tail_fraction == f
            @test m_tail.centroid ≈ μ - f * 50
        end
        w90_f = [peak_width_metrics(v_tail(f, sw)).w90 for f in (0.0, 0.05, 0.1, 0.2)]
        @test all(>(0), diff(w90_f))
        w90_sw = [peak_width_metrics(v_tail(0.1, s / μ)).w90 for s in (2.0, 5.0, 20.0, 50.0)]
        @test all(>(0), diff(w90_sw))

        @test_throws "parameter σ must be positive" peak_width_metrics(merge(v_gauss, (σ = 0.0,)))
        @test_throws "parameter σ must be positive" peak_width_metrics(merge(v_gauss, (σ = -1.0,)))
        @test_throws "parameter skew_width is not finite" peak_width_metrics(merge(v_gauss, (skew_width = NaN,)))
        @test_throws "parameter μ is not finite" peak_width_metrics(merge(v_gauss, (μ = Inf,)))
        @test_throws "parameter n is not finite" peak_width_metrics(merge(v_gauss, (n = NaN,)))
        @test_throws "parameter skew_fraction is not finite" peak_width_metrics(merge(v_gauss, (skew_fraction = Inf,)))
        @test_throws "containment must lie in (0, 1)" peak_width_metrics(v_gauss; containment = 1.0)
        @test_throws "containment must lie in (0, 1)" peak_width_metrics(v_gauss; containment = 0.0)
        @test_throws ArgumentError peak_width_metrics(merge(v_gauss, (skew_fraction = 1.0,)))
    end

    @testset "objective_value" begin
        m = (fwhm = 2.0u"keV", w90 = 3.0u"keV", asym = 1.5, tail_fraction = 0.1, tail_scale = 5.0u"keV", centroid = 2614.0u"keV")
        @test objective_value(m, :w90) == 3.0u"keV"
        @test objective_value(m, :fwhm) == 2.0u"keV"
        @test objective_value(m, :asym) == 1.5
        @test objective_value(m, :fwhm_asym) == 3.0u"keV"
        @test_throws "objective must be one of (:w90, :fwhm, :asym, :fwhm_asym)" objective_value(m, :sigma)
    end

    @testset "select_optimum" begin
        params = [(2, 1), (1, 3), (1, 2), (3, 0)]
        # all four within 0.5 % of the minimum: shortest first, then second element
        @test select_optimum([1.0, 1.002, 1.003, 0.999], params; rel_tol = 0.005) == 3
        @test select_optimum([1.0, 1.002, 1.003, 0.999], params; rel_tol = 0.0) == 4
        @test select_optimum([1.0, 1.02, 1.004, 1.03], params; rel_tol = 0.005) == 3
        # NaN is never selected, even when its parameters are the shortest
        @test select_optimum([1.0, 1.0, NaN, 1.0], params; rel_tol = 0.005) == 2
        @test select_optimum([NaN, 2.0, NaN, 1.0], params; rel_tol = 0.005) == 4
        @test_throws "all objective values are NaN" select_optimum([NaN, NaN], [(1, 1), (2, 2)]; rel_tol = 0.005)
        @test_throws DimensionMismatch select_optimum([1.0, 2.0], [(1, 1)]; rel_tol = 0.005)
        # indices follow the input axes
        @test select_optimum(view([9.0, 1.0, 1.0], 2:3), view([(0, 0), (2, 2), (1, 1)], 2:3); rel_tol = 0.0) == 2
    end
end
