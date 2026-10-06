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

    @testset "evaluate_energy_point" begin
        rng = MersenneTwister(1234)
        n = 4000
        qdrift = 100 .+ 200 .* rand(rng, n)
        slope = 0.01
        e = μ .+ 0.8 .* randn(rng, n) .- slope .* (qdrift .- 200)
        row = evaluate_energy_point(e, qdrift, peak, window, 2400.0, 2800.0, 0.5)
        row_raw = evaluate_energy_point(e, qdrift, peak, window, 2400.0, 2800.0, 0.5; apply_ctc = false)
        @test keys(row) == (:fwhm_before, :fwhm, :w90, :asym, :tail_fraction, :tail_scale, :centroid, :fct, :m_cal_simple, :ctc_converged, :n_events)
        @test row.fct ≈ slope rtol = 0.2
        @test row.ctc_converged
        @test row.w90 < row_raw.w90
        @test row.fwhm < row.fwhm_before
        @test unit(row.w90) == u"keV"
        @test row.m_cal_simple ≈ 1.0u"keV" rtol = 0.01
        @test row.n_events <= n
        # the ADC correction `e + fct * qdrift` removes the injected dependence (1.5 before)
        e_ctc = e .+ row.fct .* qdrift
        low, high = qdrift .< 150, qdrift .> 250
        @test abs(sum(e_ctc[low]) / count(low) - sum(e_ctc[high]) / count(high)) < 0.3
        # without correction: same histogram as the uncorrected fit inside ctc_energy
        @test row_raw.fct == 0.0
        @test row_raw.fwhm_before == row_raw.fwhm
        @test row_raw.fwhm ≈ row.fwhm_before rtol = 1e-8
        @test evaluate_energy_point(view(e, :), view(qdrift, :), peak, window, 2400.0, 2800.0, 0.5; apply_ctc = false) == row_raw
        @test_throws "at least 100 required" evaluate_energy_point(e[1:50], qdrift[1:50], peak, window, 2400.0, 2800.0, 0.5)
        @test_throws DimensionMismatch evaluate_energy_point(e, qdrift[1:10], peak, window, 2400.0, 2800.0, 0.5)
    end

    @testset "fit_fwhm_ft" begin
        rng = MersenneTwister(42)
        n = 4000
        z = randn(rng, n)
        qdrift = 100 .+ 200 .* rand(rng, n)
        e_grid_ft = range(1.0u"µs", step = 0.2u"µs", length = 11)
        σ_ft = [0.8 * (1 + 3 * ustrip(u"µs", ft - 2.0u"µs")^2) for ft in e_grid_ft]
        e_grid = μ .+ σ_ft .* z'
        e_grid[1, :] .= NaN

        result, report = fit_fwhm_ft(e_grid, e_grid_ft, qdrift, 3.0u"µs", 2400.0, 2800.0, 0.5, false)
        @test abs(mvalue(result.ft) - 2.0u"µs") <= step(e_grid_ft) * (1 + 1e-9)
        @test abs(mvalue(result.legacy_ft) - 2.0u"µs") <= step(e_grid_ft) * (1 + 1e-9)
        @test result.objective == :w90
        @test result.min_objective <= minimum(report.w90) * 1.005
        @test report.table isa Table
        @test length(report.table) == length(e_grid_ft)
        @test report.table.ft == collect(e_grid_ft)
        @test report.table.status[1] != "ok" && isnan(report.table.w90[1])
        @test all(==("ok"), report.table.status[2:end])
        @test report.e_grid_ft == collect(e_grid_ft)[2:end]
        @test all(iszero, report.fct)

        result_fwhm, _ = fit_fwhm_ft(e_grid, e_grid_ft, qdrift, 3.0u"µs", 2400.0, 2800.0, 0.5, false; objective = :fwhm)
        @test result_fwhm.min_objective == result_fwhm.min_fwhm

        @test_throws "No valid FWHM found" fit_fwhm_ft(fill(NaN, 2, n), e_grid_ft[1:2], qdrift, 3.0u"µs", 2400.0, 2800.0, 0.5, false)
        @test_throws "objective must be one of" fit_fwhm_ft(e_grid, e_grid_ft, qdrift, 3.0u"µs", 2400.0, 2800.0, 0.5, false; objective = :sigma)

        # default path with charge-trapping correction
        fts_ctc = e_grid_ft[4:8]
        e_grid_ctc = μ .+ σ_ft[4:8] .* z' .- 0.01 .* (qdrift' .- 200)
        result_ctc, report_ctc = fit_fwhm_ft(e_grid_ctc, fts_ctc, qdrift, 3.0u"µs", 2400.0, 2800.0, 0.5)
        @test abs(mvalue(result_ctc.ft) - 2.0u"µs") <= step(fts_ctc) * (1 + 1e-9)
        @test all(isapprox.(report_ctc.table.fct, 0.01; rtol = 0.2))
    end
end
