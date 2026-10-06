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
        @test_throws "parameter μ must be positive" peak_width_metrics(merge(v_gauss, (μ = 0.0,)))
        @test_throws "parameter n must be positive" peak_width_metrics(merge(v_gauss, (n = 0.0,)))
        @test_throws "parameter skew_width must be non-negative" peak_width_metrics(merge(v_gauss, (skew_width = -0.01,)))

        # long tails: w90 agrees with a reference on a grid with step σ / 500
        f_sig = LegendSpecFits.get_th228_fit_functions().gamma_sigWithTail
        function w90_reference(v; containment = 0.9)
            x = range(v.μ - max(10 * v.σ, 8 * v.skew_width * v.μ), v.μ + 6 * v.σ; step = v.σ / 500)
            y = [f_sig(xi, v) for xi in x]
            cdf = cumsum([0.0; (y[begin+1:end] .+ y[begin:end-1]) ./ 2])
            cdf ./= last(cdf)
            w = Inf
            j = firstindex(cdf)
            for i in eachindex(cdf)
                target = cdf[i] + containment
                target > last(cdf) && break
                while cdf[j] < target
                    j += 1
                end
                w = min(w, x[j-1] + (target - cdf[j-1]) / (cdf[j] - cdf[j-1]) * step(x) - x[i])
            end
            return w
        end
        for tail_scale in (100.0, 200.0)
            v_long = (μ = μ, σ = 0.8, n = 1000.0, skew_fraction = 0.05, skew_width = tail_scale / μ)
            @test peak_width_metrics(v_long).w90 ≈ w90_reference(v_long) rtol = 0.002
        end
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
        i_axes = select_optimum(view([9.0, 1.0, 1.0], Base.IdentityUnitRange(2:3)), view([(0, 0), (2, 2), (1, 1)], Base.IdentityUnitRange(2:3)); rel_tol = 0.0)
        @test i_axes == 3
        @test i_axes in 2:3
        @test_throws "smallest objective must be positive and finite" select_optimum([0.0, 1.0], [(1, 1), (2, 2)]; rel_tol = 0.005)
        @test_throws "smallest objective must be positive and finite" select_optimum([-1.0, 1.0], [(1, 1), (2, 2)]; rel_tol = 0.005)
        for bad in (-0.1, Inf, NaN)
            @test_throws "rel_tol" select_optimum([1.0, 2.0], [(1, 1), (2, 2)]; rel_tol = bad)
        end
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
        # widths are scaled to the fitted peak position, so the centroid of a symmetric peak is the peak
        @test abs(row.centroid - peak) < 5e-4 * peak
        @test abs(row_raw.centroid - peak) < 5e-4 * peak
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

        for bad in (-0.1, Inf, NaN)
            @test_throws "rel_tol" fit_fwhm_ft(e_grid, e_grid_ft, qdrift, 3.0u"µs", 2400.0, 2800.0, 0.5, false; rel_tol = bad)
        end

        # default path with charge-trapping correction
        fts_ctc = e_grid_ft[4:8]
        e_grid_ctc = μ .+ σ_ft[4:8] .* z' .- 0.01 .* (qdrift' .- 200)
        result_ctc, report_ctc = fit_fwhm_ft(e_grid_ctc, fts_ctc, qdrift, 3.0u"µs", 2400.0, 2800.0, 0.5)
        @test abs(mvalue(result_ctc.ft) - 2.0u"µs") <= step(fts_ctc) * (1 + 1e-9)
        @test all(isapprox.(report_ctc.table.fct, 0.01; rtol = 0.2))
    end

    @testset "optimize_rtft" begin
        rng = MersenneTwister(7)
        n = 3000
        z = randn(rng, n)
        qdrift = 100 .+ 200 .* rand(rng, n)
        lattice = 0.1u"µs"
        to_steps(t) = round(Int, uconvert(NoUnits, t / lattice))
        # σ grows by 2 % per lattice step squared around rt = 4.0 µs, ft = 2.0 µs; the fixed
        # draws `z` make the objective a deterministic function of the lattice point
        σ_bowl(rt, ft) = 0.8 * (1 + 0.02 * ((to_steps(rt) - 40)^2 + (to_steps(ft) - 20)^2))
        f_bowl(rt, ft) = μ .+ σ_bowl(rt, ft) .* z
        kw = (rt_bounds = (1.0u"µs", 8.0u"µs"), ft_bounds = (0.5u"µs", 4.0u"µs"), step0 = (0.4u"µs", 0.4u"µs"), lattice = lattice,
              peak = peak, window = window, min_e = 2400.0, max_e = 2800.0, rel_cut_fit = 0.5, apply_ctc = false)
        unique_points(t) = length(unique(zip(to_steps.(t.rt), to_steps.(t.ft)))) == length(t)

        result, report = optimize_rtft(f_bowl, qdrift, 3.5u"µs", 1.6u"µs"; kw...)
        @test result.converged
        @test abs(result.rt - 4.0u"µs") <= lattice * (1 + 1e-9)
        @test abs(result.ft - 2.0u"µs") <= lattice * (1 + 1e-9)
        @test result.objective == :w90 && report.objective == :w90
        @test result.n_evals == length(report.table) <= 60
        @test unique_points(report.table)
        @test count(==(:seed), report.table.stage) == 1
        @test all(==("ok"), report.table.status)
        @test to_steps.(first(report.path)) == (35, 16)
        @test to_steps.(last(report.path)) == to_steps.((result.rt, result.ft))
        @test (report.rt, report.ft) == (result.rt, result.ft)
        # the table order does not depend on thread scheduling
        _, report_again = optimize_rtft(f_bowl, qdrift, 3.5u"µs", 1.6u"µs"; kw...)
        @test collect(zip(to_steps.(report_again.table.rt), to_steps.(report_again.table.ft))) == collect(zip(to_steps.(report.table.rt), to_steps.(report.table.ft)))

        # the half-plane ft > 2.25 µs excludes the unconstrained optimum
        feasible(rt, ft) = ft > 2.25u"µs"
        result_f, report_f = optimize_rtft(f_bowl, qdrift, 3.5u"µs", 2.8u"µs"; feasible, kw...)
        @test all(feasible.(report_f.table.rt, report_f.table.ft))
        @test abs(result_f.rt - 4.0u"µs") <= lattice * (1 + 1e-9)
        @test abs(result_f.ft - 2.3u"µs") <= lattice * (1 + 1e-9)
        @test unique_points(report_f.table)

        # flat objective: all points tie, the shortest rt, then ft, of the final neighborhood wins
        f_flat(rt, ft) = μ .+ 0.8 .* z
        result_flat, _ = optimize_rtft(f_flat, qdrift, 3.5u"µs", 1.6u"µs"; kw...)
        @test result_flat.converged
        @test to_steps.((result_flat.rt, result_flat.ft)) == (34, 15)

        result_short, report_short = optimize_rtft(f_bowl, qdrift, 3.5u"µs", 1.6u"µs"; merge(kw, (max_evals = 5,))...)
        @test !result_short.converged
        @test result_short.n_evals == length(report_short.table) == 5

        # failed evaluations are recorded and the search continues around them
        f_fail(rt, ft) = ft > 1.65u"µs" ? fill(NaN, n) : f_bowl(rt, ft)
        result_fail, report_fail = optimize_rtft(f_fail, qdrift, 3.5u"µs", 1.6u"µs"; kw...)
        failed = report_fail.table.status .!= "ok"
        @test any(failed)
        @test all(isnan, report_fail.table.w90[failed])
        @test result_fail.ft <= 1.6u"µs" + lattice / 2

        # plateau noise below rel_tol moves the optimum by at most one lattice step
        noise(rt, ft) = 1 + 0.002 * sin(12.9898 * to_steps(rt) + 78.233 * to_steps(ft))
        f_noisy(rt, ft) = μ .+ (σ_bowl(rt, ft) * noise(rt, ft)) .* z
        result_noisy, _ = optimize_rtft(f_noisy, qdrift, 3.5u"µs", 1.6u"µs"; kw...)
        @test abs(to_steps(result_noisy.rt) - 40) <= 1
        @test abs(to_steps(result_noisy.ft) - 20) <= 1

        # a seed in a corner of the bounds: no evaluation outside the bounds
        result_corner, report_corner = optimize_rtft(f_bowl, qdrift, 1.0u"µs", 0.5u"µs"; kw...)
        @test all(r -> 10 <= to_steps(r) <= 80, report_corner.table.rt)
        @test all(f -> 5 <= to_steps(f) <= 40, report_corner.table.ft)
        @test length(report_corner.table) > 1

        # only the seed is feasible: it is returned after one evaluation
        feasible_seed(rt, ft) = (to_steps(rt), to_steps(ft)) == (35, 16)
        result_seed, _ = optimize_rtft(f_bowl, qdrift, 3.5u"µs", 1.6u"µs"; feasible = feasible_seed, kw...)
        @test (to_steps(result_seed.rt), to_steps(result_seed.ft)) == (35, 16)
        @test result_seed.converged
        @test result_seed.n_evals == 1

        # every neighbor of the seed fails: the seed is returned and the minimum is not certified
        f_seed_only(rt, ft) = (to_steps(rt), to_steps(ft)) == (35, 16) ? f_bowl(rt, ft) : fill(NaN, n)
        result_alone, report_alone = optimize_rtft(f_seed_only, qdrift, 3.5u"µs", 1.6u"µs"; kw...)
        @test (to_steps(result_alone.rt), to_steps(result_alone.ft)) == (35, 16)
        @test !result_alone.converged
        @test count(==("ok"), report_alone.table.status) == 1

        # an interrupt is not an evaluation failure
        f_interrupt(rt, ft) = (to_steps(rt), to_steps(ft)) == (35, 16) ? f_bowl(rt, ft) : throw(InterruptException())
        root_error(e) = e isa TaskFailedException ? root_error(e.task.exception) : e isa CompositeException ? root_error(first(e.exceptions)) : e
        err_search = try
            optimize_rtft(f_interrupt, qdrift, 3.5u"µs", 1.6u"µs"; kw...)
            nothing
        catch e
            e
        end
        @test root_error(err_search) isa InterruptException

        for bad in (-0.1, Inf, NaN)
            @test_throws "rel_tol" optimize_rtft(f_bowl, qdrift, 3.5u"µs", 1.6u"µs"; kw..., rel_tol = bad)
        end
        @test_throws TypeError optimize_rtft(f_bowl, qdrift, 3.5u"µs", 1.6u"µs"; kw..., step0 = (0.4u"µs", 0.4u"µs", 0.4u"µs"))

        @test_throws "optimize_rtft: seed evaluation failed" optimize_rtft((rt, ft) -> fill(NaN, n), qdrift, 3.5u"µs", 1.6u"µs"; kw...)
        @test_throws "outside rt_bounds" optimize_rtft(f_bowl, qdrift, 9.0u"µs", 1.6u"µs"; kw...)
        @test_throws "lattice must be positive" optimize_rtft(f_bowl, qdrift, 3.5u"µs", 1.6u"µs"; merge(kw, (lattice = 0.0u"µs",))...)
        @test_throws "max_evals must be at least 1" optimize_rtft(f_bowl, qdrift, 3.5u"µs", 1.6u"µs"; merge(kw, (max_evals = 0,))...)
        @test_throws "eval_kwargs must contain (:peak,)" optimize_rtft(f_bowl, qdrift, 3.5u"µs", 1.6u"µs"; Base.structdiff(kw, NamedTuple{(:peak,)})...)
        @test_throws "unsupported eval_kwargs (:bogus,)" optimize_rtft(f_bowl, qdrift, 3.5u"µs", 1.6u"µs"; kw..., bogus = 1)
        @test_throws "objective must be one of" optimize_rtft(f_bowl, qdrift, 3.5u"µs", 1.6u"µs"; kw..., objective = :sigma)
    end
end
