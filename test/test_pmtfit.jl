# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
using LegendSpecFits
using Test
using Distributions
using Measurements: value as mvalue, uncertainty as muncert
using Random

@testset "fit_pmt_spe" begin
    # synthetic pulse-height spectrum: pedestal, its exponential tail, 1 p.e. and 2 p.e. peaks
    rng = MersenneTwister(1234)
    μ1, σ1 = 17.0, 6.0
    x = vcat(abs.(rand(rng, Normal(0.8, 0.3), 200_000)),   # pedestal
             rand(rng, Exponential(3.0), 30_000),           # pedestal tail
             rand(rng, Normal(μ1, σ1), 20_000),             # 1 p.e.
             rand(rng, Normal(2μ1, sqrt(2) * σ1), 2_000))   # 2 p.e.

    result, report = fit_pmt_spe(x)
    @test result.gof.converged
    @test abs(mvalue(result.μ1) - μ1) < 4 * muncert(result.μ1)
    @test abs(mvalue(result.σ1) - σ1) < 4 * muncert(result.σ1)
    @test 0 <= mvalue(result.r2) <= 1
    @test result.x_min > 0
    @test result.gof.pvalue > 1e-3
    @test report.x_min == result.x_min && report.μ1 == result.μ1
    @test isnan(report.f_fit(result.x_min - 1)) && report.f_fit(μ1) > 0
    @test report.f_fit(μ1) ≈ sum(f(μ1) for f in report.f_components)

    # without uncertainties
    result_nu, _ = fit_pmt_spe(x; uncertainty = false)
    @test result_nu.μ1 isa Real && isapprox(result_nu.μ1, mvalue(result.μ1); rtol = 1e-3)

    # 1 p.e. peak outside the default search range: the bounds follow the keywords
    x_high = vcat(abs.(rand(rng, Normal(0.8, 0.3), 100_000)), rand(rng, Exponential(3.0), 20_000),
                  rand(rng, Normal(80.0, 20.0), 20_000))
    result_high, _ = fit_pmt_spe(x_high; hist_range = (0, 300), peak_search_range = (20, 200))
    @test abs(mvalue(result_high.μ1) - 80.0) < 4 * muncert(result_high.μ1)

    # wider 1 p.e. peaks than the lower end of the search range, as for an unfiltered pulse height
    for (μ, σ) in ((17.0, 8.0), (30.0, 10.0))
        x_wide = vcat(abs.(rand(rng, Normal(0.8, 0.3), 200_000)), rand(rng, Exponential(3.0), 30_000),
                      rand(rng, Normal(μ, σ), 20_000), rand(rng, Normal(2μ, sqrt(2) * σ), 2_000))
        result_wide, _ = fit_pmt_spe(x_wide)
        @test result_wide.gof.converged
        @test abs(mvalue(result_wide.μ1) - μ) < 4 * muncert(result_wide.μ1)
        @test abs(mvalue(result_wide.σ1) - σ) < 4 * muncert(result_wide.σ1)
    end

    # no 1 p.e. peak: the pedestal tail only falls
    x_tail = vcat(abs.(rand(rng, Normal(0.8, 0.3), 100_000)), rand(rng, Exponential(3.0), 5_000))
    @test_throws ArgumentError fit_pmt_spe(x_tail)

    # pedestal only: no signal above the pedestal
    @test_throws ArgumentError fit_pmt_spe(abs.(rand(rng, Normal(0.8, 0.3), 10_000)))
    @test_throws ArgumentError fit_pmt_spe(x; peak_search_range = (6, 200))
    @test_throws ArgumentError fit_pmt_spe(x; peak_search_range = (6.0, 6.4))
    @test_throws ArgumentError fit_pmt_spe(x; bin_width = 3.0)
end
