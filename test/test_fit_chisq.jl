
using LegendSpecFits
using Measurements
using Measurements: value as mvalue, uncertainty as muncert
using LinearAlgebra: Diagonal, diag, ⋅
using Test

@testset "fit_chisq" begin
    par_true = [5, 2]
    f_lin(x,p1,p2)  = p1 + p2 * x 
    x = [1,2,3,4,5,6,7,8,9,10]
    y = f_lin.(x,par_true...) .+ 0.5.*randn(10)
    @info "chisq fit without uncertainties on x and y "
    result, report       = chi2fit(1, x, y; uncertainty=true) 
    @test isapprox(result.par[1], par_true[1], atol = 0.2*par_true[1])
    @test isapprox(result.par[2], par_true[2], atol = 0.2*par_true[2])

    x = measurement.([1,2,3,4,5,6,7,8,9,10], ones(10))
    y = f_lin.(x,par_true...) .+ 0.5.*randn(10)
    @info "chisq fit with uncertainties on x and y"
    result, report       = chi2fit(1, x, y; uncertainty=true) 
    @test isapprox(result.par[1], par_true[1], atol = 0.2*par_true[1])
    @test isapprox(result.par[2], par_true[2], atol = 0.2*par_true[2])

    # the parameters carry the full covariance of the fit, so quantities derived from them, such as
    # the fit function, propagate the parameter correlations; `correlated=false` returns independent
    # parameters with the same marginal uncertainties
    @test Measurements.cov(result.par) ≈ result.gof.covmat
    @test muncert(report.f_fit(5.0)) ≈ sqrt([1, 5.0]' * result.gof.covmat * [1, 5.0])
    result_ind, report_ind = chi2fit(1, x, y; uncertainty=true, correlated=false)
    @test mvalue.(result_ind.par) ≈ mvalue.(result.par)
    @test muncert.(result_ind.par) ≈ muncert.(result.par)
    @test Measurements.cov(result_ind.par) ≈ Diagonal(diag(result.gof.covmat))
    @test muncert(report_ind.f_fit(5.0)) ≈ sqrt([1, 5.0] .^ 2 ⋅ diag(result.gof.covmat))

    x = measurement.([1,2,3,4,5,6,7,8,9,10], ones(10))
    y = f_lin.(x,par_true...) .+ 0.5.*randn(10)
    @info "chisq fit with uncertainties on x and y"
    result, report       = chi2fit(1, x, y; pull_t = [(mean = par_true[1], std= 0.1),(mean = par_true[2],std= 0.1)], uncertainty=true) 
    @test isapprox(result.par[1], par_true[1], atol = 0.2*par_true[1])
    @test isapprox(result.par[2], par_true[2], atol = 0.2*par_true[2])
    # a pull term constrains a parameter like an additional data point
    @test result.gof.dof == length(x)

    x = [1,2]
    y = f_lin.(x,par_true...) .+ 0.5.*randn(2)
    @info "chisq fit with 2 fit parameter on 2 data points (test of gof)"
    result, report = @test_logs (:warn,) match_mode=:any chi2fit(1, x, y; uncertainty=true)
    @test iszero(report.gof.dof)
    @test isnan(report.gof.pvalue) # check that the p-value is NaN
    result, report = chi2fit(1, x, y; pull_t = [(mean = par_true[1], std= 0.1),(mean = par_true[2],std= 0.1)], uncertainty=true)
    @test report.gof.dof == 2
    @test !isnan(report.gof.pvalue)
end
