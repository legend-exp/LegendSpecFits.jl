
using LegendSpecFits
using Measurements
using Test

@testset "fit_chisq" begin
    par_true = [5, 2]
    f_lin(x,p1,p2)  = p1 +  p2 * x 
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

    x = measurement.([1,2,3,4,5,6,7,8,9,10], ones(10))
    y = f_lin.(x,par_true...) .+ 0.5.*randn(10)
    @info "chisq fit with uncertainties on x and y"
    result, report       = chi2fit(1, x, y; pull_t = [(mean = par_true[1], std= 0.1),(mean = par_true[2],std= 0.1)], uncertainty=true) 
    @test isapprox(result.par[1], par_true[1], atol = 0.2*par_true[1])
    @test isapprox(result.par[2], par_true[2], atol = 0.2*par_true[2])
end
