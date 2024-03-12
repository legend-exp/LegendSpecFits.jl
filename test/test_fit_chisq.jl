
using LegendSpecFits 
using Test

@testset "fit_chisq" begin
    #  linear fit : simple case with and without pull term 
    f_lin(x,p1,p2)  = p1 .* x .+ p2

    x           = [1.0,2.0,3.0,4.0,5.0]
    y           = f_lin(x,5.0,10.0)
    yerr        = sqrt.(y)
    @info "linear chisq fit"
    result = fit_chisq(x,y,yerr,f_lin) 
    @info "linear chisq fit with pull term"
    result = fit_chisq(x,y,yerr,f_lin;pull_t = [NamedTuple(), (mean = 10.0, std = 0.1)])

  
    #  quadratic fit : simple case with and without pull term 
    f_quad(x,p1,p2,p3)  = p1 .* x.^2 .+ p2 .* x .+ p3
    x           = [1.0,2.0,3.0,4.0,5.0]
    y           = f_quad(x,2.0,5.0,10.0)
    yerr        = sqrt.(y)
    @info "quadratic chisq fit"
    result = fit_chisq(x,y,yerr,f_quad)
    @info "quadratic chisq fit with pull term"
    result = fit_chisq(x,y,yerr,f_quad;pull_t = [(mean = 2.0, std = 0.1),NamedTuple() ,NamedTuple(), ])
end