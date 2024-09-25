
"""
    fit_chisq(x::AbstractVector{<:Real},y::AbstractVector{<:Real},yerr::AbstractVector{<:Real}, f_fit::Function;pull_t::Vector{<:NamedTuple} = fill(NamedTuple(), first(methods(f_fit)).nargs - 2), v_init::Vector = [])
Least square fit with chi2 minimization
# Input:
- x : x-values
- y : y-values
- yerr : 1 sigma uncertainty on y
- f_fit : fit/model function. e.g. for a linear function: f_lin(x,p1,p2)  = p1 .* x .+ p2   
The numer of fit parameter is determined with `first(methods(f_fit)).nargs - 2`. That's why it's important that f_fit has the synthax f(x,arg1,arg2,arg3,...)
pull_t : pull term, a vector of NamedTuple with fields `mean` and `std`. A Gaussian pull term is added to the chi2 function to account for systematic uncertainties. If left blank, no pull term is used.
v_init : initial value for fit parameter optimization. If left blank, the initial value is set to 1 or guessed roughly for all fit parameters
# Return:
- result : NamedTuple with the optimized fit parameter and the fit function
- report: 

""" 
function chi2fit(f_fit::Function, x::AbstractVector{<:Union{Real,Measurement{<:Real}}}, y::AbstractVector{<:Union{Real,Measurement{<:Real}}}; pull_t::Vector{<:NamedTuple}=fill(NamedTuple(), first(methods(f_fit)).nargs - 2), v_init::Vector = [], uncertainty::Bool=true )
    # prepare pull terms
    f_pull(v::Number,pull_t::NamedTuple) = isempty(pull_t) ? zero(v) : (v .- pull_t.mean) .^2 ./ pull_t.std.^2  # pull term is zero if pull_t is zero
    f_pull(v::Vector,pull_t::Vector)     = sum(f_pull.(v,pull_t))
    pull_t_sum = Base.Fix2(f_pull, pull_t)

    # get rid of measurements 
    X_val = mvalue.(x) 
    Y_val = mvalue.(y) 
    X_err= muncert.(x) 
    Y_err = muncert.(y)
    if all(X_err .== 0) && all(Y_err .== 0)
        Y_err = ones(length(Y_val))
    end 
    # muncert_res(x) = x==0 ? 1.0 : muncert(x) # if uncertainties are zero use 1.0 instead (equal weights)

    # chi2-function 
    function chi2xy(f_fit, x_val, x_err, y_val, y_err, pars)
        dual_x = ForwardDiff.Dual{UncertTag}(x_val, x_err)
        dual_y = f_fit(dual_x, pars...)
        y_pred_val = ForwardDiff.value(UncertTag, dual_y)
        y_pred_err = only(ForwardDiff.partials(UncertTag, dual_y))
        chi2 = (y_val - y_pred_val)^2 / (y_err^2 + y_pred_err^2)
        return chi2
    end

    # function that is minimized -> chi-squared with pull terms 
    f_opt = let X_val = X_val, Y_val = Y_val, X_err = X_err, Y_err = Y_err, f_pull = pull_t_sum, f_fit = f_fit
        v -> sum(chi2xy.(Ref(f_fit), X_val, X_err, Y_val, Y_err, Ref(v))) + f_pull(v)
    end

    # init guess for fit parameter: this could be improved. 
    npar = length(pull_t) # number of fit parameter (including nuisance parameters)
    if isempty(v_init) 
        v_init = ones(npar)
    end

    # minimization and error estimation
    opt_r = optimize(f_opt, v_init, LBFGS(linesearch = MoreThuente()), Optim.Options(iterations = 3000, allow_f_increases=false, show_trace=contains(get(ENV, "JULIA_DEBUG", ""), "LegendSpecFits"), callback=advanced_time_and_memory_control()), autodiff=:forward)
    if !Optim.converged(opt_r)
        opt_r = optimize(f_opt, v_init, NelderMead(), Optim.Options(callback=advanced_time_and_memory_control(time_limit=20), iterations = 3000))
    end
    
    converged = Optim.converged(opt_r)
    @debug opt_r

    # get best fit results
    v_chi2  = Optim.minimizer(opt_r)
    
    if !converged @warn "Fit did not converge" end
    par = measurement.(v_chi2,Ref(NaN)) # if ucnertainty is not calculated, return NaN
    result = (par = par, converged = converged) # fit function with optimized parameters
    report = (par = result.par, f_fit = x -> f_fit(x, v_chi2...), x = x, y = y, gof = NamedTuple())
    
    if uncertainty && converged
        try
            covmat = inv(ForwardDiff.hessian(f_opt, v_chi2))
            v_chi2_err = sqrt.(diag(abs.(covmat)))#mvalue.(sqrt.(diag(abs.(covmat))))
            par = measurement.(v_chi2, v_chi2_err)
            
            @debug "Best Fit parameters: $par"
            # gof 
            chi2min = minimum(opt_r)
            dof = length(x) - length(v_chi2)
            pvalue = ccdf(Chisq(dof), chi2min)
            function get_y_pred_err(f_fit, x_val, x_err, pars)  # get final uncertainties for normalized residuals
                dual_x = ForwardDiff.Dual{UncertTag}(x_val, x_err)
                dual_y = f_fit(dual_x, pars...)
                y_pred_err = only(ForwardDiff.partials(UncertTag, dual_y))
                return y_pred_err
            end
            Y_pred_err = get_y_pred_err.(Ref(f_fit), X_val, X_err, Ref(v_chi2))
            Y_err_tot = sqrt.(Y_err.^2 .+ Y_pred_err.^2)
            Y_err_tot = ifelse(all(Y_err_tot .== 0), ones(length(Y_val)), Y_err_tot)
            residuals_norm = (Y_val - f_fit(X_val, v_chi2...)) ./ Y_err_tot

            @debug "Covariance matrix: $covmat"
            @debug "P-value: $(pvalue)"
            @debug "Chi2/Ndof: $(chi2min)/$(dof)"

            result = (par = par, gof = (converged = converged, pvalue = pvalue, chi2min = chi2min, dof = dof, covmat = covmat, residuals_norm = residuals_norm))
            report = merge(report, (par = par, gof = result.gof, f_fit = x -> f_fit(x, par...)))
        catch e
            @warn "Error in calculating uncertainties: $e"
        end
    else
        @debug "Best Fit parameters: $par"
    end
    return result, report 
end

function chi2fit(n_poly::Int, x, y; pull_t::Vector{<:NamedTuple}=fill(NamedTuple(), n_poly+1), kwargs...) 
    @assert length(pull_t) == n_poly+1 "Length of pull_t does not match the order of the polynomial"
    chi2fit((x, a...) -> PolCalFunc(a...).(x), x, y; pull_t = pull_t, kwargs...)
end

function chi2fit(f_outer::Function, n_poly::Int, x, y; pull_t::Vector{<:NamedTuple}=fill(NamedTuple(), n_poly + 1), kwargs...) 
    @assert length(pull_t) == n_poly + 1 "Length of pull_t does not match the order of the polynomial"
    chi2fit((x, a...) -> f_outer.(PolCalFunc(a...).(x)), x, y; pull_t = pull_t, kwargs...)
end

chi2fit(f_fit, x::AbstractVector, y::AbstractVector{<:Real}, yerr::AbstractVector{<:Real}; kwargs...) = chi2fit(f_fit, x, measurement.(y,yerr); kwargs...)
chi2fit(f_fit, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, yerr::AbstractVector{<:Real}, xerr::AbstractVector{<:Real}; kwargs...) = chi2fit(f_fit, measurement.(x, xerr), measurement.(y, yerr); kwargs...)

export chi2fit



#f5(x,p1,p2,p3) =  PolCalFun(p1,p2,p3)