
"""
fit_chisq(x::AbstractVector{<:Real},y::AbstractVector{<:Real},yerr::AbstractVector{<:Real}, f_fit::Function;pull_t::Vector{<:NamedTuple} = fill(NamedTuple(), first(methods(f_fit)).nargs - 2), v_init::Vector = [])
- least square fit with chi2 minimization
# input:
- x : x-values
- y : y-values
- yerr : 1 sigma uncertainty on y
- f_fit : fit/model function. e.g. for a linear function: f_lin(x,p1,p2)  = p1 .* x .+ p2   
The numer of fit parameter is determined with `first(methods(f_fit)).nargs - 2`. That's why it's important that f_fit has the synthax f(x,arg1,arg2,arg3,...)
pull_t : pull term, a vector of NamedTuple with fields `mean` and `std`. A Gaussian pull term is added to the chi2 function to account for systematic uncertainties. If left blank, no pull term is used.
v_init : initial value for fit parameter optimization. If left blank, the initial value is set to 1 or guessed roughly for all fit parameters
""" 
function fit_chisq(x::AbstractVector{<:Real},y::AbstractVector{<:Real},yerr::AbstractVector{<:Real}, f_fit::Function;pull_t::Vector{<:NamedTuple} = fill(NamedTuple(), first(methods(f_fit)).nargs - 2), v_init::Vector = [])
    # prepare pull terms
    f_pull(v::Number,pull_t::NamedTuple) =  isempty(pull_t) ? zero(v) : (v .- pull_t.mean) .^2 ./ pull_t.std.^2  # pull term is zero if pull_t is zero
    f_pull(v::Vector,pull_t::Vector)     = sum(f_pull.(v,pull_t))
    pull_t_sum = Base.Fix2(f_pull, pull_t)

    # chi2 function
    f_chi2 = let x = x, y = y, yerr = yerr, f_fit = f_fit, f_pull = pull_t_sum
        v -> sum((y - f_fit(x, v...) ).^2 ./ yerr.^2) + f_pull(v)
    end

   # init guess for fit parameter: this could be improved. 
    npar = first(methods(f_fit)).nargs - 2 # number of fit parameter (including nuisance parameters)
    if isempty(v_init) 
        if npar==2 
            v_init = [y[1]/x[1],1.0] # linear fit : guess slope 
        else
             v_init = ones(npar)
        end 
    end 
    
     # minimization and error estimation
    opt_r = optimize(f_chi2, v_init)
    v_chi2 =  Optim.minimizer(opt_r)
    covmat = inv(ForwardDiff.hessian(f_chi2, v_chi2))
    v_chi2_err = sqrt.(diag(abs.(covmat)))
  
    # gof 
    chi2min = minimum(opt_r)
    dof = length(x)-length(v_chi2)
    pvalue = ccdf(Chisq(dof),chi2min)  

    # result 
    result = (par = v_chi2,
             err = v_chi2_err,
             gof = (pvalue = pvalue, chi2min = chi2min, dof = dof, covmat = covmat))

    return result 
end

export fit_chisq