
"""
    chi2fit(f_fit::Function, x, y; pull_t, v_init, lower_bound, upper_bound, pseudo_prior, uncertainty, correlated)
    chi2fit(f_fit, x, y, yerr; kwargs...)
    chi2fit(f_fit, x, y, yerr, xerr; kwargs...)
    chi2fit(n_poly::Int, x, y; kwargs...)
    chi2fit(f_outer::Function, n_poly::Int, x, y; kwargs...)

Fits the model `f_fit` to the data `(x, y)` by minimizing a chi-squared function
with the Minuit minimizer (`migrad`).

`x` and `y` are vectors of numbers or `Measurement`s. The uncertainty of each
`x` is propagated through the model with forward-mode automatic differentiation
and added in quadrature to the uncertainty of `y`. If all uncertainties are
zero, all points have equal weight. Separate `yerr` (and `xerr`) vectors are
converted to `Measurement`s.

## Arguments
* `f_fit`: Model function `f(x, p1, p2, ...)`. The number of fit parameters is
  `first(methods(f_fit)).nargs - 2`, so `f_fit` needs a single method with
  explicit parameter arguments.
* `n_poly::Int`: Fits a polynomial of degree `n_poly` instead, with parameters ordered from the
  constant term upward (see [`PolCalFunc`](@ref)). With `f_outer`, the model is `f_outer(poly(x))`.

## Keywords
* `pull_t::Vector{<:NamedTuple}`: One entry per parameter. A `NamedTuple` `(mean, std)` adds a
  Gaussian pull term `((p - mean) / std)^2` to the chi-squared function; an empty `NamedTuple()` adds none.
  The default adds no pull terms.
* `v_init::Vector{<:Real}`: Initial parameter values. Default is one for every parameter.
* `lower_bound::Vector{<:Real}`, `upper_bound::Vector{<:Real}`: Parameter limits. Default is no limit.
* `pseudo_prior::Union{ContinuousMultivariateDistribution, Nothing} = nothing`: If given, the
  minimization runs in a transformed space where one unit is one prior standard deviation, which
  keeps parameters in the support of the prior.
* `uncertainty::Bool = true`: Calculate parameter uncertainties and goodness of fit from the Hessian of the chi-squared function.
* `correlated::Bool = true`: If `true`, the returned parameters carry their full covariance, so that
  uncertainties of derived quantities, including `report.f_fit`, account for the parameter correlations.
  If `false`, the parameters are independent `Measurement`s with the same marginal
  uncertainties, which overestimates the uncertainty of derived quantities when the parameters are correlated.

## Returns
A tuple `(result, report)`. `result` is a `NamedTuple` with the fields
* `par`: best-fit parameters as `Measurement`s. The uncertainty is `NaN` if `uncertainty` is `false`
  or the fit did not converge.
* `converged`: whether the minimizer converged. A warning is issued if not.
* `gof`: goodness of fit, only present if the uncertainties were calculated, as a `NamedTuple` with
  `converged`, `pvalue`, `chi2min`, `dof`, `covmat` and `residuals_norm`. The number of degrees
  of freedom counts each pull term as one data point; `pvalue` is `NaN` if it is zero.

`report` contains `par`, `f_fit` (the fitted model as a function of `x`), `x`, `y` and `gof`,
for plotting with `LegendMakie.lplot`.
""" 
function chi2fit(f_fit::Function, x::AbstractVector{<:Union{Real,Measurement{<:Real}}}, y::AbstractVector{<:Union{Real,Measurement{<:Real}}}; 
                    pull_t::Vector{<:NamedTuple}=fill(NamedTuple(), first(methods(f_fit)).nargs - 2), 
                    v_init::Vector{<:Real} = ones(length(pull_t)),
                    lower_bound::Vector{<:Real}=fill(-Inf, length(pull_t)),
                    upper_bound::Vector{<:Real}=fill(Inf, length(pull_t)),
                    pseudo_prior::Union{ContinuousMultivariateDistribution, Nothing}=nothing,
                    uncertainty::Bool=true,
                    correlated::Bool=true)
    @assert length(x) == length(y) "x and y must have the same length"
    @assert length(pull_t) == length(v_init) == length(lower_bound)  == length(upper_bound) "Length of pull_t, v_init, lower_bound and upper_bound does not match."

    # prepare pull terms
    f_pull(v::Number,pull_t::NamedTuple) = isempty(pull_t) ? zero(v) : (v .- pull_t.mean) .^2 ./ pull_t.std.^2  # pull term is zero if pull_t is zero
    f_pull(v::Vector,pull_t::Vector)     = sum(f_pull.(v,pull_t))
    pull_t_sum = Base.Fix2(f_pull, pull_t)

    @debug "Initial guess for fit parameters: $v_init"
    @debug "Pull terms: $pull_t"
    @debug "Lower bound: $lower_bound"
    @debug "Upper bound: $upper_bound"

    f_trafo = if isnothing(pseudo_prior)
        identity
    else
        BAT.DistributionTransform(Normal, pseudo_prior)
    end

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

    # minimize in the pseudo-prior space u, where one unit is one prior standard deviation
    f_inv = inverse(f_trafo)
    f_u = f_opt ∘ f_inv
    u_init = f_trafo(v_init)
    step = isnothing(pseudo_prior) ? max.(0.1 .* abs.(v_init), 1e-3) : ones(npar)
    limits = [isfinite(lb) || isfinite(ub) ? (isfinite(lb) ? lb : nothing, isfinite(ub) ? ub : nothing) : nothing for (lb, ub) in zip(lower_bound, upper_bound)]
    # strategy 2 seeds with the Hessian at the start point; a first step scaled by `errors` alone can
    # overshoot into the flat tails of the prior transform, from which the fit cannot recover
    m = Minuit(f_u, u_init; errors = step, limits, up = 1.0, strategy = 2, grad = u -> ForwardDiff.gradient(f_u, u), check_gradient = false)
    migrad!(m; maxfcn = 5000)
    converged = m.valid

    # get best fit results
    u_chi2 = collect(m.values)
    v_chi2 = f_inv(u_chi2)
    
    if !converged @warn "Fit did not converge" end
    par = measurement.(v_chi2,Ref(NaN)) # if ucnertainty is not calculated, return NaN
    result = (par = par, converged = converged) # fit function with optimized parameters
    report = (par = result.par, f_fit = x -> f_fit(x, v_chi2...), x = x, y = y, gof = NamedTuple())
    
    if uncertainty && converged
        try
            # cov = 2 H⁻¹ for a χ², in the fit parameters: the prior transform is too nonlinear in its
            # tails to propagate a covariance from the pseudo-prior space
            covmat = 2 .* inv(ForwardDiff.hessian(f_opt, v_chi2))
            # correlated parameters share unit normals, so derived quantities propagate the full covariance
            par = if correlated
                v_chi2 .+ cholesky(Hermitian(covmat)).L * [measurement(0, 1) for _ in v_chi2]
            else
                measurement.(v_chi2, sqrt.(diag(covmat)))
            end
            
            @debug "Best Fit parameters: $par"
            # gof 
            chi2min = m.fval
            # each pull term constrains one parameter like an additional data point
            dof = length(x) - length(v_chi2) + count(!isempty, pull_t)
            pvalue = if iszero(dof)
                @warn "The number of fit parameters is equivalent to number of data points --> dof = 0 ; p-value not meaningful, set to NaN"
                NaN
            else 
                ccdf(Chisq(dof), chi2min)
            end 

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