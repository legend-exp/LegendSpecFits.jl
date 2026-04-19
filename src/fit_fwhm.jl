"""
    fitFWHM(fit_fwhm(peaks::Vector{T}, fwhm::Vector{T}) where T<:Real
Fit the FWHM of the peaks to a quadratic function.
# Returns
    * `qbb`: the FWHM at 2039 keV
    * `err`: the uncertainties of the fit parameters
    * `v`: the fit result parameters
    * `f_fit`: the fitted function
"""
function fit_fwhm end
export fit_fwhm

function fit_fwhm(pol_order::Int, peaks::Vector{<:Unitful.Energy{<:Real}}, fwhm::Vector{<:Unitful.Energy{<:Real}}; e_type_cal::Symbol=:e_cal, e_expression::Union{Symbol, String}="e", uncertainty::Bool=true)
    @assert length(peaks) == length(fwhm) "Peaks and FWHM must have the same length"
    @assert pol_order != 1 || pol_order != 2 "Only 1, 2 order polynominal calibration is supported"
    
    
    @debug "Fit resolution curve with $(pol_order)-order polynominal function"
    # get initial guess for ENC and Fano factor using pre-fit
    enc_guess, fano_guess = _get_enc_fano_guess(peaks, fwhm)
    @debug "Initial guess for ENC: $enc_guess, Fano factor: $fano_guess"
    p_start = append!(mvalue.([enc_guess, fano_guess]), fill(0.0, pol_order-1))
    @debug "Initial parameters: $p_start"
    pseudo_prior = get_fit_fwhm_pseudo_prior(pol_order, enc_guess, fano_guess)
    @debug "Pseudo prior: $pseudo_prior"

    # fit FWHM fit function as a square root of a polynomial
    # result_chi2, report_chi2 = chi2fit(x -> LegendSpecFits.heaviside(x)*sqrt(abs(x)), pol_order, ustrip.(e_unit, peaks), ustrip.(e_unit, fwhm); v_init=p_start, pseudo_prior=pseudo_prior, uncertainty=uncertainty)
    result_chi2, report_chi2_linear = chi2fit(pol_order, ustrip.(e_unit, peaks), ustrip.(e_unit, fwhm).^2; v_init=p_start, pseudo_prior=pseudo_prior, uncertainty=uncertainty)
    report_chi2 = NamedTuple{keys(report_chi2_linear)}(merge(report_chi2_linear, (y = ustrip.(e_unit, fwhm), f_fit = x -> sqrt(report_chi2_linear.f_fit(x)))))

    # get pars and apply unit
    par =  result_chi2.par
    par_unit = par .* [e_unit^i for i in pol_order:-1:0]

    # built function in string
    func     = "sqrt($(join(["$(mvalue(par[i])) * ($(e_expression))^$(i-1)" for i in eachindex(par)], " + ")))$e_unit"
    func_err = "sqrt($(join(["($(par[i])) * ($(e_expression))^$(i-1)" for i in eachindex(par)], " + ")))$e_unit"
    func_cal = "sqrt($(join(["$(mvalue(par[i])) * $(e_type_cal)^$(i-1) * keV^$(3-i)" for i in eachindex(par)], " + ")))"
    func_cal_err = "sqrt($(join(["($(par[i])) * $(e_type_cal)^$(i-1) * keV^$(3-i)" for i in eachindex(par)], " + ")))"

    # get fwhm at Qbb 
    # Qbb from: https://www.researchgate.net/publication/253446083_Double-beta-decay_Q_values_of_74Se_and_76Ge
    qbb = report_chi2.f_fit(measurement(2039.061, 0.007)) * e_unit
    result = merge(result_chi2, (par = par_unit , qbb = qbb, func = func, func_err = func_err, func_cal = func_cal, func_cal_err = func_cal_err, peaks = peaks, fwhm = fwhm))
    report = merge(report_chi2, (e_unit = e_unit, par = result.par, qbb = result.qbb, type = :fwhm))

    return result, report
end
fit_fwhm(peaks::Vector{<:Unitful.Energy{<:Real}}, fwhm::Vector{<:Unitful.Energy{<:Real}}; kwargs...) = fit_fwhm(1, peaks, fwhm; kwargs...)


function _simple_linear_fit(x::Vector{<:Real}, y::Vector{<:Real})
    # Creates a matrix where column 1 is all 1s, column 2 is x
    X = hcat(ones(length(x)), x)

    # solve linear regression using the normal equation: β = (X'X)^(-1) X'y
    β = X \ y

    # n = number of observations, p = number of parameters
    n, p = size(X) 
    # dof = n - p Degrees of freedom
    dof = n - p

    # Calculate residuals
    y_pred = X * β
    residuals = y .- y_pred

    # Calculate residual variance (Mean Squared Error)
    sigma_sq = sum(residuals.^2) / dof

    # Calculate the Variance-Covariance matrix
    # We need the inverse of (X' * X). We can do this by solving (X' * X) \\ I
    I_mat = [i == j ? 1.0 : 0.0 for i in 1:p, j in 1:p]

    cov_matrix = sigma_sq * ((X' * X) \ I_mat)

    # The standard errors are the square roots of the diagonal elements
    se = sqrt.(abs.([cov_matrix[i, i] for i in 1:p]))
    measurement.(β, se)
end

function _get_enc_fano_guess(peaks::Vector{<:Unitful.Energy{<:Real}}, fwhm::Vector{<:Unitful.Energy{<:Real}})
    # strip units, only use central values, square y-values to fit a square root function
    enc_guess, fano_guess = _simple_linear_fit(mvalue.(ustrip.(e_unit, peaks)), mvalue.(ustrip.(e_unit, fwhm).^2))

    # sanity checks to make sure initial guesses are strictly positive, otherwise set to small positive values to avoid issues with the fit function
    enc_guess, fano_guess = if enc_guess < 0 # if the ENC is negative, set it to a small positive value (e.g. 0.01) to avoid issues with the fit function
        @warn "ENC is negative in initial guess, trying different intial guess strategy"
        enc_guess, fano_guess_non_squared = _simple_linear_fit(mvalue.(ustrip.(e_unit, peaks)), mvalue.(ustrip.(e_unit, fwhm)))
        if enc_guess < 0.0 # if the ENC is still negative, set it to first FWHM value as very rough estimate
            @warn "ENC is still negative in initial guess lowest FWHM"
            measurement(mvalue(ustrip(e_unit, fwhm[sortperm(peaks)])), mvalue(ustrip(e_unit, fwhm[sortperm(peaks)]*0.8))), fano_guess_non_squared
        else
            enc_guess, fano_guess_non_squared
        end
    else
        enc_guess, fano_guess
    end
    
    # if fano factor is still negative, set it to literature value for germanium (e.g. 0.11) to avoid issues with the fit function
    if fano_guess < 0 # if the fano factor is negative, set it to a small positive value (e.g. 0.01) to avoid issues with the fit function
        @warn "Fano factor is negative in initial guess, setting it to 2.96e-2*0.11"
        enc_guess, measurement(2.96e-2*0.11, 0.8*2.96e-2*0.11)
    else
        enc_guess, fano_guess
    end
end

function get_fit_fwhm_pseudo_prior(pol_order::Int, enc_guess::Measurement, fano_guess::Measurement)
    # create pseudo prior for fit parameters using initial fit pars for pseudo priors
    # fano_guess = 2.96e-2*0.11
    pprior_base = NamedTupleDist(
        enc = truncated(weibull_from_mx(mvalue(enc_guess), mvalue(enc_guess) + ifelse(muncert(enc_guess) > 0.05, muncert(enc_guess), 1.2*mvalue(enc_guess))).untruncated, ifelse(mvalue(enc_guess) < 0.3, 0.3*mvalue(enc_guess), 0.3), Inf),
        fano = weibull_from_mx(mvalue(fano_guess), 10*mvalue(fano_guess)),
        # for a √ of a quadratic function, the function is concave if fano^2/(4*enc) < ct, and convex if fano^2/(4*enc) >  ct, so let's take 0.5 as a conservative guess for the upper limit of ct, and 0 as the lower limit
        ct = Uniform(0, mvalue(fano_guess^2/(4*enc_guess)/2))
    )

    # extract prior base
    (; enc, fano, ct) = pprior_base

    unshaped(if pol_order == 1
        NamedTupleDist(; enc, fano)
    elseif pol_order == 2
        NamedTupleDist(; enc, fano, ct)
    else
        throw(ArgumentError("Only 1, 2 order polynominal calibration is supported"))
    end)
end