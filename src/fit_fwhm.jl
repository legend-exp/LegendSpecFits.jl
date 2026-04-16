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

function fit_fwhm(pol_order::Int, peaks::Vector{<:Unitful.Energy{<:Real}}, fwhm::Vector{<:Unitful.Energy{<:Real}}; e_type_cal::Symbol=:e_cal, e_expression::Union{Symbol, String}="e", uncertainty::Bool=true, fano_term::Unitful.Energy{<:Real}=2.96e-2*0.11u"keV")
    @assert length(peaks) == length(fwhm) "Peaks and FWHM must have the same length"
    @assert pol_order != 1 || pol_order != 2 "Only 1, 2 order polynominal calibration is supported"
    

    # pre-fit linear model to get a better initial guess for the intercept
    X = hcat(ones(length(peaks)), mvalue.(ustrip.(e_unit, peaks))) # Creates a matrix where column 1 is all 1s, column 2 is x
    β = X \ mvalue.(ustrip.(e_unit, fwhm)).^2
    intercept = β[1]

    enc_guess = if intercept > 0.1
        β[1]
    else
        mvalue(ustrip(e_unit, fwhm[argmin(peaks)]))
    end
    fano_guess = ustrip(e_unit, fano_term)
    # fit FWHM fit function
    @debug "Fit resolution curve with $(pol_order)-order polynominal function"
    p_start = append!([enc_guess, fano_guess], fill(0.0, pol_order-1))
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

function get_fit_fwhm_pseudo_prior(pol_order::Int, enc_guess::Real, fano_term::Real)
    unshaped(if pol_order == 1
        NamedTupleDist(
            enc = weibull_from_mx(enc_guess, 1.5*enc_guess),
            fano = weibull_from_mx(fano_term, 1.33*fano_term)
        )
    elseif pol_order == 2
        NamedTupleDist(
            enc = weibull_from_mx(enc_guess, 1.2*enc_guess),
            fano = weibull_from_mx(fano_term, 1.2*fano_term),
            # for a √ of a quadratic function, the function is concave if fano^2/(4*enc) < ct, and convex if fano^2/(4*enc) >  ct, so let's take 0.5 as a conservative guess for the upper limit of ct, and 0 as the lower limit
            ct = Uniform(0, fano_term^2/(4*enc_guess)/2)
        )
    else
        throw(ArgumentError("Only 1, 2 order polynominal calibration is supported"))
    end)
end