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
    @assert pol_order >= 1 "The polynomial order must be greater than 0"
    
    # fit FWHM fit function
    e_unit = u"keV"
    _linear_intercept(x1::Float64, x2::Float64, y1::Float64, y2::Float64) = y1 - ((y2 - y1) / (x2 - x1)) * x1
    intercept_first_two_points = _linear_intercept(mvalue.(ustrip.(e_unit,sort(peaks)[1:2]))..., mvalue.(ustrip.(e_unit, fwhm[sortperm(peaks)[1:2]]))...)
    intercept_guess = if intercept_first_two_points > 0.1
        intercept_first_two_points
    else
        0.9*mvalue(ustrip(e_unit, fwhm[argmin(peaks)]))
    end
    @debug "Fit resolution curve with $(pol_order)-order polynominal function"
    p_start = append!([intercept_guess, 2.96e-3*0.11], fill(0.0, pol_order-1))
    @debug "Initial parameters: $p_start"
    pseudo_prior = get_fit_fwhm_pseudo_prior(pol_order, intercept_guess)
    @debug "Pseudo prior: $pseudo_prior"

    # fit FWHM fit function as a square root of a polynomial
    result_chi2, report_chi2 = chi2fit(x -> LegendSpecFits.heaviside(x)*sqrt(abs(x)), pol_order, ustrip.(e_unit, peaks), ustrip.(e_unit, fwhm); v_init=p_start, pseudo_prior=pseudo_prior, uncertainty=uncertainty)
    
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

function get_fit_fwhm_pseudo_prior(pol_order::Int, intercept_guess::Real; fano_term::Float64=2.96e-3*0.11)
    unshaped(if pol_order == 1
        NamedTupleDist(
            enc = weibull_from_mx(intercept_guess, 1.2*intercept_guess),
            fano = Normal(fano_term, 0.2*fano_term)
        )
    elseif pol_order == 2
        NamedTupleDist(
            enc = weibull_from_mx(intercept_guess, 1.2*intercept_guess),
            fano = Normal(fano_term, 0.2*fano_term),
            ct = weibull_from_mx((0.01*fano_term)^2, (0.05*fano_term)^2)
        )
    else
        throw(ArgumentError("Only 0, 1, 2 order polynominal calibration is supported"))
    end)
end