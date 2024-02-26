# f_fwhm(x, p) = sqrt.((x .* x .* p[3] .+ x .* p[2] .+ p[1]) .* heaviside.(x .* x .* p[3] .+ x .* p[2] .+ p[1]))
f_fwhm(x::T, p::AbstractArray{<:T}) where T<:Unitful.RealOrRealQuantity = sqrt((x * x * p[3] + x * p[2] + p[1]) * heaviside(x^2 * p[3] + x * p[2] + p[1]))
f_fwhm(x::Array{<:T}, p::AbstractArray{<:T}) where T<:Unitful.RealOrRealQuantity = Base.Fix2(f_fwhm, p).(x)

"""
    fitFWHM(fit_fwhm(peaks::Vector{T}, fwhm::Vector{T}) where T<:Real
Fit the FWHM of the peaks to a quadratic function.
# Returns
    * `qbb`: the FWHM at 2039 keV
    * `err`: the uncertainties of the fit parameters
    * `v`: the fit result parameters
    * `f_fit`: the fitted function
"""
function fit_fwhm(peaks::Vector{<:Unitful.Energy{<:Real}}, fwhm::Vector{<:Unitful.Energy{<:Real}})
    # fit FWHM fit function
    e_unit = u"keV"
    p_start = [1*e_unit^2, 0.001*e_unit, 0.0*e_unit]
    lower_bound = [0.0*e_unit^2, 0.0*e_unit, 0.0*e_unit]
    fwhm_fit_result = curve_fit(f_fwhm, ustrip.(e_unit, peaks), ustrip.(e_unit, fwhm), ustrip.(p_start), lower=ustrip.(lower_bound))

    # get FWHM at Qbb with error
    err = standard_errors(fwhm_fit_result)
    enc, fano, ct = fwhm_fit_result.param[1]*e_unit^2, fwhm_fit_result.param[2]*e_unit, fwhm_fit_result.param[3]
    fwhm_qbb = f_fwhm(2039u"keV", [enc, fano, ct])
    fwhm_pars_rand = rand(MvNormal(fwhm_fit_result.param, estimate_covar(fwhm_fit_result)), 10000)
    # prevent negative parameter values in fit
    # TODO: check if this is the right way to do this
    fwhm_pars_rand_cleaned = fwhm_pars_rand[:, findall(x -> all(x .> 0), eachcol(fwhm_pars_rand))]
    fwhm_pars_rand_cleaned = [[p1, p2, p3] for (p1, p2, p3) in zip(fwhm_pars_rand_cleaned[1,:] .*e_unit^2, fwhm_pars_rand_cleaned[2,:] .* e_unit, fwhm_pars_rand_cleaned[3,:])]
    # get all FWHM at Qbb with error
    fwhm_qbb_rand = f_fwhm.(2039u"keV", fwhm_pars_rand_cleaned)
    fwhm_qbb_err = std(fwhm_qbb_rand)

    result = (
        qbb = measurement(fwhm_qbb, fwhm_qbb_err),
        enc = measurement(fwhm_fit_result.param[1]*e_unit^2, err[1]*e_unit^2), 
        fano = measurement(fwhm_fit_result.param[2]*e_unit, err[2]*e_unit), 
        ct = measurement(fwhm_fit_result.param[3], err[3])
    )
    report = (
        qbb = result.qbb,
        v = [enc, fano, ct],
        f_fit = x -> Base.Fix2(f_fwhm, [enc, fano, ct])(x)
    )
    return result, report
end
export fit_fwhm