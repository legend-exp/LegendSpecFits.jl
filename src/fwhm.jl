f_fwhm(x, p) = sqrt.(x.*p[2] .+ p[1])
"""
    fitFWHM
Fit the FWHM of the peaks to a quadratic function.
Returns
    * `fwhm_qbb`: the FWHM of the 2039 keV line
    * `fwhm_qbb_err`: the uncertainty on `fwhm_qbb`
    * `fwhm_fit_result.param`: the fit result parameters
"""
function fitFWHM(fwhm_vals::Dict{Float64, Float64})
    fwhm_vals_noDEPSEP = filter(((k,v),) -> k != 1592.53 && k != 2103.53 && v > 0.0, fwhm_vals)
    fwhm_fit_result = curve_fit(f_fwhm, collect(keys(fwhm_vals_noDEPSEP)), collect(values(fwhm_vals_noDEPSEP)), [1, 0.01], lower=[0.0, 0.0])

    fwhm_qbb = f_fwhm(2039, fwhm_fit_result.param)
    fwhm_pars_rand = rand(MvNormal(fwhm_fit_result.param, estimate_covar(fwhm_fit_result)), 1000)
    fwhm_qbb_rand = f_fwhm.(2039, eachcol(fwhm_pars_rand))
    fwhm_qbb_err = std(fwhm_qbb_rand)

    return fwhm_fit_result.param, fwhm_qbb, fwhm_qbb_err
end
export fitFWHM