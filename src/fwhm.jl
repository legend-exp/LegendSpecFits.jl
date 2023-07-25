f_fwhm(x, p) = sqrt.(x .* p[2] .+ p[1])
"""
    fitFWHM
Fit the FWHM of the peaks to a quadratic function.
# Returns
    * `qbb`: the FWHM at§ 2039 keV
    * `err.qbb`: the uncertainty on `qbb`
    * `v`: the fit result parameters
"""
function fit_fwhm(peaks::Vector{T}, fwhm::Vector{T}) where T<:Real
    # get rid of the DEP and SEP peak
    noDEPSEP = (peaks .!= 2103.53 .&& peaks .!= 1592.53)
    fwhm_noDEPSEP  = fwhm[noDEPSEP]
    peaks_noDEPSEP = peaks[noDEPSEP]

    # fit FWHM fit function
    fwhm_fit_result = curve_fit(f_fwhm, peaks_noDEPSEP, fwhm_noDEPSEP, [1, 0.01], lower=[0.0, 0.0])

    # get FWHM at Qbb with error
    fwhm_qbb = f_fwhm(2039, fwhm_fit_result.param)
    fwhm_pars_rand = rand(MvNormal(fwhm_fit_result.param, estimate_covar(fwhm_fit_result)), 1000)
    # prevent negative parameter values in fit
    # TODO: check if this is the right way to do this
    for v in eachcol(fwhm_pars_rand)
        v[1] = max(v[1], 0)
        v[2] = max(v[2], 0)
    end
    fwhm_qbb_rand = f_fwhm.(2039, eachcol(fwhm_pars_rand))
    fwhm_qbb_err = std(fwhm_qbb_rand)

    result = (
        qbb = fwhm_qbb,
        err = (qbb = fwhm_qbb_err,), 
        v = fwhm_fit_result.param
    )
    report = (
        qbb = result.qbb,
        err = (qbb = result.err.qbb,),
        v = result.v,
        f_fit = x -> Base.Fix2(f_fwhm, result.v)(x)
    )
    return result, report
end
export fit_fwhm