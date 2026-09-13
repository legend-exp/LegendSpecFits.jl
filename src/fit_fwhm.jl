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

function fit_fwhm(pol_order::Int, peaks::Vector{<:Unitful.Energy{<:Real}}, fwhm::Vector{<:Unitful.Energy{<:Real}}; e_type_cal::Symbol=:e_cal, e_expression::Union{Symbol, String}="e", uncertainty::Bool=true, scale_err_by_chi2red::Bool=false)
    @assert length(peaks) == length(fwhm) "Peaks and FWHM must have the same length"
    @assert pol_order in (1, 2) "Only 1, 2 order polynominal calibration is supported"

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
    # the ct bound in the prior is built from the pre-fit values; check concavity on the fitted ones (always true for pol_order 1)
    concave = pol_order == 1 || 4 * mvalue(par[1]) * mvalue(par[3]) < mvalue(par[2])^2
    concave || @warn "FWHM resolution curve is not concave: 4·enc·ct = $(4 * mvalue(par[1]) * mvalue(par[3])) ≥ fano² = $(mvalue(par[2])^2)"
    par_unit = par .* [e_unit^i for i in pol_order:-1:0]

    # built function in string
    func     = "sqrt($(join(["$(mvalue(par[i])) * ($(e_expression))^$(i-1)" for i in eachindex(par)], " + ")))$e_unit"
    func_err = "sqrt($(join(["($(par[i])) * ($(e_expression))^$(i-1)" for i in eachindex(par)], " + ")))$e_unit"
    func_cal = "sqrt($(join(["$(mvalue(par[i])) * $(e_type_cal)^$(i-1) * keV^$(3-i)" for i in eachindex(par)], " + ")))"
    func_cal_err = "sqrt($(join(["($(par[i])) * $(e_type_cal)^$(i-1) * keV^$(3-i)" for i in eachindex(par)], " + ")))"

    # get fwhm at Qbb 
    # Qbb from: https://www.researchgate.net/publication/253446083_Double-beta-decay_Q_values_of_74Se_and_76Ge
    # with the parameter covariance: enc and fano are ~80 % anti-correlated, propagating them as independent
    # Measurements (report_chi2.f_fit) overestimates the error by up to a factor 2
    qbb = _fwhm_at(result_chi2, 2039.061, 0.007; scale_err_by_chi2red) * e_unit
    result = merge(result_chi2, (par = par_unit , qbb = qbb, concave = concave, func = func, func_err = func_err, func_cal = func_cal, func_cal_err = func_cal_err, peaks = peaks, fwhm = fwhm))
    report = merge(report_chi2, (e_unit = e_unit, par = result.par, qbb = result.qbb, type = :fwhm))

    return result, report
end
fit_fwhm(peaks::Vector{<:Unitful.Energy{<:Real}}, fwhm::Vector{<:Unitful.Energy{<:Real}}; kwargs...) = fit_fwhm(1, peaks, fwhm; kwargs...)

# FWHM = √p(E) at E ± e_err from the fwhm² polynomial fit, error from the full parameter covariance. With
# scale_err_by_chi2red the covariance is inflated by χ²/ndf if > 1 (PDG scale factor: the points scatter more than
# their errors allow). No covariance (uncertainty=false or failed error estimate) gives NaN as the error
function _fwhm_at(result::NamedTuple, e::Real, e_err::Real; scale_err_by_chi2red::Bool=false)
    p = mvalue.(result.par); f = sqrt(evalpoly(e, p))
    hasproperty(result, :gof) || return measurement(f, NaN)
    scale = scale_err_by_chi2red && result.gof.dof > 0 ? max(1.0, result.gof.chi2min / result.gof.dof) : 1.0
    g = [e^(i-1) for i in eachindex(p)] ./ (2f)                                  # ∂f/∂pᵢ
    df_de = evalpoly(e, p[2:end] .* (1:length(p)-1)) / (2f)                       # ∂f/∂E
    var = scale * (g' * result.gof.covmat * g) + (df_de * e_err)^2
    measurement(f, var >= 0 ? sqrt(var) : NaN)
end


function _simple_linear_fit(x::Vector{<:Real}, y::Vector{<:Union{Real, Measurement{<:Real}}})
    # weighted least squares of y = β₁ + β₂ x with the y uncertainties as weights; the parameter covariance is the
    # inverse weighted normal matrix (exact for known weights). Without uncertainties: unit weights, scaled by the
    # residual variance if there are degrees of freedom left
    X = hcat(ones(length(x)), x)
    y_val, y_err = mvalue.(y), muncert.(y)
    w = all(>(0), y_err) ? y_err .^ -2 : ones(length(y))
    cov_matrix = inv(X' * (w .* X))
    β = cov_matrix * (X' * (w .* y_val))
    dof = length(y) - 2
    s2 = all(>(0), y_err) ? 1.0 : (dof > 0 ? sum(w .* (y_val .- X * β) .^ 2) / dof : 1.0)
    measurement.(β, sqrt.(s2 .* diag(cov_matrix)))
end

function _get_enc_fano_guess(peaks::Vector{<:Unitful.Energy{<:Real}}, fwhm::Vector{<:Unitful.Energy{<:Real}})
    # strip units, square y-values to fit a square root function; the FWHM uncertainties (propagated to fwhm²) weight the fit
    enc_guess, fano_guess = _simple_linear_fit(mvalue.(ustrip.(e_unit, peaks)), ustrip.(e_unit, fwhm).^2)

    # a negative intercept or slope means the FWHM values do not follow √(enc + fano·E); any fit from a substitute start
    # value is invented - the peak fits of this channel need a look (QC / override) instead
    (mvalue(enc_guess) > 0 && mvalue(fano_guess) > 0) || throw(ArgumentError("FWHM pre-fit gives enc = $(mvalue(enc_guess)) keV², fano = $(mvalue(fano_guess)) keV - the FWHM values are inconsistent with √(enc + fano·E); check the peak fits of this channel"))
    return enc_guess, fano_guess
end

function get_fit_fwhm_pseudo_prior(pol_order::Int, enc_guess::Measurement, fano_guess::Measurement)
    # create pseudo prior for fit parameters using initial fit pars for pseudo priors
    # fano_guess = 2.96e-2*0.11
    pprior_base = NamedTupleDist(
        # mode at the pre-fit value, 68 % quantile 3σ above it; the Weibull support already enforces enc > 0
        enc = weibull_from_mx(mvalue(enc_guess), mvalue(enc_guess) + 3 * muncert(enc_guess)),
        fano = weibull_from_mx(mvalue(fano_guess), 10*mvalue(fano_guess)),
        # √(enc + fano·E + ct·E²) is concave for all E iff 4·enc·ct < fano²: allow ct up to half that bound (from the pre-fit values)
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
