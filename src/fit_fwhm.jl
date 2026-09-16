"""
    fit_fwhm(pol_order::Int, peaks::Vector{<:Unitful.Energy}, fwhm::Vector{<:Unitful.Energy}; e_type_cal, e_expression, uncertainty, correlated, pull_fano)
Fit the resolution curve fwhm(E) = √(enc + fano·E [+ ct·E²]) to the FWHM of the peaks, with `pol_order` the
degree of the polynomial under the square root. With `pull_fano = true`, the quadratic fit ties the
linear term to Fano statistics of germanium (F = 0.112 ± 0.015) by a pull term, so that the E² term takes
up the other contributions; by default, and in the linear fit, the linear term is free.
# Returns
    * `par`: `enc`, the width √enc in keV that the curve approaches at zero energy; `fano`, the Fano
      factor extracted from the linear term with the pair creation energy of germanium; and, for the
      quadratic fit, the dimensionless coefficient `ct` of E²
    * `qbb`: the FWHM at 2039 keV
    * `func`, `func_cal`: the resolution curve as LEGEND Julia expressions of `e_expression` and `e_type_cal`
    * `func_err`, `func_cal_err`: the same expressions with the uncertainty of the curve written out as
      `value ± error`, which includes the correlations between the fit parameters (see `chi2fit`).
      Evaluating them at an energy that is itself a `Measurement` nests the two uncertainties.
    * `f_fit` (report): the fitted function
"""
function fit_fwhm end
export fit_fwhm

# Fano statistics of germanium at 77 K, fwhm² = fwhm_prefactor² · F · ε · E: F = 0.112 (Croft and Bond,
# Appl. Radiat. Isot. 42 (1991) 1009), its uncertainty spans published values from 0.106 to 0.129
const fwhm_prefactor = 2 * sqrt(2 * log(2))
const fano_factor_ge = measurement(0.112, 0.015)
const pair_creation_energy_ge = 2.96u"eV"
const fwhm_fano_term_ge = fwhm_prefactor^2 * fano_factor_ge * pair_creation_energy_ge

# energies enter the fit in MeV with fwhm² in keV², which keeps the polynomial coefficients of comparable
# size and the Hessian of the χ² invertible
const e_fit_fwhm_unit = u"MeV"

function fit_fwhm(pol_order::Int, peaks::Vector{<:Unitful.Energy{<:Real}}, fwhm::Vector{<:Unitful.Energy{<:Real}}; e_type_cal::Symbol=:e_cal, e_expression::Union{Symbol, String}="e", uncertainty::Bool=true, correlated::Bool=true, pull_fano::Bool=false)
    @assert length(peaks) == length(fwhm) "Peaks and FWHM must have the same length"
    @assert pol_order == 1 || pol_order == 2 "Only 1, 2 order polynominal calibration is supported"

    @debug "Fit resolution curve with $(pol_order)-order polynominal function"
    # initial guess for the polynomial coefficients (ENC, Fano term, quadratic term) from a pre-fit
    par_guess = _get_enc_fano_guess(peaks, fwhm, pol_order)
    @debug "Initial guess for fit parameters: $par_guess"
    # the linear term is pulled to Fano statistics only when an E² term takes up the other contributions
    pull_t = NamedTuple[NamedTuple() for _ in 1:pol_order+1]
    if pol_order == 2 && pull_fano
        fano_unit = e_unit^2 / e_fit_fwhm_unit
        pull_t[2] = (mean = ustrip(fano_unit, mvalue(fwhm_fano_term_ge)), std = ustrip(fano_unit, muncert(fwhm_fano_term_ge)))
    end
    @debug "Pull terms: $pull_t"
    p_start = mvalue.(par_guess)
    pseudo_prior = get_fit_fwhm_pseudo_prior(pol_order, par_guess)
    @debug "Pseudo prior: $pseudo_prior"

    # fit fwhm² with a polynomial; the report exposes the fwhm curve as its square root, in keV
    result_chi2, report_chi2_linear = chi2fit(pol_order, ustrip.(e_fit_fwhm_unit, peaks), ustrip.(e_unit, fwhm).^2; v_init=p_start, pseudo_prior=pseudo_prior, pull_t=pull_t, uncertainty, correlated)
    report_chi2 = NamedTuple{keys(report_chi2_linear)}(merge(report_chi2_linear,
        (x = ustrip.(e_unit, peaks), y = ustrip.(e_unit, fwhm), f_fit = x -> sqrt(report_chi2_linear.f_fit(ustrip(e_fit_fwhm_unit, x * e_unit))))))

    # the coefficient of E^(i-1) is fitted in keV²/MeV^(i-1) and reported in keV^(3-i)
    par = [ustrip(e_unit^(3-i), p * e_unit^2 / e_fit_fwhm_unit^(i-1)) for (i, p) in enumerate(result_chi2.par)]

    # √(enc + fano·E + ct·E²) is concave for all E iff 4·enc·ct ≤ fano²
    if pol_order == 2
        enc, fano, ct = mvalue.(par)
        4 * enc * ct > fano^2 && @warn "Resolution curve is convex: ct = $ct exceeds fano^2/(4*enc) = $(fano^2 / (4*enc))"
    end

    # √enc is the FWHM at zero energy; the Fano factor follows from the linear term
    par_fit = (enc = sqrt(par[1]) * e_unit, fano = uconvert(NoUnits, par[2] * e_unit / (fwhm_prefactor^2 * pair_creation_energy_ge)))
    pol_order == 2 && (par_fit = merge(par_fit, (ct = par[3],)))

    # fwhm = √p(E) with uncertainty √d(E) / (2√p(E)), d(E) = Σᵢⱼ Cᵢⱼ E^(i+j-2) from the covariance C;
    # written out because separate `±` literals cannot carry the parameter correlations
    C = Measurements.cov(par)
    d = [sum(C[i, j] for i in eachindex(par), j in eachindex(par) if i + j - 1 == k) for k in 1:2length(par)-1]
    p_str(e) = join(["$(mvalue(par[i])) * ($e)^$(i-1)" for i in eachindex(par)], " + ")
    d_str(e) = join(["$(d[k]) * ($e)^$(k-1)" for k in eachindex(d)], " + ")
    p_str_cal = join(["$(mvalue(par[i])) * $(e_type_cal)^$(i-1) * keV^$(3-i)" for i in eachindex(par)], " + ")
    d_str_cal = join(["$(d[k]) * $(e_type_cal)^$(k-1) * keV^$(5-k)" for k in eachindex(d)], " + ")
    func     = "sqrt($(p_str(e_expression)))$e_unit"
    func_err = "(sqrt($(p_str(e_expression))) ± (sqrt($(d_str(e_expression))) / (2 * sqrt($(p_str(e_expression))))))$e_unit"
    func_cal = "sqrt($p_str_cal)"
    func_cal_err = "sqrt($p_str_cal) ± (sqrt($d_str_cal) / (2 * sqrt($p_str_cal)))"

    # get fwhm at Qbb 
    # Qbb from: https://www.researchgate.net/publication/253446083_Double-beta-decay_Q_values_of_74Se_and_76Ge
    qbb = report_chi2.f_fit(measurement(2039.061, 0.007)) * e_unit
    result = merge(result_chi2, (par = par_fit, qbb = qbb, func = func, func_err = func_err, func_cal = func_cal, func_cal_err = func_cal_err, peaks = peaks, fwhm = fwhm))
    report = merge(report_chi2, (e_unit = e_unit, par = par, qbb = result.qbb, type = :fwhm))

    return result, report
end
fit_fwhm(peaks::Vector{<:Unitful.Energy{<:Real}}, fwhm::Vector{<:Unitful.Energy{<:Real}}; kwargs...) = fit_fwhm(1, peaks, fwhm; kwargs...)


function _simple_linear_fit(x::AbstractVector{<:Real}, y::AbstractVector{<:Measurement}, n_poly::Int=1)
    # weighted least squares of the polynomial y = Σᵢ βᵢ xⁱ (i = 0…n_poly) with weights 1/σ_y²;
    # the covariance (XᵀWX)⁻¹ depends only on x and σ_y, not on the fit residuals
    all(>(0), muncert.(y)) || throw(ArgumentError("All y values need a positive uncertainty for the weighted linear fit"))
    X = reduce(hcat, (x .^ i for i in 0:n_poly))
    w = muncert.(y) .^ -2
    cov = inv(X' * (w .* X))
    β = cov * (X' * (w .* mvalue.(y)))
    measurement.(β, sqrt.(diag(cov)))
end

function _get_enc_fano_guess(peaks::Vector{<:Unitful.Energy{<:Real}}, fwhm::Vector{<:Unitful.Energy{<:Real}}, pol_order::Int=1)
    # fwhm² = enc + fano * e (+ ct * e²) is linear in the fit parameters, on the scale the fit uses
    par_fit = _simple_linear_fit(mvalue.(ustrip.(e_fit_fwhm_unit, peaks)), ustrip.(e_unit, fwhm) .^ 2, pol_order)

    # median and 84.1% quantile of N(μ, σ) restricted to (0, ∞): μ and μ + σ far from zero, a small
    # positive value of scale σ for μ ≲ 0
    function _positive(p::Measurement)
        d = truncated(Normal(mvalue(p), muncert(p)), 0.0, Inf)
        m = median(d)
        measurement(m, quantile(d, 0.8413) - m)
    end
    par_guess = _positive.(par_fit)

    # a pre-fit quadratic term beyond the concavity limit of its prior starts at the center of the range
    if pol_order == 2
        ct_max = mvalue(par_guess[2])^2 / (4 * mvalue(par_guess[1]))
        if mvalue(par_guess[3]) >= ct_max
            par_guess[3] = measurement(ct_max / 2, ct_max / 4)
        end
    end
    par_guess
end

function get_fit_fwhm_pseudo_prior(pol_order::Int, par_guess::AbstractVector{<:Measurement})
    @assert length(par_guess) == pol_order + 1 "Need one guess per polynomial coefficient"
    # Weibull priors: strictly positive support without a hard lower cut-off;
    # median at the guess, 84.1% quantile one uncertainty above it
    _weibull(g::Measurement) = weibull_from_mx(mvalue(g), mvalue(g) + muncert(g), 0.8413)

    unshaped(if pol_order == 1
        enc, fano = _weibull.(par_guess)
        NamedTupleDist(; enc, fano)
    elseif pol_order == 2
        enc, fano = _weibull.(par_guess[1:2])
        # concave for all E iff 4·enc·ct ≤ fano²; truncated at zero rather than at the floor of
        # `stabilize_dist` so that the whole positive range is reachable
        ct = truncated(_weibull(par_guess[3]).untruncated, 0, mvalue(par_guess[2])^2 / (4 * mvalue(par_guess[1])))
        NamedTupleDist(; enc, fano, ct)
    else
        throw(ArgumentError("Only 1, 2 order polynominal calibration is supported"))
    end)
end
