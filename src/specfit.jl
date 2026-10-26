# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    fit_peaks(
        peakhists::Array, peakstats::StructArray, th228_lines::Vector;
        calib_type::Symbol, kwargs...
    )

Fits a gamma peak shape to each histogram in `peakhists`, one per calibration
line in `th228_lines`, using [`fit_single_peak_th228`](@ref) with the initial
values in `peakstats`. The fits run in parallel on the available threads.

## Arguments
* `peakhists`: One histogram per line.
* `peakstats`: `StructArray` of initial peak statistics, one entry per line, as returned by
  [`estimate_single_peak_stats`](@ref) (`:th228` variant).
* `th228_lines`: Calibration line energies; they are used as keys of the result.

## Keywords
* `calib_type::Symbol`: Required. Only `:th228` is supported; other values throw an error.
* `e_unit::Union{Nothing, Unitful.EnergyUnits} = nothing`: Energy unit attached to the fitted
  `μ`, `σ`, `fwhm` and `centroid`. Without it, they are unitless.
* `uncertainty::Bool = true`: Calculate parameter uncertainties and goodness of fit.
* `low_e_tail::Bool = true`: Fit the low-energy tail.
* `iterative_fit::Bool = false`: If `true` and the covariance matrix of a fit is not
  positive definite, repeat that fit without low-energy tail.
* `fit_func::Vector{Symbol}`: Fit function of each line (see [`fit_single_peak_th228`](@ref)).
  Default is `:gamma_def` for all lines.
* `pseudo_prior::NamedTupleDist = NamedTupleDist(empty = true)`: Replaces fields of the default
  prior of every fit (see [`fit_single_peak_th228`](@ref)).
* `m_cal_simple = 1.0`: Simple calibration constant; `μ`, `σ`, `fwhm` and `centroid` are
  divided by it, which converts them back to the scale before the simple calibration.

## Returns
A tuple `(result, report)` of two `OrderedDict`s keyed by the line energies in
`th228_lines`. The values of `result` are the `result` `NamedTuple`s of
[`fit_single_peak_th228`](@ref) and the values of `report` are its `report`
`NamedTuple`s, for plotting with `LegendMakie.lplot`.
"""
function fit_peaks(peakhists::Array, peakstats::StructArray, th228_lines::Vector; kwargs...)
    # remove calib type from kwargs
    @assert haskey(kwargs, :calib_type) "Calibration type not specified"
    calib_type = kwargs[:calib_type]
    # remove :calib_type from kwargs
    kwargs = pairs(NamedTuple(filter(k -> !(:calib_type in k), kwargs)))
    if calib_type == :th228
        return fit_peaks_th228(peakhists, peakstats, th228_lines,; kwargs...)
    else
        error("Calibration type not supported")
    end
end
export fit_peaks

function fit_peaks_th228(peakhists::Array, peakstats::StructArray, th228_lines::Vector{T},; e_unit::Union{Nothing, Unitful.EnergyUnits}=nothing, uncertainty::Bool=true, low_e_tail::Bool=true, iterative_fit::Bool=false,
    fit_func::Vector{Symbol}= fill(:gamma_def, length(th228_lines)), pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true),  m_cal_simple::MaybeWithEnergyUnits = 1.0) where T
    
    e_unit = ifelse(isnothing(e_unit), NoUnits, e_unit)

    # create return and result dicts
    v_result = Vector{NamedTuple}(undef, length(th228_lines))
    v_report = Vector{NamedTuple}(undef, length(th228_lines))

    # iterate throuh all peaks
    Threads.@threads for i in eachindex(th228_lines)
        # get histogram and peakstats
        peak = th228_lines[i]
        h  = peakhists[i]
        ps = peakstats[i]
        f = fit_func[i] 

        # fit peak
        result_peak, report_peak = fit_single_peak_th228(h, ps; uncertainty=uncertainty, low_e_tail=low_e_tail, fit_func = f, pseudo_prior = pseudo_prior)

        # check covariance matrix for being semi positive definite (no negative uncertainties)
        if uncertainty
            if iterative_fit && !isposdef(result_peak.covmat)
                @warn "Covariance matrix not positive definite for peak $peak - repeat fit without low energy tail"
                pval_save = result_peak.pval
                result_peak, report_peak = fit_single_peak_th228(h, ps, ; uncertainty=uncertainty, low_e_tail=false, fit_func = f, pseudo_prior = pseudo_prior)
                @info "New covariance matrix is positive definite: $(isposdef(result_peak.covmat))"
                @info "p-val with low-energy tail  p=$(round(pval_save,digits=5)) , without low-energy tail: p=$(round((result_peak.pval),digits=5))"
                end
        end
        # save results 
        keys_with_unit = [:μ, :σ, :fwhm, :centroid]
        result_peak = merge(result_peak, NamedTuple{Tuple(keys_with_unit)}([result_peak[k] .* e_unit ./ m_cal_simple for k in keys_with_unit]...))

        v_result[i] = result_peak
        v_report[i] = report_peak
    end

    # create return and result dicts
    result = OrderedDict{T, NamedTuple}(th228_lines .=> v_result)
    report = OrderedDict{T, NamedTuple}(th228_lines .=> v_report)

    return result, report
end


"""
    fit_single_peak_th228(
        h::Histogram, ps::NamedTuple;
        uncertainty::Bool = true, low_e_tail::Bool = true, fixed_position::Bool = false,
        pseudo_prior::NamedTupleDist = NamedTupleDist(empty = true),
        fit_func::Symbol = :gamma_def, background_center = ps.peak_pos, m_cal_simple::Real = 1.0
    )

Fits a gamma peak shape to the histogram `h` by maximizing the binned Poisson
likelihood, starting from the peak statistics `ps`, and calculates the FWHM of
the fitted shape.

`ps` is the `NamedTuple` returned by [`estimate_single_peak_stats`](@ref) for
`calib_type = :th228`, with the fields `peak_pos`, `peak_fwhm`, `peak_sigma`,
`peak_counts`, `bin_width`, `mean_background`, `mean_background_step` and
`mean_background_std`. It sets the location and width of the default prior of
each parameter. The FWHM and its uncertainty are determined by Monte Carlo
error propagation with [`get_peak_fwhm_th228`](@ref).

## Keywords
* `uncertainty::Bool = true`: Calculate parameter uncertainties (from the Hessian of the
  likelihood) and goodness of fit. The uncertainties are `NaN` if `false` or if the fit did not converge.
* `low_e_tail::Bool = true`: Fit the low- and high-energy tails. If `false`, the tail fractions are fixed to zero.
* `fixed_position::Bool = false`: Fix the peak position `μ` to `ps.peak_pos`.
* `pseudo_prior::NamedTupleDist = NamedTupleDist(empty = true)`: Replaces fields of the default prior.
  Its keys must be parameters of `fit_func`.
* `fit_func::Symbol = :gamma_def`: Peak shape, built from [`gamma_peakshape`](@ref):
  - `:gamma_def`: Gaussian signal, low-energy tail, flat background and step background.
  - `:gamma_tails`: `:gamma_def` plus a high-energy tail (`skew_fraction_highE`, `skew_width_highE`).
  - `:gamma_bckSlope`: `:gamma_def` plus a linear background slope (`background_slope`).
  - `:gamma_bckExp`: `:gamma_def` plus an exponential background (`background_exp`).
  - `:gamma_bckFlat`: `:gamma_def` without the step background (no `step_amplitude`).
  - `:gamma_tails_bckFlat`: `:gamma_tails` without the step background.
  - `:gamma_minimal`: Gaussian signal and flat background only (`μ`, `σ`, `n`, `background`).
* `background_center::Union{Real, Nothing} = ps.peak_pos`: Reference position of the slope and
  exponential background terms.
* `m_cal_simple::Real = 1.0`: Simple calibration constant; `μ`, `σ`, `fwhm` and `centroid`
  are divided by it after the fit.

## Returns
A tuple `(result, report)`. `result` is a `NamedTuple` with
* the fit parameters of `fit_func` (`μ`, `σ`, `n`, `skew_fraction`, `skew_width`, `background`,
  `step_amplitude`, ...) as `Measurement`s;
* `fwhm`: FWHM of the fitted signal including the low-energy tail;
* `centroid`: mean position of the peak including the tails (see [`peak_centroid`](@ref));
* `fit_func`: the peak shape that was used;
* `gof`: `NamedTuple` with `converged` and, if the uncertainties were calculated, `pvalue`,
  `chi2`, `dof` (Poisson likelihood-ratio based, see [`p_value_poissonll`](@ref)), `covmat`,
  and `mean_residuals`, `median_residuals`, `std_residuals` of the normalized residuals.

`report` contains the best-fit values `v`, the histogram `h`, the fit function
`f_fit`, the peak components `f_components` and `gof` (including `residuals`
and `residuals_norm`), for plotting with `LegendMakie.lplot`.
"""
function fit_single_peak_th228(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :bin_width, :mean_background, :mean_background_step, :mean_background_std), NTuple{8, T}}; 
    uncertainty::Bool=true, low_e_tail::Bool=true, fixed_position::Bool=false, pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true),
    fit_func::Symbol=:gamma_def, background_center::Union{Real,Nothing} = ps.peak_pos, m_cal_simple::Real = 1.0) where T<:Real
    # create standard pseudo priors
    pseudo_prior = get_pseudo_prior(h, ps, fit_func; pseudo_prior = pseudo_prior, fixed_position = fixed_position, low_e_tail = low_e_tail)
    
    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = Vector(mean(f_trafo.target_dist)) 

    # get fit function with background center
    fit_function = get_th228_fit_functions(; background_center = background_center)[fit_func]

    # create loglikehood function: f_loglike(v) that can be evaluated for any set of v (fit parameter)
    f_loglike = let f_fit = fit_function, h = h
        v -> hist_loglike(x -> x in Interval(extrema(h.edges[1])...) ? f_fit(x, v) : 0, h)
    end

    # MLE
    optf = OptimizationFunction((u, p) -> ((-) ∘ f_loglike ∘ inverse(f_trafo))(u), AutoForwardDiff())
    optpro = OptimizationProblem(optf, v_init, ())
    res = solve(optpro, LBFGSB(), maxiters = 3000) #, maxtime=optim_time_limit)

    converged = (res.retcode == ReturnCode.Success)
    if !converged @warn "Fit did not converge" end

    # best fit results
    v_ml = inverse(f_trafo)(res.u)

    f_loglike_array = let f_fit=fit_function, h=h, v_keys = keys(pseudo_prior) #same loglikelihood function as f_loglike, but has array as input instead of NamedTuple
        v ->  - hist_loglike(x -> f_fit(x, NamedTuple{v_keys}(v)), h)
    end

    if uncertainty && converged
        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml))

        # Calculate the parameter covariance matrix
        param_covariance_raw = inv(H)
        param_covariance = nearestSPD(param_covariance_raw)

        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)

        # calculate p-value
        pval, chi2, dof = p_value_poissonll(fit_function, h, v_ml) # based on likelihood ratio 

        # calculate normalized residuals
        residuals, residuals_norm, _, _ = get_residuals(fit_function, h, v_ml)

        # get fwhm of peak
        fwhm, fwhm_err = 
            try
                get_peak_fwhm_th228(v_ml, param_covariance)
            catch e
                get_peak_fwhm_th228(v_ml, v_ml_err)
            end

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"
        @debug "n: $(v_ml.n) ± $(v_ml_err.n)"
        @debug "p: $pval , chi2 = $(chi2) with $(dof) dof"
        @debug "FWHM: $(fwhm) ± $(fwhm_err)"
    
        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml)]...),
                (fwhm = measurement(fwhm, fwhm_err), fit_func = fit_func, 
                    gof = (pvalue = pval, 
                            chi2 = chi2, 
                            dof = dof, 
                            covmat = param_covariance, 
                            mean_residuals = mean(residuals_norm),
                            median_residuals = median(residuals_norm),
                            std_residuals = std(residuals_norm),
                            converged = converged)
                        )
                    )
        report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(fit_function, result)(x),
            f_components = peakshape_components(fit_func, v_ml; background_center = background_center),
            gof = merge(result.gof, (residuals = residuals, residuals_norm = residuals_norm,))
        )
    else
        # get fwhm of peak
        fwhm, fwhm_err = get_peak_fwhm_th228(v_ml, v_ml, false)

        @debug "Best Fit values"
        @debug "μ: $(v_ml.μ)"
        @debug "σ: $(v_ml.σ)"
        @debug "n: $(v_ml.n)"
        @debug "FWHM: $(fwhm)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], NaN) for k in keys(v_ml)]...),
            (fwhm = measurement(fwhm, NaN), fit_func = fit_func, ), (gof = (converged = converged,),))
        report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(fit_function, v_ml)(x),
            f_components = peakshape_components(fit_func, v_ml; background_center = background_center),
            gof = NamedTuple()
        )
    end

    # convert µ, centroid and sigma, fwhm back to [ADC]
    centroid = peak_centroid(result)/m_cal_simple
    result = merge(result, (µ = result.µ/m_cal_simple, fwhm = result.fwhm/m_cal_simple, σ = result.σ/m_cal_simple, centroid = centroid))
    return result, report
end
export fit_single_peak_th228

"""
    peak_centroid(v::NamedTuple)

Calculates the centroid (mean position) of a gamma peak from the fit parameters
`v`: the Gaussian position `μ` shifted by the low-energy tail,
`-skew_fraction * μ * skew_width`, and by the high-energy tail,
`+skew_fraction_highE * μ * skew_width_highE`. A tail is only included if its
parameters are fields of `v`.
"""
function peak_centroid(v::NamedTuple)
    centroid = if haskey(v, :skew_fraction)
       v.μ - v.skew_fraction * (v.µ * v.skew_width)
    else
        v.μ
    end 
    if haskey(v, :skew_fraction_highE)
        centroid += v.skew_fraction_highE * (v.µ * v.skew_width_highE)
    end
    return centroid
end
export peak_centroid
"""
    estimate_fwhm(v::NamedTuple)

Calculates the FWHM of a peak from the fit parameters `v`.

If `v` has the field `skew_fraction`, the FWHM is determined numerically from the
Gaussian signal plus low-energy tail. Otherwise it is `2√(2 ln 2) σ`. Returns
`NaN` if the half-maximum positions cannot be found or `v` has no field `σ`.
"""
function estimate_fwhm(v::NamedTuple)
    if haskey(v, :skew_fraction)
        f_sigWithTail = Base.Fix2(get_th228_fit_functions().gamma_sigWithTail,v)
        try
            e_low, e_high = v.skew_fraction <= 0.5 ? (v.μ - v.σ, v.μ + v.σ) : (v.μ * (1 - v.skew_width), v.μ * (1 + v.skew_width))
            # the maximum is the zero of the derivative inside the bracket
            max_sig = f_sigWithTail(find_zero(x -> ForwardDiff.derivative(f_sigWithTail, x), (e_low, e_high), Bisection()))
            half_max_sig = max_sig/2
            
            tmp = x -> f_sigWithTail(x) - half_max_sig
            roots_low = find_zero(tmp, e_low, maxiter=100)
            roots_high = find_zero(tmp, e_high, maxiter=100)
            return roots_high - roots_low
        catch
            return NaN
        end
    elseif haskey(v, :σ)
        return 2 * sqrt(2 * log(2)) * v.σ
    else
        return NaN
    end
end
"""
    get_peak_fwhm_th228(v_ml::NamedTuple, v_ml_err::Union{Matrix, NamedTuple}, uncertainty::Bool = true)

Calculates the FWHM of a gamma peak from the best-fit parameters `v_ml` and its
uncertainty by Monte Carlo error propagation.

`v_ml_err` is either the covariance matrix of the parameters (10 000 correlated
samples) or a `NamedTuple` of standard deviations (1000 independent samples).
With `uncertainty = false`, no sampling is done and the uncertainty is `NaN`.

## Returns
A tuple `(fwhm, fwhm_err)` of the FWHM and its standard deviation over the finite samples.
"""
function get_peak_fwhm_th228(v_ml::NamedTuple, v_ml_err::Union{Matrix,NamedTuple}, uncertainty::Bool=true)
    # get fwhm for peak fit
    fwhm = estimate_fwhm(v_ml)
    if !uncertainty
        return fwhm, NaN
    end

    # get MC for FWHM err
    if isa(v_ml_err, Matrix)# use correlated fit parameter uncertainties 
        v_mc = get_mc_value_shapes(v_ml, v_ml_err, 10000)
    elseif isa(v_ml_err, NamedTuple) # use uncorrelated fit parameter uncertainties 
        v_mc = get_mc_value_shapes(v_ml, v_ml_err, 1000)
    end
    fwhm_mc = estimate_fwhm.(v_mc)
    fwhm_err = std(fwhm_mc[isfinite.(fwhm_mc)])
    return fwhm, fwhm_err
end
export get_peak_fwhm_th228



"""
    fit_subpeaks_th228(
        h_survived::Histogram, h_cut::Histogram, h_result::NamedTuple;
        uncertainty::Bool = false, low_e_tail::Bool = true, fix_σ::Bool = true,
        fix_skew_fraction::Bool = true, fix_skew_width::Bool = true,
        pseudo_prior::NamedTupleDist = NamedTupleDist(empty = true),
        fit_func::Symbol = :gamma_def, background_center::Real = h_result.μ
    )

Fits the two histograms `h_survived` and `h_cut`, which together form a
histogram that was already fitted with [`fit_single_peak_th228`](@ref) (result
`h_result`), simultaneously. The fit determines the signal survival fraction
`sf` and the background survival fraction `bsf` (the fraction of the peak and
background of `h_result` that remains in `h_survived`).

The position, amplitude and background level are fixed to the values in
`h_result`. `fix_σ`, `fix_skew_fraction` and `fix_skew_width` fix the peak width
and the tail parameters of both histograms to the values in `h_result`; if
`false`, they are fitted separately for each histogram.

## Returns
A tuple `(result, report)`. `result` holds the fit parameters as `Measurement`s,
in particular the signal survival fraction `sf` and the background survival
fraction `bsf` (and `sasf`, the survival fraction of the step amplitude, if
`h_result` has a `step_amplitude`), the `fwhm`, the `fit_func` and `gof` with
`converged` and the entries `survived` and `cut` (`pvalue`, `chi2`, `dof`, `covmat`).
`report` has the entries `survived` and `cut` (each with `v`, `h`, `f_fit`,
`f_components`, `gof`) and `sf` and `bsf`.
"""
function fit_subpeaks_th228(
    h_survived::Histogram, h_cut::Histogram, h_result::NamedTuple; 
    uncertainty::Bool=false, low_e_tail::Bool=true, fix_σ::Bool = true, fix_skew_fraction::Bool = true, fix_skew_width::Bool = true, 
    pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true), fit_func::Symbol= :gamma_def, background_center::Real = h_result.μ
)

    # create pseudo priors
    pseudo_prior = get_subpeaks_pseudo_prior(h_survived, h_cut, h_result, fit_func; pseudo_prior = pseudo_prior, fix_σ = fix_σ, fix_skew_fraction = fix_skew_fraction, fix_skew_width = fix_skew_width, low_e_tail = low_e_tail)

    # get fit function with background center
    fit_function = get_th228_fit_functions(; background_center = background_center)[fit_func]

    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = Vector(mean(f_trafo.target_dist))

    # create loglikehood function: f_loglike(v) that can be evaluated for any set of v (fit parameter)
    f_loglike = let f_fit=fit_function, h_cut=h_cut, h_survived=h_survived, get_v_ml=Base.Fix2(get_subpeaks_v_ml, fit_func)
        v -> begin
            v_survived, v_cut = get_v_ml(v)
            hist_loglike(Base.Fix2(f_fit, v_survived), h_survived) + hist_loglike(Base.Fix2(f_fit, v_cut), h_cut)
        end
    end

    # MLE
    optf = OptimizationFunction((u, p) -> ((-) ∘ f_loglike ∘ inverse(f_trafo))(u), AutoForwardDiff())
    optpro = OptimizationProblem(optf, v_init, ())
    res = solve(optpro, LBFGSB(), maxiters = 3000) #, maxtime=optim_time_limit)

    converged = (res.retcode == ReturnCode.Success)
    if !converged @warn "Fit did not converge" end

    # best fit results
    v_ml = inverse(f_trafo)(res.u)
    
    v_ml_survived, v_ml_cut = get_subpeaks_v_ml(v_ml, fit_func)

    gof_survived = NamedTuple()
    gof_cut = NamedTuple()

    if uncertainty && converged

        f_loglike_array = let v_keys = keys(pseudo_prior)
            v ->  -f_loglike(NamedTuple{v_keys}(v))
        end

        # Calculate the Hessian matrix using ForwardDiff
        H = ForwardDiff.hessian(f_loglike_array, tuple_to_array(v_ml))

        # Calculate the parameter covariance matrix
        param_covariance_raw = inv(H)
        param_covariance = nearestSPD(param_covariance_raw)

        # Extract the parameter uncertainties
        v_ml_err = array_to_tuple(sqrt.(abs.(diag(param_covariance))), v_ml)
            
        # calculate all of this for each histogram individually
        gofs = [
            begin
                
            h_part = Dict("survived" => h_survived, "cut" => h_cut)[part]
            v_ml_part = Dict("survived" => v_ml_survived, "cut" => v_ml_cut)[part]
            
            # calculate p-value
            pval, chi2, dof = p_value_poissonll(fit_function, h_part, v_ml_part)
        
            # calculate normalized residuals
            residuals, residuals_norm, _, bin_centers = get_residuals(fit_function, h_part, v_ml_part)
                
            gof = (
                pvalue = pval, chi2 = chi2, dof = dof,
                covmat = param_covariance,
                residuals = residuals, residuals_norm = residuals_norm,
                bin_centers = bin_centers,
                converged = converged
            )
                    
            end for part in ("survived", "cut")
        ]
        # get gofs
        gof_survived, gof_cut = gofs

        # get fwhm of peak
        fwhm, fwhm_err = try
                get_peak_fwhm_th228(v_ml, param_covariance)
            catch e
                get_peak_fwhm_th228(v_ml, v_ml_err)
            end

        @debug "Best Fit values"
        @debug "SF: $(v_ml.sf) ± $(v_ml_err.sf)"
        @debug "BSF: $(v_ml.bsf) ± $(v_ml_err.bsf)"
        @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        @debug "σ survived: $(v_ml.σ_survived) ± $(v_ml_err.σ_survived)"
        @debug "σ cut     : $(v_ml.σ_cut) ± $(v_ml_err.σ_cut)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml)]...),
                (fwhm = measurement(fwhm, fwhm_err), fit_func = fit_func, ), #NamedTuple{(:gof_survived, :gof_cut)}(gofs))
                (gof = (converged = converged,
                    survived = (pvalue = gofs[1].pvalue, chi2 = gofs[1].chi2, dof = gofs[1].dof, covmat = gofs[1].covmat),
                    cut = (pvalue = gofs[2].pvalue, chi2 = gofs[2].chi2, dof = gofs[2].dof, covmat = gofs[2].covmat)
                ), ))
    else
        # get fwhm of peak
        fwhm, fwhm_err = get_peak_fwhm_th228(v_ml, v_ml, false)

        @debug "Best Fit values"
        @debug "SF: $(v_ml.sf)"
        @debug "BSF: $(v_ml.bsf)"
        @debug "μ: $(v_ml.μ)"
        @debug "σ survived: $(v_ml.σ_survived)"
        @debug "σ cut     : $(v_ml.σ_cut)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], NaN) for k in keys(v_ml)]...),
        (fwhm = measurement(fwhm, NaN), fit_func = fit_func, gof = (converged = converged, survived = NamedTuple(), cut = NamedTuple())))
    end

    report = (
        survived = (
            v = v_ml_survived,
            h = h_survived,
            f_fit = x -> Base.Fix2(fit_function, v_ml_survived)(x),
            f_components = peakshape_components(fit_func, v_ml_survived; background_center = background_center),
            gof = gof_survived
        ),
        cut = (
            v = v_ml_cut,
            h = h_cut,
            f_fit = x -> Base.Fix2(fit_function, v_ml_cut)(x),
            f_components = peakshape_components(fit_func, v_ml_cut; background_center = background_center),
            gof = gof_cut
        ),
        sf = result.sf,
        bsf = result.bsf
    )

    return result, report
end