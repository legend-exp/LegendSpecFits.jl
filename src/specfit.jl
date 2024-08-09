# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).
"""
    estimate_single_peak_stats(h::Histogram, calib_type::Symbol=:th228)

Estimate statistics/parameters for a single peak in the given histogram `h`.

`h` must only contain a single peak. The peak should have a Gaussian-like
shape.
`calib_type` specifies the calibration type. Currently only `:th228` is implemented.
If you want get the peak statistics for a PSD calibration, use `:psd`.

# Returns 
`NamedTuple` with the fields
    * `peak_pos`: estimated position of the peak (in the middle of the peak)
    * `peak_fwhm`: full width at half maximum (FWHM) of the peak
    * `peak_sigma`: estimated standard deviation of the peak
    * `peak_counts`: estimated number of counts in the peak
    * `mean_background`: estimated mean background value
"""
function estimate_single_peak_stats(h::Histogram,; calib_type::Symbol=:th228)
    if calib_type == :th228
        return estimate_single_peak_stats_th228(h)
    elseif calib_type == :psd
        return estimate_single_peak_stats_psd(h)
    else
        error("Calibration type not supported")
    end
end
export estimate_single_peak_stats

function estimate_single_peak_stats_th228(h::Histogram{T}) where T<:Real
    W = h.weights
    E = first(h.edges)
    bin_width = step(E)
    peak_amplitude, peak_idx = findmax(W)
    fwhm_idx_left = findfirst(w -> w >= (first(W) + peak_amplitude) / 2, W)
    fwhm_idx_right = findlast(w -> w >= (last(W) + peak_amplitude) / 2, W)
    peak_max_pos = (E[peak_idx] + E[peak_idx+1]) / 2
    peak_mid_pos = (E[fwhm_idx_right] + E[fwhm_idx_left]) / 2
    peak_pos = (peak_max_pos + peak_mid_pos) / 2.0
    peak_fwhm = (E[fwhm_idx_right] - E[fwhm_idx_left]) / 1.0
    peak_sigma = peak_fwhm * inv(2*√(2log(2)))
    peak_fwqm = NaN
    # make sure that peakstats have non-zero sigma and fwhm values to prevent fit priors from being zero
    if peak_fwhm == 0
        fwqm_idx_left = findfirst(w -> w >= (first(W) + peak_amplitude) / 4, W)
        fwqm_idx_right = findlast(w -> w >= (last(W) + peak_amplitude) / 4, W)
        peak_fwqm = (E[fwqm_idx_right] - E[fwqm_idx_left]) / 1.0
        peak_sigma = peak_fwqm * inv(2*√(2log(4)))
        peak_fwhm  = peak_sigma * 2*√(2log(2))
    end
    if peak_sigma == 0
        peak_sigma = 1.0
        peak_fwhm = 2.0
    end
    # peak_area = peak_amplitude * peak_sigma * sqrt(2*π)
    # calculate mean background and step
    idx_bkg_left = something(findfirst(x -> x >= peak_pos - 15*peak_sigma, E[2:end]), 7)
    idx_bkg_right = something(findfirst(x -> x >= peak_pos + 15*peak_sigma, E[2:end]), length(W) - 7)
    mean_background_left, mean_background_right = mean(view(W, 1:idx_bkg_left)), mean(view(W, idx_bkg_right:length(W)))
    
    mean_background_step = (mean_background_left - mean_background_right) / bin_width
    mean_background = mean_background_right / bin_width #(mean_background_left + mean_background_right) / 2 / bin_width
    mean_background_std = 0.5*(std(view(W, 1:idx_bkg_left)) + std(view(W, idx_bkg_right:length(W)))) / bin_width
    #mean_background_err = 0.5*(std(view(W, 1:idx_bkg_left))/sqrt(length(1:idx_bkg_left)) + std(view(W, idx_bkg_right:length(W)))/sqrt(length(idx_bkg_right:length(W))) ) / bin_width # error of the mean 
    
    # sanity checks
    mean_background = ifelse(mean_background == 0, 0.01, mean_background)
    mean_background_step = ifelse(mean_background_step < 1e-2, 1e-2, mean_background_step)
    mean_background_std = ifelse(!isfinite(mean_background_std) || mean_background_std == 0, sqrt(mean_background), mean_background_std)

    peak_counts = inv(0.761) * (sum(view(W,fwhm_idx_left:fwhm_idx_right)) - mean_background * peak_fwhm)
    peak_counts = ifelse(peak_counts < 0.0, inv(0.761) * sum(view(W,fwhm_idx_left:fwhm_idx_right)), peak_counts)
    if !isnan(peak_fwqm)
        peak_counts = inv(0.904) * (sum(view(W,fwqm_idx_left:fwqm_idx_right)) - mean_background * peak_fwqm)
        peak_counts = ifelse(peak_counts < 0.0, inv(0.904) * sum(view(W,fwqm_idx_left:fwqm_idx_right)), peak_counts)
    end
    (
        peak_pos = peak_pos, 
        peak_fwhm = peak_fwhm,
        peak_sigma = peak_sigma, 
        peak_counts = peak_counts, 
        mean_background = mean_background,
        mean_background_step = mean_background_step,
        mean_background_std = mean_background_std,
    )
end

"""
    fit_peaks(peakhists::Array, peakstats::StructArray, th228_lines::Array,; calib_type::Symbol=:th228, uncertainty::Bool=true, low_e_tail::Bool=true)
Perform a fit of the peakshape to the data in `peakhists` using the initial values in `peakstats` to the calibration lines in `th228_lines`. 
# Returns
    * `peak_fit_plots`: array of plots of the peak fits
    * `return_vals`: dictionary of the fit results
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
     fit_func::Symbol= :f_fit, pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true),  m_cal_simple::Union{<:Unitful.AbstractQuantity{<:Real}, <:Real} = 1.0) where T<:Any
    # create return and result dicts
    result = Dict{T, NamedTuple}()
    report = Dict{T, NamedTuple}()
    # iterate throuh all peaks
    for (i, peak) in enumerate(th228_lines)
        # get histogram and peakstats
        h  = peakhists[i]
        ps = peakstats[i]
        # fit peak
        result_peak, report_peak = fit_single_peak_th228(h, ps; uncertainty=uncertainty, low_e_tail=low_e_tail, fit_func = fit_func, pseudo_prior = pseudo_prior, m_cal_simple = m_cal_simple)

        # check covariance matrix for being semi positive definite (no negative uncertainties)
        if uncertainty
            if iterative_fit && !isposdef(result_peak.covmat)
                @warn "Covariance matrix not positive definite for peak $peak - repeat fit without low energy tail"
                pval_save = result_peak.pval
                result_peak, report_peak = fit_single_peak_th228(h, ps, ; uncertainty=uncertainty, low_e_tail=false, fit_func = fit_func, pseudo_prior = pseudo_prior,  m_cal_simple = m_cal_simple)
                @info "New covariance matrix is positive definite: $(isposdef(result_peak.covmat))"
                @info "p-val with low-energy tail  p=$(round(pval_save,digits=5)) , without low-energy tail: p=$(round((result_peak.pval),digits=5))"
                end
        end
        # save results 
        if !isnothing(e_unit) 
            keys_with_unit = [:μ, :σ, :fwhm, :μ_cen]
            result_peak = merge(result_peak, NamedTuple{Tuple(keys_with_unit)}([result_peak[k] .* e_unit for k in keys_with_unit]...))
        end
        result[peak] = result_peak
        report[peak] = report_peak
    end
    return result, report
end


"""
    fit_single_peak_th228(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background), NTuple{5, T}};, uncertainty::Bool=true, fixed_position::Bool=false, low_e_tail::Bool=true) where T<:Real
Perform a fit of the peakshape to the data in `h` using the initial values in `ps` while using the `gamma_peakshape` with low-E tail.
Also, FWHM is calculated from the fitted peakshape with MC error propagation. The peak position can be fixed to the value in `ps` by setting `fixed_position=true`. If the low-E tail should not be fitted, it can be disabled by setting `low_e_tail=false`.
# Returns
    * `result`: NamedTuple of the fit results containing values and errors
    * `report`: NamedTuple of the fit report which can be plotted
"""
function fit_single_peak_th228(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background, :mean_background_step, :mean_background_std), NTuple{7, T}}; 
    uncertainty::Bool=true, low_e_tail::Bool=true, fixed_position::Bool=false, pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true),
    fit_func::Symbol=:f_fit, background_center::Real = ps.peak_pos, m_cal_simple::Union{<:Unitful.AbstractQuantity{<:Real}, <:Real} = 1.0) where T<:Real
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
        v -> hist_loglike(Base.Fix2(f_fit, v), h)
    end

    # MLE
    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), v_init, Optim.Options(time_limit = 60, iterations = 3000))
    converged = Optim.converged(opt_r)

    # best fit results
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

    f_loglike_array = let f_fit=fit_function, h=h, v_keys = keys(pseudo_prior) #same loglikelihood function as f_loglike, but has array as input instead of NamedTuple
        v ->  - hist_loglike(x -> f_fit(x, NamedTuple{v_keys}(v)), h) 
    end

    if uncertainty
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
                (fwhm = measurement(fwhm, fwhm_err), gof = (pvalue = pval, chi2 = chi2, dof = dof, covmat = param_covariance, converged = converged))
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
            (fwhm = measurement(fwhm, NaN), ), (gof = (converged = converged,),))
        report = (
            v = v_ml,
            h = h,
            f_fit = x -> Base.Fix2(fit_function, v_ml)(x),
            f_components = peakshape_components(fit_func, v_ml; background_center = background_center),
            gof = NamedTuple()
        )
    end

    # convert µ, centroid and sigma, fwhm back to [ADC]
    µ_cen = peak_centroid(result)/m_cal_simple
    result = merge(result, (µ = result.µ/m_cal_simple, fwhm = result.fwhm/m_cal_simple, σ = result.σ/m_cal_simple, µ_cen = µ_cen))
    return result, report
end
export fit_single_peak_th228

"""
    peak_centroid(v::NamedTuple)
calculate centroid of gamma peak from fit parameters
"""
function peak_centroid(v::NamedTuple)
    centroid = v.μ - v.skew_fraction * (v.µ * v.skew_width)
    if haskey(v, :skew_fraction_highE)
        centroid += v.skew_fraction_highE * (v.µ * v.skew_width_highE)
    end
    return centroid
end
export peak_centroid
"""
    estimate_fwhm(v::NamedTuple, v_err::NamedTuple)
Get the FWHM of a peak from the fit parameters.

# Returns
    * `fwhm`: the FWHM of the peak
"""
function estimate_fwhm(v::NamedTuple)
    # get FWHM
    f_sigWithTail = Base.Fix2(get_th228_fit_functions().f_sigWithTail,v)
    try
        half_max_sig = maximum(f_sigWithTail.(v.μ - v.σ:0.001:v.μ + v.σ))/2
        roots_low = find_zero(x -> f_sigWithTail(x) - half_max_sig, v.μ - v.σ, maxiter=100)
        roots_high = find_zero(x -> f_sigWithTail(x) - half_max_sig, v.μ + v.σ, maxiter=100)
        return roots_high - roots_low
    catch e
        return NaN
    end
end
"""
    get_peak_fwhm_th228(v_ml::NamedTuple, v_ml_err::NamedTuple)
Get the FWHM of a peak from the fit parameters while performing a MC error propagation.

# Returns
    * `fwhm`: the FWHM of the peak
    * `fwhm_err`: the uncertainty of the FWHM of the peak
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
    fit_single_peak_th228(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background), NTuple{5, T}};, uncertainty::Bool=true, fixed_position::Bool=false, low_e_tail::Bool=true) where T<:Real
    
Perform a simultaneous fit of two peaks (`h_survived` and `h_cut`) that together would form a histogram `h`, from which the result `h_result` was already determined using `fit_single_peak_th228`.
Also, FWHM is calculated from the fitted peakshape with MC error propagation. The peak position can be fixed to the value in `ps` by setting `fixed_position=true`. If the low-E tail should not be fitted, it can be disabled by setting `low_e_tail=false`.
# Returns
    * `result`: NamedTuple of the fit results containing values and errors, in particular the signal survival fraction `sf` and the background survival frachtion `bsf`.
    * `report`: NamedTuple of the fit report which can be plotted
"""
function fit_subpeaks_th228(
    h_survived::Histogram, h_cut::Histogram, h_result; 
    uncertainty::Bool=false, low_e_tail::Bool=true, fix_σ::Bool = true, fix_skew_fraction::Bool = true, fix_skew_width::Bool = true, 
    pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true), fit_func::Symbol=:f_fit, background_center::Real = h_result.μ
)

    # create standard pseudo priors
    standard_pseudo_prior = let ps = h_result, ps_cut = estimate_single_peak_stats(h_cut), ps_survived = estimate_single_peak_stats(h_survived)
        NamedTupleDist(
            μ = ConstValueDist(mvalue(ps.μ)),
            σ_survived = ifelse(fix_σ, ConstValueDist(mvalue(ps.σ)), weibull_from_mx(mvalue(ps.σ), 2*mvalue(ps.σ))),
            σ_cut = ifelse(fix_σ, ConstValueDist(mvalue(ps.σ)), weibull_from_mx(mvalue(ps.σ), 2*mvalue(ps.σ))),
            n = ConstValueDist(mvalue(ps.n)),
            sf = Uniform(0,1), # signal survival fraction
            bsf = Uniform(0,1), # background survival fraction 
            sasf = Uniform(0,1), # step amplitude survival fraction
            step_amplitude = ConstValueDist(mvalue(ps.step_amplitude)),
            skew_fraction_survived = ifelse(low_e_tail, ifelse(fix_skew_fraction, ConstValueDist(mvalue(ps.skew_fraction)), truncated(weibull_from_mx(0.01, 0.05), 0.0, 0.1)), ConstValueDist(0.0)),
            skew_fraction_cut = ifelse(low_e_tail, ifelse(fix_skew_fraction, ConstValueDist(mvalue(ps.skew_fraction)), truncated(weibull_from_mx(0.01, 0.05), 0.0, 0.1)), ConstValueDist(0.0)),
            skew_width_survived = ifelse(low_e_tail, ifelse(fix_skew_width, mvalue(ps.skew_width), weibull_from_mx(0.001, 1e-2)), ConstValueDist(1.0)),
            skew_width_cut = ifelse(low_e_tail, ifelse(fix_skew_width, mvalue(ps.skew_width), weibull_from_mx(0.001, 1e-2)), ConstValueDist(1.0)),
            background = ConstValueDist(mvalue(ps.background))
        )
    end

     # get fit function with background center
     fit_function = get_th228_fit_functions(; background_center = background_center)[fit_func]

    # use standard priors in case of no overwrites given
    if !(:empty in keys(pseudo_prior))
        # check if input overwrite prior has the same fields as the standard prior set
        @assert all(f -> f in keys(standard_pseudo_prior), keys(pseudo_prior)) "Pseudo priors can only have $(keys(standard_pseudo_prior)) as fields."
        # replace standard priors with overwrites
        pseudo_prior = merge(standard_pseudo_prior, pseudo_prior)
    else
        # take standard priors as pseudo priors with overwrites
        pseudo_prior = standard_pseudo_prior    
    end

    # transform back to frequency space
    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    # start values for MLE
    v_init = Vector(mean(f_trafo.target_dist))

    # create loglikehood function: f_loglike(v) that can be evaluated for any set of v (fit parameter)
    f_loglike = let f_fit=fit_function, h_cut=h_cut, h_survived=h_survived
        v -> begin
            v_survived = (μ = v.μ, σ = v.σ_survived, n = v.n * v.sf, 
                step_amplitude = v.step_amplitude * v.sasf,
                skew_fraction = v.skew_fraction_survived,
                skew_width = v.skew_width_survived,
                background = v.background * v.bsf
            )
            v_cut = (μ = v.μ, σ = v.σ_cut, n = v.n * (1 - v.sf), 
                step_amplitude = v.step_amplitude * (1 - v.sasf),
                skew_fraction = v.skew_fraction_cut,
                skew_width = v.skew_width_cut,
                background = v.background * (1 - v.bsf)
            )
            hist_loglike(Base.Fix2(f_fit, v_survived), h_survived) + hist_loglike(Base.Fix2(f_fit, v_cut), h_cut)
        end
    end

    # MLE
    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), v_init, Optim.Options(time_limit = 60, iterations = 3000))
    converged = Optim.converged(opt_r) 

    # best fit results
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))
    
    v_ml_survived = (
        μ = v_ml.μ, 
        σ = v_ml.σ_survived, 
        n = v_ml.n * v_ml.sf, 
        step_amplitude = v_ml.step_amplitude * v_ml.sasf,
        skew_fraction = v_ml.skew_fraction_survived,
        skew_width = v_ml.skew_width_survived,
        background = v_ml.background * v_ml.bsf
    ) 
            
    v_ml_cut = (
        μ = v_ml.μ, 
        σ = v_ml.σ_cut, 
        n = v_ml.n * (1 - v_ml.sf), 
        step_amplitude = v_ml.step_amplitude * (1 - v_ml.sasf),
        skew_fraction = v_ml.skew_fraction_cut,
        skew_width = v_ml.skew_width_cut,
        background = v_ml.background * (1 - v_ml.bsf)
    )

    gof_survived = NamedTuple()
    gof_cut = NamedTuple()

    if uncertainty

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
            
        # get fwhm of peak
        fwhm, fwhm_err = try
                get_peak_fwhm_th228(v_ml, param_covariance)
            catch e
                get_peak_fwhm_th228(v_ml, v_ml_err)
            end

        # @debug "Best Fit values for $(part)"
        # @debug "μ: $(v_ml.μ) ± $(v_ml_err.μ)"
        # @debug "σ: $(v_ml.σ) ± $(v_ml_err.σ)"
        # @debug "n: $(v_ml.n) ± $(v_ml_err.n)"
        # @debug "p: $pval , chi2 = $(chi2) with $(dof) dof"
        # @debug "FWHM: $(fwhm) ± $(fwhm_err)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], v_ml_err[k]) for k in keys(v_ml)]...),
                (fwhm = measurement(fwhm, fwhm_err),), NamedTuple{(:gof_survived, :gof_cut)}(gofs))
                    
    else
        # get fwhm of peak
        fwhm, fwhm_err = get_peak_fwhm_th228(v_ml, v_ml, false)

        # @debug "Best Fit values"
        # @debug "μ: $(v_ml.μ)"
        # @debug "σ: $(v_ml.σ)"
        # @debug "n: $(v_ml.n)"
        # @debug "FWHM: $(fwhm)"

        result = merge(NamedTuple{keys(v_ml)}([measurement(v_ml[k], NaN) for k in keys(v_ml)]...),
        (fwhm = measurement(fwhm, NaN), gof_survived = NamedTuple(), gof_cut = NamedTuple()))
    end

    report = (
        survived = (
            v = v_ml_survived,
            h = h_survived,
            f_fit = x -> Base.Fix2(fit_function, v_ml_survived)(x),
            f_components = peakshape_components(fit_func, v_ml; background_center = background_center),
            gof = result.gof_survived
        ),
        cut = (
            v = v_ml_cut,
            h = h_cut,
            f_fit = x -> Base.Fix2(fit_function, v_ml_cut)(x),
            f_components = peakshape_components(fit_func, v_ml; background_center = background_center),
            gof = result.gof_cut
        ),
        sf = v_ml.sf,
        bsf = v_ml.bsf
    )

    return result, report
end