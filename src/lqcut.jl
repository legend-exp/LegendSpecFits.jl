"""
    lq_norm::Vector{<:AbstractFloat}, dt_eff::Vector{<:Unitful.RealOrRealQuantity}, e_cal::Vector{<:Unitful.Energy{<:Real}}, DEP_µ::Unitful.AbstractQuantity, DEP_σ::Unitful.AbstractQuantity; 
    ctc_dep_edgesigma::Float64=3.0 , ctc_driftime_cutoff_method::Symbol=:percentile, lq_outlier_sigma::Float64 = 2.0, drift_time_outlier_sigma::Float64 = 2.0, prehist_sigma::Float64=2.5, lq_e_corr_expression::Union{String,Symbol}="(lq / e)", dt_eff_expression::Union{String,Symbol}="(qdrift / e)" ,ctc_dt_eff_low_quantile::Float64=0.15, ctc_dt_eff_high_quantile::Float64=0.95, pol_fit_order::Int=1) 

Perform the drift time correction on the LQ data using the DEP peak. The function cuts outliers in lq and drift time, then performs a linear fit on the remaining data. The data is Corrected by subtracting the linear fit from the lq data.
# Returns
    * `result`: NamedTuple of the function used for lq classifier construction
    * `report`: NamedTuple of the histograms used for the fit, the cutoff values and the DEP edges
"""
function lq_ctc_correction(
    lq_norm::Vector{<:AbstractFloat}, dt_eff::Vector{<:Unitful.RealOrRealQuantity}, e_cal::Vector{<:Unitful.Energy{<:Real}}, DEP_µ::Unitful.AbstractQuantity, DEP_σ::Unitful.AbstractQuantity; 
    ctc_dep_edgesigma::Float64=3.0 , ctc_driftime_cutoff_method::Symbol=:percentile, lq_outlier_sigma::Float64 = 2.0, drift_time_outlier_sigma::Float64 = 2.0, prehist_sigma::Float64=2.5, lq_e_corr_expression::Union{String,Symbol}="(lq / e)", dt_eff_expression::Union{String,Symbol}="(qdrift / e)" ,ctc_dt_eff_low_quantile::Float64=0.15, ctc_dt_eff_high_quantile::Float64=0.95, pol_fit_order::Int=1) 

    # calculate DEP edges
    DEP_left = DEP_µ - ctc_dep_edgesigma * DEP_σ
    DEP_right = DEP_µ + ctc_dep_edgesigma * DEP_σ

    # cut data to DEP peak
    lq_DEP = lq_norm[DEP_left .< e_cal .< DEP_right]
    dt_eff_DEP = ustrip.(dt_eff[DEP_left .< e_cal .< DEP_right])

    lq_precut = cut_single_peak(lq_DEP, minimum(lq_DEP), maximum(lq_DEP))

    # truncated gaussian fit
    lq_result, lq_report = fit_single_trunc_gauss(lq_DEP, lq_precut)
    µ_lq = mvalue(lq_result.μ)
    σ_lq = mvalue(lq_result.σ)

    #set cutoff in lq dimension for later fit
    lq_lower = µ_lq - lq_outlier_sigma * σ_lq 
    lq_upper = µ_lq + lq_outlier_sigma * σ_lq 


    # dt_eff_DEP cutoff; method dependant on detector type
    
    if ctc_driftime_cutoff_method == :percentile # standard method; can be used for all detectors
        #set cutoff; default at the 15% and 95% percentile
        t_lower = quantile(dt_eff_DEP, ctc_dt_eff_low_quantile)
        t_upper = quantile(dt_eff_DEP, ctc_dt_eff_high_quantile)
        drift_prehist = nothing
        drift_report = nothing

    elseif ctc_driftime_cutoff_method == :gaussian # can't be used for detectors with double peaks
        
        ideal_bin_width = get_friedman_diaconis_bin_width(dt_eff_DEP)

        drift_prehist = fit(Histogram, dt_eff_DEP, range(minimum(dt_eff_DEP), stop=maximum(dt_eff_DEP), step=ideal_bin_width))
        drift_prestats = estimate_single_peak_stats(drift_prehist)
        drift_start = drift_prestats.peak_pos - prehist_sigma * drift_prestats.peak_sigma
        drift_stop = drift_prestats.peak_pos + prehist_sigma * drift_prestats.peak_sigma
        
        drift_edges = range(drift_start, stop=drift_stop, step=ideal_bin_width)
        drift_hist_DEP = fit(Histogram, dt_eff_DEP, drift_edges)
        
        drift_result, drift_report = fit_binned_trunc_gauss(drift_hist_DEP)
        µ_t = mvalue(drift_result.μ)
        σ_t = mvalue(drift_result.σ)

        #set cutoff in drift time dimension for later fit
        t_lower = µ_t - drift_time_outlier_sigma * σ_t
        t_upper = µ_t + drift_time_outlier_sigma * σ_t

    elseif ctc_driftime_cutoff_method == :double_gaussian # can be used for detectors with double peaks; not optimized yet
        #create histogram for drift time
        ideal_length = get_number_of_bins(dt_eff_DEP)
        drift_prehist = fit(Histogram, dt_eff_DEP, range(minimum(dt_eff_DEP), stop=maximum(dt_eff_DEP), length=ideal_length))
        drift_prestats = estimate_single_peak_stats(drift_prehist)

        #fit histogram with double gaussian
        drift_result, drift_report = fit_binned_double_gauss(drift_prehist, drift_prestats)
        
        #set cutoff at the x-value where the fit function is 10% of its maximum value
        x_values = -1000:0.5:5000  
        max_value = maximum(drift_report.f_fit.(x_values))
        threshold = 0.1 * max_value

        g(x) = drift_report.f_fit(x) - threshold
        x_at_threshold = find_zeros(g, minimum(x_values), maximum(x_values))

        t_lower = minimum(x_at_threshold)
        t_upper = maximum(x_at_threshold)
    else
        error("Driftime cutoff method $ctc_driftime_cutoff_method not supported")
    end

    #store cutoff values in box to return later    
    box = (lq_lower = lq_lower, lq_upper = lq_upper, t_lower = t_lower, t_upper = t_upper)

    #cut data according to cutoff values
    lq_cut = lq_DEP[lq_lower .< lq_DEP .< lq_upper .&& t_lower .< dt_eff_DEP .< t_upper]
    t_cut = dt_eff_DEP[lq_lower .< lq_DEP .< lq_upper .&& t_lower .< dt_eff_DEP .< t_upper]

    #polynomial fit
    result_µ, report_µ = chi2fit(pol_fit_order, t_cut, lq_cut; uncertainty=true)
    par = mvalue(result_µ.par)
    drift_time_func(x) =  sum((mvalue(par[i])) * x^(i-1) for i in eachindex(par))

    #property function for drift time correction
    lq_class_func = "$lq_e_corr_expression - " * join(["$(mvalue(par[i])) * $dt_eff_expression^$(i-1)" for i in eachindex(par)], " - ")
    lq_class_func_generic = "lq / e  - (slope * qdrift / e + y_inter)"

    #create result and report
    result = (
    func = lq_class_func,
    func_generic = lq_class_func_generic,
    )

    report = (
    lq_report = lq_report, 
    drift_prehist = drift_prehist, 
    drift_report = drift_report,
    lq_box = box,
    drift_time_func = drift_time_func,
    DEP_left = DEP_left,
    DEP_right = DEP_right,
    )
    

    return result, report
end
export lq_ctc_correction

"""
    DEP_µ::Unitful.Energy, DEP_σ::Unitful.Energy, e_cal::Vector{<:Unitful.Energy}, lq_classifier::Vector{<:AbstractFloat}; cut_sigma::Float64=3.0, dep_sideband_sigma::Float64=4.5, cut_truncation_sigma::Float64=2.0)

Evaluates the cutoff value for the LQ cut. The function performs a binned gaussian fit on the sidebandsubtracted LQ histogram and evaluates the cutoff value difined at 3σ of the fit.
# Returns
    * `result`: NamedTuple of the cutoff value
    * `report`: NamedTuple of the fit result, fit report and temporary histograms
"""
function lq_cut(
    DEP_µ::Unitful.Energy, DEP_σ::Unitful.Energy, e_cal::Vector{<:Unitful.Energy}, lq_classifier::Vector{<:AbstractFloat}; cut_sigma::Float64=3.0, dep_sideband_sigma::Float64=4.5, cut_truncation_sigma::Float64=2.0)

    # Define sidebands
    lq_DEP = lq_classifier[DEP_µ - dep_sideband_sigma * DEP_σ .< e_cal .< DEP_µ + dep_sideband_sigma * DEP_σ]
    lq_sb1 = lq_classifier[DEP_µ -  2 * dep_sideband_sigma * DEP_σ .< e_cal .< DEP_µ - dep_sideband_sigma * DEP_σ]
    lq_sb2 = lq_classifier[DEP_µ + dep_sideband_sigma * DEP_σ .< e_cal .< DEP_µ + 2 * dep_sideband_sigma * DEP_σ]
    
    # Generate values for histogram edges
    combined = [lq_DEP; lq_sb1; lq_sb2]
    ideal_bin_width = get_friedman_diaconis_bin_width(combined)
    edges = range(start=minimum(combined), stop=maximum(combined), step=ideal_bin_width)

    # Create histograms with the same bin edges
    hist_DEP = fit(Histogram, lq_DEP, edges)
    hist_sb1 = fit(Histogram, lq_sb1, edges)
    hist_sb2 = fit(Histogram, lq_sb2, edges)
    
    # Subtract histograms
    weights_subtracted = hist_DEP.weights .- hist_sb1.weights .- hist_sb2.weights
    hist_subtracted = Histogram(edges, weights_subtracted)

    # Replace negative bins with 0
    weights_corrected = max.(weights_subtracted, 0)
    hist_corrected = Histogram(edges, weights_corrected)
    
    # Create a named tuple of histograms for crosschecks
    temp_hists = (hist_DEP = hist_DEP, hist_sb1 = hist_sb1, hist_sb2 = hist_sb2, hist_subtracted = hist_subtracted, hist_corrected = hist_corrected)

    #get truncate values for fit; needed if outliers are present after in sideband subtracted histogram
    lq_prestats = estimate_single_peak_stats(hist_corrected)
    lq_start = lq_prestats.peak_pos - cut_truncation_sigma * lq_prestats.peak_sigma
    lq_stop = lq_prestats.peak_pos + cut_truncation_sigma * lq_prestats.peak_sigma

    #simplecuts benutzen

    # Fit the sideband subtracted histogram
    fit_result, fit_report = fit_binned_trunc_gauss(hist_corrected, (low=lq_start, high=lq_stop, max=NaN))

    #final cutoff value defined by "cut_sigma"
    cut_3σ = fit_result.μ + cut_sigma * fit_result.σ

    result = (
        cut = cut_3σ,
    )

    report = (
        cut = cut_3σ,
        fit_result = fit_result,
        temp_hists = temp_hists,
        fit_report = fit_report,
    )

    return result, report
end
export lq_cut
