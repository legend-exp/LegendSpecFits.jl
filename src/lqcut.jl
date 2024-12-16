"""
    lq_drift_time_correction(
    lq_norm::Vector{<:AbstractFloat}, dt_eff::Vector{<:Unitful.RealOrRealQuantity}, e_cal::Vector{<:Unitful.Energy{<:Real}}, DEP_µ::Unitful.AbstractQuantity, DEP_σ::Unitful.AbstractQuantity; 
    DEP_edgesigma::Float64=3.0 , mode::Symbol=:percentile, drift_cutoff_sigma::Float64 = 2.0, prehist_sigma::Float64=2.5, e_expression::Union{String,Symbol}="e", dt_eff_low_quantile::Float64=0.15, dt_eff_high_quantile::Float64=0.95)

Perform the drift time correction on the LQ data using the DEP peak. The function cuts outliers in lq and drift time, then performs a linear fit on the remaining data. The data is corrected by subtracting the linear fit from the lq data.

# Arguments
    * `lq_norm`: Normalized inverted cut logic 
    * `dt_eff`: Effective drift time
    * `e_cal`: Calibrated energies
    * `DEP_µ`: Double escape peak mean
    * `DEP_σ`: Double escape peak sigma

# Keywords
    * `DEP_edgesigma`: Double escape peak of the edge sigma
    * `mode`: Mode symbol 
    * `drift_cutoff_sigma`: Drift cutoff standard deviation
    * `prehist_sigma`: Prehistogram standard deviation
    * `e_expression`: Energy expression
    * `dt_eff_low_quantile`: Effctive drift time low quantile
    * `dt_eff_high_quantile`: Effective drift time high quantile


# Returns
    * `func`: Function
    * `func_generic`: Generic function
    * `lq_prehist`: Inverted cut logic prehistogram
    * `lq_report`: Inverted cut logic report
    * `drift_prehist`: Pre-histogram drift
    * `drift_report`: Drift reprot
    * `lq_box`: Cutoff values
    * `drift_time_func`: Drift time function
    * `DEP_left`: Left double escape peak edge
    * `DEP_right`: Right double escape peak edge
"""
function lq_drift_time_correction(
    lq_norm::Vector{<:AbstractFloat}, dt_eff::Vector{<:Unitful.RealOrRealQuantity}, e_cal::Vector{<:Unitful.Energy{<:Real}}, DEP_µ::Unitful.AbstractQuantity, DEP_σ::Unitful.AbstractQuantity; 
    DEP_edgesigma::Float64=3.0 , mode::Symbol=:percentile, drift_cutoff_sigma::Float64 = 2.0, prehist_sigma::Float64=2.5, e_expression::Union{String,Symbol}="e", dt_eff_low_quantile::Float64=0.15, dt_eff_high_quantile::Float64=0.95)

    # get energy units to remve later
    e_unit = unit(first(e_cal))

    # calculate DEP edges
    DEP_left = DEP_µ - DEP_edgesigma * DEP_σ
    DEP_right = DEP_µ + DEP_edgesigma * DEP_σ

    # cut data to DEP peak
    lq_DEP = lq_norm[DEP_left .< e_cal .< DEP_right]
    dt_eff_DEP = ustrip.(dt_eff[DEP_left .< e_cal .< DEP_right])

    # Calculate range to truncate lq data for outlier removal
    ideal_bin_width = get_friedman_diaconis_bin_width(lq_DEP)
    lq_prehist = fit(Histogram, lq_DEP, range(minimum(lq_DEP), maximum(lq_DEP), step=ideal_bin_width))
    lq_prestats = estimate_single_peak_stats(lq_prehist)
    lq_start = lq_prestats.peak_pos - prehist_sigma * lq_prestats.peak_sigma
    lq_stop = lq_prestats.peak_pos + prehist_sigma * lq_prestats.peak_sigma

    # truncated gaussian fit
    lq_result, lq_report = fit_single_trunc_gauss(lq_DEP, (low=lq_start, high=lq_stop, max=NaN))
    µ_lq = mvalue(lq_result.μ)
    σ_lq = mvalue(lq_result.σ)

    #set cutoff in lq dimension for later fit
    lq_lower = µ_lq - drift_cutoff_sigma * σ_lq 
    lq_upper = µ_lq + drift_cutoff_sigma * σ_lq 


    # dt_eff_DEP cutoff; method dependant on detector type
    
    if mode == :percentile # standard method; can be used for all detectors
        #set cutoff; default at the 15% and 95% percentile
        t_lower = quantile(dt_eff_DEP, dt_eff_low_quantile)
        t_upper = quantile(dt_eff_DEP, dt_eff_high_quantile)
        drift_prehist = nothing
        drift_report = nothing

    elseif mode == :gaussian # can't be used for detectors with double peaks
        
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
        t_lower = µ_t - drift_cutoff_sigma * σ_t
        t_upper = µ_t + drift_cutoff_sigma * σ_t

    elseif mode == :double_gaussian # can be used for detectors with double peaks; not optimized yet
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
        error("Mode $mode not supported")
    end

    #store cutoff values in box to return later    
    box = (lq_lower = lq_lower, lq_upper = lq_upper, t_lower = t_lower, t_upper = t_upper)

    #cut data according to cutoff values
    lq_cut = lq_DEP[lq_lower .< lq_DEP .< lq_upper .&& t_lower .< dt_eff_DEP .< t_upper]
    t_cut = dt_eff_DEP[lq_lower .< lq_DEP .< lq_upper .&& t_lower .< dt_eff_DEP .< t_upper]

    #linear fit
    result_µ, report_µ = chi2fit(1, t_cut, lq_cut; uncertainty=true)
    parameters = mvalue(result_µ.par)
    drift_time_func(x) = parameters[1] .+ parameters[2] .* x

    #property function for drift time correction
    lq_class_func = "(lq / ($(e_expression))$(e_unit)^-1) - ($(parameters[2]) * (qdrift / ($(e_expression))$(e_unit)^-1) + $(parameters[1]))"  # removes the units to get unitless lq classifier
    lq_class_func_generic = "lq / e  - (slope * qdrift / e + y_inter)"

    #create result and report
    result = (
    func = lq_class_func,
    func_generic = lq_class_func_generic,
    )

    report = (
    lq_prehist = lq_prehist, 
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
export lq_drift_time_correction

"""
    LQ_cut(
    DEP_µ::Unitful.Energy, DEP_σ::Unitful.Energy, e_cal::Vector{<:Unitful.Energy}, lq_classifier::Vector{<:AbstractFloat}; cut_sigma::Float64=3.0, truncation_sigma::Float64=2.0)

Evaluates the cutoff value for the LQ cut. The function performs a binned gaussian fit on the sidebandsubtracted LQ histogram and evaluates the cutoff value defined at 3σ of the fit.

# Arguments
    * `DEP_µ`:Double escape peak mean
    * `DEP_σ`: Double escape peak sigma
    * `e_cal`: Calibrated energies
    * `lq_classifier`: Inverted cut logic classifier
    
# Keywords
    * `cut_sigma`: Cutoff value
    * `truncation_sigma`: Truncated value

# Returns
    * `result`: NamedTuple of the cutoff value
    * `report`: NamedTuple of the fit result, fit report and temporary histograms
"""
function LQ_cut(
    DEP_µ::Unitful.Energy, DEP_σ::Unitful.Energy, e_cal::Vector{<:Unitful.Energy}, lq_classifier::Vector{<:AbstractFloat}; cut_sigma::Float64=3.0, truncation_sigma::Float64=2.0)

    # Define sidebands
    lq_DEP = lq_classifier[DEP_µ - 4.5 * DEP_σ .< e_cal .< DEP_µ + 4.5 * DEP_σ]
    lq_sb1 = lq_classifier[DEP_µ -  2 * 4.5 * DEP_σ .< e_cal .< DEP_µ - 4.5 * DEP_σ]
    lq_sb2 = lq_classifier[DEP_µ + 4.5 * DEP_σ .< e_cal .< DEP_µ + 2 * 4.5 * DEP_σ]
    
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
    lq_start = lq_prestats.peak_pos - truncation_sigma * lq_prestats.peak_sigma
    lq_stop = lq_prestats.peak_pos + truncation_sigma * lq_prestats.peak_sigma

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
export LQ_cut
