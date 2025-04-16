"""
    lq_ctc_correction(lq::Vector{<:AbstractFloat}, dt_eff::Vector{<:Unitful.RealOrRealQuantity}, e_cal::Vector{<:Unitful.Energy{<:Real}}, dep_µ::Unitful.AbstractQuantity, dep_σ::Unitful.AbstractQuantity; 
    ctc_dep_edgesigma::Float64=3.0 , ctc_lq_precut_relative_cut::Float64=0.5, lq_outlier_sigma::Float64 = 2.0, ctc_driftime_cutoff_method::Symbol=:percentile, dt_eff_outlier_sigma::Float64 = 2.0, lq_e_corr_expression::Union{String,Symbol}="(lq / e)", dt_eff_expression::Union{String,Symbol}="(qdrift / e)" ,ctc_dt_eff_low_quantile::Float64=0.15, ctc_dt_eff_high_quantile::Float64=0.95, pol_fit_order::Int=1) )

    Perform the drift time correction on the LQ data using the DEP peak. The function cuts outliers in lq and drift time, then performs a polynomial fit on the remaining data. The data is Corrected by subtracting the polynomial fit from the lq data.

# Arguments 
    * `lq`: Energy corrected lq parameter
    * `dt_eff`: Effective drift time
    * `e_cal`: Energy
    * `dep_µ`: Mean of the DEP peak
    * `dep_σ`: Standard deviation of the DEP peak

# Keywords
    * `ctc_dep_edgesigma`: Number of standard deviations used to define the DEP edges
    * `ctc_lq_precut_relative_cut`: Relative cut for cut_single_peak function
    * `ctc_driftime_cutoff_method`: Method used to define the drift time cutoff
    * `lq_outlier_sigma`: Number of standard deviations used to define the lq cutoff
    * `dt_eff_outlier_sigma`: Number of standard deviations used to define the drift time cutoff
    * `lq_e_corr_expression`: Expression for the energy corrected lq classifier 
    * `dt_eff_expression`: Expression for the effective drift time 
    * `ctc_dt_eff_low_quantile`: Lower quantile used to define the drift time cutoff
    * `ctc_dt_eff_high_quantile`: Higher quantile used to define the drift time cutoff
    * `pol_fit_order`: Order of the polynomial fit used for the drift time correction

# Returns
    * `result`: NamedTuple of the function used for the drift time correction
    * `report`: NamedTuple of the histograms used for the fit, the cutoff values and the DEP edges

"""
function lq_ctc_correction(
    lq::Vector{<:AbstractFloat}, dt_eff::Vector{<:Unitful.RealOrRealQuantity}, e_cal::Vector{<:Unitful.Energy{<:Real}}, dep_µ::Unitful.AbstractQuantity, dep_σ::Unitful.AbstractQuantity; 
    ctc_dep_edgesigma::Float64=3.0, ctc_lq_precut_relative_cut::Float64=0.25, lq_outlier_sigma::Float64 = 2.0, ctc_driftime_cutoff_method::Symbol=:percentile, dt_eff_outlier_sigma::Float64 = 2.0, lq_e_corr_expression::Union{String,Symbol}="(lq / e)", dt_eff_expression::Union{String,Symbol}="(qdrift / e)" ,ctc_dt_eff_low_quantile::Float64=0.15, ctc_dt_eff_high_quantile::Float64=0.95, pol_fit_order::Int=1, uncertainty::Bool=false) 

    # calculate DEP edges
    dep_left = dep_µ - ctc_dep_edgesigma * dep_σ
    dep_right = dep_µ + ctc_dep_edgesigma * dep_σ

    # cut data to DEP peak
    dep_finite = (dep_left .< e_cal .< dep_right .&& isfinite.(lq) .&& isfinite.(dt_eff))
    lq_dep = lq[dep_finite]
    dt_eff_dep = ustrip.(dt_eff[dep_finite])
   
    # precut lq data for fit
    lq_precut = cut_single_peak(lq_dep, minimum(lq_dep), quantile(lq_dep, 0.99); relative_cut=ctc_lq_precut_relative_cut)

    # truncated gaussian fit
    lq_result, lq_report = fit_single_trunc_gauss(lq_dep, lq_precut; uncertainty)
    µ_lq = mvalue(lq_result.μ)
    σ_lq = mvalue(lq_result.σ)

    # set cutoff in lq dimension for later fit
    lq_lower = µ_lq - lq_outlier_sigma * σ_lq 
    lq_upper = µ_lq + lq_outlier_sigma * σ_lq 


    # dt_eff_dep cutoff; method dependant on detector type
    
    if ctc_driftime_cutoff_method == :percentile # standard method; can be used for all detectors
        #set cutoff; default at the 15% and 95% percentile
        t_lower = quantile(dt_eff_dep, ctc_dt_eff_low_quantile)
        t_upper = quantile(dt_eff_dep, ctc_dt_eff_high_quantile)
        drift_report = nothing

    elseif ctc_driftime_cutoff_method == :gaussian # can't be used for detectors with double peaks
        
        dt_eff_precut = cut_single_peak(dt_eff_dep, minimum(lq_dep), maximum(lq_dep))
        drift_result, drift_report = fit_single_trunc_gauss(dt_eff_dep, dt_eff_precut; uncertainty)
        µ_t = mvalue(drift_result.μ)
        σ_t = mvalue(drift_result.σ)

        #set cutoff in drift time dimension for later fit
        t_lower = µ_t - dt_eff_outlier_sigma * σ_t
        t_upper = µ_t + dt_eff_outlier_sigma * σ_t

    elseif ctc_driftime_cutoff_method == :double_gaussian # can be used for detectors with double peaks; not optimized yet
        #create histogram for drift time
        ideal_length = get_number_of_bins(dt_eff_dep)
        drift_prehist = fit(Histogram, dt_eff_dep, range(minimum(dt_eff_dep), stop=maximum(dt_eff_dep), length=ideal_length))
        drift_prestats = estimate_single_peak_stats(drift_prehist)

        #fit histogram with double gaussian
        drift_result, drift_report = fit_binned_double_gauss(drift_prehist, drift_prestats; uncertainty)
        
        #set cutoff at the x-value where the fit function is 10% of its maximum value
        x_values = -1000:0.5:5000  
        max_value = maximum(drift_report.f_fit.(x_values))
        threshold = 0.1 * max_value

        g(x) = drift_report.f_fit(x) - threshold
        x_at_threshold = find_zeros(g, minimum(x_values), maximum(x_values))

        t_lower = minimum(x_at_threshold)
        t_upper = maximum(x_at_threshold)
    else
        throw(ArgumentError("Drift time cutoff method $ctc_driftime_cutoff_method not supported"))
    end

    # store cutoff values in box to return later    
    box = (;lq_lower, lq_upper, t_lower, t_upper)

    # cut data according to cutoff values
    lq_cut = lq_dep[lq_lower .< lq_dep .< lq_upper .&& t_lower .< dt_eff_dep .< t_upper]
    t_cut = dt_eff_dep[lq_lower .< lq_dep .< lq_upper .&& t_lower .< dt_eff_dep .< t_upper]

    # polynomial fit
    result_µ, report_µ = chi2fit(pol_fit_order, t_cut, lq_cut; uncertainty)
    par = mvalue(result_µ.par)
    pol_fit_func = report_µ.f_fit

    # property function for drift time correction
    lq_class_func = "$lq_e_corr_expression - " * join(["$(par[i]) * ($dt_eff_expression)^$(i-1)" for i in eachindex(par)], " - ")
    lq_class_func_generic = "lq / e  - (slope * qdrift / e + y_inter)"

    # create result and report
    result = (
    func = lq_class_func,
    func_generic = lq_class_func_generic,
    fit_result = result_µ,
    box_constraints = box,
    )

    report = (
    lq_report = lq_report, 
    drift_report = drift_report,
    lq_box = box,
    drift_time_func = pol_fit_func,
    dep_left = dep_left,
    dep_right = dep_right,
    )
    

    return result, report
end
export lq_ctc_correction

"""
    lq_cut(dep_µ::Unitful.Energy, dep_σ::Unitful.Energy, e_cal::Vector{<:Unitful.Energy}, lq_classifier::Vector{<:AbstractFloat}; cut_sigma::Float64=3.0, dep_sideband_sigma::Float64=4.5, cut_truncation_sigma::Float64=2.0) )

    Evaluates the cutoff value for the LQ cut. The function performs a binned gaussian fit on the sidebandsubtracted LQ histogram and evaluates the cutoff value difined at 3σ of the fit.

# Arguments
    * `dep_µ`: Mean of the DEP peak
    * `dep_σ`: Standard deviation of the DEP peak
    * `e_cal`: Energy
    * `lq_classifier`: LQ classifier

# Keywords
    * `cut_sigma`: Number of standard deviations used to define the final cutoff value
    * `dep_sideband_sigma`: Number of standard deviations used to define the sideband edges
    * `cut_truncation_sigma`: Number of standard deviations used for the precut of sideband subtracted histogram

# Returns
    * `result`: NamedTuple of the cutoff value
    * `report`: NamedTuple of the fit result, fit report and temporary histograms

"""
function lq_cut(
    dep_µ::Unitful.Energy, dep_σ::Unitful.Energy, e_cal::Vector{<:Unitful.Energy}, lq_classifier::Vector{<:AbstractFloat}; cut_sigma::Float64=3.0, dep_sideband_sigma::Float64=4.5, cut_truncation_sigma::Float64=3.5, uncertainty::Bool=true, lq_class_expression::Union{String,Symbol}="lq / e  - (slope * qdrift / e + y_inter)"
    )

    # define sidebands; different for low and high energy resolution detectors to avoid sb reaching into 212-Bi FEP
    DEP_edge_left  = dep_µ - dep_sideband_sigma * dep_σ
    DEP_edge_right = dep_µ + dep_sideband_sigma * dep_σ

    if dep_σ < 2.0u"keV"
        sb1_edge = dep_µ - 2 * dep_sideband_sigma * dep_σ
        sb2_edge = dep_µ + 2 * dep_sideband_sigma * dep_σ  

        lq_dep = lq_classifier[DEP_edge_left .< e_cal .< DEP_edge_right]
        lq_sb1 = lq_classifier[sb1_edge .< e_cal .< DEP_edge_left]
        lq_sb2 = lq_classifier[DEP_edge_right .< e_cal .< sb2_edge]
    else
        sb1_edge = dep_µ - 2 * dep_sideband_sigma * dep_σ  
        sb2_edge = dep_µ - 3 * dep_sideband_sigma * dep_σ

        lq_dep = lq_classifier[DEP_edge_left .< e_cal .< DEP_edge_right]
        lq_sb1 = lq_classifier[sb1_edge .< e_cal .< DEP_edge_left]
        lq_sb2 = lq_classifier[sb2_edge .< e_cal .< sb1_edge]
    end

    # save edges for crosschecks
    edges_for_crosschecks = (;DEP_edge_left, DEP_edge_right, sb1_edge, sb2_edge)

    # generate values for histogram edges
    combined = filter(isfinite,[lq_dep; lq_sb1; lq_sb2])
    ideal_bin_width = get_friedman_diaconis_bin_width(combined)
    edges = range(start=minimum(combined), stop=maximum(combined), step=ideal_bin_width)

    # create histograms with the same bin edges
    hist_dep = fit(Histogram, lq_dep, edges)
    hist_sb1 = fit(Histogram, lq_sb1, edges)
    hist_sb2 = fit(Histogram, lq_sb2, edges)
    
    # subtract histograms
    weights_subtracted = hist_dep.weights .- hist_sb1.weights .- hist_sb2.weights
    hist_subtracted = Histogram(edges, weights_subtracted)

    # replace negative bins with 0
    weights_corrected = max.(weights_subtracted, 0)
    hist_corrected = Histogram(edges, weights_corrected)
    
    # create a named tuple of histograms for crosschecks
    temp_hists = (;hist_dep, hist_sb1, hist_sb2, hist_subtracted, hist_corrected)

    # get truncate values for fit; needed if outliers are present after in sideband subtracted histogram
    lq_prestats = estimate_single_peak_stats(hist_corrected)
    lq_start = lq_prestats.peak_pos - cut_truncation_sigma * lq_prestats.peak_sigma
    lq_stop = lq_prestats.peak_pos + cut_truncation_sigma * lq_prestats.peak_sigma

    # fit the sideband subtracted histogram
    fit_result, fit_report = fit_binned_trunc_gauss(hist_corrected, (low=lq_start, high=lq_stop, max=NaN); uncertainty)

    # final cutoff value defined by "cut_sigma"
    cut_3σ = fit_result.μ + cut_sigma * fit_result.σ

    # normalize lq classifier
    lq_norm_func = " ( $(lq_class_expression)  - $(mvalue(fit_result.μ)) ) / ( $(mvalue(fit_result.σ)) )"

    result = (
        cut = cut_3σ,
        cut_fit_result = fit_result,
        func = lq_norm_func,
    )

    report = (
        cut = cut_3σ,
        fit_result = fit_result,
        temp_hists = temp_hists,
        fit_report = fit_report,
        dep_σ = dep_σ,
        edges = edges_for_crosschecks,
    )

    return result, report
end
export lq_cut
