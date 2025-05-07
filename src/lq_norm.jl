"""
    lq_norm(dep_µ::Unitful.Energy, dep_σ::Unitful.Energy, e_cal::Vector{<:Unitful.Energy}, lq_classifier::Vector{<:AbstractFloat}; dep_sideband_sigma::Float64=4.5, cut_truncation_sigma::Float64=2.0,  uncertainty::Bool=true, lq_class_expression::Union{String,Symbol}="lq / e  - (slope * qdrift / e + y_inter)" )

    Performs normalization of the charge-trapping-corrected LQ classifier using the double-escape peak (DEP) region of the Th calibration. It subtracts sidebands around the DEP, fits a truncated Gaussian to the resulting histogram, and generates a normalization expression so that a numerical value of one corresponds to one standard deviation (σ) of the fitted Gaussian.

# Arguments
    * `dep_µ`: Mean of the DEP peak
    * `dep_σ`: Standard deviation of the DEP peak
    * `e_cal`: Vector of Energy values
    * `lq_classifier`: LQ classifier (typically charge-trapping-corrected)

# Keywords
    * `dep_sideband_sigma`: Number of standard deviations used to define the sideband edges
    * `cut_truncation_sigma`: Number of standard deviations used for the precut of sideband subtracted histogram
    * `uncertainty`: Boolean flag to include uncertainty in the fit (default: true)
    * `lq_class_expression`: Expression for the used LQ classifier

# Returns
    * `result`: NamedTuple of the fit result and normalization function
    * `report`: NamedTuple of the fit result, fit report and temporary histograms

"""
function lq_norm(
    dep_µ::Unitful.Energy, dep_σ::Unitful.Energy, e_cal::Vector{<:Unitful.Energy}, lq_classifier::Vector{<:AbstractFloat}; dep_sideband_sigma::Float64=4.5, cut_truncation_sigma::Float64=3.5, uncertainty::Bool=true, lq_class_expression::Union{String,Symbol}="lq / e  - (slope * qdrift / e + y_inter)"
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

    # normalize lq classifier
    lq_norm_func = " ( ($(lq_class_expression))  - $(mvalue(fit_result.μ)) ) / ( $(mvalue(fit_result.σ)) )"

    result = (
        fit_result = fit_result,
        func = lq_norm_func,
    )

    report = (
        fit_result = fit_result,
        temp_hists = temp_hists,
        fit_report = fit_report,
        dep_σ = dep_σ,
        edges = edges_for_crosschecks,
    )

    return result, report
end
export lq_norm
