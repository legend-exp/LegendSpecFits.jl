function lq_drift_time_correction(lq_norm::Vector{Float64}, tdrift, e_cal)

    #DEP needs to be changed to be dynamic and use the DEP peak fit results from e calibration, then e_cal can be removed from fucntion arguments; could not do this yet, because i did not copy the DEP peak fit results to the TUM server
    lq_DEP_dt = lq_norm[1589u"keV" .< e_cal .< 1596u"keV"]
    t_tcal = ustrip.(tdrift[1589u"keV" .< e_cal .< 1596u"keV"])

    #lq box
    #sort array to exclude outliers
    sort_lq = sort(lq_DEP_dt)
    low_cut = Int(round(length(sort_lq) *0.02)) # cut lower 2%
    high_cut = Int(round(length(sort_lq) *0.995)) # cut upper 0.5%
    lq_prehist = fit(Histogram, lq_DEP_dt, range(sort_lq[low_cut], sort_lq[high_cut], length=100))
    lq_prestats = estimate_single_peak_stats(lq_prehist)
    lq_start = lq_prestats.peak_pos - 3*lq_prestats.peak_sigma
    lq_stop = lq_prestats.peak_pos + 3*lq_prestats.peak_sigma

    lq_edges = range(lq_start, stop=lq_stop, length=51) 
    lq_hist_DEP = fit(Histogram, lq_DEP_dt, lq_edges)

    lq_DEP_stats = estimate_single_peak_stats(lq_hist_DEP)
    lq_result, lq_report = LegendSpecFits.fit_binned_gauss(lq_hist_DEP, lq_DEP_stats)
    µ_lq = mvalue(lq_result.μ)
    σ_lq = mvalue(lq_result.σ)

    #t_tcal box

    drift_prehist = fit(Histogram, t_tcal, range(minimum(t_tcal), stop=maximum(t_tcal), length=100))
    drift_prestats = estimate_single_peak_stats(drift_prehist)
    drift_start = drift_prestats.peak_pos - 3*drift_prestats.peak_sigma
    drift_stop = drift_prestats.peak_pos + 3*drift_prestats.peak_sigma
    
    drift_edges = range(drift_start, stop=drift_stop, length=71)
    drift_hist_DEP = fit(Histogram, t_tcal, drift_edges)
    
    drift_DEP_stats = estimate_single_peak_stats(drift_hist_DEP)
    drift_result, drift_report = LegendSpecFits.fit_binned_gauss(drift_hist_DEP, drift_DEP_stats)
    µ_t = mvalue(drift_result.μ)
    σ_t = mvalue(drift_result.σ)

    #create box 
    box = (lq_lower = µ_lq - 2 * σ_lq, 
    lq_upper = µ_lq + 2 * σ_lq, 
    t_lower = µ_t - 2 * σ_t, 
    t_upper = µ_t + 2 * σ_t)

    lq_box = lq_DEP_dt[box.lq_lower .< lq_DEP_dt .< box.lq_upper .&& box.t_lower .< t_tcal .< box.t_upper]
    t_box = t_tcal[box.lq_lower .< lq_DEP_dt .< box.lq_upper .&& box.t_lower .< t_tcal .< box.t_upper]

    #linear fit
    result_µ, report_µ = chi2fit(1, t_box, lq_box; uncertainty=true)
    parameters = mvalue(result_µ.par)
    drift_time_func(x) = parameters[2] .* x .+ parameters[1] 

    lq_classifier = lq_norm .- drift_time_func(ustrip.(tdrift))

    result = (
    lq_classifier = lq_classifier, 
    lq_box = box, 
    drift_time_func = drift_time_func
    )

    report = (
    lq_prehist = lq_prehist, 
    lq_report = lq_report, 
    drift_prehist = drift_prehist, 
    drift_report = drift_report
    )
    

    return result, report
end
export lq_drift_time_correction



function LQ_cut(DEP_µ, DEP_σ, e_cal, lq_norm)
    # Define sidebands
    lq_DEP = lq_norm[DEP_µ - 4.5 * DEP_σ .< e_cal .< DEP_µ + 4.5 * DEP_σ]
    lq_sb1 = lq_norm[DEP_µ -  2 * 4.5 * DEP_σ .< e_cal .< DEP_µ - 4.5 * DEP_σ]
    lq_sb2 = lq_norm[DEP_µ + 4.5 * DEP_σ .< e_cal .< DEP_µ + 2 * 4.5 * DEP_σ]
    
    # Generate values for histogram edges
    #exclude outliers
    sort_lq = sort(lq_DEP)
    low_cut = Int(round(length(sort_lq) *0.02)) # cut lower 2%
    high_cut = Int(round(length(sort_lq) *0.995)) # cut upper 0.5%
    prehist = fit(Histogram, lq_DEP, range(sort_lq[low_cut], sort_lq[high_cut], length=100))
    prestats = estimate_single_peak_stats(prehist)
    start = prestats.peak_pos - 3*prestats.peak_sigma
    stop = prestats.peak_pos + 3*prestats.peak_sigma
    edges = range(start, stop=stop, length=71) 

    # Create histograms with the same bin edges
    hist_DEP = fit(Histogram, lq_DEP, edges)
    hist_sb1 = fit(Histogram, lq_sb1, edges)
    hist_sb2 = fit(Histogram, lq_sb2, edges)
    
    # Subtract histograms
    weights_subtracted = hist_DEP.weights - hist_sb1.weights - hist_sb2.weights
    hist_subtracted = Histogram(edges, weights_subtracted)

    # Replace negative bins with 0
    weights_corrected = max.(weights_subtracted, 0)
    hist_corrected = Histogram(edges, weights_corrected)
    
    # Create a named tuple of temporary histograms for crosschecks
    temp_hists = (prehist = prehist, hist_DEP = hist_DEP, hist_sb1 = hist_sb1, hist_sb2 = hist_sb2, hist_subtracted = hist_subtracted, hist_corrected = hist_corrected)

    lq_DEP_stats = estimate_single_peak_stats(hist_corrected)
    fit_result, fit_report = fit_binned_gauss(hist_corrected, lq_DEP_stats)
    cut_3σ = fit_result.μ - 3fit_result.σ
    
    report = (
        temp_hists = temp_hists,
        fit_report = fit_report,
    )

    result = (
        cut = cut_3σ,
        fit_result = fit_result,
    )

    return result, report
end
export LQ_cut