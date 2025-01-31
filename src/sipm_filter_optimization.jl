
"""
    fit_sipm_wl(trig_max_grid::VectorOfVectors{<:Real}, e_grid_wl::StepRangeLen)

Fit the SiPM spectrum for different window lengths and return the optimal window length.

# Arguments
- `trig_max_grid`: grid of trigger maxima for different window lengths
- `e_grid_wl`: range of window lengths to sweep through

# Returns
- `result`: optimal window length and corresponding gain, resolution and position of 1pe peak
- `report`: report with all window lengths and corresponding gains, resolutions and positions of 1pe peaks
"""
function fit_sipm_wl(trig_max_grid::VectorOfVectors{<:Real}, e_grid_wl::StepRangeLen, thresholds::Vector{<:Real}=zeros(length(e_grid_wl));
    min_pe_peak::Int=1, max_pe_peak::Int=5, n_fwhm_noise_cut::Real=2.0, peakfinder_threshold::Real=5.0, initial_max_amp::Real = 50.0, initial_max_bin_width_quantile::Real=0.9999, 
    peakfinder_rtol::Real=0.1, peakfinder_α::Real=0.1, peakfinder_σ::Real=-1.0, 
    min_pe_fit::Real=0.6, max_pe_fit::Real=3.5, Δpe_peak_assignment::Real=0.3)
    
    gain_wl    = Vector{Measurement{Float64}}(undef, length(e_grid_wl))
    res_1pe_wl = Vector{Measurement{Float64}}(undef, length(e_grid_wl))
    pos_1pe_wl = Vector{Measurement{Float64}}(undef, length(e_grid_wl))
    success    = Vector{Bool}(zeros(length(e_grid_wl)))
    reports_simple     = Vector{NamedTuple}(undef, length(e_grid_wl))
    reports_fit        = Vector{NamedTuple}(undef, length(e_grid_wl))

    # for each window lenght, calculate gain, resolution and position of 1pe peak
    Threads.@threads for w in eachindex(e_grid_wl)
        wl = e_grid_wl[w]
        trig_max = filter(isfinite, collect(trig_max_grid[w]))
        threshold = thresholds[w]
        try
            result_simple, report_simple = sipm_simple_calibration(trig_max; initial_min_amp=threshold, initial_max_amp=initial_max_amp, initial_max_bin_width_quantile=initial_max_bin_width_quantile,
                                            min_pe_peak=min_pe_peak, max_pe_peak=max_pe_peak, n_fwhm_noise_cut=n_fwhm_noise_cut, peakfinder_threshold=peakfinder_threshold, 
                                            peakfinder_rtol=peakfinder_rtol, peakfinder_α=peakfinder_α, peakfinder_σ=peakfinder_σ)

            result_fit, report_fit = fit_sipm_spectrum(result_simple.pe_simple_cal, min_pe_fit, max_pe_fit; f_uncal=result_simple.f_simple_uncal, Δpe_peak_assignment=Δpe_peak_assignment)

            # gain_wl[w] = minimum(result_simple.peakpos) - ifelse(threshold == 0.0, result_simple.noisepeakpos, threshold)
            gain_wl[w] = minimum(result_simple.peakpos) - result_simple.noisepeakpos
            res_1pe_wl[w] = first(result_fit.resolutions)
            pos_1pe_wl[w] = first(result_fit.positions)
            reports_simple[w] = report_simple
            reports_fit[w] = report_fit
            success[w] = true
        catch e
            @warn "Failed to process wl: $wl: $e"
        end
    end

    thrs = if all(thresholds .== 0.0) ones(length(e_grid_wl)) else thresholds end
    obj = sqrt.(res_1pe_wl[success]) .* sqrt.(thrs[success]) ./ gain_wl[success]
    wls = collect(e_grid_wl)[success]

    if isempty(obj)
        @error "No valid gain found"
        throw(ErrorException("No valid gain found, could not determine optimal window length"))
    end
    min_obj    = minimum(obj)
    wl_min_obj = wls[findmin(obj)[2]]
    min_res1pe = res_1pe_wl[success][findmin(obj)[2]]
    min_gain   = gain_wl[success][findmin(obj)[2]]
    min_pos1pe = pos_1pe_wl[success][findmin(obj)[2]]
    min_threshold = thresholds[success][findmin(obj)[2]]
    min_report_simple = reports_simple[success][findmin(obj)[2]]
    min_report_fit = reports_fit[success][findmin(obj)[2]]

    # generate result and report
    result = (
        wl = measurement(wl_min_obj, step(e_grid_wl)),
        obj = min_obj,
        res_1pe = min_res1pe,
        gain = min_gain,
        pos_1pe = min_pos1pe,
        threshold = min_threshold
    )
    report = (
        wl = result.wl,
        min_obj = result.obj,
        gain = gain_wl,
        res_1pe = res_1pe_wl,
        pos_1pe = pos_1pe_wl,
        threshold = thresholds[success],
        a_grid_wl_sg = wls,
        obj = obj,
        report_simple = min_report_simple,
        report_fit = min_report_fit,
    )
    return result, report
end
export fit_sipm_wl



"""
    fit_sipm_threshold(thresholds::Vector{<:Real}, min_cut::Real=minimum(thresholds), max_cut::Real=maximum(thresholds); n_bins::Int=-1, relative_cut::Real=0.2, fit_thresholds::Bool=true, uncertainty::Bool=true)

Fit the SiPM threshold spectrum and return the optimal threshold.

# Arguments
- `thresholds`: vector of thresholds
- `min_cut`: minimum threshold
- `max_cut`: maximum threshold
- `n_bins`: number of bins for histogram
- `relative_cut`: relative cut for threshold
- `fit_thresholds`: fit thresholds
- `uncertainty`: calculate uncertainty

# Returns
- `result`: optimal threshold and corresponding gain, resolution and position of 1pe peak
- `report`: report with all thresholds and corresponding gains, resolutions and positions of 1pe peaks
"""
function fit_sipm_threshold(thresholds::Vector{<:Real}, min_cut::Real=minimum(thresholds), max_cut::Real=maximum(thresholds); n_bins::Int=-1, relative_cut::Real=0.2, fit_thresholds::Bool=true, uncertainty::Bool=true)
    # cut out thresholds
    filter!(in(min_cut .. max_cut), thresholds)

    # get bin_width
    h = if n_bins < 1
        fit(Histogram, thresholds, min_cut:get_friedman_diaconis_bin_width(thresholds):max_cut)
    else
        fit(Histogram, thresholds, n_bins)
    end
    # get simple thresholds
    result_simple = (μ_simple = mean(thresholds), σ_simple = std(thresholds))
    
    # fit histogram
    result_trig, report_trig = if fit_thresholds
        # generate cuts for thresholds
        cuts_thres = cut_single_peak(thresholds, min_cut, max_cut; n_bins=n_bins, relative_cut=relative_cut)
        # fit histogram
        fit_binned_trunc_gauss(h, cuts_thres; uncertainty=uncertainty)
    else
        (μ = result_simple.μ_simple, σ = result_simple.σ_simple), h
    end
    # get simple std and mu values
    result = merge(result_trig, result_simple)
    return result, report_trig
end
export fit_sipm_threshold