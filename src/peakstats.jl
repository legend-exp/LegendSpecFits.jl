"""
    estimate_single_peak_stats(args...; calib_type::Symbol=:th228)

Estimate statistics/parameters for a single peak in the given histogram `h`.

`h` must only contain a single peak. The peak should have a Gaussian-like
shape.
`calib_type` specifies the calibration type. Currently `:th228`, `:psd` and `:simple` is implemented..

# Arguments
    * `h`: Histogram data containing a single peak

# Keywords
    * `calib_type`: Calibration type

# Returns 
`NamedTuple` with the fields
    * `peak_pos`: estimated position of the peak (in the middle of the peak)
    * `peak_fwhm`: full width at half maximum (FWHM) of the peak
    * `peak_sigma`: estimated standard deviation of the peak
    * `peak_counts`: estimated number of counts in the peak
    * `mean_background`: estimated mean background value
    * `mean_background_step`: estimated mean background step value
    * `mean_background_std`: estimated mean background standard deviation

 
"""
function estimate_single_peak_stats(args...; calib_type::Symbol=:th228)
    if calib_type == :th228
        return estimate_single_peak_stats_th228(args...)
    elseif calib_type == :psd
        return estimate_single_peak_stats_psd(args...)
    elseif calib_type == :simple
        return estimate_single_peak_stats_simple(args...)
    else
        error("Calibration type not supported")
    end
end
export estimate_single_peak_stats

"""
    _get_hist_peakpos_fwhm(E::AbstractVector, W::Vector{<:Real})

Get the histogram's peak position at full width, half max.

# Arguments
    * `E`: 
    * `W`:

# Returns
    * Histogram components

TO DO: Arguments, description
"""

function _get_hist_peakpos_fwhm(E::AbstractVector, W::Vector{<:Real})
    peak_amplitude, peak_idx = findmax(W)
    fwhm_idx_left = findfirst(w -> w >= (first(W) + peak_amplitude) / 2, W)
    fwhm_idx_right = findlast(w -> w >= (last(W) + peak_amplitude) / 2, W)
    peak_max_pos = (E[peak_idx] + E[peak_idx+1]) / 2
    peak_mid_pos = (E[fwhm_idx_right] + E[fwhm_idx_left]) / 2
    peak_pos = (peak_max_pos + peak_mid_pos) / 2.0
    peak_fwhm = (E[fwhm_idx_right] - E[fwhm_idx_left]) / 1.0
    peak_sigma = peak_fwhm * inv(2*√(2log(2)))
    return peak_fwhm, peak_sigma, peak_pos, fwhm_idx_left, fwhm_idx_right
end
_get_hist_peakpos_fwhm(h::Histogram) = _get_hist_peakpos_fwhm(first(h.edges), h.weights)

"""
    _get_hist_fwqm(E::AbstractVector, W::Vector{<:Real})

Gets the the full width at quarter max of a histogram.

# Arguments
    * `E`:
    * `W`:

TO DO: arguments, description
"""

function _get_hist_fwqm(E::AbstractVector, W::Vector{<:Real})
    peak_amplitude, peak_idx = findmax(W)
    fwqm_idx_left = findfirst(w -> w >= (first(W) + peak_amplitude) / 4, W)
    fwqm_idx_right = findlast(w -> w >= (last(W) + peak_amplitude) / 4, W)
    peak_fwqm = (E[fwqm_idx_right] - E[fwqm_idx_left]) / 1.0
    peak_sigma = peak_fwqm * inv(2*√(2log(4)))
    peak_fwhm  = peak_sigma * 2*√(2log(2))
    return peak_fwqm, peak_sigma, peak_fwhm, fwqm_idx_left, fwqm_idx_right
end

"""
Same function as below, but a different method. 

"""

function estimate_single_peak_stats_th228(e::Vector{<:Real}, bin_width::Real)
    h = fit(Histogram, e, minimum(e):bin_width:maximum(e))
    peak_fwhm, _, _, _, _ = _get_hist_peakpos_fwhm(h)
    while peak_fwhm == 0.0
        bin_width /= 2
        @debug "Decrease bin width for peak estimation to $bin_width"
        h = fit(Histogram, e, minimum(e):bin_width:maximum(e))
        peak_fwhm, _, _, _, _ = _get_hist_peakpos_fwhm(h)
    end
    estimate_single_peak_stats_th228(h)
end


"""
    estimate_single_peak_stats_th228(h::Histogram{T}) where T<:Real

Estimate statistics/parameters for a single peak in the given histogram `h` for Th228 calibration.

# Arguments
    * `h`: Histogram data

# Returns 
`NamedTuple` with the fields
    * `peak_pos`: estimated position of the peak (in the middle of the peak)
    * `peak_fwhm`: full width at half maximum (FWHM) of the peak
    * `peak_sigma`: estimated standard deviation of the peak
    * `peak_counts`: estimated number of counts in the peak
    * `mean_background`: estimated mean background value
    * `mean_background_step`: estimated mean background step value
    * `mean_background_std`: estimated mean background standard deviation
"""
function estimate_single_peak_stats_th228(h::Histogram{T}) where T<:Real
    W = h.weights
    E = first(h.edges)
    bin_width = step(E)
    peak_amplitude, peak_idx = findmax(W)
    peak_fwhm, peak_sigma, peak_pos, fwhm_idx_left, fwhm_idx_right = _get_hist_peakpos_fwhm(E, W)
    peak_fwqm = NaN
    # make sure that peakstats have non-zero sigma and fwhm values to prevent fit priors from being zero
    if peak_fwhm == 0
        peak_fwqm, peak_sigma, peak_fwhm, fwqm_idx_left, fwqm_idx_right = _get_hist_fwqm(E, W)
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
    # mean_background_err = 0.5*(std(view(W, 1:idx_bkg_left))/sqrt(length(1:idx_bkg_left)) + std(view(W, idx_bkg_right:length(W)))/sqrt(length(idx_bkg_right:length(W))) ) / bin_width # error of the mean 
    
    # sanity checks
    mean_background = ifelse(mean_background == 0, 0.01, mean_background)
    mean_background_step = ifelse(mean_background_step < 1e-2, 1e-2, mean_background_step)
    mean_background_std = ifelse(!isfinite(mean_background_std) || mean_background_std == 0, sqrt(mean_background), mean_background_std)
    peak_counts = inv(0.761) * (sum(view(W,fwhm_idx_left:fwhm_idx_right)) - mean_background * peak_fwhm)
    if peak_counts <= 0.0
        peak_counts = inv(0.761) * sum(view(W,fwhm_idx_left:fwhm_idx_right))
    end
    if peak_counts <= 0.0 && !isnan(peak_fwqm)
        peak_counts = inv(0.904) * (sum(view(W,fwqm_idx_left:fwqm_idx_right)) - mean_background * peak_fwqm)
    end
    if peak_counts <= 0.0 && !isnan(peak_fwqm)
        peak_counts = inv(0.904) * sum(view(W,fwqm_idx_left:fwqm_idx_right))
    end
    if peak_counts <= 0.0
        if peak_idx <= 2
            peak_counts = inv(0.761) * sum(view(W,peak_idx:peak_idx + 4))
        elseif peak_idx >= length(W) - 1
            peak_counts = inv(0.761) * sum(view(W,peak_idx - 4:peak_idx))
        else
            peak_counts = inv(0.761) * sum(view(W,peak_idx - 2:peak_idx + 2))
        end
    end
    if peak_counts <= 0.0
        peak_counts = 2.0
    end
    (
        peak_pos = peak_pos, 
        peak_fwhm = peak_fwhm,
        peak_sigma = peak_sigma, 
        peak_counts = peak_counts,
        bin_width = bin_width,
        mean_background = mean_background,
        mean_background_step = mean_background_step,
        mean_background_std = mean_background_std,
    )
end

"""
    estimate_single_peak_stats_psd(h::Histogram{T}) where T<:Real

Estimate statistics/parameters for a single peak in the given histogram `h` for psd calibration.

# Arguments
    * `h`: Histogram data

# Returns
    * `peak_pos`: estimated position of the peak (in the middle of the peak)
    * `peak_fwhm`: full width at half maximum (FWHM) of the peak
    * `peak_sigma`: estimated standard deviation of the peak
    * `peak_counts`: estimated number of counts in the peak
    * `mean_background`: estimated mean background value
"""


function estimate_single_peak_stats_psd(h::Histogram{T}) where T<:Real
    W = h.weights
    E = first(h.edges)
    peak_amplitude, peak_idx = findmax(W)
    fwhm_idx_left = findfirst(w -> w >= (first(W) + peak_amplitude) /2, W)
    fwhm_idx_right = findlast(w -> w >= (last(W) + peak_amplitude) /2, W)
    peak_max_pos = (E[peak_idx] + E[peak_idx+1]) / 2
    peak_mid_pos = (E[fwhm_idx_right] + E[fwhm_idx_left]) / 2
    peak_pos = (peak_max_pos + peak_mid_pos) / 2.0
    peak_fwhm = (E[fwhm_idx_right] - E[fwhm_idx_left]) / 1.0
    peak_sigma = peak_fwhm * inv(2*√(2log(2)))
    # make sure that peakstats have non-zero sigma and fwhm values to prevent fit priors from being zero
    if peak_fwhm == 0
        fwqm_idx_left = findfirst(w -> w >= (first(W) + peak_amplitude) / 4, W)
        fwqm_idx_right = findlast(w -> w >= (last(W) + peak_amplitude) / 4, W)
        peak_fwqm = (E[fwqm_idx_right] - E[fwqm_idx_left]) / 1.0
        peak_sigma = peak_fwqm * inv(2*√(2log(4)))
        peak_fwhm  = peak_sigma * 2*√(2log(2))
    end
    three_sigma_idx_left = findfirst(e -> e >= peak_pos - 3*peak_sigma, E)
    mean_background = convert(typeof(peak_pos), (sum(view(W, 1:three_sigma_idx_left))))
    mean_background = ifelse(mean_background == 0.0, 100.0, mean_background)
    peak_counts = 2*sum(view(W,peak_idx:lastindex(W)))

    (
        peak_pos = peak_pos, 
        peak_fwhm = peak_fwhm,
        peak_sigma = peak_sigma, 
        peak_counts = peak_counts, 
        mean_background = mean_background
    )
end

"""
    estimate_single_peak_stats_simple(h::Histogram{T}) where T<:Real

Estimate statistics/parameters for a single peak in the given histogram `h` for a simple calibration.

# Arguments
    * `h`: Histogram data

# Returns
    * `peak_pos`: estimated position of the peak (in the middle of the peak)
    * `peak_fwhm`: full width at half maximum (FWHM) of the peak
    * `peak_sigma`: estimated standard deviation of the peak
    * `peak_counts`: estimated number of counts in the peak
"""

function estimate_single_peak_stats_simple(h::Histogram{T}) where T<:Real
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

    peak_counts = inv(0.761) * (sum(view(W,fwhm_idx_left:fwhm_idx_right)))
    peak_counts = ifelse(peak_counts < 0.0, inv(0.761) * sum(view(W,fwhm_idx_left:fwhm_idx_right)), peak_counts)
    if !isnan(peak_fwqm)
        peak_counts = inv(0.904) * (sum(view(W,fwqm_idx_left:fwqm_idx_right)))
        peak_counts = ifelse(peak_counts < 0.0, inv(0.904) * sum(view(W,fwqm_idx_left:fwqm_idx_right)), peak_counts)
    end
    (
        peak_pos = peak_pos, 
        peak_fwhm = peak_fwhm,
        peak_sigma = peak_sigma, 
        peak_counts = peak_counts, 
    )
end