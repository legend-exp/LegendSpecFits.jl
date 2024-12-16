"""
    sipm_simple_calibration(pe_uncal::Array)

Perform a simple calibration for the uncalibrated p.e. spectrum array `pe_uncal`
using just the 1 p.e. and 2 p.e. peak positions estimated by a peakfinder.

Inputs:
    * `pe_uncal`: array of uncalibrated peak amplitudes
kwargs:
    * `initial_min_amp`: uncalibrated amplitude value as a left boundary to build the uncalibrated histogram where the peak search is performed on.
                        For the peak search with noise peak, this value is consecutively increased i.o.t exclude the noise peak from the histogram.
    * `initial_max_quantile`: quantile of the uncalibrated amplitude array to used as right boundary to build the uncalibrated histogram
    * `peakfinder_σ`: sigma value in number of bins for peakfinder
    * `peakfinder_threshold`: threshold value for peakfinder

Returns 
    * `pe_simple_cal`: array of the calibrated pe array with the simple calibration
    * `func`: function to use for the calibration (`pe_simple_cal = pe_uncal .* c .+ offset`)
    * `c`: calibration factor
    * `offset`: calibration offset 
    * `peakpos`: 1 p.e. and 2 p.e. peak positions in uncalibrated amplitude
    * `peakpos_cal`: 1 p.e. and 2 p.e. peak positions in calibrated amplitude
    * `h_uncal`: histogram of the uncalibrated pe array
    * `h_calsimple`: histogram of the calibrated pe array with the simple calibration
"""
function sipm_simple_calibration end
export sipm_simple_calibration

function sipm_simple_calibration(pe_uncal::Vector{<:Real};
                                kwargs...)
    
    h_uncal, peakpos = find_peaks(pe_uncal; kwargs...)

    # simple calibration
    sort!(peakpos)
    gain = peakpos[2] - peakpos[1]
    c = 1/gain
    offset = - (peakpos[1] * c - 1) 

    f_simple_calib = x -> x .* c .+ offset
    f_simple_uncal = x -> (x .- offset) ./ c
    pe_simple_cal = pe_uncal .* c .+ offset
    peakpos_cal = peakpos .* c .+ offset

    bin_width_cal = get_friedman_diaconis_bin_width(filter(in(0.5..1.5), pe_simple_cal))
    bin_width_uncal = get_friedman_diaconis_bin_width(filter(in( (0.5 - offset) / c .. (1.5 - offset) / c), pe_simple_cal))

    h_calsimple = fit(Histogram, pe_simple_cal, 0.0:bin_width_cal:6.0)
    h_uncal = fit(Histogram, pe_uncal, 0.0:bin_width_uncal:(6.0 - offset) / c)

    result = (
        pe_simple_cal = pe_simple_cal,
        peakpos = peakpos,
        f_simple_calib = f_simple_calib,
        f_simple_uncal = f_simple_uncal,
        c = c,
        offset = offset
    )
    report = (
        peakpos = peakpos,
        peakpos_cal = peakpos_cal,
        h_uncal = h_uncal, 
        h_calsimple = h_calsimple
    )
    return result, report
end


function find_peaks(
    amps::Vector{<:Real}; initial_min_amp::Real=1.0, initial_max_quantile::Real=0.99, 
    peakfinder_σ::Real=2.0, peakfinder_threshold::Real=10.0
)
    # Start with a big window where the noise peak is included
    min_amp = initial_min_amp
    max_quantile = initial_max_quantile
    max_amp = quantile(amps, max_quantile)
    bin_width = get_friedman_diaconis_bin_width(filter(in(min_amp..quantile(amps, 0.9)), amps))

    # Initial peak search
    h_uncal = fit(Histogram, amps, min_amp:bin_width:max_amp)
    h_decon, peakpos = peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=false, threshold=peakfinder_threshold)

    # Ensure at least 2 peaks
    num_peaks = length(peakpos)
    if num_peaks == 0
        error("No peaks found.")
    end

    # Determine the 1 p.e. peak position based on the assumption that it is the highest
    peakpos_idxs = StatsBase.binindex.(Ref(h_decon), peakpos)
    cts_peakpos = h_decon.weights[peakpos_idxs]
    first_pe_peak_pos = peakpos[argmax(cts_peakpos)]

    # Remove all peaks with x vals < x pos of 1p.e. peak
    filter!(x -> x >= first_pe_peak_pos, peakpos)
    num_peaks = length(peakpos)

    # while less than two peaks found, or second peakpos smaller than 1 pe peakpos, or gain smaller than 1 (peaks too close)
    while num_peaks < 2 || peakpos[2] <= first_pe_peak_pos || (peakpos[2] - peakpos[1]) <= 1.0
        # Adjust σ and recheck peaks
        if peakfinder_σ < 25.0
            @debug "Increasing peakfinder_σ: $peakfinder_σ"
            peakfinder_σ += 0.5
        else
            # If σ can't increase further, reduce threshold
            @debug "Adjusting peakfinder_threshold: $peakfinder_threshold"
            peakfinder_threshold -= 1.0
            peakfinder_σ = 2.0  # Reset σ for new threshold

            # Safety check to avoid lowering threshold too much
            if peakfinder_threshold < 2.0
                throw(ErrorException("Unable to find two peaks within reasonable quantile range."))
            end
        end

        # Find peaks with updated parameters
        h_decon, peakpos = peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)
        filter!(x -> x >= first_pe_peak_pos, peakpos)
        num_peaks = length(peakpos)

        # Safety check to avoid infinite loops
        if peakfinder_σ >= 10.0 && peakfinder_threshold < 2.0
            throw(ErrorException("Unable to find two peaks within reasonable quantile range."))
        end
    end

    return h_decon, peakpos
end