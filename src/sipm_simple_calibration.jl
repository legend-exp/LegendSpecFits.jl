"""
    sipm_simple_calibration(pe_uncal::Array)

Perform a simple calibration for the uncalibrated p.e. spectrum array `pe_uncal`
using just the 1 p.e. and 2 p.e. peak positions estimated by a peakfinder.

Inputs:
    * `pe_uncal`: array of uncalibrated peak amplitudes
kwargs:
    * `expect_noise_peak`: bool value stating whether noise peak exists in the uncalibrated spectrum or not
    * `initial_min_amp`: uncalibrated amplitude value as a left boundary to build the uncalibrated histogram where the peak search is performed on.
                        For the peak search with noise peak, this value is consecutively increased i.o.t exclude the noise peak from the histogram.
    * `initial_max_quantile`: quantile of the uncalibrated amplitude array to used as right boundary to build the uncalibrated histogram
    * `step_size_min_amp`: only for if noise peak exists, this gives the step size for the minimal amplitude increase, to exclude the noise peak from the uncalibrated histogram eventually
    * `peakfinder_σ`: sigma value in number of bins for peakfinder
    * `peakfinder_threshold`: threshold value for peakfinder

Returns 
    * `pe_simple_cal`: array of the calibrated pe array with the simple calibration
    * `func`: function to use for the calibration
    * `c`: calibration factor
    * `peakpos`: 1 p.e. and 2 p.e. peak positions in uncalibrated amplitude
    * `h_uncal`: histogram of the uncalibrated pe array
    * `h_calsimple`: histogram of the calibrated pe array with the simple calibration
"""
function sipm_simple_calibration end
export sipm_simple_calibration

function sipm_simple_calibration(pe_uncal::Vector{<:Real}; 
                                expect_noise_peak::Bool=false, 
                                kwargs...)
    
    h_uncal, peakpos = if expect_noise_peak
        find_peaks_noise_peak_exists(pe_uncal; kwargs...)
    else
        find_peaks(pe_uncal; kwargs...)
    end

    # simple calibration
    sort!(peakpos)
    gain = peakpos[2] - peakpos[1]
    c = 1/gain
    offset = - (peakpos[1] * c - 1) 

    f_simple_calib = Base.Fix1(*, c)
    pe_simple_cal = pe_uncal .* c .+ offset
    peakpos_cal = peakpos .* c .+ offset

    bin_width_cal = get_friedman_diaconis_bin_width(filter(in(0.5..1.5), pe_simple_cal))
    bin_width_uncal = get_friedman_diaconis_bin_width(filter(in( (0.5 - offset) / c .. (1.5 - offset) / c), pe_simple_cal))

    h_calsimple = fit(Histogram, pe_simple_cal, 0.0:bin_width_cal:6.0)
    h_uncal = fit(Histogram, pe_uncal, 0.0:bin_width_uncal:(6.0 - offset) / c)

    result = (
        pe_simple_cal = pe_simple_cal,
        peakpos = peakpos,
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



# search the 1 p.e. and 2 p.e. peak, noise peak exists
function find_peaks_noise_peak_exists(
    amps::Vector{<:Real}; initial_min_amp::Real=1.0, initial_max_quantile::Real=0.99, 
    step_size_min_amp::Real=1.0, peakfinder_σ::Real=2.0, peakfinder_threshold::Real=10.0
)
    # Start with a big window where the noise peak is included
    min_amp = initial_min_amp
    max_quantile = initial_max_quantile
    max_amp = quantile(amps, max_quantile)
    bin_width = get_friedman_diaconis_bin_width(filter(in(quantile(amps, 0.01)..quantile(amps, 0.9)), amps))

    # Initial peak search
    h_uncal = fit(Histogram, amps, min_amp:bin_width:max_amp)
    h_decon, peakpos = peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)

    num_peaks = length(peakpos)

    # Determine the noise peak position
    peakpos_idxs = StatsBase.binindex.(Ref(h_decon), peakpos)
    cts_peakpos = h_decon.weights[peakpos_idxs]
    noise_peak_pos = peakpos[argmax(cts_peakpos)]

    # helper function
    function is_within_range(x0, x_list, range=0.5)
        for x in x_list
            if abs(x - x0) <= range
                return true
            end
        end
        return false
    end

    # Adjust min_amp to exclude the noise peak
    while is_within_range(noise_peak_pos, peakpos)
        min_amp += step_size_min_amp

        h_uncal = fit(Histogram, amps, min_amp:bin_width:max_amp)
        h_decon, peakpos = peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)
        @debug("Current peak positions: ", peakpos)

        num_peaks = length(peakpos)

        # Safety check to avoid infinite loops
        if min_amp >= 50
            @error("Unable to exclude noise peak within reasonable min_amp range.")
        end
    end

    # If more than two peaks are found, reduce max_quantile to find exactly two peaks
    if num_peaks > 2
        @debug("Found more than 2 peaks. Reducing max range. Currently: quantile $max_quantile")

        while num_peaks != 2
            max_quantile -= 0.01
            max_amp = quantile(amps, max_quantile)

            h_uncal = fit(Histogram, amps, min_amp:bin_width:max_amp)
            h_decon, peakpos = peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)

            num_peaks = length(peakpos)

            # Safety check to avoid infinite loops
            if max_quantile <= 0.5
                @error("Unable to find exactly two peaks within reasonable quantile range.")
            end
        end
    end

    return h_uncal, peakpos
end



function find_peaks(
    amps::Vector{<:Real}; initial_min_amp::Real=1.0, initial_max_quantile::Real=0.99, 
    peakfinder_σ::Real=2.0, peakfinder_threshold::Real=10.0
)
    # Start with a big window where the noise peak is included
    min_amp = initial_min_amp
    max_quantile = initial_max_quantile
    max_amp = quantile(amps, max_quantile)
    bin_width = get_friedman_diaconis_bin_width(filter(in(quantile(amps, 0.01)..quantile(amps, 0.9)), amps))

    # Initial peak search
    h_uncal = fit(Histogram, amps, min_amp:bin_width:max_amp)
    h_decon, peakpos = peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)

    num_peaks = length(peakpos)

    # Determine the 1 p.e. peak position based on the assumption that it is the highest
    peakpos_idxs = StatsBase.binindex.(Ref(h_decon), peakpos)
    cts_peakpos = h_decon.weights[peakpos_idxs]
    first_pe_peak_pos = peakpos[argmax(cts_peakpos)]

    # remove all peaks < 1p.e. peak
    filter!(x -> x >= first_pe_peak_pos, peakpos)
    num_peaks = length(peakpos)

    while num_peaks < 2
        # Try increasing σ first
        while num_peaks < 2 && peakfinder_σ < 5.0
            h_decon, peakpos = peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)
            num_peaks = length(peakpos)

            if num_peaks < 2
                peakfinder_σ += 0.5
            end
        end

        # If increasing σ doesn't find 2 peaks, start lowering the threshold
        if num_peaks < 2
            peakfinder_threshold -= 1.0
            
            # Safety check to avoid threshold becoming too low
            if peakfinder_threshold < 3.0
                @error("Unable to find more than one peak within reasonable quantile range.")
            end

            # Reset σ to its initial value after adjusting threshold
            peakfinder_σ = 1.0
        end

        # Safety check to avoid infinite loops
        if peakfinder_σ >= 5.0 && peakfinder_threshold < 3.0
            @error("Unable to find more than one peak within reasonable quantile range.")
        end
    end

    # helper function
    function is_within_range(x0, x_list, range=0.5)
        for x in x_list
            if abs(x - x0) <= range
                return true
            end
        end
        return false
    end

    # If more than two peaks are found, reduce max_quantile to find exactly two peaks
    if num_peaks > 2
        @debug("Found more than 2 peaks. Reducing max range. Currently: quantile $max_quantile")

        while num_peaks != 2
            max_quantile -= 0.01
            max_amp = quantile(amps, max_quantile)

            h_uncal = fit(Histogram, amps, min_amp:bin_width:max_amp)
            h_decon, peakpos = peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)

            num_peaks = length(peakpos)

            # Safety check to avoid infinite loops
            if max_quantile <= 0.5
                @error("Unable to find exactly two peaks within reasonable quantile range.")
            end
        end
    end

    return h_uncal, peakpos
end