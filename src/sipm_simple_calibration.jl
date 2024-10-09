"""
    sipm_simple_calibration(pe_uncal::Array)


Perform a simple calibration for the uncalibrated p.e. spectrum array `pe_uncal`
using just the 1 p.e. and 2 p.e. peak positions estimated by a peakfinder.

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
    
    if expect_noise_peak
        h_uncal, peakpos = find_peaks_noise_peak_exists(pe_uncal; kwargs...)
    else
        h_uncal, peakpos = find_peaks(pe_uncal; kwargs...)
    end

    # simple calibration
    gain = peakpos[2] - peakpos[1]
    c = 1/gain
    offset = - (peakpos[1] * c - 1) 

    f_simple_calib = Base.Fix1(*, c)
    pe_simple_cal = pe_uncal .* c .+ offset

    h_calsimple = histogram(pe_simple_cal, bins=0.5:.01:4.5)

    result = (
        pe_simple_cal = pe_simple_cal,
        func = f_simple_calib, 
        c = c,
        offset = offset
    )
    report = (
        peakpos = peakpos,
        h_uncal = h_uncal, 
        h_calsimple = h_calsimple
    )
    return result, report
end



# search the 1 p.e. and 2 p.e. peak, noise peak exists
function find_peaks_noise_peak_exists(
    amps::Vector{<:Real}; initial_min_amp::Real=1.0, initial_max_quantile::Real=0.99, 
    step_size_min_amp::Real=1.0, peakfinder_σ::Real=1.0, peakfinder_threshold::Real=10.0
)
    # Start with a big window where the noise peak is included
    min_amp = initial_min_amp
    max_quantile = initial_max_quantile
    max_amp = quantile(amps, max_quantile)
    bin_width = LegendSpecFits.get_friedman_diaconis_bin_width(filter(in(quantile(amps, 0.01)..quantile(amps, 0.9)), amps))

    # Initial peak search
    h_uncal = fit(Histogram, amps, min_amp:bin_width:max_amp)
    h_decon, peakpos = RadiationSpectra.peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)

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
        h_decon, peakpos = RadiationSpectra.peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)
        println("Current peak positions: ", peakpos)

        num_peaks = length(peakpos)

        # Safety check to avoid infinite loops
        if min_amp >= 50
            error("Unable to exclude noise peak within reasonable min_amp range.")
        end
    end

    # If more than two peaks are found, reduce max_quantile to find exactly two peaks
    if num_peaks > 2
        println("Found more than 2 peaks. Reducing max range. Currently: quantile $max_quantile")

        while num_peaks != 2
            max_quantile -= 0.01
            max_amp = quantile(amps, max_quantile)

            h_uncal = fit(Histogram, amps, min_amp:bin_width:max_amp)
            h_decon, peakpos = RadiationSpectra.peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)

            num_peaks = length(peakpos)

            # Safety check to avoid infinite loops
            if max_quantile <= 0.5
                error("Unable to find exactly two peaks within reasonable quantile range.")
            end
        end
    end

    return h_uncal, peakpos
end



function find_peaks(
    amps::Vector{<:Real}; initial_min_amp::Real=1.0, initial_max_quantile::Real=0.99, 
    peakfinder_σ::Real=1.0, peakfinder_threshold::Real=10.0
)
    # Start with a big window where the noise peak is included
    min_amp = initial_min_amp
    max_quantile = initial_max_quantile
    max_amp = quantile(amps, max_quantile)
    bin_width = LegendSpecFits.get_friedman_diaconis_bin_width(filter(in(quantile(amps, 0.01)..quantile(amps, 0.9)), amps))

    # Initial peak search
    h_uncal = fit(Histogram, amps, min_amp:bin_width:max_amp)
    h_decon, peakpos = RadiationSpectra.peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)

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
            h_decon, peakpos = RadiationSpectra.peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)
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
                error("Unable to find more than one peak within reasonable quantile range.")
            end

            # Reset σ to its initial value after adjusting threshold
            peakfinder_σ = 1.0
        end

        # Safety check to avoid infinite loops
        if peakfinder_σ >= 5.0 && peakfinder_threshold < 3.0
            error("Unable to find more than one peak within reasonable quantile range.")
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
        println("Found more than 2 peaks. Reducing max range. Currently: quantile $max_quantile")

        while num_peaks != 2
            max_quantile -= 0.01
            max_amp = quantile(amps, max_quantile)

            h_uncal = fit(Histogram, amps, min_amp:bin_width:max_amp)
            h_decon, peakpos = RadiationSpectra.peakfinder(h_uncal, σ=peakfinder_σ, backgroundRemove=true, threshold=peakfinder_threshold)

            num_peaks = length(peakpos)

            # Safety check to avoid infinite loops
            if max_quantile <= 0.5
                error("Unable to find exactly two peaks within reasonable quantile range.")
            end
        end
    end

    return h_uncal, peakpos
end