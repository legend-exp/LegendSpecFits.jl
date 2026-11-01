# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    fit_sipm_template(pulses::AbstractVector{<:AbstractVector{<:Real}}, noise::AbstractVector{<:AbstractVector{<:Real}}, dt::Unitful.Time{<:Real}, t_start::Unitful.Time{<:Real}; kwargs...)

Build the single-photon template and the noise spectrum of a SiPM channel from trigger-aligned, baseline-subtracted
single pulses and from baseline-subtracted trigger-free traces.

Every pulse is normalised to its own maximum before the median per sample is taken, so the template is the common
pulse shape whatever PE band dominates the channel; its peak lies slightly below 1 because the maximum of a single
pulse includes the noise. An exponential is fitted to the template in
`tail_fit_window` (least squares in amplitude, the decay time through its logarithm, bounded between one sample and the
length of the noise traces); the
template continues beyond the window with that fit: `tail_level` is its value at the window end, `tail_tau` its decay
time. A template that has already decayed into the noise gets a `tail_level` compatible with zero.

The noise spectrum is the median of the one-sided Hann-windowed periodograms of the traces (ADC²/Hz, so that the
spectrum integrated over frequency is the variance), divided by `log(2)` to turn the median of the χ²-distributed
periodogram bins into their mean. The traces must all have the same length; its frequency grid has the step `df`.
The noise σ `baseline_σ` is the median σ of the traces; a Gaussian fitted to the core (±3 σ) of all their samples
gives the noise model `μ`, `σ` of the channel in ADC.

The `amplitude` is the raw pulse height in ADC (most frequent single-pulse maximum, parabolically refined; biased high
by the noise of the single-sample maximum), the ADC scale of the template for the kernel design. The report also carries the aligned
pulses as heatmaps in ADC and normalised, the residuals of the normalised pulses against the template (all pulses
and per PE band) and the target stick (Gaussian of `target_fwhm`, peak 1 at `t = 0`).

# Arguments
- `pulses`: baseline-subtracted pulse windows, one per single-trigger waveform, all of the same length
- `noise`: baseline-subtracted trigger-free traces, all of the same length
- `dt`: sampling interval
- `t_start`: time of the first window sample relative to the trigger

# Keywords
- `target_fwhm`: FWHM of the target stick
- `tail_fit_window`: time range after the trigger of the exponential fit to the template tail
- `end_length`: length of the window end over which the end level of the template is averaged (report only)
- `min_pulses`, `min_noise_traces`: minimum numbers of pulses and traces, an `ArgumentError` is thrown below

# Returns
- `result`: `template` (`t0`, `dt`, `pulse`, `amplitude` in ADC, `n_pulses`, `tail_level`, `tail_tau`, `baseline_σ` in ADC) and `noise` (`df`, `psd` in ADC²/Hz, `n_noise`, Gaussian `μ`, `σ` in ADC)
- `report`: `t`, `pulse`, `target`, `tail`, `amplitude`, `h_heatmap`, `h_heatmap_norm`, `h_residual`, `residual_bands`, `h_amp`, `baseline_σ`, `n_pulses`, `freq`, `psd`, `n_noise`, `h_noise`, `noise_fit`
"""
function fit_sipm_template(pulses::AbstractVector{<:AbstractVector{<:Real}}, noise::AbstractVector{<:AbstractVector{<:Real}}, dt::Unitful.Time{<:Real}, t_start::Unitful.Time{<:Real};
        target_fwhm::Unitful.Time{<:Real} = 75.0u"ns", tail_fit_window::Tuple{<:Unitful.Time{<:Real}, <:Unitful.Time{<:Real}} = (20.0u"µs", 30.0u"µs"),
        end_length::Unitful.Time{<:Real} = 1.0u"µs",
        heatmap_window::Tuple{<:Unitful.Time{<:Real}, <:Unitful.Time{<:Real}} = (-1.0u"µs", 6.0u"µs"), heatmap_bins::Int = 200,
        band_width::Real = 0.25, n_bands::Int = 3, min_pulses::Int = 1000, min_noise_traces::Int = 100)
    length(pulses) >= min_pulses || throw(ArgumentError("Only $(length(pulses)) single pulses, at least $min_pulses are needed for the template"))
    length(noise) >= min_noise_traces || throw(ArgumentError("Only $(length(noise)) trigger-free traces, at least $min_noise_traces are needed for the noise spectrum"))
    n_samples = length(first(pulses))
    all(==(n_samples), length.(pulses)) || throw(ArgumentError("All pulses must have the same length"))
    n_noise_samples = length(first(noise))
    all(==(n_noise_samples), length.(noise)) || throw(ArgumentError("All noise traces must have the same length"))

    # pulses as columns of one matrix; only pulses with a positive maximum can be normalised to it
    M = reshape(flatview(VectorOfVectors(pulses)), n_samples, length(pulses))
    amps = vec(maximum(M; dims = 1))
    M = M[:, amps .> 0]
    amps = amps[amps .> 0]
    n_pulses = length(amps)
    n_pulses >= min_pulses || throw(ArgumentError("Only $n_pulses single pulses with a positive maximum, at least $min_pulses are needed for the template"))
    N = M ./ amps'
    pulse = vec(median(N; dims = 2))
    t = t_start .+ (0:n_samples-1) .* dt
    t_us = ustrip.(u"µs", t)

    # exponential tail: fit in the tail fit window, time relative to the window end so that the first parameter is the
    # level at the window end; the decay time through its logarithm stays positive (plain µs: Unitful ranges of mixed
    # time units convert wrongly)
    t_end = last(t)
    end_level = mean(pulse[t .> t_end - end_length])
    sel = findall(i -> first(tail_fit_window) <= t[i] <= last(tail_fit_window), eachindex(t))
    length(sel) >= 2 || throw(ArgumentError("The tail fit window $(tail_fit_window) holds fewer than two template samples"))
    tail_model(τ, p) = p[1] .* exp.(-τ ./ exp(p[2]))
    log_tau_range = log.(ustrip.(u"µs", (dt, n_noise_samples * dt)))    # a decay time between one sample and the trace length
    tail_fit = curve_fit(tail_model, t_us[sel] .- last(t_us), pulse[sel], [mean(pulse[sel]), log(ustrip(u"µs", last(tail_fit_window) - first(tail_fit_window)))];
                         lower = [-Inf, first(log_tau_range)], upper = [Inf, last(log_tau_range)])
    tail_level, tail_tau = tail_fit.param[1], exp(tail_fit.param[2]) * u"µs"
    # continuation of the template beyond the window, for the report
    t_tail_us = last(t_us) .+ (0:n_samples-1) .* ustrip(u"µs", dt)
    tail = (t = t_tail_us .* u"µs", pulse = tail_model(t_tail_us .- last(t_us), tail_fit.param), window = tail_fit_window,
            tau = tail_tau, level = tail_level, end_level = end_level, t_end = t_end)

    # noise spectrum: median of the one-sided Hann periodograms, corrected for the median of the χ² distribution
    fs = ustrip(u"Hz", 1 / dt)
    w = [0.5 * (1 - cos(2π * k / (n_noise_samples - 1))) for k in 0:n_noise_samples-1]
    norm = fs * sum(abs2, w)
    P = reduce(hcat, abs2.(rfft(y .* w)) ./ norm for y in noise)
    P[2:(iseven(n_noise_samples) ? end - 1 : end), :] .*= 2
    psd = vec(median(P; dims = 2)) ./ log(2)
    freq = collect(rfftfreq(n_noise_samples, fs)) .* u"Hz"
    df = fs / n_noise_samples * u"Hz"
    baseline_σ = median(std.(noise))

    # raw pulse height: most frequent pulse maximum (bins of half the noise σ), parabolically refined
    h_amp = fit(Histogram, amps, 0:(baseline_σ / 2):quantile(amps, 0.999))
    wmax, i = findmax(h_amp.weights)
    amplitude = first(h_amp.edges)[i] + step(first(h_amp.edges)) / 2
    if 1 < i < length(h_amp.weights)
        yl, yr = h_amp.weights[i-1], h_amp.weights[i+1]
        amplitude += step(first(h_amp.edges)) * (yl - yr) / (2 * (yl - 2wmax + yr))
    end

    # all samples of the noise traces and a Gaussian fitted to their core (within ±3 baseline σ): the noise model of
    # the channel; light left in the traces shows as a tail above the Gaussian
    h_noise = fit(Histogram, flatview(VectorOfVectors(noise)), (-8 * baseline_σ):(baseline_σ / 4):max(8 * baseline_σ, 3 * amplitude))
    x_noise = midpoints(first(h_noise.edges))
    core = findall(x -> abs(x) <= 3 * baseline_σ, x_noise)
    gauss_model(x, p) = p[1] .* exp.(-0.5 .* ((x .- p[2]) ./ p[3]).^2)
    gauss_fit = curve_fit(gauss_model, x_noise[core], Float64.(h_noise.weights[core]), [Float64(maximum(h_noise.weights)), 0.0, baseline_σ])
    noise_fit = (μ = gauss_fit.param[2], σ = abs(gauss_fit.param[3]), amplitude = gauss_fit.param[1])

    # target stick at the alignment point, peak 1
    σ_target = ustrip(u"µs", target_fwhm) / (2 * sqrt(2 * log(2)))
    target = exp.(-0.5 .* (t_us ./ σ_target).^2)

    # heatmaps of the aligned pulses (ADC and normalised) and of the residuals against the template
    sel_hm = findall(i -> first(heatmap_window) <= t[i] <= last(heatmap_window), eachindex(t))
    edges_t = t_us[first(sel_hm)] - ustrip(u"µs", dt) / 2 : ustrip(u"µs", dt) : t_us[last(sel_hm)] + ustrip(u"µs", dt) / 2
    t_rep = repeat(t_us[sel_hm], n_pulses)
    h_heatmap = fit(Histogram, (t_rep, vec(view(M, sel_hm, :))), (edges_t, range(-0.5 * amplitude, 5 * amplitude; length = heatmap_bins + 1)))
    h_heatmap_norm = fit(Histogram, (t_rep, vec(view(N, sel_hm, :))), (edges_t, range(-0.3, 1.3; length = heatmap_bins + 1)))
    R = view(N, sel_hm, :) .- view(pulse, sel_hm)
    h_residual = fit(Histogram, (t_rep, vec(R)), (edges_t, range(-0.3, 0.3; length = heatmap_bins + 1)))

    # median residual per PE band (pulses with a maximum within ±band_width of k × amplitude)
    residual_bands = [begin
        in_band = findall(a -> abs(a / (k * amplitude) - 1) <= band_width, amps)
        (pe = k, n = length(in_band), residual = isempty(in_band) ? zeros(length(sel_hm)) : vec(median(view(R, :, in_band); dims = 2)))
    end for k in 1:n_bands]

    result = (template = (t0 = t_start, dt = dt, pulse = pulse, amplitude = amplitude, n_pulses = n_pulses, tail_level = tail_level, tail_tau = tail_tau, baseline_σ = baseline_σ),
        noise = (df = df, psd = psd, n_noise = length(noise), μ = noise_fit.μ, σ = noise_fit.σ))
    report = (t = t, pulse = pulse, target = target, tail = tail, amplitude = amplitude, h_heatmap = h_heatmap, h_heatmap_norm = h_heatmap_norm,
        h_residual = h_residual, residual_bands = residual_bands, h_amp = h_amp, baseline_σ = baseline_σ, n_pulses = n_pulses,
        freq = freq, psd = psd, n_noise = length(noise), h_noise = h_noise, noise_fit = noise_fit)
    return result, report
end
export fit_sipm_template
