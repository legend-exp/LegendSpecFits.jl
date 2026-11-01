# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    fit_sipm_kernel(template::NamedTuple, noise::NamedTuple; kwargs...)

Design the inverse (Wiener) kernel of a SiPM channel from its single-photon `template` (`t0`, `dt`, `pulse`,
`tail_level`, `tail_tau`, `amplitude`) and its `noise` spectrum (`df`, `psd`), both as `fit_sipm_template` returns them.

The template is placed on the grid of the noise traces (circular, lag 0 at the first sample, the samples before the
trigger at the end), continued beyond its window with its exponential tail and scaled to ADC with `amplitude`, so
that the kernel turns one photon into the target `g`, a Gaussian stick of `target_fwhm` with unit sum: the filtered
trace counts photons. The kernel is `H = G P* / (|P|² + λ N)` with `P`, `G` the spectra of template and target and
`N` the noise power per frequency bin. For every `λ` of the logarithmic grid `lambda_range` × `n_lambda`, the response
`q = h ∗ p` of the template is compared with the target in the tail window from `tail_start_n_sigma` σ of the target
to `tail_end`: the largest deviation `|q - g|` relative to the target peak (what a following photon sees with a short
shaping) and the largest running integral of `q - g` relative to the target sum (what it sees with a long shaping; it
also rejects a kernel so smooth that it spreads the photon itself into the window). The largest `λ` with both below
`tail_max` is chosen; an `ArgumentError` is thrown if none is. The kernel is cut to `±kernel_half_length` (`taps`, lag 0
in the middle) and the deviations, the stick FWHM and the noise per sample (relative to the target peak, from `|H|² N`)
are measured again with the cut kernel, the noise also after four moving averages of `trigger_shaping` relative to the
peak of the target after the same shaping. The single `pulses` the template
was built from (ADC, on the template grid) are sent through the cut kernel (linear convolution, zero outside the
window) and the same four moving averages, scaled so that one photon has the peak 1, and reported as a heatmap within
`heatmap_window` with their median.

# Returns
- `result`: `lambda`, `half_length`, `dt`, `taps`, `target_fwhm`, `tail_local`, `tail_integral`, `fwhm`, `noise_per_sample`, `noise_shaped`, `trigger_shaping`, `response_sum`, `truncation_ok`
- `report`: the λ scan, the responses, the spectra before and after the kernel, the taps and the deconvolved pulses
"""
function fit_sipm_kernel(template::NamedTuple, noise::NamedTuple;
        target_fwhm::Unitful.Time{<:Real} = 75.0u"ns", tail_start_n_sigma::Real = 4.0, tail_end::Unitful.Time{<:Real} = 5.0u"µs",
        tail_max::Real = 0.01, lambda_range::Tuple{<:Real, <:Real} = (1e-5, 1e2), n_lambda::Int = 71,
        kernel_half_length::Unitful.Time{<:Real} = 16.0u"µs", trigger_shaping::Unitful.Time{<:Real} = 160.0u"ns",
        pulses::AbstractVector{<:AbstractVector{<:Real}} = Vector{Float64}[],
        heatmap_window::Tuple{<:Unitful.Time{<:Real}, <:Unitful.Time{<:Real}} = (-1.0u"µs", 2.0u"µs"), heatmap_bins::Int = 200)
    # grid of the noise traces
    n_bins = length(noise.psd)
    N = 2 * (n_bins - 1)
    dt_us = ustrip(u"µs", template.dt)
    fs = ustrip(u"Hz", 1 / template.dt)
    isapprox(ustrip(u"Hz", noise.df), fs / N; rtol = 1e-6) ||
        throw(ArgumentError("The noise spectrum ($(noise.df) per bin, $n_bins bins) does not match the template sampling $(template.dt)"))

    # template in ADC on the circular grid: window samples, exponential continuation, pre-trigger samples at the end
    t_us = ustrip(u"µs", template.t0) .+ (0:length(template.pulse)-1) .* dt_us
    n_pre = count(<(0), t_us)
    p = zeros(N)
    for (tk, pk) in zip(t_us, template.pulse)
        p[mod(round(Int, tk / dt_us), N) + 1] = template.amplitude * pk
    end
    t_end = last(t_us)
    for j in (round(Int, t_end / dt_us) + 2):(N - n_pre)
        p[j] = template.amplitude * template.tail_level * exp(-((j - 1) * dt_us - t_end) / ustrip(u"µs", template.tail_tau))
    end

    # target: Gaussian stick with unit sum at lag 0
    σ_s = ustrip(u"µs", target_fwhm) / (2 * sqrt(2 * log(2))) / dt_us
    lag = [j <= N ÷ 2 ? j - 1 : j - 1 - N for j in 1:N]
    g = exp.(-0.5 .* (lag ./ σ_s).^2)
    g ./= sum(g)
    g_peak = maximum(g)

    # spectra: noise power per DFT bin of an unwindowed trace from the one-sided spectrum
    P = rfft(p)
    G = rfft(g)
    Nd = noise.psd .* (N * fs / 2)
    Nd[1] *= 2
    Nd[end] *= 2
    freq = collect(rfftfreq(N, fs)) .* u"Hz"

    # the trigger shaping after the kernel: four moving averages of trigger_shaping, amplitude transfer M4 per bin
    L = round(Int, ustrip(u"µs", trigger_shaping) / dt_us)
    M4 = [k == 0 ? 1.0 : (sin(π * k * L / N) / (L * sin(π * k / N)))^4 for k in 0:(N ÷ 2)]
    g_peak_shaped = maximum(irfft(G .* M4, N))
    noise_σ(W) = sqrt((W[1] * Nd[1] + 2 * sum(W[2:end-1] .* Nd[2:end-1]) + W[end] * Nd[end]) / N^2)

    # the tail window in positive lags
    i_tail = (ceil(Int, tail_start_n_sigma * σ_s) + 1):(round(Int, ustrip(u"µs", tail_end) / dt_us) + 1)
    function measure(H)
        q = irfft(H .* P, N)
        r = q[i_tail] .- g[i_tail]
        above = findall(>(maximum(q) / 2), circshift(q, N ÷ 2))    # the stick straddles lag 0, so centre it first
        (q = q, tail_local = maximum(abs.(r)) / g_peak, tail_integral = maximum(abs.(cumsum(r))), noise_per_sample = noise_σ(abs2.(H)) / g_peak,
         noise_shaped = noise_σ(abs2.(H) .* M4 .^ 2) / g_peak_shaped, fwhm = (maximum(above) - minimum(above) + 1) * dt_us * u"µs")
    end
    kernel(λ) = G .* conj.(P) ./ (abs2.(P) .+ λ .* Nd)

    # the λ scan: the largest λ that keeps both deviations within tail_max
    lambdas = 10 .^ range(log10(first(lambda_range)), log10(last(lambda_range)); length = n_lambda)
    scan = [measure(kernel(λ)) for λ in lambdas]
    ok = findall(s -> s.tail_local <= tail_max && s.tail_integral <= tail_max, scan)
    isempty(ok) && throw(ArgumentError("No λ in $(lambda_range) keeps the response tail within $(tail_max): smallest deviations " *
                                       "$(round(minimum(s.tail_local for s in scan), sigdigits = 3)) (local) and " *
                                       "$(round(minimum(s.tail_integral for s in scan), sigdigits = 3)) (integral)"))
    i_λ = maximum(ok)
    λ = lambdas[i_λ]
    H = kernel(λ)

    # the kernel cut to ±kernel_half_length, measured again
    h = irfft(H, N)
    m = round(Int, ustrip(u"µs", kernel_half_length) / dt_us)
    taps = vcat(h[N-m+1:N], h[1:m+1])
    hc = zeros(N)
    hc[1:m+1] .= h[1:m+1]
    hc[N-m+1:N] .= h[N-m+1:N]
    Hc = rfft(hc)
    cut = measure(Hc)

    # the single pulses through the cut kernel (zero outside their window) and the trigger shaping, one photon = peak 1:
    # heatmap and median
    function mwa4(v)
        for pass in 1:4
            c = cumsum(v)
            v = [(c[i] - (i > L ? c[i-L] : 0.0)) / min(i, L) for i in eachindex(v)]
            v = reverse(v)
        end
        v
    end
    n_p = length(template.pulse)
    all(==(n_p), length.(pulses)) || throw(ArgumentError("All pulses must have the length of the template ($n_p samples)"))
    L_pad = n_p + 2m
    T = rfft(vcat(taps, zeros(L_pad - 2m - 1)))
    D = Matrix{Float64}(undef, n_p, length(pulses))
    for (k, y) in enumerate(pulses)
        D[:, k] = mwa4(irfft(rfft(vcat(y, zeros(L_pad - n_p))) .* T, L_pad)[m+1:m+n_p]) ./ g_peak_shaped
    end
    hw = ustrip.(u"µs", heatmap_window)
    sel_hm = findall(i -> first(hw) <= t_us[i] <= last(hw), eachindex(t_us))
    edges_t = t_us[first(sel_hm)] - dt_us / 2 : dt_us : t_us[last(sel_hm)] + dt_us / 2
    h_pulses = fit(Histogram, (repeat(t_us[sel_hm], length(pulses)), vec(D[sel_hm, :])), (edges_t, range(-0.5, 4.0; length = heatmap_bins + 1)))
    target_shaped = irfft(G .* M4, N) ./ g_peak_shaped
    pulses_median = isempty(pulses) ? zeros(length(sel_hm)) : vec(median(view(D, sel_hm, :); dims = 2))

    result = (lambda = λ, half_length = m * dt_us * u"µs", dt = template.dt, taps = taps, target_fwhm = target_fwhm, tail_local = cut.tail_local,
              tail_integral = cut.tail_integral, fwhm = cut.fwhm, noise_per_sample = cut.noise_per_sample, noise_shaped = cut.noise_shaped,
              trigger_shaping = trigger_shaping, response_sum = sum(cut.q), truncation_ok = cut.tail_local <= tail_max && cut.tail_integral <= tail_max)
    report = (lambdas = lambdas, tail_local = [s.tail_local for s in scan], tail_integral = [s.tail_integral for s in scan],
              noise_per_sample = [s.noise_per_sample for s in scan], noise_shaped = [s.noise_shaped for s in scan], trigger_shaping = trigger_shaping,
              i_lambda = i_λ, lambda = λ, tail_max = tail_max,
              lag = lag .* dt_us .* u"µs", p = p, g = g, q_full = scan[i_λ].q, q_cut = cut.q,
              tail_window = (first(i_tail) - 1) * dt_us * u"µs" => (last(i_tail) - 1) * dt_us * u"µs",
              freq = freq, pulse_spectrum = abs2.(P), noise_spectrum = Nd, target_spectrum = abs2.(G),
              filtered_pulse = abs2.(Hc .* P), filtered_noise = abs2.(Hc) .* Nd, taps = taps, taps_lag = (-m:m) .* dt_us .* u"µs",
              h_pulses = h_pulses, pulses_median = pulses_median, t_pulses = t_us[sel_hm] .* u"µs", n_pulses = length(pulses),
              target_shaped = target_shaped, result = result)
    return result, report
end
export fit_sipm_kernel
