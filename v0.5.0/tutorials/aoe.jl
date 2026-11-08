using CairoMakie
using Distributions
using LegendMakie
using LegendSpecFits
using Measurements
using Measurements: value as mvalue
using Random
using StatsBase
using Unitful
using Unitful: keV

CairoMakie.activate!(type = "png")

Random.seed!(2024)

lines = (
    # energy (keV), number of events, multi-site fraction
    (583.2, 60_000, 0.3), (727.3, 20_000, 0.3), (860.6, 8_000, 0.3),
    (1592.5, 25_000, 0.12), (1620.5, 6_000, 0.4), (2103.5, 30_000, 0.75),
    (2614.5, 120_000, 0.35),
)
n_continuum = 1_200_000
resolution(E) = 0.75 + 0.00022 * E

e_lines = reduce(vcat, [E .+ resolution(E) .* randn(n) for (E, n, _) in lines])
e_continuum = 200 .+ rand(truncated(Exponential(800), 0, 2400), n_continuum)
e_keV = vcat(e_lines, e_continuum)
mse_fraction = vcat(reduce(vcat, [fill(f, n) for (_, n, f) in lines]), fill(0.25, n_continuum))
nothing #hide

μA, μB, σA, σB = 1.0, -4.0e-6, 5.0e-3, 12.0
sse_mean(E) = μA + μB * E
sse_sigma(E) = sqrt(σA^2 + σB^2 / E^2)

σ_sse = sse_sigma.(e_keV)
aoe_scale = 55.0
aoe_raw = aoe_scale .* (sse_mean.(e_keV) .+ σ_sse .* randn(length(e_keV)) .-
    (rand(length(e_keV)) .< mse_fraction) .* σ_sse .* rand(Exponential(6), length(e_keV)))
e_cal = e_keV .* u"keV"
nothing #hide

cuts_aoe = cut_single_peak(aoe_raw, 0.0, quantile(filter(isfinite, aoe_raw), 0.99); n_bins = -1)
aoe_expression = "aoe / $(cuts_aoe.max)"
aoe = aoe_raw ./ cuts_aoe.max
nothing #hide

h_aoe_raw = fit(Histogram, (e_keV, aoe), (0:5.0:3000, 0.5:0.005:1.3))
lhist(h_aoe_raw, figsize = (670, 400), watermark = false, rasterize = true,
    title = "Normalized A/E", xlabel = "Energy (keV)", ylabel = "A/E (a.u.)")

compton_bands = [500, 540, 620, 660, 760, 800, 900, 940, 1000, 1040, 1100, 1140, 1200, 1240,
    1300, 1340, 1400, 1440, 1500, 1540, 1660, 1700, 1760, 1800, 1860, 1900, 1960, 2000,
    2060, 2140, 2200, 2240, 2300, 2340] .* u"keV"
compton_window = 20.0u"keV"

compton_band_peakhists = generate_aoe_compton_bands(aoe, e_cal, compton_bands, compton_window)
nothing #hide

lhist(compton_band_peakhists.peakhists[1], watermark = false,
    title = "A/E Compton band at $(compton_bands[1])", xlabel = "A/E (a.u.)", ylabel = "Counts")

result_fit, report_fit = fit_aoe_compton(compton_band_peakhists.peakhists,
    compton_band_peakhists.peakstats, compton_bands; uncertainty = true)
result_fit[compton_bands[1]]

lplot(report_fit[compton_bands[1]], watermark = false)

good_bands = findall(b -> haskey(result_fit, b) && result_fit[b].gof.converged, compton_bands)
compton_bands = compton_bands[good_bands]
μ = [result_fit[b].μ for b in compton_bands]
σ = [result_fit[b].σ for b in compton_bands]
length(good_bands)

result_fit_single, report_fit_single = fit_aoe_corrections(compton_bands, μ, σ;
    aoe_expression = aoe_expression, e_expression = "e")
result_fit_single.µ_compton.par

lplot(report_fit_single.report_μ, watermark = false)

lplot(report_fit_single.report_σ, watermark = false)

result_fit_combined, report_fit_combined = fit_aoe_compton_combined(
    compton_band_peakhists.peakhists[good_bands], compton_band_peakhists.peakstats[good_bands],
    compton_bands, result_fit_single; aoe_expression = aoe_expression, e_expression = "e", uncertainty = true)
result_fit_combined

lplot(report_fit_single.report_μ, report_fit_combined.report_µ, watermark = false,
    legend_position = :lb, title = "A/E μ")

correct_aoe = Core.eval(@__MODULE__, Meta.parse("(aoe, e) -> " * result_fit_combined.func))
aoe_corr = Base.invokelatest.(correct_aoe, aoe_raw, e_cal)
nothing #hide

h_aoe_corr = fit(Histogram, (e_keV, aoe_corr), (0:5.0:3000, -25:0.1:10))
lhist(h_aoe_corr, figsize = (670, 400), watermark = false, rasterize = true,
    title = "Corrected A/E", xlabel = "Energy (keV)", ylabel = "A/E (corrected)")

result_cut, report_cut = get_low_aoe_cut(aoe_corr, e_cal; dep = 1592.5u"keV",
    window = [20.0, 20.0]u"keV", bin_width_window = 3.0u"keV", cut_search_interval = (-25.0, 0.0),
    dep_sf = 0.9, rtol = 1e-5, maxiters = 200, sigma_high_sided = 3.0, fit_func = :gamma_def, uncertainty = true)
result_cut.lowcut

lplot(report_cut, figsize = (750, 400), watermark = false)

peaks = [1592.5, 1620.5, 2103.5, 2614.5] .* u"keV"
peak_names = [:DEP, :Bi212FEP, :SEP, :FEP]
left_windows, right_windows = fill(20.0u"keV", 4), fill(20.0u"keV", 4)

result_peaks_low, report_peaks_low = get_peaks_survival_fractions(aoe_corr, e_cal, peaks, peak_names,
    left_windows, right_windows; low_cut = mvalue(result_cut.lowcut), bin_width_window = 3.0u"keV",
    fit_funcs = [:gamma_def, :gamma_def, :gamma_def, :gamma_def], uncertainty = true)
result_peaks_low[:SEP].sf

lplot(report_peaks_low[:SEP], watermark = false)

result_peaks_ds, _ = get_peaks_survival_fractions(aoe_corr, e_cal, peaks, peak_names,
    left_windows, right_windows; low_cut = mvalue(result_cut.lowcut), high_cut = mvalue(result_cut.highcut),
    bin_width_window = 3.0u"keV", fit_funcs = [:gamma_def, :gamma_def, :gamma_def, :gamma_def], uncertainty = true)
result_peaks_ds[:SEP].sf

result_qbb, report_qbb = get_continuum_survival_fraction(aoe_corr, e_cal, 2039.06u"keV", 35.0u"keV";
    low_cut = mvalue(result_cut.lowcut), high_cut = mvalue(result_cut.highcut))
result_qbb.sf
