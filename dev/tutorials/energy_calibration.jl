using LegendSpecFits
using CairoMakie, LegendMakie
using Distributions
using Measurements: value as mvalue, uncertainty as muncert
using Random
using Unitful
using Unitful: keV  # the function strings of the results refer to `keV`

CairoMakie.activate!(type = "png")

Random.seed!(1234)

f_peak = LegendSpecFits.get_th228_fit_functions().gamma_def

# Peak shape parameters of every line; σ² grows linearly with the energy
th228_lines = [583.191, 727.330, 860.564, 1592.513, 1620.50, 2103.512, 2614.511]u"keV"
shape_pars = [
    (σ = 0.73, n = 1865.5, step_amplitude = 3.0,  skew_fraction = 0.02, skew_width = 0.003, background = 3),
    (σ = 0.78, n = 452.9,  step_amplitude = 1.0,  skew_fraction = 0.01, skew_width = 0.003, background = 3),
    (σ = 0.83, n = 355.8,  step_amplitude = 0.5,  skew_fraction = 0.01, skew_width = 0.003, background = 3),
    (σ = 1.02, n = 206.8,  step_amplitude = 0.1,  skew_fraction = 0.01, skew_width = 0.003, background = 2),
    (σ = 1.04, n = 130.3,  step_amplitude = 0.1,  skew_fraction = 0.01, skew_width = 0.003, background = 2),
    (σ = 1.14, n = 385.1,  step_amplitude = 0.1,  skew_fraction = 0.01, skew_width = 0.003, background = 2),
    (σ = 1.25, n = 3130.4, step_amplitude = 0.1,  skew_fraction = 0.02, skew_width = 0.003, background = 1),
]

function sample_line(μ, pars; n_events, half_width = 60.0, n_grid = 10_000)
    x = range(μ - half_width, μ + half_width, length = n_grid)
    cdf = cumsum(f_peak.(x, Ref((; μ, pars...))))
    cdf ./= last(cdf)
    u = rand(n_events)
    # linear interpolation of the inverse cumulative distribution function
    map(u) do ui
        i = clamp(searchsortedfirst(cdf, ui), 2, n_grid)
        x[i-1] + (ui - cdf[i-1]) / (cdf[i] - cdf[i-1]) * step(x)
    end
end

# The relative peak heights follow the model heights; each line gets its share of 200_000 events
peak_heights = map(zip(th228_lines, shape_pars)) do (μ, pars)
    maximum(f_peak.(range(ustrip(μ) - 60, ustrip(μ) + 60, length = 2000), Ref((; μ = ustrip(μ), pars...))))
end
n_events = round.(Int, 200_000 .* peak_heights ./ sum(peak_heights))
e_lines = reduce(vcat, [sample_line(ustrip(μ), pars; n_events = n) for (μ, pars, n) in zip(th228_lines, shape_pars, n_events)])
length(e_lines)

c_true = 0.01u"keV"            # keV per ADC unit
n_continuum = 150_000
e_continuum = rand(truncated(Exponential(700.0), 100.0, 2614.5), n_continuum)

e_true = vcat(e_lines, e_continuum)
qdrift = rand(Normal(1000.0, 300.0), length(e_true))
e_uncal = @. e_true / ustrip(c_true) - 0.5 * (qdrift - 1000.0)
length(e_uncal)

th228_names = [:Tl208a, :Bi212a, :Tl208b, :Tl208DEP, :Bi212FEP, :Tl208SEP, :Tl208FEP]
left_window_sizes  = [25, 25, 25, 25, 15, 25, 35]u"keV"
right_window_sizes = [25, 25, 25, 15, 25, 25, 30]u"keV"
fit_funcs = fill(:gamma_def, length(th228_names))
binning_peak_window = 4u"keV"      # energy range around a peak used to choose the histogram bin width
quantile_perc = 0.91               # fraction of events below the 2614.5 keV peak
cal_pol_order = 1
fwhm_pol_order = 1
excluded_peaks = [:Tl208DEP, :Tl208SEP]  # escape peaks are not used for the calibration and resolution curves

th228_lines_dict = Dict(th228_names .=> th228_lines)

result_simple, report_simple = simple_calibration(
    e_uncal, th228_lines, left_window_sizes, right_window_sizes;
    calib_type = :th228, binning_peak_window, quantile_perc
)
m_cal_simple = result_simple.c

lplot(report_simple, cal = true, title = "Simple Calibration", watermark = false)

result_ctc, report_ctc = ctc_energy(
    e_uncal .* m_cal_simple, qdrift, 2614.511u"keV", (35u"keV", 30u"keV"), m_cal_simple;
    e_expression = "e_uncal", pol_order = 1
)
result_ctc.converged || error("The charge-trapping correction did not converge")
result_ctc.func

(fct = only(result_ctc.fct), fwhm_before = mvalue(result_ctc.fwhm_before), fwhm_after = mvalue(result_ctc.fwhm_after))

lplot(report_ctc, figsize = (600, 600), title = "Charge-Trapping Correction", watermark = false)

e_uncal_ctc = e_uncal .+ only(result_ctc.fct) .* qdrift;

result_simple, report_simple = simple_calibration(
    e_uncal_ctc, th228_lines, left_window_sizes, right_window_sizes;
    calib_type = :th228, binning_peak_window, quantile_perc
)
m_cal_simple = result_simple.c

lplot(report_simple, cal = true, title = "Simple Calibration After Correction", watermark = false)

result_fit, report_fit = fit_peaks(
    result_simple.peakhists, result_simple.peakstats, th228_names;
    e_unit = result_simple.unit, calib_type = :th228, fit_func = fit_funcs, m_cal_simple
)
all(r -> r.gof.converged, values(result_fit)) || error("A peak fit did not converge")

result_fit[:Tl208FEP].centroid, result_fit[:Tl208FEP].fwhm

lplot(report_fit, figsize = (600, 400 * length(report_fit)), title = "Peak Fits", watermark = false)

cal_names = [p for p in th228_names if p ∉ excluded_peaks]
result_calib, report_calib = fit_calibration(
    cal_pol_order,
    [result_fit[p].centroid for p in cal_names],
    [th228_lines_dict[p] for p in cal_names];
    e_expression = "e_uncal + $(only(result_ctc.fct)) * qdrift"
)
result_calib.gof.converged || error("The calibration curve fit did not converge")
result_calib.par

result_calib.func

lplot(
    report_calib, xerrscaling = 100, title = "Calibration Curve", watermark = false,
    additional_pts = (
        μ = [result_fit[p].centroid for p in excluded_peaks],
        peaks = [th228_lines_dict[p] for p in excluded_peaks]
    )
)

f_cal_widths(x) = report_calib.f_fit(x) .* report_calib.e_unit .- first(report_calib.par)
fwhm_names = [p for p in th228_names if p ∉ excluded_peaks]

result_fwhm, report_fwhm = fit_fwhm(
    fwhm_pol_order,
    [th228_lines_dict[p] for p in fwhm_names],
    f_cal_widths.([result_fit[p].fwhm for p in fwhm_names]);
    e_type_cal = :e_cal, uncertainty = true
)
result_fwhm.gof.converged || error("The resolution curve fit did not converge")

result_fwhm.qbb

result_fwhm.func_cal

lplot(
    report_fwhm, title = "Resolution Curve", watermark = false,
    additional_pts = (
        peaks = [th228_lines_dict[p] for p in excluded_peaks],
        fwhm = f_cal_widths.([result_fit[p].fwhm for p in excluded_peaks])
    )
)

calibrate = Core.eval(@__MODULE__, Meta.parse("(e_uncal, qdrift) -> " * result_calib.func))
e_cal = ustrip.(u"keV", Base.invokelatest.(calibrate, e_uncal, qdrift))

fig = Figure(size = (700, 400))
ax = Axis(
    fig[1, 1], xlabel = "Energy (keV)", ylabel = "Counts / 1 keV", yscale = log10,
    limits = ((0, 3000), (1, 2e4)), xticks = 0:500:3000, title = "Calibrated Spectrum"
)
stephist!(ax, e_cal, bins = 0:1:3000)
vlines!(ax, ustrip.(u"keV", th228_lines), color = (:red, 0.3), linestyle = :dash, label = "Literature energies")
axislegend(ax, position = :rt)
fig
