using LegendSpecFits
using CairoMakie, LegendMakie
using Distributions
using Measurements: value as mvalue, uncertainty as muncert
using Random
using Unitful
using Unitful: keV  # the function strings of the results refer to `keV`

CairoMakie.activate!(type = "png")

Random.seed!(1234)

th228_lines = [583.191, 727.330, 860.564, 1592.513, 1620.50, 2103.512, 2614.511]u"keV"

# energy (keV), counts in the FEP, Compton events per FEP event; escape peaks have no own continuum
gamma_lines = [
    (510.77,  6_000, 2.8), (583.191, 25_000, 2.9), (727.330, 4_500, 3.1), (785.37,  700, 3.2),
    (860.564, 3_000, 3.3), (1078.62, 400, 3.6),    (1592.513, 2_500, 0.0), (1620.50, 800, 4.4),
    (2103.512, 4_000, 0.0), (2614.511, 20_000, 5.9),
]
resolution(E) = sqrt(0.42 + 2.7e-4 * E)   # Gaussian σ in keV
tail_fraction, tail_scale = 0.05, 0.001     # low-energy tail: fraction of the FEP, exponential scale relative to E
multiscatter_fraction = 0.03                # fraction of the continuum events between the Compton edge and the line

# Energy deposited by a single Compton scattering, sampled from the Klein-Nishina cross section
function sample_compton(E; m_e = 510.999)
    ε = E / m_e
    T_edge = E * 2ε / (1 + 2ε)
    dσdT(T) = (s = T / E; 2 + s^2 / (ε^2 * (1 - s)^2) + s / (1 - s) * (s - 2 / ε))
    f_max = maximum(dσdT, range(0, T_edge, length = 100))
    while true
        T = rand() * T_edge
        rand() * f_max < dσdT(T) && return T
    end
end
compton_edge(E) = E * (1 - 1 / (1 + 2E / 510.999))

e_true = Float64[]
for (E, n_fep, compton_per_fep) in gamma_lines
    tail = rand(n_fep) .< tail_fraction
    append!(e_true, E .- tail .* rand(Exponential(tail_scale * E), n_fep))
    n_compton = round(Int, compton_per_fep * n_fep)
    multiscatter = rand(n_compton) .< multiscatter_fraction
    append!(e_true, [m ? rand(Uniform(compton_edge(E), E)) : sample_compton(E) for m in multiscatter])
end

# gammas scattered in the materials around the detector add a continuum that rises towards low energies;
# random coincidences (pile-up) add a small flat continuum, also above 2614.5 keV
append!(e_true, rand(Exponential(250.0), 150_000))
append!(e_true, rand(Uniform(0, 3300), 8_000))
e_true .+= resolution.(e_true) .* randn(length(e_true))

# the detector trigger threshold removes the lowest energies
filter!(>(100.0), e_true)
length(e_true)

c_true = 0.01u"keV"            # keV per ADC unit
qdrift = rand(Normal(1000.0, 300.0), length(e_true))
e_uncal = @. e_true / ustrip(c_true) - 0.5 * (qdrift - 1000.0)
length(e_uncal)

fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1], xlabel = "Energy (keV)", ylabel = "Counts / 1 keV", yscale = log10, limits = ((0, 3000), (1, 2e4)))
stephist!(ax, e_true, bins = 0:1:3000)
fig

th228_names = [:Tl208a, :Bi212a, :Tl208b, :Tl208DEP, :Bi212FEP, :Tl208SEP, :Tl208FEP]
left_window_sizes  = [25, 25, 25, 25, 15, 25, 35]u"keV"
right_window_sizes = [25, 25, 25, 15, 25, 25, 30]u"keV"
fit_funcs = fill(:gamma_def, length(th228_names))
binning_peak_window = 4u"keV"      # energy range around a peak used to choose the histogram bin width
cal_pol_order = 1
fwhm_pol_order = 1
excluded_peaks = [:Tl208DEP, :Tl208SEP]  # escape peaks are not used for the calibration and resolution curves

th228_lines_dict = Dict(th228_names .=> th228_lines)

result_simple, report_simple = simple_calibration(
    e_uncal, th228_lines, left_window_sizes, right_window_sizes;
    calib_type = :th228, binning_peak_window
)
m_cal_simple = result_simple.c

lplot(report_simple, cal = true, title = "Simple Calibration", watermark = false)

using Logging #hide
disable_logging(Logging.Warn) #hide
result_ctc, report_ctc = ctc_energy(
    e_uncal .* m_cal_simple, qdrift, 2614.511u"keV", (35u"keV", 30u"keV"), m_cal_simple;
    e_expression = "e_uncal", pol_order = 1
)
disable_logging(Logging.BelowMinLevel) #hide
result_ctc.converged || error("The charge-trapping correction did not converge")
result_ctc.func

(fct = only(result_ctc.fct), fwhm_before = mvalue(result_ctc.fwhm_before), fwhm_after = mvalue(result_ctc.fwhm_after))

lplot(report_ctc, figsize = (600, 600), title = "Charge-Trapping Correction", watermark = false)

e_uncal_ctc = e_uncal .+ only(result_ctc.fct) .* qdrift;

result_simple, report_simple = simple_calibration(
    e_uncal_ctc, th228_lines, left_window_sizes, right_window_sizes;
    calib_type = :th228, binning_peak_window
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
