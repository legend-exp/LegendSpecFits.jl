# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits, RadiationSpectra
using LegendHDF5IO
using StatsBase, Distributions
using LinearAlgebra
using StructArrays
using ValueShapes, InverseFunctions, BAT, Optim
using Plots

# Some LH5-file with uncalibrated calibration data enery depositions:
input_filename = ENV["LEGEND_CALTEST_EDEP_UNCAL"]

detector_no = 40

lhd = LHDataStore(input_filename)
E_uncal = lhd[string(detector_no)][:]

h_uncal = fit(Histogram, E_uncal, nbins = 20000)

#th228_lines = [583.191, 727.330, 860.564, 1592.53, 1620.50, 2103.53, 2614.50]
th228_lines = [583.191, 727.330, 860.564, 2103.53, 2614.50]
h_cal, h_deconv, peakpos, threshold, c, c_precal = RadiationSpectra.calibrate_spectrum(
    h_uncal, th228_lines,
    σ = 2.0, threshold = 5.0
)

peakhists = RadiationSpectra.subhist.(Ref(h_cal), (x -> (x-25, x+25)).(th228_lines))
peakstats = StructArray(estimate_single_peak_stats.(peakhists))

plot(
    (
        plot(normalize(h_uncal, mode = :density), st = :stepbins, yscale = :log10);
        vline!(peakpos)
    ),
    plot.(normalize.(peakhists, mode = :density), st = :stepbins, yscale = :log10)...
)


# Peak-by-peak fit:

peak_fit_plots = Plots.Plot[]

for i in eachindex(peakhists)
    h = peakhists[i]
    ps = peakstats[i]

    pseudo_prior = NamedTupleDist(
        μ = Uniform(ps.peak_pos-10, ps.peak_pos+10),
        σ = weibull_from_mx(ps.peak_sigma, 2*ps.peak_sigma),
        n = weibull_from_mx(ps.peak_counts, 2*ps.peak_counts),
        step_amplitude = weibull_from_mx(ps.mean_background, 2*ps.mean_background),
        skew_fraction = Uniform(0.01, 0.25),
        skew_width = LogUniform(0.001, 0.1),
        background = weibull_from_mx(ps.mean_background, 2*ps.mean_background),
    )

    f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

    v_init = mean(pseudo_prior)

    f_fit(x, v) = gamma_peakshape(x, v.μ, v.σ, v.n, v.step_amplitude, v.skew_fraction, v.skew_width) + v.background

    f_loglike = let f_fit=f_fit, h=h
        v -> hist_loglike(Base.Fix2(f_fit, v), h)
    end

    opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init))
    v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))
    plt = plot(normalize(h, mode = :density), st = :stepbins, yscale = :log10)
    plot!(minimum(h.edges[1]):0.1:maximum(h.edges[1]), Base.Fix2(f_fit, v_ml))
    push!(peak_fit_plots, plt)
end

plot(peak_fit_plots...)


# Global fit over all calibration gamma lines:

th228_lines = [583.191, 727.330, 860.564, 2614.50]
peakhists = RadiationSpectra.subhist.(Ref(h_cal), (x -> (x-25, x+25)).(th228_lines))
peakstats = StructArray(estimate_single_peak_stats.(peakhists))

function f_fit(x, v)
    μ = v.cal_offs + v.cal_slope * v.expected_μ + v.cal_sqr * v.expected_μ^2
    σ = sqrt(v.σ_enc^2 + (sqrt(μ) * v.σ_fano)^2)
    gamma_peakshape(
        x, μ, σ,
        v.n, v.step_amplitude, v.skew_fraction, v.skew_width
    ) + v.background
end

function empirical_prior_from_peakstats(peakstats::StructArray{<:NamedTuple})
    ps = peakstats
    mean_rel_sigma = mean(peakstats.peak_sigma ./ sqrt.(peakstats.peak_pos))
    NamedTupleDist(
        expected_μ = ConstValueDist(th228_lines),
        cal_offs = Exponential(5.0),
        cal_slope = weibull_from_mx.(1.0, 1.01),
        cal_sqr = Exponential(0.000001),
        σ_fano = weibull_from_mx.(mean_rel_sigma, 2 .* mean_rel_sigma),
        σ_enc = Exponential(0.5),
        n = product_distribution(weibull_from_mx.(ps.peak_counts, 2 .* ps.peak_counts)),
        step_amplitude = product_distribution(weibull_from_mx.(ps.mean_background, 2 .* ps.mean_background)),
        skew_fraction = product_distribution(fill(Uniform(0.01, 0.25), length(peakstats))),
        skew_width = product_distribution(fill(LogUniform(0.001, 0.1), length(peakstats))),
        background = product_distribution(weibull_from_mx.(ps.mean_background, 2 .* ps.mean_background)),
    )
end

pseudo_prior = empirical_prior_from_peakstats(peakstats)

f_trafo = BAT.DistributionTransform(Normal, pseudo_prior)

v_init = mean(pseudo_prior)

f_loglike = let f_fit=f_fit, peakhists=peakhists
    v -> sum(hist_loglike.(Base.Fix2.(f_fit, expand_vars(v)), peakhists))
end

opt_r = optimize((-) ∘ f_loglike ∘ inverse(f_trafo), f_trafo(v_init))
v_ml = inverse(f_trafo)(Optim.minimizer(opt_r))

plot([begin
    h = peakhists[i]
    v = expand_vars(v_ml)[i]
    plot(normalize(h, mode = :density), st = :stepbins, yscale = :log10)
    plot!(minimum(h.edges[1]):0.1:maximum(h.edges[1]), Base.Fix2(f_fit, v))
end; for i in eachindex(peakhists)]...)
