# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using PrecompileTools: @setup_workload, @compile_workload
import Logging

# Calibration chains on small synthetic spectra, covering the method
# specializations that the energy, A/E and LQ calibration processors call.
@setup_workload begin
    @compile_workload begin
        Logging.with_logger(Logging.NullLogger()) do
            # Th228 energy calibration, run level and partition level
            let
                rng = Random.Xoshiro(1)
                lines = [583.187, 727.33, 860.557, 1592.513, 1620.5, 2103.512, 2614.511]u"keV"
                names = [:Tl208a, :Bi212a, :Tl208b, :Tl208DEP, :Bi212FEP, :Tl208SEP, :Tl208FEP]
                fit_funcs = [:gamma_def, :gamma_def, :gamma_def, :gamma_tails_bckFlat, :gamma_def, :gamma_tails_bckFlat, :gamma_def]
                left_windows = [25.0, 25.0, 25.0, 25.0, 15.0, 25.0, 35.0]u"keV"
                right_windows = [25.0, 25.0, 25.0, 15.0, 25.0, 25.0, 30.0]u"keV"
                weights = [5000.0, 450.0, 350.0, 200.0, 130.0, 385.0, 1500.0]
                n_events = 50_000
                n_peaks = round.(Int, n_events .* weights ./ sum(weights))
                e_true = Float64[]
                for (line, n) in zip(ustrip.(lines), n_peaks)
                    append!(e_true, line .+ (0.8 + 4e-4 * line) .* randn(rng, round(Int, 0.85 * n)))
                    append!(e_true, line .+ 30 .* (2 .* rand(rng, round(Int, 0.15 * n)) .- 1))
                end
                c_cal = 0.123u"keV"
                e_cal = e_true .* u"keV"
                e_uncal = ustrip.(e_cal ./ c_cal)

                result_simple, _ = simple_calibration(e_uncal, lines, left_windows, right_windows; calib_type = :th228, quantile_perc = NaN,
                                                      binning_peak_window = 4.0u"keV")
                result_fit, _ = fit_peaks(result_simple.peakhists, result_simple.peakstats, names; e_unit = result_simple.unit, calib_type = :th228,
                                          fit_func = fit_funcs, m_cal_simple = result_simple.c)
                peakhists, peakstats, _, _ = peakhists_gamma(e_cal, lines, left_windows, right_windows)
                result_fit, _ = fit_peaks(peakhists, peakstats, names; e_unit = u"keV", calib_type = :th228, fit_func = fit_funcs)
                fit_peaks(peakhists, peakstats, names; e_unit = u"keV", calib_type = :th228, fit_func = fit_funcs, uncertainty = false)

                mu_fit = [result_fit[p].centroid for p in names]
                fit_calibration(1, mu_fit, lines; uncertainty = true)
                fit_calibration(1, mu_fit, lines; e_expression = "e_uncal", m_cal_simple = result_simple.c)
                fwhm_fit = [result_fit[p].fwhm for p in names]
                fit_fwhm(2, lines, fwhm_fit; e_type_cal = :e_cal, uncertainty = true)
                fit_fwhm(1, lines, fwhm_fit; e_type_cal = :e_cal, e_expression = "e_uncal", uncertainty = true)
            end

            # A/E Compton band fits, corrections and charge trapping correction
            let
                rng = Random.Xoshiro(2)
                n_events = 150_000
                e_true = 500 .+ 2000 .* rand(rng, n_events)
                mu_aoe(e) = 1.01 - 4e-6 * e
                sigma_aoe(e) = sqrt(5e-3^2 + 12.0^2 / e^2)
                aoe = [mu_aoe(e) + sigma_aoe(e) * randn(rng) - (rand(rng) < 0.2 ? 5 * sigma_aoe(e) * randexp(rng) : 0.0) for e in e_true]
                e_cal = e_true .* u"keV"
                compton_bands = collect((550:70:2350)u"keV")
                compton_window = 20.0u"keV"

                cut_single_peak(aoe, 0.0, quantile(filter(isfinite, aoe), 0.99); n_bins = -1)
                band_hists = generate_aoe_compton_bands(aoe, e_cal, compton_bands, compton_window)
                result_bands, _ = fit_aoe_compton(band_hists.peakhists, band_hists.peakstats, compton_bands; uncertainty = true)
                mu_fit = [result_bands[band].μ for band in compton_bands]
                sigma_fit = [result_bands[band].σ for band in compton_bands]
                result_single, _ = fit_aoe_corrections(compton_bands, mu_fit, sigma_fit; aoe_expression = "a_sg / e_101010", e_expression = "e_cusp_ctc_cal")
                fit_aoe_compton_combined(band_hists.peakhists, band_hists.peakstats, compton_bands, result_single; e_expression = "e_cusp_ctc_cal",
                                         aoe_expression = "a_sg / e_101010", uncertainty = true)

                aoe_corr = [(a - mu_aoe(e)) / sigma_aoe(e) for (a, e) in zip(aoe, e_true)]
                qdrift_e = 100 .+ 10 .* randn(rng, n_events)
                aoe_corr .-= 0.02 .* (qdrift_e .- 100)
                ctc_aoe(aoe_corr, e_cal, qdrift_e, compton_bands; aoe_expression = "aoe_corr", qdrift_expression = "qdrift / e_101010")
            end

            # A/E cut and survival fractions
            let
                rng = Random.Xoshiro(3)
                lines = [1592.53, 1620.5, 2103.53, 2614.51]u"keV"
                names = [:Tl208DEP, :Bi212FEP, :Tl208SEP, :Tl208FEP]
                fit_funcs = [:gamma_tails_bckFlat, :gamma_def, :gamma_tails_bckFlat, :gamma_def]
                left_windows = [20.0, 25.0, 25.0, 35.0]u"keV"
                right_windows = [20.0, 25.0, 25.0, 25.0]u"keV"
                e_true = 1400 .+ 1400 .* rand(rng, 40_000)
                for (line, n) in zip(ustrip.(lines), [6000, 2500, 3500, 4000])
                    append!(e_true, line .+ (1.0 + 4e-4 * line) .* randn(rng, n))
                end
                e_cal = e_true .* u"keV"
                aoe_corr = randn(rng, length(e_true)) .- [rand(rng) < 0.15 ? 5 * randexp(rng) : 0.0 for _ in e_true]
                low_cut = -3.0
                high_cut = 3.0

                result_cut, _ = get_low_aoe_cut(aoe_corr, e_cal; dep = 1592.53u"keV", window = [20.0, 20.0]u"keV", cut_search_interval = (-25.0, 0.0),
                                                bin_width_window = 3.0u"keV", rtol = 1e-5, maxiters = 200, dep_sf = 0.9, fixed_position = false,
                                                sigma_high_sided = 3.0, fit_func = :gamma_tails_bckFlat, uncertainty = true)
                get_peaks_survival_fractions(aoe_corr, e_cal, lines, names, left_windows, right_windows; low_cut = result_cut.lowcut,
                                             bin_width_window = 3.0u"keV", fit_funcs, uncertainty = true)
                get_peaks_survival_fractions(aoe_corr, e_cal, lines, names, left_windows, right_windows; low_cut = result_cut.lowcut,
                                             high_cut = result_cut.highcut, bin_width_window = 3.0u"keV", fit_funcs, uncertainty = true)
                get_continuum_survival_fraction(aoe_corr, e_cal, 2039.0u"keV", 35.0u"keV"; low_cut = result_cut.lowcut)
                get_continuum_survival_fraction(aoe_corr, e_cal, 2039.0u"keV", 35.0u"keV"; low_cut = result_cut.lowcut, high_cut = result_cut.highcut)
                get_peaks_survival_fractions(-aoe_corr, e_cal, lines, names, left_windows, right_windows; high_cut = 3.0, bin_width_window = 3.0u"keV",
                                             fit_funcs, uncertainty = true)
                get_continuum_survival_fraction(-aoe_corr, e_cal, 2039.0u"keV", 35.0u"keV"; high_cut = 3.0)
            end

            # Energy charge trapping correction
            let
                rng = Random.Xoshiro(4)
                n_events = 30_000
                qdrift = 100 .+ 15 .* randn(rng, n_events)
                e_true = 2614.51 .+ 1.4 .* randn(rng, n_events)
                e_cal = (e_true .- 0.02 .* (qdrift .- 100)) .* u"keV"
                ctc_energy(e_cal, qdrift, 2614.51u"keV", (35.0u"keV", 30.0u"keV"), 0.123u"keV"; e_expression = "e_cusp", pol_order = 1)
            end

            # LQ drift time correction and normalization
            let
                rng = Random.Xoshiro(5)
                n_events = 40_000
                dep_mu = 1592.53u"keV"
                dep_σ = 1.1u"keV"
                e_cal = (1592.53 .+ 1.1 .* randn(rng, n_events)) .* u"keV"
                dt_eff = 500 .+ 100 .* randn(rng, n_events)
                lq = 0.002 .* dt_eff .+ 3 .+ 0.1 .* randn(rng, n_events)
                lq_ctc_correction(lq, dt_eff, e_cal, dep_mu, dep_σ; ctc_dep_edgesigma = 3.0, ctc_lq_precut_relative_cut = 0.25, lq_outlier_sigma = 2.0,
                                  ctc_driftime_cutoff_method = :percentile, dt_eff_outlier_sigma = 2.0, lq_e_corr_expression = "lq / e",
                                  dt_eff_expression = "qdrift / e", ctc_dt_eff_low_quantile = 0.15, ctc_dt_eff_high_quantile = 0.95, pol_fit_order = 1,
                                  uncertainty = false)
                e_sb = vcat(e_cal, (1592.53 .+ 25 .* (2 .* rand(rng, n_events ÷ 2) .- 1)) .* u"keV")
                lq_ctc = vcat(lq, 3.5 .+ 1.0 .* randn(rng, n_events ÷ 2)) .- 3.0
                lq_norm(dep_mu, dep_σ, e_sb, lq_ctc; dep_sideband_sigma = 4.5, cut_truncation_sigma = 2.5, uncertainty = true, lq_class_expression = "lq")
            end
        end
    end
end
