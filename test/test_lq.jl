using Test
using LegendSpecFits
using Unitful
using Distributions
using Plots

# Define the test function
@testset "LQ_cut Test" begin
    # Define energy peak parameters
    DEP_µ = 1000.0u"keV"
    DEP_σ = 1.0u"keV"
    n_peak = 5000  # Peak events
    n_bg = n_peak ÷ 10  # Background events

    # Energy calibration (fixed at DEP_µ for all events)
    e_cal = vcat(fill(DEP_µ, n_peak + n_bg), fill(DEP_µ - 5* DEP_σ, n_bg ÷ 2), fill(DEP_µ + 5* DEP_σ, n_bg ÷ 2) )  # Fixed energy value for peak and background

    # LQ Classifier
    # Peak 1: Normally distributed LQ values
    lq_classifier_peak1 = randn(n_peak)
    # Peak 2: Flat background within the peak
    lq_classifier_peak2 = -4 .+ 14 .* rand(n_bg)

    # Below: Flat background below the peak
    lq_classifier_below = -4 .+ 14 .* rand(n_bg ÷ 2)
    lq_classifier_above = -4 .+ 14 .* rand(n_bg ÷ 2)

    # Combine all cases into the LQ classifier array
    lq_classifier_combined = vcat(lq_classifier_peak1, lq_classifier_peak2, lq_classifier_below, lq_classifier_above)

    # Call the LQ_cut function
    result, report = LQ_cut(DEP_µ, DEP_σ, e_cal, lq_classifier_combined)

    plot(report.temp_hists.hist_DEP, label="LQ SEP")
    plot!(report.temp_hists.hist_sb1, label="LQ SB1")
    plot!(report.temp_hists.hist_sb2, label="LQ SB2")
    plot!(report.temp_hists.hist_subtracted, label="DEP Subtracted")
    plot(report.temp_hists.hist_corrected, label="original histogram")
    plot!(report.fit_report.f_fit, label="Fit function")

    # Extract the cutoff value
    report.fit_result.μ
    report.fit_result.σ
    cut_3σ = result.cut
  
    # Calculate the expected mean, sigma and cutoff value
    expected_mean = mean(lq_classifier_peak1)
    expected_sigma = std(lq_classifier_peak1)
    expected_cut = expected_mean + 3 * expected_sigma

    # Test the parameters
    @test isapprox(report.fit_result.μ, expected_mean, atol=0.05)
    @test isapprox(report.fit_result.σ, expected_sigma, atol=0.05)
    @test isapprox(cut_3σ, expected_cut, atol=0.1)
end


@testset "Test lq_drift_time_correction with Tail" begin

    # Generate 10000 data points
    N = 10000

    # Generate dt_eff (drift time) with a Gaussian distribution
    dt_mean = 500.0
    dt_std = 100.0
    dt_eff = (dt_mean .+ dt_std .* randn(N)) .* u"µs"

    # Generate lq_norm with a linear dependence on dt_eff + some noise
    true_slope = 0.002
    true_intercept = 3
    noise_level = 0.1
    lq_norm = true_slope .* ustrip.(dt_eff) .+ true_intercept .+ randn(N) .* noise_level

    # Introduce a tail for low drift times (below 350)
    tail_threshold = 350.0
    tail_factor = 0.1

    lq_norm_tail = [lq + (tail_factor * (tail_threshold - dt)^2 / tail_threshold) * (dt < tail_threshold ? 1.0 : 0.0) for (lq, dt) in zip(lq_norm, ustrip.(dt_eff))]

    # Generate energies at DEP
    e_cal = 1600.0u"keV" .+ randn(N) .* 1.0u"keV"

    # DEP parameters
    DEP_µ = 1600.0u"keV"
    DEP_σ = 1.0u"keV"

    # Call the function with mode=:percentile
    result, report = lq_drift_time_correction(lq_norm_tail, dt_eff, e_cal, DEP_µ, DEP_σ; mode=:percentile)

    # Retrieve the fitted parameters
    fitted_slope = report.drift_time_func(1.0) - report.drift_time_func(0.0)
    fitted_intercept = report.drift_time_func(0.0)

    # Check if the fitted slope and intercept match the true values (linear region)
    @test isapprox(fitted_slope, true_slope, atol=0.0001)
    @test isapprox(fitted_intercept, true_intercept, atol=0.1)
end


