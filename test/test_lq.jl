using Test
using LegendSpecFits
using Unitful
using Distributions

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

    # Generate energies at dep
    e_cal = 1600.0u"keV" .+ randn(N) .* 1.0u"keV"

    # dep parameters
    dep_µ = 1600.0u"keV"
    dep_σ = 1.0u"keV"

    # Call the function with mode=:percentile
    result, report = lq_ctc_correction(lq_norm_tail, dt_eff, e_cal, dep_µ, dep_σ; ctc_driftime_cutoff_method=:percentile)

    # Retrieve the fitted parameters
    fitted_slope = report.drift_time_func(1.0) - report.drift_time_func(0.0)
    fitted_intercept = report.drift_time_func(0.0)

    # Check if the fitted slope and intercept match the true values (linear region)
    @test isapprox(fitted_slope, true_slope, atol=0.0001)
    @test isapprox(fitted_intercept, true_intercept, atol=0.1)
end


# Define the test function
@testset "lq_cut Test" begin
    # Define energy peak parameters
    dep_µ = 1000.0u"keV"
    dep_σ = 1.0u"keV"
    n_peak = 50000  # Peak events
    n_bg = div(n_peak, 10)  # Background events

    # Energy calibration (fixed at dep_µ for all events)
    e_cal = vcat(fill(dep_µ, n_peak + n_bg), fill(dep_µ - 5* dep_σ, n_bg ÷ 2), fill(dep_µ + 5* dep_σ, n_bg ÷ 2) )  # Fixed energy value for peak and background

    # LQ Classifier
    # Peak 1: Normally distributed LQ values
    peak_mean = 2.5
    peak_std = 1.5
    lq_classifier_peak1 = peak_mean .+ peak_std .* randn(n_peak)

    # Bkg peak: Flat background within the peak
    lq_classifier_peak2 = -4 .+ 14 .* rand(n_bg)

    # Bkg sidebands: Flat background below the peak
    lq_classifier_below = -4 .+ 14 .* rand(n_bg ÷ 2)
    lq_classifier_above = -4 .+ 14 .* rand(n_bg ÷ 2)

    # Combine all cases into the LQ classifier array
    lq_classifier_combined = vcat(lq_classifier_peak1, lq_classifier_peak2, lq_classifier_below, lq_classifier_above)

    # Call the LQ_cut function
    result, report = lq_cut(dep_µ, dep_σ, e_cal, lq_classifier_combined)

    # Extract the cutoff value
    cut_3σ = result.cut
  
    # Calculate the expected mean, sigma and cutoff value
    expected_mean = mean(lq_classifier_peak1)
    expected_sigma = std(lq_classifier_peak1)
    expected_cut = expected_mean + 3 * expected_sigma

    # Test the parameters
    @test isapprox(report.fit_result.μ, expected_mean, atol=0.05)
    @test isapprox(report.fit_result.σ, expected_sigma, atol=0.05)
    @test isapprox(cut_3σ, expected_cut, atol=0.1)

    # Test the normalization
    lq_norm = (lq_classifier_peak1 .- report.fit_result.μ ) ./ report.fit_result.σ
    @test isapprox(mean(lq_norm), 0.0, atol=0.05)
    @test isapprox(std(lq_norm), 1.0, atol=0.05) 
end
