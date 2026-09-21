using Test
using LegendSpecFits
using Unitful
using Distributions
using TypedTables
using LegendDataManagement

@testset "Test lq_drift_time_correction with Tail" begin
    # Generate 10000 data points
    N = 10000

    # Generate q_drift (effective drift time) with a Gaussian distribution
    dt_mean = 500.0
    dt_std = 100.0
    q_drfit = (dt_mean .+ dt_std .* randn(N))

    # Generate lq_norm with a linear dependence on q_drfit + some noise
    true_slope = 0.002
    true_intercept = 3
    noise_level = 0.1
    lq_norm = true_slope .* ustrip.(q_drfit) .+ true_intercept .+ randn(N) .* noise_level

    # Introduce a tail for low drift times (below 350)
    tail_threshold = 350.0
    tail_factor = 0.1

    lq_norm_tail = [lq + (tail_factor * (tail_threshold - dt)^2 / tail_threshold) * (dt < tail_threshold ? 1.0 : 0.0) for (lq, dt) in zip(lq_norm, ustrip.(q_drfit))]

    # Generate energies at dep
    e_cal = 1600.0u"keV" .+ randn(N) .* 1.0u"keV"

    # dep parameters
    dep_µ = 1600.0u"keV"
    dep_σ = 1.0u"keV"

    @testset "LQ minimizer ctc" begin
        result, report = ctc_lq(lq_norm_tail, e_cal, q_drfit, dep_µ, dep_σ; pol_order=1)

        # Get the optimized correction factor
        fct = report.fct[1]

        # Check if the fitted slope and intercept match the true values (linear region)
        @test isapprox(fct, true_slope, atol=0.0001)
    end

    @testset "linear LQ ctc" begin
        # Call the function with mode=:percentile
        result, report = lq_ctc_lin_fit(lq_norm_tail, q_drfit, e_cal, dep_µ, dep_σ; pol_fit_order = 1, ctc_driftime_cutoff_method=:percentile)

        # Retrieve the fitted parameters
        fitted_slope = report.drift_time_func(1.0) - report.drift_time_func(0.0)
        fitted_intercept = report.drift_time_func(0.0)

        # Check if the fitted slope and intercept match the true values (linear region)
        @test isapprox(fitted_slope, true_slope, atol=0.0001)
        @test isapprox(fitted_intercept, true_intercept, atol=0.1)#
    end
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
    result, report = lq_norm(dep_µ, dep_σ, e_cal, lq_classifier_combined; lq_class_expression = :lq)
  
    # Calculate the expected mean, sigma and cutoff value
    expected_mean = mean(lq_classifier_peak1)
    expected_sigma = std(lq_classifier_peak1)

    # Test the parameters
    @test isapprox(report.fit_result.μ, expected_mean, atol=0.05)
    @test isapprox(report.fit_result.σ, expected_sigma, atol=0.05)

    # Test the normalization
    lq_table = Table(lq = lq_classifier_peak1)
    lq_normalized = ljl_propfunc(result.func).(lq_table)
    @test isapprox(mean(lq_normalized), 0.0, atol=0.05)
    @test isapprox(std(lq_normalized), 1.0, atol=0.05)
end
