# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits
using Test

using BAT, DensityInterface, Distributions, MeasureBase
using StatsBase
using SpecialFunctions: erfc
import ForwardDiff, LinearAlgebra

using LegendSpecFits: GaussPeak, LowETailPeak, HighETailPeak, GaussianStep, PolynomialShape, ExpDecay


@testset "shape_measures" begin
    x = 2.35

    @test densityof(GaussPeak(2.0, 0.5), x) ≈ LegendSpecFits.gauss_pdf(x, 2.0, 0.5)
    @test logdensityof(GaussPeak(2.0, 0.5), x) ≈ logpdf(Normal(2.0, 0.5), x)
    @test densityof(LowETailPeak(2.0, 0.5, 0.3), x) ≈ LegendSpecFits.ex_gauss_pdf(-x, -2.0, 0.5, 0.3)
    @test densityof(HighETailPeak(2.0, 0.5, 0.3), x) ≈ LegendSpecFits.ex_gauss_pdf(x, 2.0, 0.5, 0.3)
    @test densityof(GaussianStep(2.0, 0.5), x) ≈ erfc((x - 2.0) / (sqrt(2) * 0.5)) / 2
    @test densityof(GaussianStep(2.0, 0.5), -5.0) ≈ 1.0
    @test densityof(PolynomialShape(1.0, 0.5, 0.1), x) ≈ 1.0 + 0.5x + 0.1x^2
    @test densityof(PolynomialShape(1.0, 0.5, 0.1, x0 = 2.0), x) ≈ 1.0 + 0.5 * (x - 2.0) + 0.1 * (x - 2.0)^2
    @test densityof(ExpDecay(2.0, 1.5), x) ≈ exp(-(x - 2.0) / 1.5)

    # Stabilized step log-density far above the edge:
    @test isfinite(logdensityof(GaussianStep(2.0, 0.5), 50.0))
    @test logdensityof(GaussianStep(2.0, 0.5), 5.0) ≈ log(densityof(GaussianStep(2.0, 0.5), 5.0))

    # Native log-space EMG evaluation: matches the peakshape functions,
    # stays finite and AD-friendly deep in the tail (where the density
    # itself underflows) and degrades to a Gaussian for zero tail scale:
    @test logdensityof(LowETailPeak(2.0, 0.5, 0.3), x) ≈ log(LegendSpecFits.ex_gauss_pdf(-x, -2.0, 0.5, 0.3))
    @test logdensityof(HighETailPeak(2.0, 0.5, 0.3), x) ≈ log(LegendSpecFits.ex_gauss_pdf(x, 2.0, 0.5, 0.3))
    @test logdensityof(HighETailPeak(0.0, 1.0, 0.01), 200.0) == logdensityof(LowETailPeak(0.0, 1.0, 0.01), -200.0)
    @test logdensityof(LowETailPeak(0.0, 1.0, 0.01), -200.0) < -1000
    @test isfinite(logdensityof(LowETailPeak(0.0, 1.0, 0.01), -200.0))
    @test isfinite(ForwardDiff.derivative(x -> logdensityof(LowETailPeak(0.0, 1.0, 0.01), x), -200.0))
    @test logdensityof(LowETailPeak(2.0, 0.5, 0.0), x) ≈ logpdf(Normal(2.0, 0.5), x)

    @test massof(GaussPeak(2.0, 0.5)) == 1
    @test massof(LowETailPeak(2.0, 0.5, 0.3)) == 1
    @test massof(HighETailPeak(2.0, 0.5, 0.3)) == 1
    # Unit masses must not promote the precision of amplitude arithmetic:
    @test massof(GaussPeak(2.0, 0.5)) * 1.0f0 === 1.0f0
    @test massof(GaussianStep(2.0, 0.5)) isa MeasureBase.UnknownMass
    @test LinearAlgebra.normalize(GaussPeak(2.0, 0.5)) === GaussPeak(2.0, 0.5)
    @test_throws ArgumentError LinearAlgebra.normalize(GaussianStep(2.0, 0.5))

    # Invalid shape parameters are rejected at construction:
    @test_throws ArgumentError GaussPeak(2.0, -0.5)
    @test_throws ArgumentError GaussPeak(2.0, 0.0)
    @test_throws ArgumentError LowETailPeak(2.0, -0.5, 0.1)
    @test_throws ArgumentError LowETailPeak(2.0, 0.5, -0.1)
    @test_throws ArgumentError HighETailPeak(2.0, 0.0, 0.1)
    @test_throws ArgumentError GaussianStep(2.0, -1.0)
    @test_throws ArgumentError ExpDecay(2.0, 0.0)
    # Infinite scales do not define the advertised unit-mass measures:
    @test_throws ArgumentError GaussPeak(0.0, Inf)
    @test_throws ArgumentError LowETailPeak(2.0, 0.5, Inf)
    @test_throws ArgumentError HighETailPeak(2.0, 0.5, Inf)
    @test_throws ArgumentError HighETailPeak(2.0, Inf, 0.1)
    @test_throws ArgumentError GaussianStep(2.0, Inf)
    @test LowETailPeak(2.0, 0.5, 0.0) isa LowETailPeak  # Gaussian limit
    @test ExpDecay(2.0, Inf) isa ExpDecay  # constant-density limit

    # Model algebra uses the BAT measure combinators:
    model = 100.0 * ((1 - 0.1) * GaussPeak(2.0, 0.5) + 0.1 * LowETailPeak(2.0, 0.5, 0.3)) +
        PolynomialShape(5.0, 0.1) + 20.0 * GaussianStep(2.0, 0.5)
    @test model isa BAT.BATSuperpositionMeasure
    @test densityof(model, x) ≈ 90 * densityof(GaussPeak(2.0, 0.5), x) +
        10 * densityof(LowETailPeak(2.0, 0.5, 0.3), x) +
        densityof(PolynomialShape(5.0, 0.1), x) + 20 * densityof(GaussianStep(2.0, 0.5), x)
end


@testset "poisson_process" begin
    edges = 0.0:0.5:4.0
    d = Normal(2.0, 0.7)
    masses = LegendSpecFits._bin_masses(BAT.batmeasure(d), collect(edges))

    # Simpson's rule bin masses against exact CDF differences:
    @test masses ≈ diff(cdf.(d, collect(edges))) rtol = 1e-3

    # Bin masses distribute over the measure tree:
    m1 = GaussPeak(2.0, 0.5)
    m2 = PolynomialShape(5.0, 0.1)
    @test LegendSpecFits._bin_masses(2.0 * m1 + m2, collect(edges)) ≈
        2.0 * LegendSpecFits._bin_masses(m1, collect(edges)) + LegendSpecFits._bin_masses(m2, collect(edges))

    # Shapes are integrated via analytic smf differences, so a peak far
    # narrower than a bin keeps its full mass (Simpson returned zero here):
    @test LegendSpecFits._bin_masses(GaussPeak(0.25, 0.001), [0.0, 1.0]) ≈ [1.0]
    @test sum(LegendSpecFits._bin_masses(GaussPeak(2.0, 0.005), collect(1.9:0.01:2.1))) ≈ 1.0
    @test smf(GaussPeak(2.0, 0.5), 2.35) ≈ cdf(Normal(2.0, 0.5), 2.35)
    # smf differences match resolved Simpson quadrature for all shapes:
    edges_f = collect(1.8:0.001:2.2)
    for m in (LowETailPeak(2.0, 0.05, 0.03), HighETailPeak(2.0, 0.05, 0.03),
              GaussianStep(2.0, 0.5), ExpDecay(2.0, 1.5))
        @test LegendSpecFits._bin_masses(m, edges_f) ≈ LegendSpecFits._simpson_bin_masses(m, edges_f) rtol = 1e-6
    end
    # EMG tail-peak masses: near-unit total, and mirror symmetry about μ:
    @test smf(LowETailPeak(2.0, 0.05, 0.03), 3.0) - smf(LowETailPeak(2.0, 0.05, 0.03), 1.0) ≈ 1.0
    @test smf(HighETailPeak(2.0, 0.05, 0.03), 2.1) - smf(HighETailPeak(2.0, 0.05, 0.03), 1.9) ≈
        smf(LowETailPeak(2.0, 0.05, 0.03), 2.1) - smf(LowETailPeak(2.0, 0.05, 0.03), 1.9)
    # Analytic checks for the background shapes:
    @test LegendSpecFits._bin_masses(PolynomialShape(1.0, 0.5, 0.1, x0 = 2.0), [1.0, 3.0]) ≈ [2 + 0.2 / 3]
    @test LegendSpecFits._bin_masses(ExpDecay(2.0, Inf), [0.0, 0.5, 1.0]) ≈ [0.5, 0.5]
    @test only(LegendSpecFits._bin_masses(GaussianStep(2.0, 0.5), [-100.0, -99.0])) ≈ 1.0
    # AD through the exact bin masses:
    @test isfinite(ForwardDiff.derivative(σ -> only(LegendSpecFits._bin_masses(GaussPeak(2.0, σ), [1.9, 2.0])), 0.05))
    @test isfinite(ForwardDiff.derivative(θ -> only(LegendSpecFits._bin_masses(LowETailPeak(2.0, 0.05, θ), [1.8, 2.0])), 0.03))

    # Binned Poisson process against an explicit computation, the variates
    # are the vectors of bin counts:
    h = Histogram(edges, fill(3, length(edges) - 1))
    bpp = BinnedPoissonProcess(100.0 * m1 + m2, edges)
    λ = LegendSpecFits._bin_masses(100.0 * m1 + m2, collect(edges))
    @test logdensityof(bpp, h.weights) ≈ sum(logpdf.(Poisson.(λ), h.weights))

    # Negative rates are invalid, empty bins must not reward them (regression):
    @test LegendSpecFits._poisson_logpdf(-1.0, 0.0) == -Inf
    @test LegendSpecFits._poisson_logpdf(-1.0, 2.0) == -Inf
    @test LegendSpecFits._poisson_logpdf(0.0, 0.0) == 0.0
    @test LegendSpecFits._poisson_logpdf(0.0, 1.0) == -Inf

    # Distribution intensities are converted to measures:
    @test BinnedPoissonProcess(d, edges) isa BinnedPoissonProcess{<:BAT.BATMeasure}

    # Binned likelihood from a model kernel and from a fixed intensity:
    ℒ = binned_poisson_likelihood(p -> p.a * m1 + m2, h)
    @test logdensityof(ℒ, (a = 100.0,)) ≈ logdensityof(bpp, h.weights)
    @test isfinite(ForwardDiff.derivative(a -> logdensityof(ℒ, (a = a,)), 100.0))
    ℒ0 = binned_poisson_likelihood(100.0 * m1 + m2, h)
    @test logdensityof(ℒ0, NamedTuple()) ≈ logdensityof(bpp, h.weights)

    # Only non-negative integer counts are valid Poisson observations,
    # integer-valued floating-point counts are accepted:
    n_bins = length(edges) - 1
    @test_throws ArgumentError binned_poisson_likelihood(100.0 * m1 + m2, Histogram(edges, [1.5; fill(3.0, n_bins - 1)]))
    @test_throws ArgumentError binned_poisson_likelihood(100.0 * m1 + m2, Histogram(edges, [-1; fill(3, n_bins - 1)]))
    ℒf = binned_poisson_likelihood(100.0 * m1 + m2, Histogram(edges, fill(3.0, n_bins)))
    @test logdensityof(ℒf, NamedTuple()) ≈ logdensityof(bpp, h.weights)
end


