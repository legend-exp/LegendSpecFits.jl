# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

using LegendSpecFits
using Test

using BAT, DensityInterface, Distributions, MeasureBase
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


