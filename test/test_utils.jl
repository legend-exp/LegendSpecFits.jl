# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    generate_mc_spectrum(n_tot::Int=200000,; f_fit::Base.Callable=th228_fit_functions.f_fit)
Sample Legend200 calibration data based on "Inverse Transform Sampling" method
# Method:
* pdf of th228 calibration calibration peak is estimated from fit model function f_fit from LegendSpecFits
* calculate the cumulative distribution function F(x)
* generate a random number u from a uniform distribution between 0 and 1.
* find the value x such that  F(x) = u  by solving for  x . done by interpolation of the inverse cdf
* repeat for many u : energy samples 
"""
function generate_mc_spectrum(n_tot::Int=200000,; f_fit::Base.Callable=LegendSpecFits.get_th228_fit_functions().f_fit)

    th228_lines =  [583.191,  727.330,  860.564,  1592.53,    1620.50,    2103.53,    2614.51]

    v = [ 
        (μ = 2103.53,   σ = 2.11501,    n = 385.123,    step_amplitude = 1e-242,    skew_fraction = 0.005,  skew_width = 0.1,   background = 55),
        (μ = 860.564,   σ = 0.817838,   n = 355.84,     step_amplitude = 1.2,       skew_fraction = 0.005,  skew_width = 0.099, background = 35),
        (μ = 727.33,    σ = 0.721594,   n = 452.914,    step_amplitude = 5.4,       skew_fraction = 0.005,  skew_width = 0.1,   background = 28),
        (μ = 1620.5,    σ = 1.24034,    n = 130.256,    step_amplitude = 1e-20,     skew_fraction = 0.005,  skew_width = 0.1,   background = 16),
        (μ = 583.191,   σ = 0.701544,   n = 1865.52,    step_amplitude = 17.9,      skew_fraction = 0.1,    skew_width = 0.1,   background = 16),
        (μ = 1592.53,   σ = 2.09123,    n = 206.827,    step_amplitude = 1e-21,     skew_fraction = 0.005,  skew_width = 0.1,   background = 17),
        (μ = 2614.5,   σ = 1.51289,    n = 3130.43,    step_amplitude = 1e-101,    skew_fraction = 0.1,    skew_width = 0.003, background = 1)
]
    # calculate pdf and cdf functions 
    bin_centers_all     =  Array{StepRangeLen,1}(undef, length(th228_lines))
    model_counts_all    =  Array{Array{Float64},1}(undef, length(th228_lines))
    model_cdf_all       =  Array{Array{Float64},1}(undef, length(th228_lines))
    energy_mc_all       =  Array{Array{Float64},1}(undef, length(th228_lines))
    PeakMax             = zeros(length(th228_lines))#u"keV"

    for i=1:length(th228_lines)  # get fine binned model function to estimate pdf 
        n_step = 5000 # fine binning 
        bin_centers_all[i] = range(v[i].µ, stop=v[i].µ+30, length=n_step)
        bw = bin_centers_all[i][2]-bin_centers_all[i][1]
        bin_widths = range(bw,bw, length=n_step) 

        # save as intermediate result 
        model_counts_all[i] = LegendSpecFits._get_model_counts(f_fit, v[i], bin_centers_all[i], bin_widths)
        PeakMax[i] = maximum(model_counts_all[i])

        # create CDF
        model_cdf_all[i] = cumsum(model_counts_all[i])
        model_cdf_all[i] = model_cdf_all[i]./maximum(model_cdf_all[i])
    end

    # weights each peak with amplitude 
    PeakMaxRel = PeakMax./sum(PeakMax)
    n_i = round.(Int,PeakMaxRel.*n_tot)

    # do the sampling: drawn from uniform distribution 
    for i=1:length(th228_lines)
        bandwidth = maximum(model_cdf_all[i])-minimum(model_cdf_all[i])
        rand_i = minimum(model_cdf_all[i]).+bandwidth.*rand(n_i[i]); # make sure sample is within model range 
        interp_cdf_inv = linear_interpolation(model_cdf_all[i], bin_centers_all[i]) # inverse cdf
        energy_mc_all[i] = interp_cdf_inv.(rand_i) 
    end

    energy_mc = fast_flatten(map(x-> x .* u"keV",energy_mc_all))
    th228_lines = th228_lines .* u"keV"
    return energy_mc, th228_lines
end

