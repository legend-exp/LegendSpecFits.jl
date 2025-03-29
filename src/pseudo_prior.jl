function get_standard_pseudo_prior(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :bin_width, :mean_background, :mean_background_step, :mean_background_std), NTuple{8, T}}, fit_func::Symbol; low_e_tail::Bool=true, fixed_position::Bool=false) where T<:Real
    # base priors common with all functions
    window_left = ps.peak_pos - minimum(h.edges[1])
    window_right = maximum(h.edges[1]) - ps.peak_pos
    # base priors common with all functions
    pprior_base = NamedTupleDist(
            μ = ifelse(fixed_position, ConstValueDist(ps.peak_pos), Normal(ps.peak_pos, 0.2*ps.peak_sigma)),
            σ = weibull_from_mx(ps.peak_sigma, 1.5*ps.peak_sigma),
            n = weibull_from_mx(ps.peak_counts, 1.5*ps.peak_counts),
            skew_fraction = ifelse(low_e_tail, truncated(weibull_from_mx(0.002, 0.008), 0.0, 0.5), ConstValueDist(0.0)),
            skew_width = ifelse(low_e_tail, weibull_from_mx(ps.peak_sigma/ps.peak_pos, 1.2*ps.peak_sigma/ps.peak_pos), ConstValueDist(1.0)),
            background = weibull_from_mx(ps.mean_background, ps.mean_background + 5*ps.mean_background_std),
            step_amplitude = weibull_from_mx(ps.mean_background_step, ps.mean_background_step + 5*ps.mean_background_std),
            skew_fraction_highE = ifelse(low_e_tail, truncated(weibull_from_mx(0.002, 0.008), 0.0, 0.1), ConstValueDist(0.0)),
            skew_width_highE = ifelse(low_e_tail, weibull_from_mx(ps.peak_sigma/ps.peak_pos, 1.2*ps.peak_sigma/ps.peak_pos), ConstValueDist(1.0)),
            background_slope = ifelse(ps.mean_background < 5, ConstValueDist(0), truncated(Normal(0, 0.1*ps.mean_background_std / (window_left + window_right)), - ps.mean_background / window_right, 0)),
            background_exp = weibull_from_mx(3e-2, 5e-2)
        )
    
    # extract single prior arguments
    (; μ, σ, n, skew_fraction, skew_width, background, step_amplitude, skew_fraction_highE, skew_width_highE, background_slope, background_exp) = pprior_base
    
    # select prior based on fit function
    if fit_func == :gamma_def
        NamedTupleDist(; μ, σ, n, skew_fraction, skew_width, background, step_amplitude)
    elseif fit_func == :gamma_bckFlat
        NamedTupleDist(; μ, σ, n, skew_fraction, skew_width, background)
    elseif fit_func == :gamma_tails
        NamedTupleDist(; μ, σ, n, skew_fraction, skew_width, background, step_amplitude, skew_fraction_highE, skew_width_highE)
    elseif fit_func == :gamma_tails_bckFlat
        NamedTupleDist(; μ, σ, n, skew_fraction, skew_width, background, skew_fraction_highE, skew_width_highE)
    elseif fit_func == :gamma_bckSlope
        NamedTupleDist(; μ, σ, n, skew_fraction, skew_width, background, step_amplitude, background_slope)
    elseif fit_func == :gamma_bckExp
        NamedTupleDist(; μ, σ, n, skew_fraction, skew_width, background, step_amplitude, background_exp)
    elseif fit_func == :gamma_minimal
        NamedTupleDist(; μ, σ, n, background)
    else
        throw(ArgumentError("Unknown fit function: $fit_func"))
    end
end

function get_pseudo_prior(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :bin_width, :mean_background, :mean_background_step, :mean_background_std), NTuple{8, T}}, fit_func::Symbol; pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true), kwargs...) where T<:Real
    standard_pseudo_prior = get_standard_pseudo_prior(h, ps, fit_func; kwargs...)
    # use standard priors in case of no overwrites given
    if !(:empty in keys(pseudo_prior))
        # check if input overwrite prior has the same fields as the standard prior set
        @assert all(f -> f in keys(standard_pseudo_prior), keys(pseudo_prior)) "Pseudo priors can only have $(keys(standard_pseudo_prior)) as fields."
        # replace standard priors with overwrites
        merge(standard_pseudo_prior, pseudo_prior)
    else
        # take standard priors as pseudo priors with overwrites
        standard_pseudo_prior
    end
end


function get_subpeaks_pseudo_prior(h_survived::Histogram, h_cut::Histogram, ps::NamedTuple, fit_func::Symbol; 
                        pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true), low_e_tail::Bool=true, 
                        fix_σ::Bool=true, fix_skew_fraction::Bool=true, fix_skew_width::Bool=true)
    
    # get standard pseudo priors for both histograms
    standard_pseudo_prior_cut = get_standard_pseudo_prior(h_cut, estimate_single_peak_stats_th228(h_cut), fit_func; low_e_tail=low_e_tail)
    standard_pseudo_prior_survived = get_standard_pseudo_prior(h_survived, estimate_single_peak_stats_th228(h_survived), fit_func; low_e_tail=low_e_tail)
    
    # create standard prior
    standard_pseudo_prior = merge(
            NamedTupleDist(
                μ = ConstValueDist(mvalue(ps.μ)),
                n = ConstValueDist(mvalue(ps.n)),
                background = ConstValueDist(mvalue(ps.background)),
                
                sf = Uniform(0,1), # signal survival fraction
                bsf = Uniform(0,1), # background survival fraction

                σ_survived = ifelse(fix_σ, ConstValueDist(mvalue(ps.σ)), weibull_from_mx(mvalue(ps.σ), 1.5*mvalue(ps.σ))),
                σ_cut = ifelse(fix_σ, ConstValueDist(mvalue(ps.σ)), weibull_from_mx(mvalue(ps.σ), 1.5*mvalue(ps.σ))),
                
                skew_fraction_survived = ifelse(fix_skew_fraction, ConstValueDist(mvalue(ps.skew_fraction)), standard_pseudo_prior_survived.skew_fraction),
                skew_fraction_cut = ifelse(fix_skew_fraction, ConstValueDist(mvalue(ps.skew_fraction)), standard_pseudo_prior_cut.skew_fraction),
                skew_width_survived = ifelse(fix_skew_width, mvalue(ps.skew_width), standard_pseudo_prior_survived.skew_width),
                skew_width_cut = ifelse(fix_skew_width, mvalue(ps.skew_width), standard_pseudo_prior_cut.skew_width),
            ),
        
            if haskey(ps, :step_amplitude)
                NamedTupleDist(
                    step_amplitude = ConstValueDist(mvalue(ps.step_amplitude)),
                    sasf = Uniform(0,1), # step amplitude survival fraction
                )
            else
                NamedTupleDist(
                    μ = ConstValueDist(mvalue(ps.μ)),
                )
            end,

            if haskey(ps, :skew_fraction_highE) && haskey(ps, :skew_width_highE)
                NamedTupleDist(
                    skew_fraction_highE_survived = ifelse(fix_skew_fraction, ConstValueDist(mvalue(ps.skew_fraction_highE)), standard_pseudo_prior_survived.skew_fraction_highE),
                    skew_fraction_highE_cut = ifelse(fix_skew_fraction, ConstValueDist(mvalue(ps.skew_fraction_highE)), standard_pseudo_prior_cut.skew_fraction_highE),
                    skew_width_highE_survived = ifelse(fix_skew_width, mvalue(ps.skew_width_highE), standard_pseudo_prior_survived.skew_width_highE),
                    skew_width_highE_cut = ifelse(fix_skew_width, mvalue(ps.skew_width_highE), standard_pseudo_prior_cut.skew_width_highE),
                )
            else
                NamedTupleDist(
                    μ = ConstValueDist(mvalue(ps.μ)),
                )
            end,

            if haskey(ps, :background_slope)
                NamedTupleDist(
                    background_slope_survived = ifelse(fix_skew_fraction, ConstValueDist(mvalue(ps.background_slope)), standard_pseudo_prior_survived.background_slope),
                    background_slope_cut = ifelse(fix_skew_fraction, ConstValueDist(mvalue(ps.background_slope)), standard_pseudo_prior_cut.background_slope)
                )
            else
                NamedTupleDist(
                    μ = ConstValueDist(mvalue(ps.μ)),
                )
            end,

            if haskey(ps, :background_exp)
                NamedTupleDist(
                    background_exp_survived = ifelse(fix_skew_fraction, ConstValueDist(mvalue(ps.background_exp)), standard_pseudo_prior_survived.background_exp),
                    background_exp_cut = ifelse(fix_skew_fraction, ConstValueDist(mvalue(ps.background_exp)), standard_pseudo_prior_survived.background_exp),
                )
            else
                NamedTupleDist(
                    μ = ConstValueDist(mvalue(ps.μ)),
                )
            end
    )

    # use standard priors in case of no overwrites given
    if !(:empty in keys(pseudo_prior))
        # check if input overwrite prior has the same fields as the standard prior set
        @assert all(f -> f in keys(standard_pseudo_prior), keys(pseudo_prior)) "Pseudo priors can only have $(keys(standard_pseudo_prior)) as fields."
        # replace standard priors with overwrites
        merge(standard_pseudo_prior, pseudo_prior)
    else
        # take standard priors as pseudo priors with overwrites
        standard_pseudo_prior
    end
end


function get_subpeaks_v_ml(v::NamedTuple, fit_func::Symbol)
    if fit_func == :gamma_def
        v_survived = (
                μ = v.μ, 
                σ = v.σ_survived, 
                n = v.n * v.sf, 
                skew_fraction = v.skew_fraction_survived,
                skew_width = v.skew_width_survived,
                background = v.background * v.bsf,
                step_amplitude = v.step_amplitude * v.sasf,
            )
            v_cut = (
                μ = v.μ, 
                σ = v.σ_cut, 
                n = v.n * (1 - v.sf), 
                skew_fraction = v.skew_fraction_cut,
                skew_width = v.skew_width_cut,
                background = v.background * (1 - v.bsf),
                step_amplitude = v.step_amplitude * (1 - v.sasf),
            )
        v_survived, v_cut
    elseif fit_func == :gamma_bckFlat
        v_survived = (
                μ = v.μ, 
                σ = v.σ_survived, 
                n = v.n * v.sf, 
                skew_fraction = v.skew_fraction_survived,
                skew_width = v.skew_width_survived,
                background = v.background * v.bsf,
            )
            v_cut = (
                μ = v.μ, 
                σ = v.σ_cut, 
                n = v.n * (1 - v.sf), 
                skew_fraction = v.skew_fraction_cut,
                skew_width = v.skew_width_cut,
                background = v.background * (1 - v.bsf),
            )
        v_survived, v_cut
    elseif fit_func == :gamma_tails
        v_survived = (
                μ = v.μ, 
                σ = v.σ_survived, 
                n = v.n * v.sf, 
                skew_fraction = v.skew_fraction_survived,
                skew_width = v.skew_width_survived,
                background = v.background * v.bsf,
                step_amplitude = v.step_amplitude * v.sasf,
                skew_fraction_highE = v.skew_fraction_highE_survived,
                skew_width_highE = v.skew_width_highE_survived,
            )
            v_cut = (
                μ = v.μ, 
                σ = v.σ_cut, 
                n = v.n * (1 - v.sf), 
                skew_fraction = v.skew_fraction_cut,
                skew_width = v.skew_width_cut,
                background = v.background * (1 - v.bsf),
                step_amplitude = v.step_amplitude * (1 - v.sasf),
                skew_fraction_highE = v.skew_fraction_highE_cut,
                skew_width_highE = v.skew_width_highE_cut,
            )
        v_survived, v_cut
    elseif fit_func == :gamma_tails_bckFlat
        v_survived = (
                μ = v.μ, 
                σ = v.σ_survived, 
                n = v.n * v.sf, 
                skew_fraction = v.skew_fraction_survived,
                skew_width = v.skew_width_survived,
                background = v.background * v.bsf,
                skew_fraction_highE = v.skew_fraction_highE_survived,
                skew_width_highE = v.skew_width_highE_survived,
            )
            v_cut = (
                μ = v.μ, 
                σ = v.σ_cut, 
                n = v.n * (1 - v.sf), 
                skew_fraction = v.skew_fraction_cut,
                skew_width = v.skew_width_cut,
                background = v.background * (1 - v.bsf),
                skew_fraction_highE = v.skew_fraction_highE_cut,
                skew_width_highE = v.skew_width_highE_cut,
            )
        v_survived, v_cut
    elseif fit_func == :gamma_bckSlope
        v_survived = (
                μ = v.μ, 
                σ = v.σ_survived, 
                n = v.n * v.sf, 
                skew_fraction = v.skew_fraction_survived,
                skew_width = v.skew_width_survived,
                background = v.background * v.bsf,
                step_amplitude = v.step_amplitude * v.sasf,
                background_slope = v.background_slope_survived,
            )
            v_cut = (
                μ = v.μ, 
                σ = v.σ_cut, 
                n = v.n * (1 - v.sf), 
                skew_fraction = v.skew_fraction_cut,
                skew_width = v.skew_width_cut,
                background = v.background * (1 - v.bsf),
                step_amplitude = v.step_amplitude * (1 - v.sasf),
                background_slope = v.background_slope_cut,
            )
        v_survived, v_cut
    elseif fit_func == :gamma_bckExp
        v_survived = (
                μ = v.μ, 
                σ = v.σ_survived, 
                n = v.n * v.sf, 
                skew_fraction = v.skew_fraction_survived,
                skew_width = v.skew_width_survived,
                background = v.background * v.bsf,
                step_amplitude = v.step_amplitude * v.sasf,
                background_exp = v.background_exp_survived,
            )
            v_cut = (
                μ = v.μ, 
                σ = v.σ_cut, 
                n = v.n * (1 - v.sf), 
                skew_fraction = v.skew_fraction_cut,
                skew_width = v.skew_width_cut,
                background = v.background * (1 - v.bsf),
                step_amplitude = v.step_amplitude * (1 - v.sasf),
                background_exp = v.background_exp_cut,
            )
        v_survived, v_cut
    elseif fit_func == :gamma_minimal 
        v_survived = (
            μ = v.μ, 
            σ = v.σ_survived, 
            n = v.n * v.sf, 
            background = v.background * v.bsf,
        )
        v_cut = (
            μ = v.μ, 
            σ = v.σ_cut, 
            n = v.n * (1 - v.sf), 
            background = v.background * (1 - v.bsf),
        )
        v_survived, v_cut
    else
        throw(ArgumentError("Unknown fit function: $fit_func"))
    end
end