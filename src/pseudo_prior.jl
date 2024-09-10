function get_standard_pseudo_prior(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background, :mean_background_step, :mean_background_std), NTuple{7, T}}, fit_func::Symbol; low_e_tail::Bool=true, fixed_position::Bool=false) where T<:Real
    pprior_base = NamedTupleDist(
        μ = ifelse(fixed_position, ConstValueDist(ps.peak_pos), Normal(ps.peak_pos, 1*ps.peak_sigma)),
        σ = weibull_from_mx(ps.peak_sigma, 2*ps.peak_sigma),
        n = weibull_from_mx(ps.peak_counts, 2*ps.peak_counts),
        step_amplitude = weibull_from_mx(ps.mean_background_step, ps.mean_background_step + 5*ps.mean_background_std),
        skew_fraction = ifelse(low_e_tail, truncated(weibull_from_mx(0.002, 0.008), 0.0, 0.5), ConstValueDist(0.0)),
        skew_width = ifelse(low_e_tail, weibull_from_mx(ps.peak_sigma/ps.peak_pos, 2*ps.peak_sigma/ps.peak_pos), ConstValueDist(1.0)),
        background = weibull_from_mx(ps.mean_background, ps.mean_background + 5*ps.mean_background_std),
        )
    if fit_func == :f_fit
        return pprior_base
    else
        window_left = ps.peak_pos - minimum(h.edges[1])
        window_right = maximum(h.edges[1]) - ps.peak_pos
        return merge(pprior_base, 
        if fit_func == :f_fit_bckSlope
            NamedTupleDist(
             background_slope = ifelse(ps.mean_background < 5, ConstValueDist(0), 
             truncated(Normal(0, 0.1*ps.mean_background_std / (window_left + window_right)), - ps.mean_background / window_right, 0)),
            )
        elseif fit_func ==:f_fit_bckExp
            NamedTupleDist(background_exp = weibull_from_mx(3e-2, 5e-2))
        elseif fit_func == :f_fit_tails
            NamedTupleDist(
            skew_fraction_highE = ifelse(low_e_tail, truncated(weibull_from_mx(0.01, 0.05), 0.0, 0.1), ConstValueDist(0.0)),
            skew_width_highE = ifelse(low_e_tail, weibull_from_mx(0.001, 1e-2), ConstValueDist(1.0)),
            )
        end
        )
    end
end

function get_pseudo_prior(h::Histogram, ps::NamedTuple{(:peak_pos, :peak_fwhm, :peak_sigma, :peak_counts, :mean_background, :mean_background_step, :mean_background_std), NTuple{7, T}}, fit_func::Symbol; pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true), kwargs...) where T<:Real
    standard_pseudo_prior = get_standard_pseudo_prior(h, ps, fit_func; kwargs...)
    # use standard priors in case of no overwrites given
    if !(:empty in keys(pseudo_prior))
        # check if input overwrite prior has the same fields as the standard prior set
        @assert all(f -> f in keys(standard_pseudo_prior), keys(pseudo_prior)) "Pseudo priors can only have $(keys(standard_pseudo_prior)) as fields."
        # replace standard priors with overwrites
        pseudo_prior = merge(standard_pseudo_prior, pseudo_prior)
    else
        # take standard priors as pseudo priors with overwrites
        pseudo_prior = standard_pseudo_prior    
    end
    return pseudo_prior
end