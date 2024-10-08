function get_aoe_standard_pseudo_prior(h::Histogram, ps::NamedTuple, fit_func::Symbol; fixed_position::Bool=false)
    pprior_base = NamedTupleDist(
        μ = ifelse(fixed_position, ConstValueDist(ps.peak_pos), Normal(ps.peak_pos, 0.5*ps.peak_sigma)),
        σ = weibull_from_mx(ps.peak_sigma, 2*ps.peak_sigma),
        n = LogUniform(0.01*ps.peak_counts, 5*ps.peak_counts),
        B = LogUniform(0.1*ps.mean_background, 10*ps.mean_background),
        δ = LogUniform(0.001, 10.0)
    )
    if fit_func == :f_fit
        return pprior_base
    else
        throw(ArgumentError("fit_func $fit_func not supported for aoe peakshapes"))
    end
end

function get_aoe_pseudo_prior(h::Histogram, ps::NamedTuple, fit_func::Symbol; pseudo_prior::NamedTupleDist=NamedTupleDist(empty = true), kwargs...)
    standard_pseudo_prior = get_aoe_standard_pseudo_prior(h, ps, fit_func; kwargs...)
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