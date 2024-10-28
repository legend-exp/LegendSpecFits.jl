function get_aoe_standard_pseudo_prior(h::Histogram, ps::NamedTuple, fit_func::Symbol; fixed_position::Bool=false)
    pprior_base = NamedTupleDist(
        μ = ifelse(fixed_position, ConstValueDist(ps.peak_pos), Normal(ps.peak_pos, 0.5*ps.peak_sigma)),
        σ = weibull_from_mx(ps.peak_sigma, 1.5*ps.peak_sigma),
        n = weibull_from_mx(ps.peak_counts, 1.5*ps.peak_counts),
        B = LogUniform(0.1*ps.mean_background, 10*ps.mean_background),
        δ = LogUniform(0.001, 10.0),
        μ2 = Normal(-15, 5),
        σ2 = weibull_from_mx(10, 15),
        B2 = weibull_from_mx(ps.mean_background, 1.5*ps.mean_background),
        δ2 = weibull_from_mx(5.0, 10.0),
    )

    # extract single prior arguments
    (; μ, σ, n, B, δ, µ2, σ2, B2, δ2) = pprior_base

    # select prior based on fit function
    if fit_func == :aoe_one_bck
        NamedTupleDist(; μ, σ, n, B, δ)
    elseif fit_func == :aoe_two_bck
        NamedTupleDist(; μ, σ, n, B, δ, μ2, σ2, B2, δ2)
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