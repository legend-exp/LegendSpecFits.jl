

function fit_enc_sigmas(enc_grid::Matrix{T}, enc_grid_rt::StepRangeLen{Quantity{<:T}, Base.TwicePrecision{Quantity{<:T}}, Base.TwicePrecision{Quantity{<:T}}, Int64}, min_enc::T, max_enc::T, nbins::Int64, rel_cut_fit::T) where T<:Real
    @assert size(enc_grid, 1) == length(enc_grid_rt) "enc_grid and enc_grid_rt must have the same number of columns"
    
    # create empty array for results
    enc        = zeros(length(enc_grid_rt))
    enc_err    = zeros(length(enc_grid_rt))
    
    for (r, rt) in enumerate(enc_grid_rt)
        # get enc for this rt
        enc_rt = flatview(enc_grid)[r, :]
        # get cut value
        cuts = cut_single_peak(enc_rt, min_enc, max_enc, nbins, rel_cut_fit)

        # fit gaussian
        result, report = fit_single_trunc_gauss(enc_rt, cuts)

        # get sigma
        enc[r] = result.σ
        enc_err[r] = result.σ_err
    end

    # get minimal enc and rt
    min_enc     = minimum(enc[enc .> 0])
    rt_min_enc  = enc_grid_rt[enc .> 0][findmin(enc[enc .> 0])[2]]
    
    # generate result and report
    result = (
        rt = rt_min_enc, 
        min_enc = min_enc
    )
    report = (
        rt = result.rt, 
        min_enc = result.min_enc,
        enc_grid_rt = collect(enc_grid_rt),
        enc = enc,
        enc_err = enc_err
    )
    return result, report

end
export fit_enc_sigmas