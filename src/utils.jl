# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    expand_vars(v::NamedTuple)::StructArray

Expand all fields in `v` (scalars or arrays) to same array size and return
a `StructArray`.
"""
function expand_vars(v::NamedTuple)
    sz = Base.Broadcast.broadcast_shape((1,), map(size, values(v))...)
    _expand(x::Real) = Fill(x, sz)
    _expand(x::AbstractArray) = broadcast((a,b) -> b, Fill(nothing, sz), x)
    StructArray(map(_expand, v))
end
export expand_vars

"""
    subhist(h::Histogram, r::Tuple{<:Real,<:Real})

Return a new `Histogram` with the bins in the range `r`.
"""
function subhist(h::Histogram{<:Any, 1}, r::Tuple{<:Real,<:Real})
    first_bin, last_bin = (StatsBase.binindex(h, r[1]), StatsBase.binindex(h, r[2]))
    if first_bin < 1 first_bin = 1 end
    if (last_bin > length(h.weights)) last_bin = length(h.weights) end
    Histogram(h.edges[1][first_bin:last_bin+1], h.weights[first_bin:last_bin])
end
function subhist(x::Array{<:Real})
    bin_width = 2 * (quantile(x, 0.75) - quantile(x, 0.25)) / ∛(length(x))
    fit(Histogram, x, minimum(x):bin_width:maximum(x))
end
subhist(x::Array, r::Tuple{<:Real,<:Real}) = subhist(x[r[1] .< x .< r[2]])
subhist(h, i::Interval) = subhist(h, (i.left, i.right))
export subhist


"""
    get_distribution_transform(d::Distribution, pprior::Prior)

Return a `DistributionTransform` for the given `Distribution` and `Prior`.
"""
function get_distribution_transform end


"""
    tuple_to_array(nt::NamedTuple, fields::Vector{Symbol})

Return an array with the values of the fields in `nt` in the order given by
`fields`.
"""
function tuple_to_array(nt::NamedTuple)
    [nt[f] for f in fieldnames(nt)]
end


"""
    array_to_tuple(a::AbstractArray, as_nt::NamedTuple)

Return a `NamedTuple` with the values of `a` in the order given by
`fieldnames(as_nt)`.
"""
function array_to_tuple(a::AbstractArray, as_nt::NamedTuple)
    NamedTuple{fieldnames(as_nt)}(a)
end


""" 
    get_mc_value_shapes(v::NamedTuple, v_err::NamedTuple, n::Int64)
Return a `NamedTuple` with the same fields as `v` and `v_err` but with
`Normal` distributions for each field.
"""
function get_mc_value_shapes(v::NamedTuple, v_err::NamedTuple, n::Int64)
    vs = BAT.distprod(map(Normal, v, v_err))
    NamedTuple.(rand(vs, n))
end

"""
    get_mc_value_shapes(v::NamedTuple, v_err::Matrix, n::Union{Int64,Int32})
Generate `n` random samples of fit parameters using their respective best-fit values `v` and covariance matrix `v_err`
"""
function get_mc_value_shapes(v::NamedTuple, v_err::Matrix, n::Union{Int64,Int32})
    if !isposdef(v_err)
        v_err = nearestSPD(v_err)
        @debug "Covariance matrix not positive definite. Using nearestSPD"
    end
    v_err = v_err[1:6,1:6] #remove background, keep only relevant for sampling 
    v_fitpar = v[keys(v)[1:size(v_err,1)]] # only fit parameter
    dist = MvNormal([v_fitpar...], v_err) # multivariate distribution using covariance matrix)
    v_mc = rand(dist, n) # Draw samples

    # constain fit_par_samples to physical values. warning hardcoded. tbd 
    Idx_keep = findall((v_mc[3,:].>0) .*                #  positive amplitude 
                        (v_mc[5,:].<0.25).*             # skew fraction 
                        (v_mc[5,:].>0) .*    #skew fraction 
                        (v_mc[6,:].>0))                 # positive skew width
    v_mc = v_mc[:,Idx_keep];
    n = size(v_mc,2)
    v_mc = [NamedTuple{keys(v)[1:size(v_err,1)]}(v_mc[:,i]) for i=1:n] # convert back to NamedTuple 
end

"""
    get_friedman_diaconis_bin_width(x::AbstractArray)

Return the bin width for the given data `x` using the Friedman-Diaconis rule.
"""
function get_friedman_diaconis_bin_width end

function get_friedman_diaconis_bin_width(x::Vector{<:Real})
    2 * (quantile(x, 0.75) - quantile(x, 0.25)) / ∛(length(x))
end
get_friedman_diaconis_bin_width(x::Vector{<:Quantity{<:Real}}) = get_friedman_diaconis_bin_width(ustrip.(x))*unit(first(x))

"""
    get_number_of_bins(x::AbstractArray,; method::Symbol=:sqrt)

Return the number of bins for the given data `x` using the given method.
"""
function get_number_of_bins(x::AbstractArray,; method::Symbol=:sqrt)
    # all methods from https://en.wikipedia.org/wiki/Histogram#:~:text=To%20construct%20a%20histogram%2C%20the,overlapping%20intervals%20of%20a%20variable.
    if method == :sqrt
        return round(Int, sqrt(length(x)))
    elseif method == :sturges
        return round(Int, ceil(log2(length(x)) + 1))
    elseif method == :rice
        return round(Int, 2 * ∛(length(x)))
    elseif method == :scott
        return round(Int, (maximum(x) - minimum(x)) / (3.5 * std(x) * ∛(length(x))))
    elseif method == :doane
        return round(Int, 1 + log2(length(x)) + log2(1 + abs(skewness(x)) / sqrt(6 / (length(x) - 2))))
    elseif method == :fd
        return round(Int, (maximum(x) - minimum(x)) / get_friedman_diaconis_bin_width(x))
    else
        @assert false "Method not implemented"
    end
end

"""
    nearestSPD(A::Matrix{<:Real}) 
Returns the nearest positive definite matrix to A
Calculation is based on matrix factorization techniques described in https://www.sciencedirect.com/science/article/pii/0024379588902236
"""
function nearestSPD(A::Matrix{<:Real})
    B = (A + A') / 2  # make sure matrix is symmetric
    _, s, V = svd(B)  # singular value decomposition (SVD), s = singular values (~eigenvalues), V = right singular vector  (~eigenvector)
    H = V * diagm(0 => max.(s, 0)) * V' # symmetric polar factor of B
    B = (B + H) / 2 # calculate nearest positive definite matrix
    B = (B + B') / 2  # make sure matrix is symmetric
    return B
end 
export nearestSPD


Measurements.value(nt::NamedTuple) = NamedTuple{keys(nt)}([Measurements.value(nt[f]) for f in keys(nt)]...)
Measurements.value(x::AbstractArray) = Measurements.value.(x)