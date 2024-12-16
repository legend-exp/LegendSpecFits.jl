# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

heaviside(x) = x < zero(x) ? zero(x) : one(x)

"""
    expand_vars(v::NamedTuple)

Expand all fields in `v` (scalars or arrays) to same array size and return
a `StructArray`.

# Arguments
    * `v`: Variables to expand into equal array sizes

"""
function expand_vars(v::NamedTuple)
    sz = Base.Broadcast.broadcast_shape((1,), map(size, values(v))...)
    _expand(x::Real) = Fill(x, sz)
    _expand(x::AbstractArray) = broadcast((a,b) -> b, Fill(nothing, sz), x)
    StructArray(map(_expand, v))
end
export expand_vars

"""
    subhist(h::Histogram{<:Any, 1}, r::Tuple{<:Real,<:Real})

Return a new `Histogram` with the bins in the range `r`.
2 methods for the function.
    
# Arguments
    * `h`: Histogram data
    * `r`: Range of the histogram to use bins for a new histogram

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

# Arguments
    * `d`: Distribution
    * `pprior`: Pseudo prior
"""
function get_distribution_transform end


"""
    tuple_to_array(nt::NamedTuple)

Return an array with the values of the fields in `nt`.
# Arguments
    * `nt`: Field values

"""
function tuple_to_array(nt::NamedTuple)
    [nt[f] for f in fieldnames(nt)]
end


"""
    array_to_tuple(a::AbstractArray, as_nt::NamedTuple)

Return a `NamedTuple` with the values of `a` in the order given by
`fieldnames(as_nt)`.

# Arguments
    * `a`: Array of values
    * `as_nt`: Order for array
"""
function array_to_tuple(a::AbstractArray, as_nt::NamedTuple)
    NamedTuple{fieldnames(as_nt)}(a)
end


""" 
    get_mc_value_shapes(v::NamedTuple, v_err::NamedTuple, n::Integer)
Return a `NamedTuple` with the same fields as `v` and `v_err` but with
`Normal` distributions for each field.

# Arguments
    * `v`: Best-fit values
    * `v_err`: Value error
    * `n`: Number of fields
"""
function get_mc_value_shapes(v::NamedTuple, v_err::NamedTuple, n::Integer)
    vs = BAT.distprod(map(Normal, v, v_err))
    NamedTuple.(rand(vs, n))
end

"""
    get_mc_value_shapes(v::NamedTuple, v_err::Matrix, n::Integer)
Generate `n` random samples of fit parameters using their respective best-fit values `v` and covariance matrix `v_err`

# Arguments
    * `v`: Best-fit values
    * `v_err`: Value error
    * `n`: Number of random fit parameter samples

"""
function get_mc_value_shapes(v::V, v_err::Matrix{T}, n::Integer)::Vector{V} where {V <: NamedTuple, T <: AbstractFloat}

    v_mc::Matrix{T} = let err = if !isposdef(v_err)
            @debug "Covariance matrix not positive definite. Using nearestSPD"
            nearestSPD(v_err)
        else
            v_err
        end
        dist::FullNormal = MvNormal(collect(v), err) # multivariate distribution using covariance matrix)
        rand(dist, n)
    end

    return [ 
        V(view(v_mc, Colon(), i)) 
        for i = 1:n if 
        v_mc[3,i] > 0.0 &&          # positive amplitude
        0.0 < v_mc[5,i] < 0.25 &&   # skew fraction
        v_mc[6,i] > 0.0             # positive skew width
    ]
end

"""
    get_friedman_diaconis_bin_width(x::Vector{<:Real})

Return the bin width for the given data `x` using the Friedman-Diaconis rule.

# Arguments
    * `x`: Given data
"""
function get_friedman_diaconis_bin_width end

function get_friedman_diaconis_bin_width(x::Vector{<:Real})
    2 * (quantile(x, 0.75) - quantile(x, 0.25)) / ∛(length(x))
end
get_friedman_diaconis_bin_width(x::Vector{<:Quantity{<:Real}}) = get_friedman_diaconis_bin_width(ustrip.(x))*unit(first(x))

"""
    get_number_of_bins(x::AbstractArray,; method::Symbol=:sqrt)

Return the number of bins for the given data `x` using the given method.

# Arguments
    * `x`: Data 

# Keywords
    * `method`: Given method used to find the number of bins

# Returns
    * Number of bins for given data
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
    nearestSPD(A::Matrix{<:Real},;n_iter::Signed=1)
Returns the nearest positive definite matrix to A
Calculation is based on matrix factorization techniques described in https://www.sciencedirect.com/science/article/pii/0024379588902236

# Arguments
    * `A`: Given matrix
    
# Keywords
    * `n_iter`: Number of iterations

# Returns
    * `B`: Nearest positive definite matrix to the 'A' matrix
"""
function nearestSPD(A::Matrix{<:Real},;n_iter::Signed=1)
    B = (A + A') / 2  # make sure matrix is symmetric
    _, s, V = svd(B)  # singular value decomposition (SVD), s = singular values (~eigenvalues), V = right singular vector  (~eigenvector)
    H = V * diagm(0 => max.(s, 0)) * V' # symmetric polar factor of B
    B = (B + H) / 2 # calculate nearest positive definite matrix
    B = (B + B') / 2  # make sure matrix is symmetric
    if (isposdef(B)) | (n_iter > 5)
        return B
    else 
        B = nearestSPD(B;n_iter=n_iter+1)
    end
end 
export nearestSPD


function mvalue end
mvalue(x) = Measurements.value(x)
mvalue(x::AbstractArray) = mvalue.(x)
mvalue(x::AbstractString) = x
mvalue(x::Symbol) = x
mvalue(nt::NamedTuple) = NamedTuple{keys(nt)}([mvalue(nt[f]) for f in keys(nt)]...)

function muncert end
muncert(x) = Measurements.uncertainty(x)
muncert(x::AbstractArray) = muncert.(x)
muncert(x::AbstractString) = x
muncert(x::Symbol) = x
muncert(nt::NamedTuple) = NamedTuple{keys(nt)}([muncert(nt[f]) for f in keys(nt)]...)