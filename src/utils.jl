# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

"""
    heaviside(x)

Returns `zero(x)` for `x < 0` and `one(x)` otherwise (including `x == 0`).
"""
heaviside(x) = x < zero(x) ? zero(x) : one(x)

"""
    expand_vars(v::NamedTuple)::StructArray

Broadcasts all fields of `v` (scalars or arrays) to a common array size and
returns them as a `StructArray`.

Scalar fields are repeated lazily (without copying) along the common size.
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
    subhist(h, i::Interval)
    subhist(x::Array{<:Real})
    subhist(x::Array, r::Tuple{<:Real,<:Real})

Returns a histogram restricted to a range.

* For a one-dimensional `Histogram` `h`, the result contains all bins from the
  bin that contains `r[1]` to the bin that contains `r[2]`, clamped to the edges
  of `h`. An `IntervalSets.Interval` `i` is converted to the tuple
  `(i.left, i.right)`.
* For a data vector `x`, the histogram is built from the values of `x` with
  the Freedman-Diaconis bin width, spanning `minimum(x)` to `maximum(x)`. If
  `r` is given, only values strictly between `r[1]` and `r[2]` are used.
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

Returns a `DistributionTransform` for the given `Distribution` and `Prior`.

This package defines no methods for this function.
"""
function get_distribution_transform end


"""
    tuple_to_array(nt::NamedTuple)

Returns a `Vector` with the values of the fields of `nt` in field order.
"""
function tuple_to_array(nt::NamedTuple)
    [nt[f] for f in fieldnames(nt)]
end


"""
    array_to_tuple(a::AbstractArray, as_nt::NamedTuple)

Returns a `NamedTuple` with the keys of `as_nt` and the values of `a`, in order.
"""
function array_to_tuple(a::AbstractArray, as_nt::NamedTuple)
    NamedTuple{fieldnames(as_nt)}(a)
end


"""    get_mc_value_shapes(v::NamedTuple, v_err::NamedTuple, n::Integer)

Draws `n` samples of the parameters in `v`, each parameter from an independent
`Normal` distribution with mean `v[k]` and standard deviation `v_err[k]`.
Returns a vector of `n` `NamedTuple`s with the keys of `v`.
"""
function get_mc_value_shapes(v::NamedTuple, v_err::NamedTuple, n::Integer)
    vs = BAT.distprod(map(Normal, v, v_err))
    NamedTuple.(rand(vs, n))
end

"""
    get_mc_value_shapes(v::NamedTuple, v_err::Matrix, n::Integer)

Draws `n` samples of the fit parameters from a multivariate normal
distribution with mean `v` and covariance matrix `v_err`.

If `v_err` is not positive definite, the nearest positive definite matrix
(see `nearestSPD`) is used. Samples are discarded unless the third parameter
(amplitude) is positive, the fifth parameter (skew fraction) lies in
`(0, 0.25)` and the sixth parameter (skew width) is positive, so the returned
vector can be shorter than `n`. These checks assume the parameter order of the
single-peak fit models.
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
    get_friedman_diaconis_bin_width(x::Vector{<:Quantity{<:Real}})

Returns the bin width ``2\\,\\mathrm{IQR}(x)/\\sqrt[3]{n}`` of the
Freedman-Diaconis rule, where IQR is the interquartile range of `x` and `n` is
the number of values. For values with units, the result carries the same unit.
"""
function get_friedman_diaconis_bin_width end

function get_friedman_diaconis_bin_width(x::Vector{<:Real})
    2 * (quantile(x, 0.75) - quantile(x, 0.25)) / ∛(length(x))
end
get_friedman_diaconis_bin_width(x::Vector{<:Quantity{<:Real}}) = get_friedman_diaconis_bin_width(ustrip.(x))*unit(first(x))

"""
    get_number_of_bins(x::AbstractArray; method::Symbol = :sqrt)

Returns the number of histogram bins for the data `x` as an `Int`.

## Keywords
* `method::Symbol = :sqrt`: Rule for the bin count. One of `:sqrt`, `:sturges`,
  `:rice`, `:scott`, `:doane` or `:fd` (Freedman-Diaconis). Any other value
  throws an `AssertionError`.
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
    nearestSPD(A::Matrix{<:Real}; n_iter::Signed = 1)

Returns the symmetric positive definite matrix nearest to `A`.

The calculation follows the singular value decomposition method of
[Higham (1988)](https://doi.org/10.1016/0024-3795(88)90223-6).
The step is repeated recursively, with `n_iter` counting the repetitions, until
the result is positive definite or `n_iter` exceeds 5.
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


"""
    mvalue(x)

Returns the central value of a `Measurements.Measurement`. Applied element-wise
to arrays and field-wise to `NamedTuple`s; strings and symbols are returned
unchanged.
"""
function mvalue end
mvalue(x) = Measurements.value(x)
mvalue(x::AbstractArray) = mvalue.(x)
mvalue(x::AbstractString) = x
mvalue(x::Symbol) = x
mvalue(nt::NamedTuple) = NamedTuple{keys(nt)}([mvalue(nt[f]) for f in keys(nt)]...)

"""
    muncert(x)

Returns the uncertainty of a `Measurements.Measurement`. Applied element-wise
to arrays and field-wise to `NamedTuple`s; strings and symbols are returned
unchanged.
"""
function muncert end
muncert(x) = Measurements.uncertainty(x)
muncert(x::AbstractArray) = muncert.(x)
muncert(x::AbstractString) = x
muncert(x::Symbol) = x
muncert(nt::NamedTuple) = NamedTuple{keys(nt)}([muncert(nt[f]) for f in keys(nt)]...)