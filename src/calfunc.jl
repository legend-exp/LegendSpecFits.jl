# This file is a part of LegendDataManagement.jl, licensed under the MIT License (MIT).


export PolCalFunc

"""
    PolCalFunc(p0, p1, ...)

Callable polynomial `x -> p0 + p1 * x + p2 * x^2 + ...` with the given
coefficients, ordered from the constant term upward. The coefficients are
stored in `params` and can be of any `Number` type, including `Measurement`
and `Unitful.Quantity`.

## Examples
```julia
f = PolCalFunc(1.0, 2.0)  # 1 + 2x
f(3.0)                    # => 7.0
```
"""
struct PolCalFunc{N,T<:Number} <:Function
    params::NTuple{N,T}

    PolCalFunc(params::T...) where T = new{length(params),T}(params)
end


function (f::PolCalFunc{N,T})(x::U) where {N,T,U}
    R = promote_type(T, U)
    y = zero(R)
    xn = one(U)
    @inbounds for p in f.params
        y = R(fma(p, xn, y))
        xn *= x
    end
    y
end


"""
    CalFuncDict

Alias for `Dict{Int,PolCalFunc{2,Float64}}`: linear calibration functions
([`PolCalFunc`](@ref) with two coefficients) keyed by an integer identifier.
"""
const CalFuncDict = Dict{Int,PolCalFunc{2,Float64}}
export CalFuncDict
