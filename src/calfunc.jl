# This file is a part of LegendDataManagement.jl, licensed under the MIT License (MIT).


export PolCalFunc

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


const CalFuncDict = Dict{Int,PolCalFunc{2,Float64}}
export CalFuncDict
