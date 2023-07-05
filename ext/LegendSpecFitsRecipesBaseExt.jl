# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

module LegendSpecFitsRecipesBaseExt

using RecipesBase
using Unitful, Formatting

# @recipe function f(x::Vector{T}, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}) where T<:Unitful.RealOrRealQuantity
@recipe function f(report::NamedTuple{(:f_fit, :μ, :μ_err, :σ, :σ_err, :n), Tuple{Q, T, T, T, T, Int64}}, x::Vector{T}, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}) where {Q <: Function, T <: Unitful.RealOrRealQuantity}
    ylabel := "Normalized Counts"
    legend := :bottomright
    @series begin
        seriestype := :histogram
        # y := report.f_fit.(x)
        bins --> :sqrt
        normalize --> :pdf
        label := "Data"
        # label := @sprintf("μ = %s ± %s\nσ = %s ± %s\nn = %d", report.μ, report.μ_err, report.σ, report.σ_err, report.n)
        x[x .> cuts.low .&& x .< cuts.high]
    end
    @series begin
        color := :red
        label := format("Normal Fit (μ = ({:.2f} ± {:.2f})µs, σ = ({:.2f} ± {:.2f})µs", ustrip.([report.μ, report.μ_err, report.σ, report.σ_err])...)
        lw := 3
        ustrip(cuts.low):0.1:ustrip(cuts.high), t -> report.f_fit(t)
    end
end

@recipe function f(report:: NamedTuple{(:rt, :min_enc, :enc_grid_rt, :enc, :enc_err), Tuple{Q, T, Vector{Quantity{<:T}}, Vector{T}, Vector{T}}}) where {T <: Real, Q<:Quantity{T}}
    xlabel := "Rise Time"
    ylabel := "ENC (ADC)"
    grid := :true
    gridcolor := :black
    gridalpha := 0.2
    gridlinewidth := 0.5
    # xscale := :log10
    # yscale := :log10
    # xlims := (5e0, 2e1)
    @series begin
        seriestype := :scatter
        u"µs", NoUnits
    end
    @series begin
        seriestype := :scatter
        label := "ENC"
        yerror --> report.enc_err
        report.enc_grid_rt[report.enc .> 0.0]*NoUnits, report.enc[report.enc .> 0.0]
    end
    @series begin
        seriestype := :hline
        label := "Min. ENC Noise (RT: $(report.rt))"
        color := :red
        linewidth := 2.5
        [report.min_enc]
    end
end




end # module LegendSpecFitsRecipesBaseExt