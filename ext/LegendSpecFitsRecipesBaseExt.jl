# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

module LegendSpecFitsRecipesBaseExt

using RecipesBase
using Unitful, Formatting
using StatsBase, LinearAlgebra

# @recipe function f(x::Vector{T}, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}) where T<:Unitful.RealOrRealQuantity
@recipe function f(report::NamedTuple{(:f_fit, :μ, :μ_err, :σ, :σ_err, :n), Tuple{Q, T, T, T, T, Int64}}, x::Vector{T}, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}) where {Q <: Function, T <: Unitful.RealOrRealQuantity}
    ylabel := "Normalized Counts"
    legend := :bottomright
    @series begin
        seriestype := :histogram
        # y := report.f_fit.(x)
        bins --> 2000
        normalize --> :pdf
        label := "Data"
        # label := @sprintf("μ = %s ± %s\nσ = %s ± %s\nn = %d", report.μ, report.μ_err, report.σ, report.σ_err, report.n)
        x[x .> cuts.low .&& x .< cuts.high]
        # x
    end
    @series begin
        color := :red
        label := format("Normal Fit (μ = ({:.2f} ± {:.2f})µs, σ = ({:.2f} ± {:.2f})µs", ustrip.([report.μ, report.μ_err, report.σ, report.σ_err])...)
        lw := 3
        ustrip(cuts.low):0.00001:ustrip(cuts.high), t -> report.f_fit(t)
    end
end

@recipe function f(report:: NamedTuple{(:rt, :min_enc, :enc_grid_rt, :enc, :enc_err)}) where {T}
    xlabel := "Rise Time (µs)"
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

@recipe function f(report:: NamedTuple{(:ft, :min_fwhm, :e_grid_ft, :fwhm)}) where {T}
    xlabel := "Flat-Top Time (µs)"
    ylabel := "FWHM FEP (keV)"
    grid := :true
    gridcolor := :black
    gridalpha := 0.2
    gridlinewidth := 0.5
    # xscale := :log10
    # yscale := :log10
    ylims := (1, 5)
    xlims := (1, 5)
    @series begin
        seriestype := :scatter
        u"µs", NoUnits
    end
    @series begin
        seriestype := :scatter
        label := "FWHM"
        report.e_grid_ft[report.fwhm .> 0.0]*NoUnits, report.fwhm[report.fwhm .> 0.0]
    end
    @series begin
        seriestype := :hline
        label := "Min. FWHM (FT: $(report.ft))"
        color := :red
        linewidth := 2.5
        [report.min_fwhm]
    end
end

@recipe function f(report::NamedTuple{(:v, :h, :f_fit, :f_sig, :f_lowEtail, :f_bck)},; show_label::Bool=true)
    xlabel := "Energy (keV)"
    ylabel := "Counts"
    legend := :bottomright
    yscale := :log10
    ylims := (1, 1.2*report.f_sig(report.v.μ))
    @series begin
        seriestype := :stepbins
        label := ifelse(show_label, "Data", "")
        bins --> :sqrt
        LinearAlgebra.normalize(report.h, mode = :density)
    end
    @series begin
        seriestype := :line
        label := ifelse(show_label, "Best Fit", "")
        color := :red
        minimum(report.h.edges[1]):0.1:maximum(report.h.edges[1]), report.f_fit
    end
    @series begin
        seriestype := :line
        label := ifelse(show_label, "Signal", "")
        color := :green
        minimum(report.h.edges[1]):0.1:maximum(report.h.edges[1]), report.f_sig
    end
    @series begin
        seriestype := :line
        label := ifelse(show_label, "Low Energy Tail", "")
        color := :blue
        minimum(report.h.edges[1]):0.1:maximum(report.h.edges[1]), report.f_lowEtail
    end
    @series begin
        seriestype := :line
        label := ifelse(show_label, "Background", "")
        color := :black
        minimum(report.h.edges[1]):0.1:maximum(report.h.edges[1]), report.f_bck
    end
end

@recipe function f(report::NamedTuple{((:v, :h, :f_fit, :f_sig, :f_bck))})
    xlabel := "A/E (a.u.)"
    ylabel := "Counts"
    legend := :bottomright
    ylims := (1, max(1.5*report.f_sig(report.v.μ), 1.5*maximum(report.h.weights)))
    @series begin
        seriestype := :stepbins
        label := "Data"
        bins --> :sqrt
        LinearAlgebra.normalize(report.h, mode = :density)
    end
    @series begin
        seriestype := :line
        label := "Best Fit"
        color := :red
        minimum(report.h.edges[1]):1e-4:maximum(report.h.edges[1]), report.f_fit
    end
    @series begin
        seriestype := :line
        label := "Signal"
        color := :green
        minimum(report.h.edges[1]):1e-4:maximum(report.h.edges[1]), report.f_sig
    end
    @series begin
        seriestype := :line
        label := "Background"
        color := :black
        minimum(report.h.edges[1]):1e-4:maximum(report.h.edges[1]), report.f_bck
    end
end

@recipe function f(x::Vector{T}, cut::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}) where T<:Unitful.RealOrRealQuantity
    ylabel := "Counts"
    legend := :topright
    xlims := (median(x) - std(x), median(x) + std(x))
    @series begin
        seriestype := :stephist
        label := "Data"
        bins --> :sqrt
        x
    end
    @series begin
        seriestype := :vline
        label := ""
        color := :red
        linewidth := 1.5
        [cut.low, cut.high]
    end
    @series begin
        seriestype := :vspan
        label := "Cut Window"
        color := :red
        alpha := 0.1
        fillrange := cut.high
        [cut.low, cut.high]
    end
end

@recipe function f(report::NamedTuple{(:h_calsimple, :h_uncal, :c, :fep_guess, :peakhists, :peakstats)}; cal::Bool)
    ylabel := "Counts"
    legend := :topright
    yscale := :log10
    if cal
        h = LinearAlgebra.normalize(report.h_calsimple, mode = :density)
        xlabel := "Energy (keV)"
        xlims := (0, 3000)
        xticks := (0:200:3000, ["$i" for i in 0:200:3000])
        ylims := (0.2, maximum(h.weights)*1.1)
        fep_guess = 2614.5
    else
        h = LinearAlgebra.normalize(report.h_uncal, mode = :density)
        xlabel := "Energy (ADC)"
        xlims := (0, 1.2*report.fep_guess)
        xticks := (0:3000:1.2*report.fep_guess, ["$i" for i in 0:3000:1.2*report.fep_guess])
        ylims := (0.2, maximum(h.weights)*1.1)
        fep_guess = report.fep_guess
    end
    @series begin
        seriestype := :stepbins
        label := "Energy"
        h
    end
    y_vline = 0.2:1:maximum(h.weights)*1.1
    @series begin
        seriestype := :line
        label := "FEP Guess"
        color := :red
        linewidth := 1.5
        fill(fep_guess, length(y_vline)), y_vline
    end
end

@recipe function f(report_ctc::NamedTuple{(:peak, :window, :fct, :bin_width, :bin_width_qdrift, :e_peak, :e_ctc, :qdrift_peak, :h_before, :h_after)})
    layout := (2,2)
    thickness_scaling := 2.0
    size := (2400, 1600)
    @series begin
        seriestype := :histogram2d
        bins := (minimum(report_ctc.e_peak):report_ctc.bin_width:maximum(report_ctc.e_peak), quantile(report_ctc.qdrift_peak, 0.01):report_ctc.bin_width_qdrift:quantile(report_ctc.qdrift_peak, 0.99))
        color := :inferno
        xlabel := "Energy (keV)"
        ylabel := "Drift Time"
        title := "Before Correction"
        xlims := (2600, 2630)
        legend := :none
        colorbar_scale := :log10
        subplot := 1
        report_ctc.e_peak, report_ctc.qdrift_peak
    end
    @series begin
        seriestype := :histogram2d
        bins := (minimum(report_ctc.e_peak):report_ctc.bin_width:maximum(report_ctc.e_peak), quantile(report_ctc.qdrift_peak, 0.01):report_ctc.bin_width_qdrift:quantile(report_ctc.qdrift_peak, 0.99))
        color := :magma
        xlabel := "Energy (keV)"
        ylabel := "Drift Time"
        title := "After Correction"
        xlims := (2600, 2630)
        legend := :none
        colorbar_scale := :log10
        subplot := 2
        report_ctc.e_ctc, report_ctc.qdrift_peak
    end
    @series begin
        seriestype := :stepbins
        color := :red
        label := "Before CTC"
        xlabel := "Energy (keV)"
        ylabel := "Counts"
        yscale := :log10
        subplot := 3
        report_ctc.h_before
    end
    @series begin
        seriestype := :stepbins
        color := :red
        label := "Before CTC"
        xlabel := "Energy (keV)"
        ylabel := "Counts"
        yscale := :log10
        subplot := 4
        report_ctc.h_before
    end
    @series begin
        seriestype := :stepbins
        color := :green
        label := "After CTC"
        xlabel := "Energy (keV)"
        ylabel := "Counts"
        yscale := :log10
        subplot := 4
        report_ctc.h_after
    end
end


end # module LegendSpecFitsRecipesBaseExt