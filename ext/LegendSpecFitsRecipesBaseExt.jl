# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

module LegendSpecFitsRecipesBaseExt

using RecipesBase
using Unitful, Formatting, Measurements
using Measurements: value, uncertainty
using StatsBase, LinearAlgebra

# @recipe function f(x::Vector{T}, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}) where T<:Unitful.RealOrRealQuantity
@recipe function f(report::NamedTuple{(:f_fit, :μ, :σ, :n)}, x::Vector{T}, cuts::NamedTuple{(:low, :high, :max), Tuple{T, T, T}}) where {T <: Unitful.RealOrRealQuantity}
    ylabel := "Normalized Counts"
    legend := :bottomright
    @series begin
        seriestype := :histogram
        bins --> :fd
        normalize --> :pdf
        label := "Data"
        ustrip(x[x .> cuts.low .&& x .< cuts.high])
    end
    @series begin
        color := :red
        label := "Normal Fit (μ = $(round(unit(report.μ), report.μ, digits=2)), σ = $(round(unit(report.σ), report.σ, digits=2)))"
        lw := 3
        ustrip(cuts.low):ustrip(Measurements.value(report.σ / 1000)):ustrip(cuts.high), t -> report.f_fit(t)
    end
end

@recipe function f(report:: NamedTuple{(:rt, :min_enc, :enc_grid_rt, :enc)})
    xlabel := "Rise Time ($(unit(first(report.rt))))"
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
        label := "ENC"
        report.enc_grid_rt*NoUnits, report.enc
    end
    @series begin
        seriestype := :hline
        label := "Min. ENC Noise (RT: $(report.rt))"
        color := :red
        linewidth := 2.5
        [report.min_enc]
    end
end

@recipe function f(report:: NamedTuple{(:ft, :min_fwhm, :e_grid_ft, :fwhm)})
    xlabel := "Flat-Top Time $(unit(first(report.ft)))"
    ylabel := "FWHM FEP"
    grid := :true
    gridcolor := :black
    gridalpha := 0.2
    gridlinewidth := 0.5
    # xscale := :log10
    # yscale := :log10
    ylims := (1, 8)
    xlims := (0.5, 5)
    @series begin
        seriestype := :scatter
        label := "FWHM"
        report.e_grid_ft*NoUnits, report.fwhm
    end
    @series begin
        seriestype := :hline
        label := "Min. FWHM $(round(u"keV", report.min_fwhm, digits=2)) (FT: $(report.ft))"
        color := :red
        linewidth := 2.5
        [report.min_fwhm]
    end
end

@recipe function f(report:: NamedTuple{(:wl, :min_sf, :a_grid_wl_sg, :sfs)})
    xlabel := "Window Length ($(unit(first(report.a_grid_wl_sg))))"
    ylabel := "SEP Surrival Fraction ($(unit(first(report.sfs))))"
    grid := :true
    gridcolor := :black
    gridalpha := 0.2
    gridlinewidth := 0.5
    # ylims := (0, 30)
    @series begin
        seriestype := :scatter
        label := "SF"
        ustrip.(report.a_grid_wl_sg), ustrip.(report.sfs)
    end
    @series begin
        seriestype := :hline
        label := "Min. SF $(report.min_sf) (WT: $(report.wl))"
        color := :red
        linewidth := 2.5
        [ustrip(Measurements.value(report.min_sf))]
    end
    @series begin
        seriestype := :hspan
        label := ""
        color := :red
        alpha := 0.1
        ustrip.([Measurements.value(report.min_sf)-Measurements.uncertainty(report.min_sf), Measurements.value(report.min_sf)+Measurements.uncertainty(report.min_sf)])
    end
end

@recipe function f(report::NamedTuple{(:v, :h, :f_fit, :f_sig, :f_lowEtail, :f_bck)}; show_label=true, show_fit=true)
    xlabel := "Energy (keV)"
    ylabel := "Counts"
    legend := :bottomright
    yscale := :log10
    ylim_max = max(5*report.f_sig(report.v.μ), 5*maximum(report.h.weights))
    ylim_max = ifelse(ylim_max == 0.0, 1e5, ylim_max)
    ylims := (1, ylim_max)
    @series begin
        seriestype := :stepbins
        label := ifelse(show_label, "Data", "")
        bins --> :sqrt
        LinearAlgebra.normalize(report.h, mode = :density)
    end
    if show_fit
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

@recipe function f(report::NamedTuple{(:h_calsimple, :h_uncal, :c, :fep_guess, :peakhists, :peakstats)}; cal=true)
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

@recipe function f(report_ctc::NamedTuple{(:peak, :window, :fct, :bin_width, :bin_width_qdrift, :e_peak, :e_ctc, :qdrift_peak, :h_before, :h_after, :fwhm_before, :fwhm_after, :report_before, :report_after)})
    layout := (2,2)
    thickness_scaling := 2.0
    size := (2400, 1600)
    @series begin
        seriestype := :histogram2d
        bins := (ustrip.(unit(first(report_ctc.e_peak)), minimum(report_ctc.e_peak):report_ctc.bin_width:maximum(report_ctc.e_peak)), quantile(report_ctc.qdrift_peak, 0.01):report_ctc.bin_width_qdrift:quantile(report_ctc.qdrift_peak, 0.99))
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
        bins := (ustrip.(unit(first(report_ctc.e_peak)), minimum(report_ctc.e_peak):report_ctc.bin_width:maximum(report_ctc.e_peak)), quantile(report_ctc.qdrift_peak, 0.01):report_ctc.bin_width_qdrift:quantile(report_ctc.qdrift_peak, 0.99))
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
        title := "FWHM $(round(report_ctc.fwhm_before, digits=2))"
        yscale := :log10
        subplot := 3
        report_ctc.h_before
    end

    # @series begin
    #     # seriestype := :stepbins
    #     color := :red
    #     # label := "Before CTC"
    #     # xlabel := "Energy (keV)"
    #     # ylabel := "Counts"
    #     # yscale := :log10
    #     subplot := 3
    #     # report_ctc.h_before
    #     minimum(report_ctc.e_peak):0.001:maximum(report_ctc.e_peak), t -> report_ctc.report_before.f_fit(t)
    # end
    # @series begin
    #     # seriestype := :stepbins
    #     color := :red
    #     # label := "Before CTC"
    #     # xlabel := "Energy (keV)"
    #     # ylabel := "Counts"
    #     # yscale := :log10
    #     subplot := 4
    #     # report_ctc.h_before
    #     minimum(report_ctc.e_peak):0.001:maximum(report_ctc.e_peak), t -> report_ctc.report_after.f_fit(t)
    # end


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
        title := "FWHM $(round(report_ctc.fwhm_after, digits=2))"
        yscale := :log10
        subplot := 4
        report_ctc.h_after
    end
end

@recipe function f(report_window_cut::NamedTuple{(:h, :f_fit, :x_fit, :low_cut, :high_cut, :low_cut_fit, :high_cut_fit, :center, :σ)})
    xlims := (value(ustrip(report_window_cut.center - 5*report_window_cut.σ)), value(ustrip(report_window_cut.center + 5*report_window_cut.σ)))
    @series begin
        seriestype := :barbins
        alpha := 0.5
        label := "Data"
        report_window_cut.h        
    end
    @series begin
        seriestype := :line
        label := "Best Fit"
        color := :red
        linewidth := 3
        report_window_cut.x_fit, report_window_cut.f_fit
    end
    @series begin
        seriestype := :vline
        label := "Cut Window"
        color := :green
        linewidth := 3
        ustrip.([report_window_cut.low_cut, report_window_cut.high_cut])
    end
    # @series begin
    #     seriestype := :vline
    #     label := "Center"
    #     color := :blue
    #     linewidth := 3
    #     ustrip.([report_window_cut.center])
    # end
    @series begin
        seriestype := :vline
        label := "Fit Window"
        color := :orange
        linewidth := 3
        ustrip.([report_window_cut.low_cut_fit, report_window_cut.high_cut_fit])
    end
    @series begin
        seriestype := :vspan
        label := ""
        color := :lightgreen
        alpha := 0.2
        ustrip.([report_window_cut.low_cut, report_window_cut.high_cut])
    end

end


end # module LegendSpecFitsRecipesBaseExt