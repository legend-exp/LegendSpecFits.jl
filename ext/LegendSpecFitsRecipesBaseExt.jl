# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

module LegendSpecFitsRecipesBaseExt

using RecipesBase
import Plots
using Unitful, Format, Measurements, LaTeXStrings
using Measurements: value, uncertainty
using StatsBase, LinearAlgebra
using KernelDensity
function round_wo_units(x::Unitful.RealOrRealQuantity; digits::Integer=2)
    if unit(x) == NoUnits
        round(x, digits=digits)
    else
        round(unit(x), x, digits=2)
    end
end

@recipe function f(report::NamedTuple{(:f_fit, :h, :μ, :σ, :gof)})
    ylabel := "Normalized Counts"
    margins := (4, :mm)
    framestyle := :box
    legend := :bottomleft
    xlims := (ustrip(Measurements.value(report.μ - 5*report.σ)), ustrip(Measurements.value(report.μ + 5*report.σ)))
    @series begin
        label := "Data"
        subplot --> 1
        report.h
    end
    @series begin
        color := :red
        subplot --> 1
        label := "Normal Fit (μ = $(round_wo_units(report.μ, digits=2)), σ = $(round_wo_units(report.σ, digits=2)))"
        lw := 3
        bottom_margin --> (-4, :mm)
        ustrip(Measurements.value(report.μ - 10*report.σ)):ustrip(Measurements.value(report.σ / 1000)):ustrip(Measurements.value(report.μ + 10*report.σ)), t -> report.f_fit(t)
    end
    if !isempty(report.gof)
        link --> :x
        layout --> @layout([a{0.7h}; b{0.3h}])
        @series begin
            seriestype := :hline
            ribbon := 3
            subplot --> 2
            fillalpha := 0.5
            label := ""
            fillcolor := :lightgrey
            linecolor := :darkgrey
            [0.0]
        end
        @series begin
            seriestype := :hline
            ribbon := 1
            subplot --> 2
            fillalpha := 0.5
            label := ""
            fillcolor := :grey
            linecolor := :darkgrey
            [0.0]
        end
        @series begin
            seriestype := :scatter
            subplot --> 2
            label := ""
            title := ""
            markercolor --> :black
            ylabel := "Residuals (σ)"
            link --> :x
            top_margin --> (-4, :mm)
            ylims := (-5, 5)
            xlims := (ustrip(Measurements.value(report.μ - 5*report.σ)), ustrip(Measurements.value(report.μ + 5*report.σ)))
            yscale --> :identity
            yticks := ([-3, 0, 3])
            collect(report.h.edges[1])[1:end-1] .+ diff(collect(report.h.edges[1]))[1]/2 , [ifelse(abs(r) < 1e-6, 0.0, r) for r in report.gof.residuals_norm]
        end
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
    ylabel := "SEP Survival Fraction ($(unit(first(report.sfs))))"
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

@recipe function f(report::NamedTuple{(:v, :h, :f_fit, :f_components, :gof)}; show_label=true, show_fit=true, show_components=true, show_residuals=true, f_fit_x_step_scaling=1/100, _subplot=1, x_label="Energy (keV)")
    f_fit_x_step = ustrip(value(report.v.σ)) * f_fit_x_step_scaling
    bin_centers = collect(report.h.edges[1])[1:end-1] .+ diff(collect(report.h.edges[1]))[1]/2 
    if x_label == "A/E"
        legend := :bottomleft
    else
        legend := :topleft
    end
    foreground_color_legend := :silver
    background_color_legend := :white
    ylim_max = max(3*value(report.f_fit(report.v.μ)), 3*maximum(report.h.weights))
    ylim_max = ifelse(ylim_max == 0.0, 1e5, ylim_max)
    ylim_min = 0.1*minimum(filter(x -> x > 0, report.h.weights))
    framestyle --> :box
    @series begin
        seriestype := :bar
        alpha --> 1.0
        fillcolor --> :lightgrey
        linecolor --> :lightgrey
        fillrange := 1e-7
        bar_width := diff(report.h.edges[1])[1]
        label --> ifelse(show_label, "Data", "")
        yscale --> :log10
        bins --> :sqrt
        if x_label == "A/E"
            xlabel --> L"A/E\ (\sigma_{A/E}))"
        else
            xlabel --> "Energy (keV)"
        end 
        subplot --> _subplot
        bin_centers, LinearAlgebra.normalize(report.h, mode = :density).weights#LinearAlgebra.normalize(report.h, mode = :density)
    end
    if show_fit
        @series begin
            seriestype := :line
            if !isempty(report.gof)
                label := ifelse(show_label, "Best Fit (p = $(round(report.gof.pvalue, digits=2)))", "")
            else
                label := ifelse(show_label, "Best Fit", "")
            end
            linewidth := 2.3
            color := :black
            linecolor := :black
            margins --> (0, :mm)
            if !isempty(report.gof) && show_residuals
                bottom_margin --> (-4, :mm)
                xlabel := ""
                xticks --> ([])
            else
                if x_label == "A/E"
                    xlabel --> L"A/E\ (\sigma_{A/E}))"
                else
                    xlabel --> "Energy (keV)"
                end 
            end
            ylims --> (ylim_min, ylim_max)
            xlims := (minimum(report.h.edges[1]), maximum(report.h.edges[1]))
            if x_label == "A/E"
                ylabel := "Counts / $(round(step(report.h.edges[1]), digits=2))"
            else
                ylabel := "Counts / $(round(step(report.h.edges[1]), digits=2)) keV"
            end
            subplot --> _subplot
            minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), value.(report.f_fit.(minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1])))
        end
        if show_components
            for (idx, component) in  enumerate(keys(report.f_components.funcs))
                @series begin
                    seriestype := :line
                    color := report.f_components.colors[component]
                    label := ifelse(show_label, report.f_components.labels[component], "")
                    linewidth := 2
                    linestyle := report.f_components.linestyles[component]
                    if idx == length(report.f_components.funcs)
                        margins --> (0, :mm)
                        if !isempty(report.gof) && show_residuals
                            bottom_margin --> (-4, :mm)
                            xlabel := ""
                            xticks --> ([])
                        else
                            if x_label == "A/E"
                                xlabel --> L"A/E\ (\sigma_{A/E}))"
                            else
                                xlabel --> "Energy (keV)"
                            end 
                        end
                        ylims --> (ylim_min, ylim_max)
                        xlims := (minimum(report.h.edges[1]), maximum(report.h.edges[1]))
                        if x_label == "A/E"
                            ylabel := "Counts / $(round(step(report.h.edges[1]), digits=2))"
                        else
                            ylabel := "Counts / $(round(step(report.h.edges[1]), digits=2)) keV"
                        end
                    end
                    subplot --> _subplot
                    minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), report.f_components.funcs[component]
                end
            end
        end

        if !isempty(report.gof) && show_residuals
            ylims_res_max, ylims_res_min = 5, -5
            if any(report.gof.residuals_norm .> 5) || any(report.gof.residuals_norm .< -5)
                abs_max = 1.2*maximum(abs.(report.gof.residuals_norm))
                ylims_res_max, ylims_res_min = abs_max, -abs_max
            end
            layout --> @layout([a{0.8h}; b{0.2h}])
            margins --> (0, :mm)
            link --> :x
            @series begin
                seriestype := :hline
                ribbon := 3
                subplot --> _subplot + 1
                fillalpha := 0.5
                label := ""
                fillcolor := :lightgrey
                linecolor := :darkgrey
                [0.0]
            end
            @series begin
                seriestype := :hline
                ribbon := 1
                subplot --> _subplot + 1
                fillalpha := 0.5
                label := ""
                fillcolor := :grey
                linecolor := :darkgrey
                [0.0]
            end
            @series begin
                seriestype := :scatter
                subplot --> _subplot + 1
                label := ""
                title := ""
                markercolor --> :black
                ylabel --> "Residuals (σ)"
                if x_label == "A/E"
                    xlabel --> L"A/E\ (\sigma_{A/E}))"
                else
                    xlabel --> "Energy (keV)"
                end 
                link --> :x
                top_margin --> (0, :mm)
                ylims := (ylims_res_min, ylims_res_max)
                xlims := (minimum(report.h.edges[1]), maximum(report.h.edges[1]))
                yscale --> :identity
                markersize --> 3 #can be changed manually in the code
                if ylims_res_max == 5
                    yticks := ([-3, 0, 3])
                end
                bin_centers, report.gof.residuals_norm
            end
        end
    end
end

@recipe function f(report::NamedTuple{(:v, :h, :f_fit, :f_components, :gof)}, mode::Symbol)
    if mode == :cormat 
        cm = cor(report.gof.covmat)
    elseif  mode == :covmat
        cm = report.gof.covmat
    else
        @debug "mode $mode not supported - has to be :cormat or :covmat"
        return
    end

    cm_plt  = NaN .* cm
    for i in range(1, stop = size(cm)[1])
        cm_plt[i:end,i] = cm[i:end,i]
    end

    # prepare labels 
    par_names = fieldnames(report.v)
    tick_names =  String.(collect(par_names))
    tick_names[tick_names .== "step_amplitude"] .= L"\textrm{bkg}_\textrm{step}"
    tick_names[tick_names .== "background"] .= "bkg"
    tick_names[tick_names .== "skew_width"] .= L"\textrm{tail}_\textrm{skew}"
    tick_names[tick_names .== "skew_fraction"] .= L"\textrm{tail}_\textrm{frac}"

    @series begin
        seriestype := :heatmap
        # c := cgrad(:grays, rev = true)
        c := :RdBu_3#:bam
        cm_plt
    end

    # annotation for correlation coefficient 
    cm_vec = round.(vec(cm_plt), digits = 2)
    xvec  = vec(hcat([fill(i, 7) for i in 1:7]...))
    yvec  = vec(hcat([fill(i, 7) for i in 1:7]...)')
    xvec = xvec[isfinite.(cm_vec)]
    yvec = yvec[isfinite.(cm_vec)]
    cm_vec = cm_vec[isfinite.(cm_vec)]
    @series begin
        seriestype := :scatter
        xticks := (1:length(par_names),tick_names)
        yticks := (1:length(par_names),tick_names)
        yflip --> true
        markeralpha := 0
        colorbar --> false 
        label --> false
        legend --> false
        thickness_scaling := 1.0
        xtickfontsize --> 12
        xlabelfontsize --> 14
        ylabelfontsize --> 14
        tickfontsize --> 12
        legendfontsize --> 10
        xlims --> (0.5, length(par_names)+0.5)
        ylims --> (0.5, length(par_names)+0.5)
        size --> (600, 575)
        grid --> false
        title --> "Correlation Matrix" 
        series_annotations := [("$(cm_vec[i])", :center, :center, :black, 12, "Helvetica Bold") for i in eachindex(xvec)]
        xvec, yvec
    end

end

@recipe function f(report::NamedTuple{((:v, :h, :f_fit, :f_sig, :f_bck))}; f_fit_x_step_scaling=1/100)
    f_fit_x_step = ustrip(value(report.v.σ)) * f_fit_x_step_scaling
    xlabel := "A/E (a.u.)"
    ylabel := "Counts"
    legend := :topleft
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
        minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), report.f_fit
    end
    @series begin
        seriestype := :line
        label := "Signal"
        color := :green
        minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), report.f_sig
    end
    @series begin
        seriestype := :line
        label := "Background"
        color := :black
        minimum(report.h.edges[1]):f_fit_x_step:maximum(report.h.edges[1]), report.f_bck
    end
end

@recipe function f(report::NamedTuple{(:survived, :cut, :sf, :bsf)}; peak_name="")
    size --> (1400,800)
    left_margin --> (10, :mm)
    title --> "$peak_name Survival fraction: $(round(report.sf * 100, digits = 2))%"
    ylim_max = max(3*value(report.survived.f_fit(report.survived.v.μ)), 3*maximum(report.survived.h.weights), 3*value(report.cut.f_fit(report.cut.v.μ)), 3*maximum(report.cut.h.weights))
    ylim_max = ifelse(ylim_max == 0.0, 1e5, ylim_max)
    ylim_min = min(minimum(filter(x -> x > 0, report.survived.h.weights)), minimum(filter(x -> x > 0, report.cut.h.weights)))
    @series begin
        label := "Data Survived"
        _subplot := 1
        fillcolor := :darkgrey
        alpha := 1.0
        report.survived
    end
    @series begin
        label := "Data Cut"
        _subplot := 1
        alpha := 0.2
        ylims --> (ylim_min, ylim_max)
        report.cut
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

@recipe function f(report::NamedTuple{(:peakpos, :peakpos_cal, :h_uncal, :h_calsimple)}; cal=true)
    legend := :topright
    size := (1000, 600)
    thickness_scaling := 1.5
    framestyle := :box
    yformatter := :plain
    if cal
        h = LinearAlgebra.normalize(report.h_calsimple, mode = :density)
        xlabel := "Peak Amplitudes (P.E.)"
        ylabel := "Counts / $(round_wo_units(step(first(h.edges)), digits=2)) P.E."
        xticks := (0:0.5:last(first(h.edges)))
        pps = report.peakpos_cal
    else
        h = LinearAlgebra.normalize(report.h_uncal, mode = :density)
        xlabel := "Peak Amplitudes (ADC)"
        ylabel := "Counts / $(round_wo_units(step(first(h.edges)), digits=2)) ADC"
        pps = report.peakpos
    end
    xlims := (0, last(first(h.edges)))
    min_y = minimum(h.weights) == 0.0 ? 1e-3*maximum(h.weights) : 0.8*minimum(h.weights)
    ylims --> (min_y, maximum(h.weights)*1.1)
    @series begin
        seriestype := :stepbins
        label := "amps"
        h
    end
    y_vline = min_y:1:maximum(h.weights)*1.1
    for (i, p) in enumerate(pps)
        @series begin
            seriestype := :line
            if i == 1
                label := "Peak Pos. Guess"
            else
                label := ""
            end
            color := :red
            linewidth := 1.5
            fill(p, length(y_vline)), y_vline
        end
    end
end

@recipe function f(report_sipm::NamedTuple{(:h_cal, :f_fit, :f_fit_components, :min_pe, :max_pe, :bin_width, :n_mixtures, :n_pos_mixtures, :peaks, :positions, :μ , :gof)}; xerrscaling=1, show_residuals=true, show_peaks=true, show_components=false)
    legend := :topright
    size := (1000, 600)
    margins := (4, :mm)
    thickness_scaling := 1.5
    framestyle := :box
    yformatter := :plain
    foreground_color_legend := :silver
    background_color_legend := :white
    ylabel := "Counts / $(round_wo_units(report_sipm.bin_width * 1e3, digits=2))E-3 P.E."
    xlims := (first(first(report_sipm.h_cal.edges)), last(first(report_sipm.h_cal.edges)))
    xticks := (ceil(first(first(report_sipm.h_cal.edges)))-0.5:0.5:last(first(report_sipm.h_cal.edges)))
    min_y = minimum(report_sipm.h_cal.weights) == 0.0 ? 1e-3*maximum(report_sipm.h_cal.weights) : 0.8*minimum(report_sipm.h_cal.weights)
    ylims := (min_y, maximum(report_sipm.h_cal.weights)*1.1)
    bin_centers = collect(report_sipm.h_cal.edges[1])[1:end-1] .+ diff(collect(report_sipm.h_cal.edges[1]))[1]/2 
    @series begin
        yscale --> :log10
        label := "Amplitudes"
        subplot --> 1
        seriestype := :bar
        alpha --> 1.0
        fillalpha --> 0.85
        fillcolor --> :lightgrey
        linecolor --> :lightgrey
        fillrange := 1e-1
        bins --> :sqrt
        bar_width := diff(report_sipm.h_cal.edges[1])[1]
        bin_centers, report_sipm.h_cal.weights
    end
    @series begin
        seriestype := :line
        if !isempty(report_sipm.gof)
            label := "Best Fit (p = $(round(report_sipm.gof.pvalue, digits=2)))"
        else
            label := "Best Fit"
        end
        if show_residuals && !isempty(report_sipm.gof)
            xlabel := ""
            xticks := []
        else
            xlabel := "Peak Amplitudes (P.E.)"
        end
        subplot --> 1
        color := :black
        linewidth := 1.5
        report_sipm.min_pe:report_sipm.bin_width/100:report_sipm.max_pe, report_sipm.f_fit
    end
    if show_components
        for (i, μ) in enumerate(report_sipm.μ)
            @series begin
                seriestype := :line
                if i == 1
                    label := "Mixture Components"
                else
                    label := ""
                end
                if show_residuals && !isempty(report_sipm.gof)
                    xlabel := ""
                    xticks := []
                else
                    xlabel := "Peak Amplitudes (P.E.)"
                end
                subplot --> 1
                color := i + 1 + length(report_sipm.positions)
                linestyle := :dash
                linewidth := 1.3
                # fillalpha := 1
                alpha := 0.4
                xi = report_sipm.min_pe:report_sipm.bin_width/100:report_sipm.max_pe
                yi = Base.Fix2(report_sipm.f_fit_components, i).(xi)
                # ribbon := (yi .- 1, zeros(length(xi)))
                xi, yi
            end
        end
    end
    if show_peaks
        y_vline = [min_y, maximum(report_sipm.h_cal.weights)*1.1]
        for (i, p) in enumerate(report_sipm.positions)
            @series begin
                seriestype := :line
                if xerrscaling == 1
                    label := "$(report_sipm.peaks[i]) P.E. [$(report_sipm.n_pos_mixtures[i]) Mix.]"
                else
                    label := "$(report_sipm.peaks[i]) P.E. [$(report_sipm.n_pos_mixtures[i]) Mix.] (error x$xerrscaling)"
                end
                subplot --> 1
                color := i + 1
                linewidth := 1.5
                fill(value(p), length(y_vline)), y_vline
            end
            @series begin
                seriestype := :vspan
                label := ""
                fillalpha := 0.1
                subplot --> 1
                if show_residuals && !isempty(report_sipm.gof)
                    xlabel := ""
                    xticks := []
                else
                    xlabel := "Peak Amplitudes (P.E.)"
                end
                color := i + 1
                [value(p) - xerrscaling * uncertainty(p), value(p) + xerrscaling * uncertainty(p)]
            end
        end
    end
    if show_residuals && !isempty(report_sipm.gof)
        link --> :x
        layout --> @layout([a{0.7h}; b{0.3h}])
        @series begin
            seriestype := :hline
            ribbon := 3
            subplot --> 2
            fillalpha := 0.5
            label := ""
            fillcolor := :lightgrey
            linecolor := :darkgrey
            [0.0]
        end
        @series begin
            seriestype := :hline
            ribbon := 1
            subplot --> 2
            fillalpha := 0.5
            label := ""
            fillcolor := :grey
            linecolor := :darkgrey
            [0.0]
        end
        @series begin
            seriestype := :scatter
            subplot --> 2
            label := ""
            title := ""
            markercolor --> :darkgrey
            markersize --> 3.0
            markerstrokewidth := 0.1
            ylabel := "Residuals (σ)"
            xlabel := "Peak Amplitudes (P.E.)"
            link --> :x
            top_margin --> (-8, :mm)
            ylims := (-6, 6)
            xlims := (first(first(report_sipm.h_cal.edges)), last(first(report_sipm.h_cal.edges)))
            yscale --> :identity
            yticks := ([-3, 0, 3])
            report_sipm.gof.bin_centers, [ifelse(abs(r) < 1e-6, 0.0, r) for r in report_sipm.gof.residuals_norm]
        end        
    end
end

@recipe function f(report_ctc::NamedTuple{(:peak, :window, :fct, :bin_width, :bin_width_qdrift, :e_peak, :e_ctc, :qdrift_peak, :h_before, :h_after, :fwhm_before, :fwhm_after, :report_before, :report_after)})
    if !("StatsPlots" in string.(Base.loaded_modules_array()))
        throw(ErrorException("StatsPlots not loaded. Please load StatsPlots before using this recipe."))
    end
    layout := (2, 1)
    size := (1000, 1000)
    framestyle := :semi
    grid := false 
    left_margin --> (5, :mm)
    right_margin --> (5, :mm)
    bottom_margin := (-4, :mm)
    margins --> (0, :mm)
    link --> :x
    foreground_color_legend := :silver
    background_color_legend := :white
    xtickfontsize := 12
    xlabelfontsize := 14
    ylabelfontsize := 14
    ytickfontsize := 12
    legendfontsize := 12
    xl = (first(report_ctc.h_before.edges[1]), last(report_ctc.h_before.edges[1]))
    @series begin
        seriestype := :stepbins 
        fill := true
        color := :darkgrey
        label := "Before correction"
        legend := :topleft
        subplot := 1
        ylims := (0, :auto)
        report_ctc.h_before
    end
    @series begin
        seriestype := :stepbins 
        fill := true
        alpha := 0.5
        color := :purple
        label := "After correction"
        legend := :topleft
        subplot := 1
        ylims := (0, :auto)
        xlims := xl
        xlabel := ""
        xticks := ([], [])
        ylabel := "Counts / $(round_wo_units(report_ctc.bin_width, digits=2))"
        report_ctc.h_after
    end
    @series begin
        seriestype := :line
        subplot := 2
        c := :binary
        colorbar := :none
        fill := true
        label := "Before correction"
        kde((ustrip(report_ctc.e_peak), report_ctc.qdrift_peak ./ maximum(report_ctc.qdrift_peak)))
    end
    @series begin
        seriestype := :line
        subplot := 2
        c := :plasma
        colorbar := :none
        fill := false
        label := "After correction"
        xlims := xl
        ylims := (0, 1)
        yticks := 0.1:0.1:0.9
        yformatter := :plain
        xlabel := "Energy ($(unit(report_ctc.peak)))"
        ylabel := "Eff. Drift time (a.u.)"
        kde((ustrip(report_ctc.e_ctc), report_ctc.qdrift_peak ./ maximum(report_ctc.qdrift_peak)))
    end
end

@recipe function f(report_window_cut::NamedTuple{(:h, :f_fit, :x_fit, :low_cut, :high_cut, :low_cut_fit, :high_cut_fit, :center, :σ)})
    xlims := (value(ustrip(report_window_cut.center - 7*report_window_cut.σ)), value(ustrip(report_window_cut.center + 7*report_window_cut.σ)))
    ylabel := "Density"
    framestyle := :box
    @series begin
        label := "Data"
        report_window_cut.h
    end
    ylims --> (0, 1.1*maximum(report_window_cut.h.weights))
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

@recipe function f(report::NamedTuple{(:par, :f_fit, :x, :y, :gof)}; plot_ribbon=true, xerrscaling=1, yerrscaling=1, additional_pts=NamedTuple())
    thickness_scaling := 2.0
    xlims := (0, 1.2*value(maximum(report.x)))
    framestyle := :box
    xformatter := :plain
    yformatter := :plain
    margins := (0, :mm)
    if !isempty(report.gof)
        layout --> @layout([a{0.8h}; b{0.2h}])
        link --> :x
    end
    @series begin
        seriestype := :line
        subplot := 1
        xticks --> :none
        if !isempty(report.gof)
            label := "Best Fit (p = $(round(report.gof.pvalue, digits=2))| χ²/ndf = $(round(report.gof.chi2min, digits=2)) / $(report.gof.dof))"
        else
            label := "Best Fit"
        end
        color := :orange
        linewidth := 2
        fillalpha := 0.2
        if plot_ribbon
            ribbon := uncertainty.(report.f_fit.(0:1:1.2*value(maximum(report.x))))
        end
        0:1:1.2*value(maximum(report.x)), value.(report.f_fit.(0:1:1.2*value(maximum(report.x))))
    end
    @series begin
        seriestype := :scatter
        subplot := 1
        if xerrscaling == 1 && yerrscaling == 1
            label := "Data"
        elseif xerrscaling == 1
            label := "Data (y-Error x$(yerrscaling))"
        elseif yerrscaling == 1
            label := "Data (x-Error x$(xerrscaling))"
        else
            label := "Data (x-Error x$(xerrscaling), y-Error x$(yerrscaling))"
        end
        markercolor --> :black
        if !isempty(report.gof)
            xguide := ""
            xticks := []
        end
        xerror := uncertainty.(report.x) .* xerrscaling
        yerror := uncertainty.(report.y) .* yerrscaling
        value.(report.x), value.(report.y)
    end
    if !isempty(additional_pts)
        @series begin
            seriestype := :scatter
            subplot --> 1
            if xerrscaling == 1 && yerrscaling == 1
                label := "Data not used for fit"
            elseif xerrscaling == 1
                label := "Data not used for fit (y-Error x$(yerrscaling))"
            elseif yerrscaling == 1
                label := "Data not used for fit (x-Error x$(xerrscaling))"
            else
                label := "Data not used for fit (x-Error x$(xerrscaling), y-Error x$(yerrscaling))"
            end
            ms --> 3
            markershape --> :circle
            markerstrokecolor --> :black
            if !isempty(report.gof)
                xguide := ""
                xticks := []
            end
            linewidth --> 0.5
            markercolor --> :silver
            xerror := uncertainty.(additional_pts.x) .* xerrscaling
            yerror := uncertainty.(additional_pts.y) .* yerrscaling
            value.(additional_pts.x), value.(additional_pts.y)
        end
    end
    if !isempty(report.gof)
        @series begin
            seriestype := :hline
            ribbon --> 3
            subplot --> 2
            fillalpha --> 0.5
            label --> ""
            fillcolor --> :lightgrey
            linecolor --> :darkgrey
            [0.0]
        end
        @series begin
            seriestype --> :hline
            ribbon --> 1
            subplot --> 2
            fillalpha --> 0.5
            label --> ""
            fillcolor --> :grey
            linecolor --> :darkgrey
            [0.0]
        end
        if !isempty(additional_pts)
            @series begin
                seriestype := :scatter
                label --> :none
                ms --> 3
                markershape --> :circle
                markerstrokecolor --> :black
                linewidth --> 0.5
                markercolor --> :silver
                subplot --> 2
                value.(additional_pts.x), additional_pts.residuals_norm
            end
        end
        @series begin
            seriestype --> :scatter
            subplot --> 2
            label --> ""
            markercolor --> :black
            yguide := "Residuals (σ)"
            top_margin --> (-4, :mm)
            ylims --> (-5, 5)
            yticks --> ([-3, 0, 3])
            value.(report.x), report.gof.residuals_norm
        end
    end
end

@recipe function f(report::NamedTuple{(:par, :f_fit, :x, :y, :gof, :e_unit, :qbb, :type)}; xerrscaling=1, yerrscaling=1, additional_pts=NamedTuple())
    bottom_margin --> (0, :mm)
    if report.type == :fwhm
        y_max = value(maximum(report.y))
        additional_pts = if !isempty(additional_pts)
            fwhm_cal = report.f_fit.(ustrip.(additional_pts.peaks)) .* report.e_unit
            y_max = max(y_max, value(maximum(ustrip.(report.e_unit, additional_pts.fwhm))))
            (x = additional_pts.peaks, y = additional_pts.fwhm,
                residuals_norm = (value.(fwhm_cal) .- additional_pts.fwhm)./ uncertainty.(fwhm_cal))
        else
            NamedTuple()
        end
        legend := :topleft
        framestyle := :box
        xlims := (0, 3000)
        xticks := (0:500:3000, ["$i" for i in 0:500:3000])
        @series begin
            grid --> :all
            xerrscaling --> xerrscaling
            xlabel := "Energy (keV)"
            yerrscaling --> yerrscaling
            additional_pts --> additional_pts
            (par = report.par, f_fit = report.f_fit, x = report.x, y = report.y, gof = get(report, :gof, NamedTuple()))
        end
        @series begin
            seriestype := :hline
            label := L"Q_{\beta \beta}:" * " $(round(u"keV", report.qbb, digits=2))"
            color := :green
            fillalpha := 0.2
            linewidth := 2.5
            xticks := :none
            ylabel := "FWHM"
            ylims := (0, 1.2*y_max)
            subplot := 1
            ribbon := uncertainty(report.qbb)
            [value(report.qbb)]
        end
    end
end

@recipe function f(report::NamedTuple{(:par, :f_fit, :x, :y, :gof, :e_unit, :type)}; xerrscaling=1, yerrscaling=1, additional_pts=NamedTuple())
    bottom_margin --> (0, :mm)
    if report.type == :cal
        additional_pts = if !isempty(additional_pts)
            μ_strip = if unit(first(additional_pts.μ)) != NoUnits
                ustrip.(report.e_unit, additional_pts.μ)
            else
                additional_pts.μ
            end
            p_strip = if unit(first(additional_pts.peaks)) != NoUnits
                ustrip.(report.e_unit, additional_pts.peaks)
            else
                additional_pts.peaks
            end
            μ_cal = report.f_fit.(μ_strip)
            (x = μ_strip, y = p_strip,
                residuals_norm = (value.(μ_cal) .- p_strip)./ uncertainty.(μ_cal))
        else
            NamedTuple()
        end
        framestyle := :box
        xlims := (0, 1.1*maximum(value.(report.x)))
        @series begin
            xlabel := "Energy (ADC)"
            ylabel := "Energy ($(report.e_unit))"
            grid --> :all
            xerrscaling --> xerrscaling
            yerrscaling --> yerrscaling
            additional_pts := additional_pts
            (par = report.par, f_fit = report.f_fit, x = report.x, y = report.y, gof = report.gof)
        end
    end
end

@recipe function f(report::NamedTuple{(:par, :f_fit, :x, :y, :gof, :e_unit, :label_y, :label_fit)})
    layout --> @layout([a{0.8h}; b{0.2h}])
    margins --> (0, :mm)
    link --> :x
    size := (1200, 700)
    layout --> @layout([a{0.7h}; b{0.3h}])
    xmin = floor(Int, minimum(report.x)/100)*100
    xmax = ceil(Int, maximum(report.x)/100)*100
    @series begin
        seriestype := :line
        subplot --> 1
        color := :orange
        ms := 3
        linewidth := 3
        label := report.label_fit
        ribbon := uncertainty.(report.f_fit(report.x))
        report.x, value.(report.f_fit(report.x))
    end
    @series begin
            ylabel := "$(report.label_y) (a.u.)"
            seriestype := :scatter
            subplot --> 1
            color := :black 
            ms := 3
            yguidefontsize := 18
            xguidefontsize := 18
            ytickfontsize := 12
            xtickfontsize := 12
            legendfontsize := 14
            foreground_color_legend := :silver
            background_color_legend := :white
            xlims := (xmin,xmax)
            xticks := (xmin:250:xmax, fill(" ", length(xmin:250:xmax) ))
            ylims := (0.98 * (Measurements.value(minimum(report.y)) - Measurements.uncertainty(median(report.y))), 1.02 * (Measurements.value(maximum(report.y)) + Measurements.uncertainty(median(report.y)) ) )
            margin := (10, :mm)
            bottom_margin := (-7, :mm)
            framestyle := :box
            grid := :false
            label := "Compton band fits: Gaussian $(report.label_y)(A/E)"
            report.x, report.y
    end

    @series begin
        seriestype := :hline
        ribbon := 3
        subplot --> 2
        fillalpha := 0.5
        label := ""
        fillcolor := :lightgrey
        linecolor := :darkgrey
        [0.0]
    end
    @series begin
        xlabel := "Energy ($(report.e_unit))"
        ylabel := "Residuals (σ) \n"
        seriestype := :scatter
        subplot --> 2
        color := :black 
        ms := 3
        label := false
        framestyle := :box
        grid := :false
        xlims := (xmin,xmax)
        xticks := xmin:250:xmax
        ylims := (floor(minimum(report.gof.residuals_norm)-1), ceil(maximum(report.gof.residuals_norm))+1)
        yticks := [-5,0,5]
        yguidefontsize := 18
        xguidefontsize := 18
        ytickfontsize := 12
        xtickfontsize := 12
        report.x, report.gof.residuals_norm
    end
end

@recipe function f(report::NamedTuple{(:par, :f_fit, :x, :y, :gof, :e_unit, :label_y, :label_fit)}, com_report::NamedTuple{(:values, :label_y, :label_fit, :energy)})
    margins --> (0, :mm)
    link --> :x
    size := (1200, 700)
    layout --> @layout([a{0.8h}; b{0.2h}]) #or 0.7 and 0.3
    xmin = floor(Int, minimum(report.x)/100)*100
    xmax = ceil(Int, maximum(report.x)/100)*100

    yguidefontsize := 16
    xguidefontsize := 16
    ytickfontsize := 12
    xtickfontsize := 12
    legendfontsize := 12
    foreground_color_legend := :silver
    background_color_legend := :white
    framestyle := :box
    grid := :false

    ### subplot 1
    @series begin
        seriestype := :line
        subplot --> 1
        color := :orange
        markersize := 3
        linewidth := 3
        label := report.label_fit
        ribbon := uncertainty.(report.f_fit(report.x))
        report.x, value.(report.f_fit(report.x))
    end
    @series begin
        ylabel := "$(report.label_y) (a.u.)"
        seriestype := :scatter
        markersize := 3
        subplot --> 1
        color := :black 
        #ylims := (0.98 * (Measurements.value(minimum(report.y)) - Measurements.uncertainty(median(report.y))), 1.02 * (Measurements.value(maximum(report.y)) + Measurements.uncertainty(median(report.y)) ) )
        label := "Compton band fits: Gaussian $(report.label_y)(A/E)"
        report.x, report.y
    end
    @series begin #combined fits
        ylabel := "$(com_report.label_y) (a.u.)"
        seriestype := :line
        subplot --> 1
        color := :red
        linewidth := 2
        linestyle := :dash
        label := com_report.label_fit
        xlims := (xmin,xmax)
        xticks := (xmin:250:xmax, fill(" ", length(xmin:250:xmax) ))
        ylims := (0.98 * (Measurements.value(minimum(report.y)) - Measurements.uncertainty(median(report.y))), 1.02 * (Measurements.value(maximum(report.y)) + Measurements.uncertainty(median(report.y)) ) )
        margin := (10, :mm)
        bottom_margin := (-7, :mm)
        com_report.energy, com_report.values
    end

    ### subplot 2
    @series begin
        seriestype := :hline
        ribbon := 3
        subplot --> 2
        fillalpha := 0.5
        label := ""
        fillcolor := :lightgrey
        linecolor := :darkgrey
        [0.0]
    end
    @series begin
        xlabel := "Energy ($(report.e_unit))"
        ylabel := "Residuals (σ) \n"
        seriestype := :scatter
        subplot --> 2
        color := :black 
        markersize := 3
        label := false
        framestyle := :box
        grid := :false
        xlims := (xmin,xmax)
        xticks := (xmin:250:xmax)
        #ylims := (floor(minimum(report.gof.residuals_norm)-1), ceil(maximum(report.gof.residuals_norm))+1)
        ylims := (-5, 5)
        yticks := [-5,0,5]
        report.x, report.gof.residuals_norm
    end
end

@recipe function f(report::NamedTuple{(:h_before, :h_after_low, :h_after_ds, :dep_h_before, :dep_h_after_low, :dep_h_after_ds, :sf, :n0, :lowcut, :highcut, :e_unit, :bin_width)})
    legend := :topright
    foreground_color_legend := :silver
    background_color_legend := :white
    size := (1000, 600)
    xlabel := "Energy ($(report.e_unit))"
    ylabel := "Counts / $(round_wo_units(report.bin_width, digits=2))"
    framestyle := :box
    thickness_scaling := 1.2
    xticks := (0:300:3000)
    xlims := (0, 3000)
    ylim_max = 3*maximum(report.h_before.weights)
    @series begin
        seriestype := :stepbins
        subplot --> 1
        color := 1
        alpha := 0.3
        label := "Before A/E"
        yscale := :log10
        report.h_before
    end
    @series begin
        seriestype := :stepbins
        subplot --> 1
        color := 2
        alpha := 0.8
        label := "After low A/E"
        yscale := :log10
        report.h_after_low
    end
    @series begin
        seriestype := :stepbins
        subplot --> 1
        color := 3
        alpha := 0.5
        label := "After DS A/E"
        yscale := :log10
        ylims := (1, ylim_max)
        report.h_after_ds
    end

    @series begin
        seriestype := :stepbins
        subplot --> 2
        color := 1
        alpha := 0.3
        inset := (1, Plots.bbox(0.3, 0.03, 0.4, 0.2, :top))
        label := "Before A/E"
        yscale := :log10
        report.dep_h_before
    end
    @series begin
        seriestype := :stepbins
        subplot --> 2
        color := 2
        alpha := 0.8
        label := "After low A/E"
        yscale := :log10
        report.dep_h_after_low
    end
    @series begin
        seriestype := :stepbins
        legend := false
        ylabelfontsize := 8
        subplot --> 2
        color := 3
        alpha := 0.5
        label := "After DS A/E"
        yscale := :log10
        margin := (1, :mm)
        ylabel := "Counts"
        xlims := (first(report.dep_h_after_low.edges[1])), last(report.dep_h_after_low.edges[1])
        xticks := (ceil(Int, first(report.dep_h_after_low.edges[1])):15:ceil(Int, last(report.dep_h_after_low.edges[1])), ["$i" for i in ceil(Int, first(report.dep_h_after_low.edges[1])):15:ceil(Int, last(report.dep_h_after_low.edges[1]))])
        report.dep_h_after_ds
    end
end

@recipe function f(report::NamedTuple{(:h_before, :h_after_low, :h_after_ds, :window, :n_before, :n_after, :sf, :e_unit, :bin_width)})
    legend := :topright
    foreground_color_legend := :silver
    background_color_legend := :white
    yformatter := :plain
    size := (800, 500)
    xlabel := "Energy ($(report.e_unit))"
    ylabel := "Counts / $(round_wo_units(report.bin_width, digits=2))"
    framestyle := :box
    thickness_scaling := 1.2
    xlims := (first(report.h_after_low.edges[1])), last(report.h_after_low.edges[1])
    xticks := (ceil(Int, first(report.h_after_low.edges[1])):15:ceil(Int, last(report.h_after_low.edges[1])))
    ylims := (0.5*minimum(report.h_after_ds.weights), 1.5*maximum(report.h_before.weights))
    @series begin
        seriestype := :stepbins
        subplot --> 1
        linewidth := 1.5
        color := 1
        label := "Before A/E"
        yscale := :log10
        report.h_before
    end
    @series begin
        seriestype := :stepbins
        subplot --> 1
        color := 2
        linewidth := 1.5
        label := "After low A/E"
        yscale := :log10
        report.h_after_low
    end
    @series begin
        seriestype := :stepbins
        subplot --> 1
        linewidth := 1.5
        color := 3
        label := "After DS A/E"
        yscale := :log10
        report.h_after_ds
    end
end


### lq recipe functions

# recipe for the lq_drift_time_correction report

@recipe function f(report::NamedTuple{(:lq_report, :drift_report, :lq_box, :drift_time_func, :dep_left, :dep_right)}, e_cal, dt_eff, lq_e_corr, plot_type::Symbol)

    # Extract data from the report
    dep_left = report.dep_left
    dep_right = report.dep_right
    box = report.lq_box

    #cut data to DEP
    dt_dep = dt_eff[dep_left .< e_cal .< dep_right]
    lq_dep = lq_e_corr[dep_left .< e_cal .< dep_right]

    # Plot configuration: 2D histogram
    xlabel := "Drift Time"
    ylabel := "LQ (A.U.)"
    framestyle := :box
    left_margin := -2Plots.mm
    bottom_margin := -4Plots.mm
    top_margin := -3Plots.mm
    color := :viridis
    formatter := :plain
    thickness_scaling := 1.6
    size := (1200, 900)


    if plot_type == :DEP
        # Create 2D histogram with filtered data based on dep_left and dep_right
        
        # dynamic bin size dependant on fit constraint box
        t_diff = box.t_upper - box.t_lower
        lq_diff = box.lq_upper - box.lq_lower
        xmin = box.t_lower - 1*t_diff
        xmax = box.t_upper + 1*t_diff
        xstep = (xmax - xmin) / 100 
        ymin = box.lq_lower - 1*lq_diff
        ymax = box.lq_upper + 4*lq_diff
        ystep = (ymax - ymin) / 100
        nbins := (xmin:xstep:xmax, ymin:ystep:ymax)

        @series begin
            seriestype := :histogram2d
            dt_dep, lq_dep
        end
    elseif plot_type == :whole
        # Create 2D histogram with all data
        colorbar_scale := :log10

        # dynamic bin size dependant on fit constraint box
        t_diff = box.t_upper - box.t_lower
        lq_diff = box.lq_upper - box.lq_lower
        xmin = box.t_lower - 1*t_diff
        xmax = box.t_upper + 1*t_diff
        xstep = (xmax - xmin) / 400
        ymin = box.lq_lower - 8*lq_diff
        ymax = box.lq_upper + 8*lq_diff
        ystep = (ymax - ymin) / 400
        nbins := (xmin:xstep:xmax, ymin:ystep:ymax)

        @series begin
            seriestype := :histogram2d
            dt_eff, lq_e_corr
        end
    end
    
    # Add vertical and horizontal lines for the fit box limits
    @series begin
        seriestype := :vline
        label := ""
        linewidth := 1.5
        color := :red
        [box.t_lower, box.t_upper]
    end

    @series begin
        seriestype := :hline
        label := ""
        linewidth := 1.5
        color := :red
        [box.lq_lower, box.lq_upper]
    end

    # Add linear fit plot
    @series begin
        label := "Linear Fit"
        linewidth := 1.5
        color := :blue

        # Evaluate drift_time_func over the full range of drift time (x-axis)
        dt_range = range(xmin, xmax, length=100)  # 100 points across the x-axis
        lq_fit = report.drift_time_func.(dt_range)  # Apply the linear function to the full dt range
        xlims := (xmin, xmax)
        ylims := (ymin, ymax)
        dt_range, lq_fit

    end
end


# recipe for the lq_cut report

@recipe function f(report::NamedTuple{(:cut, :fit_result, :temp_hists, :fit_report)}, lq_class::Vector{Float64}, e_cal, plot_type::Symbol)

    # Extract cutvalue from the report
    cut_value = Measurements.value.(report.cut)

    # Plot configuration for all types
    left_margin := -2Plots.mm
    bottom_margin := -4Plots.mm
    top_margin := -3Plots.mm
    thickness_scaling := 1.6
    size := (1200, 900)
    framestyle := :box
    formatter := :plain

    # Plot configuration for each specific type
    if plot_type == :lq_cut
        # 2D histogram for LQ Cut
        xlabel := "Energy"
        ylabel := "LQ (A.U.)"
        #lq bins dependant on cut value
        ymin = -5 * cut_value
        ymax = 10 * cut_value
        ystep = (ymax - ymin) / 500
        nbins := (0:6:3000, ymin:ystep:ymax)
        colorbar_scale := :log10
        color := :viridis
        legend := :bottomright
        
        @series begin
            seriestype := :histogram2d
            e_cal, lq_class
        end

        @series begin
            seriestype := :hline
            label := "3σ exclusion"
            linewidth := 2
            color := :red
            [cut_value]
        end

    elseif plot_type == :energy_hist
        # Energy histogram before/after LQ cut
        xlabel := "Energy"
        ylabel := "Counts"
        nbins := 0:1:3000
        yscale := :log10

        @series begin
            seriestype := :stephist
            label := "Data before LQ Cut"
            e_cal
        end

        @series begin
            seriestype := :stephist
            label := "Surviving LQ Cut"
            e_cal[lq_class .< cut_value]
        end

        @series begin
            seriestype := :stephist
            label := "Cut by LQ Cut"
            e_cal[lq_class .> cut_value]
        end

    elseif plot_type == :cut_fraction
        # Percentual cut plot
        xlabel := "Energy"
        ylabel := "Fraction"
        legend := :topleft

        # Calculate fraction of events cut by LQ cut
        h1 = fit(Histogram, ustrip.(e_cal), 0:3:3000)
        h2 = fit(Histogram, ustrip.(e_cal[lq_class .> cut_value]), 0:3:3000)
        h_diff = h2.weights ./ h1.weights

        @series begin
            label := "Cut Fraction"
            0:3:3*length(h_diff)-1, h_diff
        end

    elseif plot_type == :fit
        # Fit plot
        xlabel := "LQ (A.U.)"
        ylabel := "Counts"
        
        ylabel := "Normalized Counts"
        margins := (4, :mm)
        framestyle := :box
        legend := :topleft
        xlims = (ustrip(Measurements.value(report.fit_report.μ - 5*report.fit_report.σ)), ustrip(Measurements.value(report.fit_report.μ + 5*report.fit_report.σ)))
        xlims := xlims
        @series begin
            label := "Data"
            subplot --> 1
            report.fit_report.h
        end
        @series begin
            color := :red
            subplot --> 1
            label := "Normal Fit (μ = $(round_wo_units(report.fit_report.μ, digits=2)), \n σ = $(round_wo_units(report.fit_report.σ, digits=2)))"
            lw := 3
            bottom_margin --> (-4, :mm)
            ustrip(Measurements.value(report.fit_report.μ - 5*report.fit_report.σ)):ustrip(Measurements.value(report.fit_report.σ / 1000)):ustrip(Measurements.value(report.fit_report.μ + 5*report.fit_report.σ)), t -> report.fit_report.f_fit(t)
        end

        @series begin
            seriestype := :vline
            label := "Cut Value"
            subplot := 1
            linewidth := 2
            color := :red
            [cut_value]
        end

        if !isempty(report.fit_report.gof)
            link --> :x
            layout --> @layout([a{0.7h}; b{0.3h}])
            @series begin
                seriestype := :hline
                ribbon := 3
                subplot --> 2
                fillalpha := 0.5
                label := ""
                fillcolor := :lightgrey
                linecolor := :darkgrey
                [0.0]
            end
            @series begin
                seriestype := :hline
                ribbon := 1
                subplot --> 2
                fillalpha := 0.5
                label := ""
                fillcolor := :grey
                linecolor := :darkgrey
                [0.0]
            end
            @series begin
                seriestype := :scatter
                subplot --> 2
                label := ""
                title := ""
                markercolor --> :black
                ylabel := "Residuals (σ)"
                link --> :x
                top_margin --> (-4, :mm)
                ylims := (-5, 5)
                xlims := xlims
                yscale --> :identity
                yticks := ([-3, 0, 3])
                
                bin_midpoints = report.fit_report.gof.bin_centers
                residuals = report.fit_report.gof.residuals_norm
                bin_midpoints, residuals             
            end
        end

    elseif plot_type == :sideband
        # Sideband histograms
        xlabel := "Lq (A.U.)"
        ylabel := "Counts"

        @series begin
            seriestype := :stepbins
            label := "Peak"
            report.temp_hists.hist_dep
        end

        @series begin
            seriestype := :stepbins
            label := "Sideband 1"
            report.temp_hists.hist_sb1
        end

        @series begin
            seriestype := :stepbins
            label := "Sideband 2"
            xlims := quantile(filter(isfinite, lq_class), 0.05), quantile(filter(isfinite, lq_class), 0.95)
            report.temp_hists.hist_sb2
        end


    end
end



end # module LegendSpecFitsRecipesBaseExt
