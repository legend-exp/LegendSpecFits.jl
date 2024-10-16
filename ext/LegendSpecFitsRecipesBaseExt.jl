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

@recipe function f(report::NamedTuple{(:v, :h, :f_fit, :f_components, :gof)}; show_label=true, show_fit=true, show_components=true, show_residuals=true, f_fit_x_step_scaling=1/100, _subplot=1, x_label="Energy (keV)")
    f_fit_x_step = ustrip(value(report.v.σ)) * f_fit_x_step_scaling
    bin_centers = collect(report.h.edges[1])[1:end-1] .+ diff(collect(report.h.edges[1]))[1]/2 
    if x_label == "A/E"
        legend := :bottomleft
    else
        legend := :topright
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
    if !isempty(report.gof)
        layout --> @layout([a{0.8h}; b{0.2h}])
        margins --> (-11.5, :mm)
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
            ylabel --> "Residuals (σ)"
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
        xlabel := "Energy (keV)"
        legend := :topleft
        framestyle := :box
        xlims := (0, 3000)
        xticks := (0:500:3000, ["$i" for i in 0:500:3000])
        @series begin
            grid --> :all
            xerrscaling --> xerrscaling
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
            μ_cal = report.f_fit.(additional_pts.μ) .* report.e_unit
            (x = additional_pts.μ, y = additional_pts.peaks, 
                residuals_norm = (value.(μ_cal) .- additional_pts.peaks)./ uncertainty.(μ_cal))
        else
            NamedTuple()
        end
        xlabel := "Energy (ADC)"
        legend := :bottomright
        framestyle := :box
        xlims := (0, 168000)
        xticks := (0:16000:176000)
        @series begin
            grid --> :all
            xerrscaling --> xerrscaling
            yerrscaling --> yerrscaling
            additional_pts := additional_pts
            (par = report.par, f_fit = report.f_fit, x = report.x, y = report.y, gof = report.gof)
        end
        @series begin
            seriestype := :hline
            label := L"Q_{\beta \beta}"
            color := :green
            fillalpha := 0.2
            linewidth := 2.5
            xticks := :none
            ylabel := "Energy ($(report.e_unit))"
            ylims := (0, 1.2*value(maximum(report.y)))
            yticks := (500:500:3000)
            subplot := 1
            [2039]
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

end # module LegendSpecFitsRecipesBaseExt
