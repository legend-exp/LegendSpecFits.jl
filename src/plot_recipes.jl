# This file is a part of LegendSpecFits.jl, licensed under the MIT License (MIT).

@recipe function f(diagnostics::LegendSpecFits.EnergyCalibrationDiagnostics)
    seriestype --> :stepbins
    diagnostics.cal_hist
end
