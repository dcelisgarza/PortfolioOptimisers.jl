module PortfolioOptimisersPlotsExt

using PortfolioOptimisers, StatsPlots, GraphRecipes, LinearAlgebra, Statistics, StatsBase,
      Clustering, Distributions, StatsAPI

import PortfolioOptimisers: ArrNum, VecNum, MatNum, Arr3Num, Option, VecNum_VecVecNum,
                            Slv_VecSlv, MatNum_Pr, PrE_Pr, Pr_RR, HClE_HCl, VecVecNum,
                            RegE_Reg, NwE_ClE_Cl, AbstractCentralityEstimator,
                            AbstractClustersEstimator, AbstractClusteringResult,
                            AbstractBaseRiskMeasure, BaseRM_VecBaseRM, VecBaseRM,
                            Scalariser, SumScalariser, measure_label, extract_pr,
                            relevant_assets, extract_fees, OptimisationResult,
                            finite_magnitudes, finite_symmetric_clim, finite_columns,
                            investable_plot_view, result_investable_mask,
                            investable_weights_view, fold_factor_returns, fold_fees,
                            result_investable_view, strip_liquidation_carriers

# A result carries its weights on the caller's universe and its prior and fee on the one
# the fit solved (ADR 0115), so every result arity below reads the three through
# `result_investable_view`, which pairs them on the result's investable universe (#884).
# A drawn figure of a result therefore draws that universe alone: the frame ADR 0118 keeps
# for a caller's prior cannot be kept for a prior that is already reduced, so each bar
# carries its own name instead. The names are the caller's, viewed at the result's
# Investable Mask, or, with none, each investable asset's index in the universe the
# result was fitted on.
function result_axis_names(imsk::Option{<:BitVector}, pr, nx::Option{<:AbstractVector})
    return if !isnothing(nx)
        nx
    elseif isnothing(imsk)
        1:size(pr.X, 2)
    else
        findall(imsk)
    end
end
function result_prior_view(res::OptimisationResult, nx::Option{<:AbstractVector} = nothing)
    imsk, _, pr, _, nx = result_investable_view(res, nothing, nothing, nx)
    return pr, result_axis_names(imsk, pr, nx)
end

include("01_CumulativeReturnsPlots.jl")
include("02_DrawdownAndHistogramPlots.jl")
include("03_MeasurePlots.jl")
include("04_CompositionPlots.jl")
include("05_ClusteringPlots.jl")
include("06_MomentPlots.jl")
include("07_FactorMomentPlots.jl")
include("08_AttributionPlots.jl")
include("09_CrossValidationPlots.jl")
include("10_FrontierPlots.jl")
include("11_FactorDiagnosticsPlots.jl")
include("12_ForecastEvaluationPlots.jl")

end
