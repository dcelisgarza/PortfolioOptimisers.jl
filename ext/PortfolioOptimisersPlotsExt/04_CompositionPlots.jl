## plot_composition
function PortfolioOptimisers.plot_composition(w::VecNum, nx::AbstractVector = 1:length(w);
                                              N::Option{<:Number} = nothing, kwargs...)
    M = length(w)
    N, idx = relevant_assets(w, M, N)
    if M > N
        sort!(view(idx, 1:N))
        fidx = view(idx, 1:N)
        w_plot = [view(w, fidx); sum(view(w, view(idx, (N + 1):M)))]
        nx_plot = [nx[fidx]; "Others"]
    else
        w_plot = w
        nx_plot = nx
    end
    return bar(w_plot; xticks = (1:length(nx_plot), nx_plot),
               title = "Portfolio Composition", xlabel = "Asset", ylabel = "Weight",
               xrotation = 90, legend = false, kwargs...)
end
function PortfolioOptimisers.plot_composition(res::OptimisationResult, rd::ReturnsResult;
                                              N::Option{<:Number} = nothing, kwargs...)
    nx = isnothing(rd.nx) ? (1:length(res.w)) : rd.nx
    return PortfolioOptimisers.plot_composition(res.w, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_composition(res::OptimisationResult,
                                              pr::PortfolioOptimisers.AbstractPriorResult;
                                              N::Option{<:Number} = nothing, kwargs...)
    return PortfolioOptimisers.plot_composition(res.w, 1:length(res.w); N = N, kwargs...)
end
function PortfolioOptimisers.plot_composition(pred::PredictionResult;
                                              N::Option{<:Number} = nothing, kwargs...)
    # The weights are on the caller's universe and the fold's `nx` on the investable one, so
    # the weights are viewed at the mask, as the fold method of `plot_risk_contribution` does.
    w = investable_weights_view(result_investable_mask(pred.res), pred.res.w)
    nx = isnothing(pred.rd.nx) ? (1:length(w)) : pred.rd.nx
    return PortfolioOptimisers.plot_composition(w, nx; N = N, kwargs...)
end
function PortfolioOptimisers.plot_composition(pred::MultiPeriodPredictionResult;
                                              N::Option{<:Number} = nothing, kwargs...)
    folds = pred.pred
    w_mat = hcat(getproperty.(getproperty.(folds, :res), :w)...)
    nx = something(folds[1].rd.nx, 1:size(w_mat, 1))
    return PortfolioOptimisers.plot_stacked_bar_composition(collect(eachcol(w_mat)), nx;
                                                            xlabel = "Fold",
                                                            title = "Walk-Forward Selection Composition",
                                                            kwargs...)
end
function PortfolioOptimisers.plot_composition(pred::PopulationPredictionResult;
                                              N::Option{<:Number} = nothing, kwargs...)
    members = pred.pred
    w_mat = hcat(map(m -> if isa(m, PredictionResult)
                         m.res.w
                     else
                         mean(hcat(getproperty.(getproperty.(m.pred, :res), :w)...);
                              dims = 2)[:]
                     end, members)...)
    rd1 = isa(members[1], PredictionResult) ? members[1].rd : members[1].pred[1].rd
    nx = something(rd1.nx, 1:size(w_mat, 1))
    return PortfolioOptimisers.plot_stacked_bar_composition(collect(eachcol(w_mat)), nx;
                                                            xlabel = "Population Member",
                                                            title = "Population Composition",
                                                            kwargs...)
end
## plot_stacked_bar_composition / plot_stacked_area_composition
function PortfolioOptimisers.plot_stacked_bar_composition(w::VecNum_VecVecNum,
                                                          nx::AbstractVector = 1:(if isa(w,
                                                                                         VecVecNum)
                                                                                      length(first(w))
                                                                                  else
                                                                                      length(w)
                                                                                  end);
                                                          kwargs...)
    wmat = isa(w, VecVecNum) ? hcat(w...) : w
    M = size(wmat, 2)
    ctg = repeat(nx; inner = M)
    return groupedbar(transpose(wmat); xticks = (1:M, 1:M), bar_position = :stack,
                      group = ctg, xlabel = "Portfolios", ylabel = "Weight",
                      title = "Portfolio Composition", legend = :outerright, kwargs...)
end
function PortfolioOptimisers.plot_stacked_bar_composition(res_vec::AbstractVector{<:OptimisationResult},
                                                          rd::ReturnsResult; kwargs...)
    w = getproperty.(res_vec, :w)
    nx = isnothing(rd.nx) ? (1:length(first(w))) : rd.nx
    return PortfolioOptimisers.plot_stacked_bar_composition(w, nx; kwargs...)
end
function PortfolioOptimisers.plot_stacked_area_composition(w::VecNum_VecVecNum,
                                                           nx::AbstractVector = 1:(if isa(w,
                                                                                          VecVecNum)
                                                                                       length(first(w))
                                                                                   else
                                                                                       length(w)
                                                                                   end);
                                                           kwargs...)
    wmat = isa(w, VecVecNum) ? hcat(w...) : w
    M = size(wmat, 2)
    return areaplot(transpose(wmat); xticks = (1:M, 1:M), label = permutedims(nx),
                    xlabel = "Portfolios", ylabel = "Weight",
                    title = "Portfolio Composition", legend = :outerright, kwargs...)
end
function PortfolioOptimisers.plot_stacked_area_composition(res_vec::AbstractVector{<:OptimisationResult},
                                                           rd::ReturnsResult; kwargs...)
    w = getproperty.(res_vec, :w)
    nx = isnothing(rd.nx) ? (1:length(first(w))) : rd.nx
    return PortfolioOptimisers.plot_stacked_area_composition(w, nx; kwargs...)
end
function PortfolioOptimisers.plot_stacked_area_composition(pred::MultiPeriodPredictionResult;
                                                           N::Option{<:Number} = nothing,
                                                           kwargs...)
    folds = pred.pred
    w_mat = hcat(getproperty.(getproperty.(folds, :res), :w)...)
    nx = something(folds[1].rd.nx, 1:size(w_mat, 1))
    return PortfolioOptimisers.plot_stacked_area_composition(collect(eachcol(w_mat)), nx;
                                                             xlabel = "Fold",
                                                             title = "Walk-Forward Selection Composition",
                                                             kwargs...)
end
function PortfolioOptimisers.plot_stacked_area_composition(pred::PopulationPredictionResult;
                                                           N::Option{<:Number} = nothing,
                                                           kwargs...)
    members = pred.pred
    w_mat = hcat(map(m -> if isa(m, PredictionResult)
                         m.res.w
                     else
                         mean(hcat(getproperty.(getproperty.(m.pred, :res), :w)...);
                              dims = 2)[:]
                     end, members)...)
    rd1 = isa(members[1], PredictionResult) ? members[1].rd : members[1].pred[1].rd
    nx = something(rd1.nx, 1:size(w_mat, 1))
    return PortfolioOptimisers.plot_stacked_area_composition(collect(eachcol(w_mat)), nx;
                                                             xlabel = "Population Member",
                                                             title = "Population Composition",
                                                             kwargs...)
end
## plot_risk_contribution
function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                    w::VecNum, rd::ReturnsResult,
                                                    fees::Option{<:Fees} = nothing;
                                                    delta::Number = 1e-6,
                                                    marginal::Bool = false,
                                                    percentage::Bool = true,
                                                    N::Option{<:Number} = nothing,
                                                    sca::Scalariser = SumScalariser(),
                                                    kwargs...)
    nx = isnothing(rd.nx) ? (1:size(rd.X, 2)) : rd.nx
    return PortfolioOptimisers.plot_risk_contribution(r, w, rd.X, fees; nx = nx,
                                                      delta = delta, marginal = marginal,
                                                      percentage = percentage, N = N,
                                                      sca = sca, kwargs...)
end
function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                    res::OptimisationResult,
                                                    rd::ReturnsResult; delta::Number = 1e-6,
                                                    marginal::Bool = false,
                                                    percentage::Bool = true,
                                                    N::Option{<:Number} = nothing,
                                                    sca::Scalariser = SumScalariser(),
                                                    kwargs...)
    _, w, rd, fees = result_investable_view(res, rd)
    return PortfolioOptimisers.plot_risk_contribution(r, w, rd, fees; delta = delta,
                                                      marginal = marginal,
                                                      percentage = percentage, N = N,
                                                      sca = sca, kwargs...)
end
function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                    res::OptimisationResult,
                                                    pr::PortfolioOptimisers.AbstractPriorResult;
                                                    nx::AbstractVector = 1:length(res.w),
                                                    delta::Number = 1e-6,
                                                    marginal::Bool = false,
                                                    percentage::Bool = true,
                                                    N::Option{<:Number} = nothing,
                                                    sca::Scalariser = SumScalariser(),
                                                    kwargs...)
    _, w, pr, fees, nx = result_investable_view(res, pr, nothing, nx)
    return PortfolioOptimisers.plot_risk_contribution(r, w, pr.X, fees; nx = nx,
                                                      delta = delta, marginal = marginal,
                                                      percentage = percentage, N = N,
                                                      sca = sca, kwargs...)
end
function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                    pred::PredictionResult{<:Any, <:Any,
                                                                           <:HeldWeightsResult},
                                                    fees::Option{<:Fees} = nothing;
                                                    kwargs...)
    # The record is on the caller's universe and the fee on the investable one, so the
    # fold views the weights and the returns at the mask, as `risk_contribution` does on a
    # fold; the fold's own `nx` is the reduced axis already.
    imsk = result_investable_mask(pred.res)
    w = investable_weights_view(imsk, pred.res.w)
    X = investable_weights_view(imsk, pred.hw.X)
    nx = isnothing(pred.rd.nx) ? (1:length(w)) : pred.rd.nx
    return PortfolioOptimisers.plot_risk_contribution(r, w, X,
                                                      fold_fees(pred.res, fees, pred.hw.X);
                                                      nx = nx, kwargs...)
end
function PortfolioOptimisers.plot_risk_contribution(::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                    ::PredictionResult{<:Any, <:Any,
                                                                       Nothing},
                                                    ::Option{<:Fees} = nothing; kwargs...)
    return throw(ArgumentError("`plot_risk_contribution(r, pred::PredictionResult)` needs the fold's asset returns, and this fold kept none: `pred.rd.X` is the portfolio return series, and `pred.hw` is absent because the fold's scheme set neither `wd` nor `pws`. Set one of them so the fold records its asset returns, or call `plot_risk_contribution(r, pred.res.w, rd::ReturnsResult, ...)` with the original returns data."))
end
## plot_factor_risk_contribution
function PortfolioOptimisers.plot_factor_risk_contribution(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                           res::OptimisationResult,
                                                           rd::ReturnsResult;
                                                           re::RegE_Reg = StepwiseRegression(),
                                                           delta::Number = 1e-6,
                                                           N::Option{<:Number} = nothing,
                                                           sca::Scalariser = SumScalariser(),
                                                           kwargs...)
    _, w, rd, fees = result_investable_view(res, rd)
    return PortfolioOptimisers.plot_factor_risk_contribution(r, w, rd.X, fees; re = re,
                                                             rd = rd, delta = delta, N = N,
                                                             sca = sca, kwargs...)
end
function PortfolioOptimisers.plot_factor_risk_contribution(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                           pred::PredictionResult{<:Any,
                                                                                  <:Any,
                                                                                  <:HeldWeightsResult},
                                                           fees::Option{<:Fees} = nothing;
                                                           rd::Option{<:ReturnsResult} = nothing,
                                                           kwargs...)
    # The same view at the mask as `factor_risk_contribution` takes on a fold, and the
    # same resolution of `rd`: a caller's is viewed, and `nothing` is built from the fold.
    imsk = result_investable_mask(pred.res)
    w = investable_weights_view(imsk, pred.res.w)
    X = investable_weights_view(imsk, pred.hw.X)
    fees = fold_fees(pred.res, fees, pred.hw.X)
    return PortfolioOptimisers.plot_factor_risk_contribution(r, w, X, fees;
                                                             rd = fold_factor_returns(imsk,
                                                                                      rd,
                                                                                      pred),
                                                             kwargs...)
end
function PortfolioOptimisers.plot_factor_risk_contribution(::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                           ::PredictionResult{<:Any, <:Any,
                                                                              Nothing},
                                                           ::Option{<:Fees} = nothing;
                                                           kwargs...)
    return throw(ArgumentError("`plot_factor_risk_contribution(r, pred::PredictionResult)` needs the fold's asset returns, and this fold kept none: `pred.rd.X` is the portfolio return series, and `pred.hw` is absent because the fold's scheme set neither `wd` nor `pws`. Set one of them so the fold records its asset returns, or call `plot_factor_risk_contribution(r, pred.res.w, rd::ReturnsResult, ...)` with the original returns data."))
end
## ────────────────────────────────────────────────────────────────────────────
## Risk contribution
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_risk_contribution(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                    w::VecNum, X::MatNum_Pr,
                                                    fees::Option{<:Fees} = nothing;
                                                    nx::AbstractVector = 1:length(w),
                                                    delta::Number = 1e-6,
                                                    marginal::Bool = false,
                                                    percentage::Bool = true,
                                                    erc::Bool = true,
                                                    N::Option{<:Number} = nothing,
                                                    sca::Scalariser = SumScalariser(),
                                                    kwargs...)
    if !(delta > zero(delta))
        throw(DomainError(delta, "delta must be > 0"))
    end
    if !isnothing(N) && !(N > zero(N))
        throw(DomainError(N, "N must be > 0"))
    end
    rc = risk_contribution(r, w, X, fees; delta = delta, marginal = marginal, sca = sca)
    if percentage
        rc = rc / sum(rc)
    end
    plt = PortfolioOptimisers.plot_composition(rc, nx; N = N, ylabel = "Contribution",
                                               title = "Risk Contribution", kwargs...)
    if erc
        plt = hline!(plt, [mean(rc)])
    end
    return plt
end
## ────────────────────────────────────────────────────────────────────────────
## Factor risk contribution
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_factor_risk_contribution(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                                           w::VecNum, X::MatNum_Pr,
                                                           fees::Option{<:Fees} = nothing;
                                                           re::RegE_Reg = StepwiseRegression(),
                                                           rd::ReturnsResult = ReturnsResult(),
                                                           nf::Option{<:AbstractVector} = nothing,
                                                           delta::Number = 1e-6,
                                                           N::Option{<:Number} = nothing,
                                                           percentage::Bool = true,
                                                           erc::Bool = true,
                                                           sca::Scalariser = SumScalariser(),
                                                           kwargs...)
    if !(delta > zero(delta))
        throw(DomainError(delta, "delta must be > 0"))
    end
    if !isnothing(N) && !(N > zero(N))
        throw(DomainError(N, "N must be > 0"))
    end
    rc = factor_risk_contribution(r, w, X, fees; re = re, rd = rd, delta = delta, sca = sca)
    # The last element is the contribution of the part of `w` with no exposure to any factor,
    # not a regression intercept, so its bar is named for that. The names apply only when they
    # count the columns of the loadings: a dimension-reduction regression fits fewer columns
    # than there are factors, and its bars are numbered.
    factor_names = if !isnothing(nf) && length(rc) == length(nf) + 1
        nf
    elseif !isnothing(rd.nf) && length(rc) == length(rd.nf) + 1
        rd.nf
    else
        string.(1:(length(rc) - 1))
    end
    factor_names = [factor_names; "Off-factor"]
    if percentage
        rc = rc / sum(rc)
    end
    plt = PortfolioOptimisers.plot_composition(rc, factor_names; N = N,
                                               title = "Factor Risk Contribution",
                                               xlabel = "Factor",
                                               ylabel = "Risk Contribution", kwargs...)
    if erc
        hline!(plt, [mean(rc)])
    end
    return plt
end
