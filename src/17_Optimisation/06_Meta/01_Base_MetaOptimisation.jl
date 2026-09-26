"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the sub-portfolio enumeration of a meta-optimiser, which states what a sub-portfolio is and which assets it sees.

A meta-optimiser solves one inner problem for each sub-portfolio, predicts the returns of each sub-portfolio, and gives the outer optimiser a synthetic universe with one asset for each sub-portfolio. One module does this for every meta-optimiser that owns an outer optimiser. The two meta-optimisers of the library differ in one respect only, and this type names it.

  - [`NestedClustered`](@ref) enumerates cluster index sets. It views its one inner optimiser onto each cluster, and it views every full-universe quantity, the Prior Result and the Fees, onto that cluster too.
  - [`Stacking`](@ref) enumerates inner optimisers. Each one sees the whole universe, so the module views nothing.

[`FullUniverse`](@ref) and [`ClusterUniverse`](@ref) declare the two. The module reads them through the four methods below, so a third meta-optimiser needs a third subtype and no copy of the module.

# Interfaces

To implement a new sub-portfolio enumeration, subtype `SubPortfolioUniverse` and implement the following methods:

  - `sub_portfolio_count(u::SubPortfolioUniverse, opti) -> Integer`: The number of sub-portfolios. `opti` is the inner optimiser field of the meta-optimiser.
  - `sub_portfolio_predict(u::SubPortfolioUniverse, opti, i::Integer, rd::ReturnsResult, cv, ex) -> MultiPeriodPredictionResult`: The cross-validation prediction of sub-portfolio `i`, one [`cross_val_predict`](@ref) call. On the combinatorial path the call returns a [`PopulationPredictionResult`](@ref) instead.
  - `sub_portfolio_view(u::SubPortfolioUniverse, x, i::Integer)`: The full-universe quantity `x`, a Prior Result or Fees or `nothing`, viewed onto sub-portfolio `i`.
  - `fold_weight_matrix(predictions::VecMPredRes, u::SubPortfolioUniverse, f::Integer, na::Integer) -> Matrix`: The weights of every sub-portfolio in fold `f`, as an `assets × sub-portfolios` matrix over all `na` assets.

## Arguments

  - `u`: The sub-portfolio enumeration.
  - `opti`: The inner optimiser field of the meta-optimiser.
  - `i`: The index of the sub-portfolio.
  - `rd`: The returns data of the meta-optimiser.
  - `cv`: The cross-validation scheme, a copy that belongs to sub-portfolio `i`.
  - `ex`: The FLoops executor.
  - `x`: A Prior Result, Fees, or `nothing`.
  - `predictions`: One prediction result for each sub-portfolio.
  - `f`: The index of the fold.
  - `na`: The number of real assets.

## Returns

  - `sub_portfolio_count`: The number of sub-portfolios.
  - `sub_portfolio_predict`: The prediction result of sub-portfolio `i`.
  - `sub_portfolio_view`: `x` viewed onto sub-portfolio `i`.
  - `fold_weight_matrix`: The `assets × sub-portfolios` weight matrix of fold `f`.

# Examples

An enumeration that splits the assets into two halves. The fold-less prediction then returns the net return of each half's equal-weight portfolio.

```jldoctest
julia> struct Halves <: PortfolioOptimisers.SubPortfolioUniverse
           n::Int
       end

julia> halves(u::Halves) = [1:(u.n ÷ 2), (u.n ÷ 2 + 1):(u.n)];

julia> PortfolioOptimisers.sub_portfolio_count(::Halves, opti) = 2

julia> function PortfolioOptimisers.sub_portfolio_view(u::Halves, x, i::Integer)
           return PortfolioOptimisers.port_opt_view(x, halves(u)[i])
       end

julia> function PortfolioOptimisers.sub_portfolio_predict(u::Halves, opti, i::Integer,
                                                          rd::ReturnsResult, cv, ex)
           return cross_val_predict(opti, rd, cv; cols = halves(u)[i], ex = ex)
       end

julia> function PortfolioOptimisers.fold_weight_matrix(predictions, u::Halves, f::Integer,
                                                       na::Integer)
           W = zeros(na, 2)
           for (i, cl) in enumerate(halves(u))
               W[cl, i] = predictions[i].pred[f].res.w
           end
           return W
       end

julia> rd = ReturnsResult(; nx = [\"A\", \"B\", \"C\", \"D\"],
                          X = [0.5 0.25 -0.25 0.125; -0.5 0.75 0.25 0.375]);

julia> res = [optimise(EqualWeighted(), PortfolioOptimisers.port_opt_view(rd, cl))
              for cl in halves(Halves(4))];

julia> W = [0.5 0.0; 0.5 0.0; 0.0 0.5; 0.0 0.5];

julia> pr = prior(EmpiricalPrior(), rd);

julia> PortfolioOptimisers.predict_outer_returns(nothing, nothing, Halves(4), rd, pr, nothing, W,
                                                 res).X
2×2 Matrix{Float64}:
 0.375  -0.0625
 0.125   0.3125
```

# Related

  - [`FullUniverse`](@ref)
  - [`ClusterUniverse`](@ref)
  - [`sub_portfolio_count`](@ref)
  - [`sub_portfolio_predict`](@ref)
  - [`sub_portfolio_view`](@ref)
  - [`fold_weight_matrix`](@ref)
  - [`predict_outer_returns`](@ref)
"""
abstract type SubPortfolioUniverse end
"""
$(DocStringExtensions.TYPEDEF)

Makes each inner optimiser a sub-portfolio that sees the whole universe.

This is the enumeration of [`Stacking`](@ref). The module views nothing onto a sub-portfolio, and an inner weight vector already covers every asset, so [`fold_weight_matrix`](@ref) adds no zeros.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`ClusterUniverse`](@ref)
  - [`Stacking`](@ref)
"""
struct FullUniverse <: SubPortfolioUniverse end
"""
$(DocStringExtensions.TYPEDEF)

Makes each cluster a sub-portfolio that sees only its own assets.

This is the enumeration of [`NestedClustered`](@ref). One inner optimiser serves every sub-portfolio, viewed onto the assets of that sub-portfolio, so an inner weight vector is as long as its cluster. [`fold_weight_matrix`](@ref) puts zeros at the other assets.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`FullUniverse`](@ref)
  - [`NestedClustered`](@ref)
"""
struct ClusterUniverse{T <: VecVecInt} <: SubPortfolioUniverse
    """
    Asset indices of each sub-portfolio. They partition the universe, so a column with zeros at the other assets is the real weight of the sub-portfolio over the whole asset axis.
    """
    cls::T
end
"""
    sub_portfolio_count(u::FullUniverse, opti)
    sub_portfolio_count(u::ClusterUniverse, opti)

Count the sub-portfolios.

A [`FullUniverse`](@ref) enumerates the inner optimisers, so it has one sub-portfolio for each optimiser in `opti`. A [`ClusterUniverse`](@ref) enumerates the clusters, and one inner optimiser serves all of them.

# Arguments

  - `u`: Sub-portfolio enumeration.
  - `opti`: The inner optimiser field of the meta-optimiser. It is a vector of optimisers for a [`FullUniverse`](@ref), and one optimiser for a [`ClusterUniverse`](@ref), which does not read it.

# Returns

  - The number of sub-portfolios.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`sub_portfolio_predict`](@ref)
"""
function sub_portfolio_count(::FullUniverse, opti)
    return length(opti)
end
function sub_portfolio_count(u::ClusterUniverse, ::Any)
    return length(u.cls)
end
"""
    sub_portfolio_predict(u::FullUniverse, opti, i, rd, cv, ex)
    sub_portfolio_predict(u::ClusterUniverse, opti, i, rd, cv, ex)

Cross-validate sub-portfolio `i`.

The method makes one [`cross_val_predict`](@ref) call, and the enumeration selects the optimiser and the assets. A [`FullUniverse`](@ref) runs `opti[i]` and passes no `cols`. The sub-portfolio is the whole universe, and the method of `cross_val_predict` for a precomputed [`OptimisationResult`](@ref) has no `cols` keyword, so a colon there raises a `MethodError`. A [`ClusterUniverse`](@ref) runs the one inner optimiser on `u.cls[i]`.

# Arguments

  - `u`: Sub-portfolio enumeration.
  - `opti`: The inner optimiser field of the meta-optimiser.
  - `i`: Sub-portfolio index.
  - `rd`: Returns data.
  - `cv`: Cross-validation scheme, the copy that belongs to this sub-portfolio.
  - `ex`: FLoops executor that controls parallelism.

# Returns

  - The cross-validation prediction result of sub-portfolio `i`.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`sub_portfolio_count`](@ref)
  - [`sub_portfolio_cv`](@ref)
  - [`cross_val_predict`](@ref)
"""
function sub_portfolio_predict(::FullUniverse, opti, i::Integer, rd::ReturnsResult, cv,
                               ex::FLoops.Transducers.Executor)
    return cross_val_predict(opti[i], rd, cv; ex = ex)
end
function sub_portfolio_predict(u::ClusterUniverse, opti, i::Integer, rd::ReturnsResult, cv,
                               ex::FLoops.Transducers.Executor)
    return cross_val_predict(opti, rd, cv; cols = u.cls[i], ex = ex)
end
"""
    sub_portfolio_view(u::FullUniverse, x, i::Integer)
    sub_portfolio_view(u::ClusterUniverse, x, i::Integer)

View a full-universe quantity onto sub-portfolio `i`.

The quantities are the two that the predicted returns of a sub-portfolio read: the Prior Result and the Fees. A [`FullUniverse`](@ref) sub-portfolio holds the whole universe, so the method returns `x` unchanged. A [`ClusterUniverse`](@ref) sub-portfolio restricts `x` to its cluster through [`port_opt_view`](@ref).

# Arguments

  - `u`: Sub-portfolio enumeration.
  - `x`: Full-universe quantity, or `nothing`.
  - `i`: Sub-portfolio index.

# Returns

  - `x`, viewed onto sub-portfolio `i`.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`port_opt_view`](@ref)
  - [`predict_outer_returns`](@ref)
"""
function sub_portfolio_view(::FullUniverse, x, ::Integer)
    return x
end
function sub_portfolio_view(u::ClusterUniverse, x, i::Integer)
    return port_opt_view(x, u.cls[i])
end
"""
    sub_portfolio_cv(cv)

Give one sub-portfolio its own copy of the cross-validation scheme.

The module cross-validates the sub-portfolios in parallel. A scheme that draws its splits from a random number generator holds mutable state, and two sub-portfolios that advance one generator at the same time get different folds on each run. So the method copies a scheme that has an `rng` field, and each copy starts from the same state. It returns a scheme with no `rng` field unchanged.

No scheme that [`OptimisationCrossValidation`](@ref) accepts has an `rng` field, so every scheme a meta-optimiser runs today takes the second branch. A scheme with an `rng` field must implement `Base.copy`.

# Arguments

  - `cv`: Cross-validation scheme.

# Returns

  - A copy of `cv` when it has an `rng` field, otherwise `cv` itself.

# Related

  - [`sub_portfolio_predictions`](@ref)
  - [`cross_val_predict`](@ref)
"""
function sub_portfolio_cv(cv)
    return !hasfield(typeof(cv), :rng) ? cv : copy(cv)
end
"""
    outer_optimisation_finaliser(wb::WeightBounds, wf::WeightFinaliser, resi::VecOpt,
                                 rco::OptimisationReturnCode, w::VecNum, wi::MatNum)
    outer_optimisation_finaliser(wb::WeightBounds, wf::WeightFinaliser, resi::VecOpt,
                                 rcos::VecOptRetCode, ws::VecVecNum, wi::MatNum)

Combine the inner and outer weights of a meta-optimiser, and finalise them under its weight bounds.

The return code reports a failure at any of the three stages: an inner solve, the outer solve, and the finalisation of the combined weights.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w} &= \\mathbf{W} \\boldsymbol{v}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: Combined weights over the ``N`` assets, before the finalisation.
  - $(math_dict[:W_inner])
  - $(math_dict[:v_outer]) [`Stacking`](@ref) passes the coefficients of [`combination_weights`](@ref) in its place.
  - $(math_dict[:N])

# Algorithm

 1. Combine the weights, giving `w = wi * w`.
 2. Finalise `w` with `wf` under `wb` through [`finalise_weight_bounds`](@ref), giving `retcode` and the finalised `w`.
 3. Read the return code of every inner result, giving `resi_retcodes`.
 4. When an inner return code, `rco` or `retcode` is an [`OptimisationFailure`](@ref), replace `retcode` with an `OptimisationFailure` whose `res` is `(; msg, opti, opto, wb)`. `msg` has one line for each stage that failed, in the order inner, outer, finalisation. `opti` is `resi_retcodes`, `opto` is `rco`, and `wb` is the return code of step 2.
 5. Return `(retcode, w)`. The method returns the weights of step 2 when a stage fails too.

The method for an efficient frontier runs these steps once for each pair of `rcos` and `ws`.

# Arguments

  - `wb`: Weight bounds over the ``N`` assets.
  - `wf`: Weight finaliser.
  - `resi`: The results of the inner optimisations.
  - `rco`: The return code of the outer optimisation.
  - `w`: Outer weights, one entry for each sub-portfolio.
  - `rcos`, `ws`: The return codes and outer weights of the points of an efficient frontier.
  - `wi`: Inner weights, `assets × sub-portfolios`.

# Returns

  - `(retcode, w)`: The return code and the finalised weights. The frontier method returns a vector of return codes and a vector of weight vectors.

# Related

  - [`NestedClustered`](@ref)
  - [`Stacking`](@ref)
  - [`finalise_weight_bounds`](@ref)
  - [`WeightBounds`](@ref)
"""
function outer_optimisation_finaliser(wb::WeightBounds, wf::WeightFinaliser, resi::VecOpt,
                                      rco::OptimisationReturnCode, w::VecNum, wi::MatNum)
    w = wi * w
    retcode, w = finalise_weight_bounds(wf, wb, w)
    wb_flag = isa(retcode, OptimisationFailure)
    opto_flag = isa(rco, OptimisationFailure)
    resi_retcodes = getproperty.(resi, :retcode)
    resi_flag = any(x -> isa(x, OptimisationFailure), resi_retcodes)
    if resi_flag || opto_flag || wb_flag
        msg = ""
        if resi_flag
            msg *= "opti failed.\n"
        end
        if opto_flag
            msg *= "opto failed.\n"
        end
        if wb_flag
            msg *= "weight bounds finalisation failed.\n"
        end
        retcode = OptimisationFailure(;
                                      res = (; msg = msg, opti = resi_retcodes, opto = rco,
                                             wb = retcode))
    end
    return retcode, w
end
function outer_optimisation_finaliser(wb::WeightBounds, wf::WeightFinaliser, resi::VecOpt,
                                      rcos::VecOptRetCode, ws::VecVecNum, wi::MatNum)
    retcode_w = [outer_optimisation_finaliser(wb, wf, resi, rco, w, wi)
                 for (rco, w) in zip(rcos, ws)]
    return map(x -> x[1], retcode_w), map(x -> x[2], retcode_w)
end
"""
    combination_weights(scale::Nothing, w::VecNum_VecVecNum)
    combination_weights(scale::VecNum, w::VecNum)
    combination_weights(scale::VecNum, w::VecVecNum)

Apply a Combination Weight to the outer weights of a meta-optimiser.

The outer optimiser decides how much of each sub-portfolio to hold. A Combination Weight is a fixed belief about the same quantity, so the two multiply, and the method rescales the products to the total that the outer optimiser chose. Schur Complement Hierarchical Risk Parity blends its parameter bundles in the same way.

The rescale makes only the ratios between the entries of a Combination Weight matter. A common factor cancels, so the weight needs no normalised form. Three cases give back `w` unchanged:

  - a uniform weight, whatever its sum;
  - a lone sub-portfolio, because one element is not a combination;
  - an outer total other than one keeps its total, so the method does not overrule a `bgt` of `0.9`.

The outer problem never sees the weight. A common factor on every synthetic return column moves the trade-off between return and risk of [`MaximumUtility`](@ref), so a uniform weight in the outer problem is not neutral. A weight in the outer problem also needs every [`predict_outer_returns`](@ref) method to apply it, and a method that drops it makes a cross-validated run disagree with a fold-less run.

# Mathematical definition

```math
\\begin{align}
c_k &= \\begin{cases}
\\dfrac{s_k v_k}{\\sum_{j=1}^{K} s_j v_j} \\sum_{j=1}^{K} v_j & \\text{if } \\sum_{j=1}^{K} s_j v_j \\neq 0 \\text{ and } \\sum_{j=1}^{K} v_j \\neq 0\\,,\\\\
s_k v_k & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:c_k_comb])
  - $(math_dict[:s_k_comb])
  - $(math_dict[:v_outer])
  - $(math_dict[:K_sub])

The first case keeps the total, ``\\sum_k c_k = \\sum_k v_k``, and the ratios, ``c_k / c_j = s_k v_k / (s_j v_j)``. In the second case no scalar rescale can keep a zero total and apply the tilt at the same time, so the tilted products stand. They are finite and keep their ratios. A denominator near zero is not a degenerate case. The large factor is the correct rescale of a combination that nearly cancels.

# Algorithm

 1. Multiply the weight into the outer weights, giving `c = scale .* w`.
 2. Divide the two totals, giving the factor `f = sum(w) / sum(c)`.
 3. Return `c` when `f` is zero or not finite, otherwise `c * f`. A factor that overflows is not finite, so it takes the second case too.

A `scale` of `nothing` returns `w`, and a frontier applies the steps to each point.

# Arguments

  - `scale`: Combination Weight, one entry for each sub-portfolio, or `nothing`.
  - `w`: Outer weights, one entry for each sub-portfolio, or a vector of them on an efficient frontier.

# Returns

  - `w` when `scale` is `nothing`, otherwise the coefficients ``\\boldsymbol{c}``.

# Related

  - [`outer_optimisation_finaliser`](@ref)
  - [`Stacking`](@ref)
"""
function combination_weights(::Nothing, w::VecNum_VecVecNum)
    return w
end
function combination_weights(scale::VecNum, w::VecNum)
    c = scale .* w
    f = sum(w) / sum(c)
    # A zero, infinite or NaN factor is every way the rescale can fail to exist: a
    # zero-total combination, a zero-total outer allocation, and the `0/0` of both at
    # once. The tilt stands unrescaled in each, which is finite and keeps its ratios.
    return iszero(f) || !isfinite(f) ? c : c * f
end
function combination_weights(scale::VecNum, w::VecVecNum)
    return [combination_weights(scale, wi) for wi in w]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Prepare the returns data of the outer problem from the inner weights `wi`.

The method collapses every per-asset quantity of `rd` onto the synthetic assets, and it allocates the buffer that the caller fills with the synthetic returns. Benchmark returns are extensive and collapse as a weighted sum. The implied volatilities and the Asset Panel are intensive and collapse as convex combinations, so the gross exposure of a sub-portfolio does not scale them.

Destructure all six returned values. Julia discards trailing values without an error, so a caller that names five binds `pnl` to the name it means for the buffer `X`, and its first write into that name fails.

# Mathematical definition

```math
\\begin{align}
\\mathbf{B}^{o} &= \\mathbf{B} \\mathbf{W}\\,,\\\\
\\mathbf{V}^{o} &= \\mathbf{V} \\tilde{\\mathbf{W}}\\,,\\\\
\\boldsymbol{a}^{o} &= \\tilde{\\mathbf{W}}^\\intercal \\boldsymbol{a}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{B}``, ``\\mathbf{B}^{o}``: Benchmark returns, `observations × assets` and `observations × sub-portfolios`. A benchmark that is not a matrix is kept as it is.
  - ``\\mathbf{V}``, ``\\mathbf{V}^{o}``: Implied volatilities, `observations × assets` and `observations × sub-portfolios`.
  - ``\\boldsymbol{a}``, ``\\boldsymbol{a}^{o}``: Implied volatility risk premium adjustment, one entry for each asset and one for each sub-portfolio. A scalar adjustment is kept as it is.
  - $(math_dict[:W_inner])
  - $(math_dict[:W_tilde_syn])

A column of zeros in ``\\mathbf{W}`` gives a column of zeros in ``\\mathbf{V}^{o}`` and a zero entry in ``\\boldsymbol{a}^{o}``.

# Algorithm

 1. When `rd.B` is a matrix, collapse it, giving `B = rd.B * wi`, and name its columns `nb = ["_b1", …]`. Otherwise keep `rd.B` and `rd.nb`.
 2. When `rd.iv` is present or `rd.ivpa` is a vector, normalise `wi` with [`synthetic_asset_weights`](@ref), giving `wn`. Collapse `iv = rd.iv * wn` when it is present, and `ivpa = transpose(wn) * rd.ivpa` when it is a vector.
 3. Collapse the Asset Panel with [`collapse_asset_panel`](@ref), giving `pnl`.
 4. Allocate the buffer `X`, `observations × sub-portfolios`, with the element type of `rd.X`.

# Arguments

  - `rd`: The returns data of the meta-optimiser.
  - `wi`: Inner weights, `assets × sub-portfolios`.

# Returns

  - `nb`: Names of the benchmark columns.
  - `B`: Benchmark returns.
  - `iv`: Implied volatilities, or `nothing`.
  - `ivpa`: Implied volatility risk premium adjustment, or `nothing`.
  - `pnl`: Asset Panel over the synthetic assets, or `nothing`.
  - `X`: Uninitialised buffer for the outer returns matrix.

# Related

  - [`ReturnsResult`](@ref)
  - [`predict_outer_returns`](@ref)
  - [`collapse_asset_panel`](@ref)
  - [`synthetic_asset_weights`](@ref)
  - [`features_are_assets`](@ref)
"""
function prepare_outer_rd(rd::ReturnsResult, wi::MatNum)
    nb, B = if !isa(rd.B, MatNum)
        rd.nb, rd.B
    else
        ["_b$(i)" for i in 1:size(wi, 2)], rd.B * wi
    end
    iv = rd.iv
    ivpa = rd.ivpa
    iv_flag = !isnothing(iv)
    ivpa_flag = isa(ivpa, AbstractVector)
    if iv_flag || ivpa_flag
        # `iv` and `ivpa` are intensive, so they collapse as convex combinations.
        wn = synthetic_asset_weights(wi)
        if iv_flag
            iv = iv * wn
        end
        if ivpa_flag
            ivpa = transpose(wn) * ivpa
        end
    end
    # Features are intensive too. When a tensor Panel Field's labels *are* the asset names
    # the contraction is two-sided, so the synthetic universe keeps a square field whose
    # labels are the synthetic asset names — which is what keeps the square case true one
    # level up.
    pnl = collapse_asset_panel(rd.pnl, wi, rd.nx)
    # `rd` is the meta-optimiser's own returns result, not a fitted prior, so this row
    # count is the panel the sub-portfolios were scored over. It is not the model-wide
    # `:T` a JuMP head registers.
    X = Matrix{eltype(rd.X)}(undef, size(rd.X, 1), size(wi, 2))
    return nb, B, iv, ivpa, pnl, X
end
"""
    assert_fold_alignment(predictions) -> VecPredRes

Check that fold `f` of every sub-portfolio covers the same observations, and return the folds of the first sub-portfolio.

A meta-optimiser runs the same cross-validation scheme over the same returns data for every sub-portfolio, so fold `f` covers the same observations for each of them. [`rebuild_returns_result`](@ref) needs this. Its `reshape(X, :, N)` lines up the `N` stacked return vectors row by row, and the weight matrix of a fold exists only when they agree. The check matters most on the combinatorial path, where the `scorer` of each sub-portfolio selects its path on its own.

# Algorithm

 1. Read the folds of the first sub-portfolio, giving `pred1`, and their number `nf`.
 2. For each sub-portfolio, check that it has `nf` folds.
 3. For each fold, compare the timestamps with those of `pred1` when the returns data has a clock, and the number of return entries when it has none.

The number of entries is the strongest check available without a clock, and it is the one `reshape` needs.

# Arguments

  - `predictions`: One [`MultiPeriodPredictionResult`](@ref) for each sub-portfolio.

# Validation

  - Every sub-portfolio has the same number of folds as the first, else `DimensionMismatch`.
  - Fold `f` of every sub-portfolio has the timestamps of fold `f` of the first, or the same number of return entries when there are no timestamps, else `DimensionMismatch`.

# Returns

  - The per-fold [`PredictionResult`](@ref) objects of the first sub-portfolio.

# Related

  - [`rebuild_returns_result`](@ref)
  - [`fold_row_indices`](@ref)
"""
function assert_fold_alignment(predictions::VecMPredRes)
    pred1 = predictions[1].pred
    nf = length(pred1)
    for (i, prediction) in enumerate(predictions)
        predi = prediction.pred
        @argcheck(length(predi) == nf,
                  DimensionMismatch("every sub-portfolio must run the same number of cross-validation folds, but sub-portfolio 1 has $(nf) and sub-portfolio $(i) has $(length(predi))"))
        for f in 1:nf
            ts1 = pred1[f].rd.ts
            aligned = if isnothing(ts1)
                length(predi[f].rd.X) == length(pred1[f].rd.X)
            else
                predi[f].rd.ts == ts1
            end
            @argcheck(aligned,
                      DimensionMismatch("sub-portfolios 1 and $(i) disagree on the observations fold $(f) covers, so their predictions cannot be laid out side by side. Every sub-portfolio of a meta-optimiser must run the same cross-validation over the same returns result."))
        end
    end
    return pred1
end
"""
    fold_row_indices(rd, pred) -> VecVecInt

Find the rows of the original returns data that each cross-validation fold covers.

The folds do not store their row indices. [`port_opt_view`](@ref) slices `ts` with the `test_idx` of the fold, so the `rd.ts` of a fold is its slice of the original clock, and [`feature_row_indices`](@ref) finds the rows from it. This works on the combinatorial path too, where [`sort_predictions!`](@ref) puts the folds of a path in split order and not in time order. The timestamps carry the order that the folds have.

Only a time-varying Asset Panel needs the rows. A static panel has no observation axis, so only the time-varying shape needs the clock.

# Arguments

  - `rd`: The original [`ReturnsResult`](@ref). The observation axis of a time-varying `rd.pnl` is parallel to its `ts`.
  - `pred`: The per-fold [`PredictionResult`](@ref) objects of one sub-portfolio.

# Validation

  - `rd.ts` is not `nothing`, else `IsNothingError`.

# Returns

  - One vector of row indices for each fold.

# Related

  - [`feature_row_indices`](@ref)
  - [`fold_feature_anchors`](@ref)
  - [`assert_fold_alignment`](@ref)
"""
function fold_row_indices(rd::ReturnsResult, pred::VecPredRes)
    @argcheck(!isnothing(rd.ts),
              IsNothingError("a time-varying Asset Panel holds its observation axis parallel to the returns result's timestamps, so collapsing it onto a meta-optimiser's synthetic assets fold by fold needs `ts` to find the rows of the panel that each fold covers. Got ts => nothing. Supply timestamps, or pass a static Asset Panel, which has no observation axis to align."))
    return [feature_row_indices(rd.pnl, p.rd.ts, rd.ts) for p in pred]
end
"""
    fold_weight_matrix(predictions, u::FullUniverse, f, na)
    fold_weight_matrix(predictions, u::ClusterUniverse, f, na)

Write the sub-portfolio weights of fold `f` as the `assets × sub-portfolios` matrix that the collapse of the Asset Panel reads.

# Mathematical definition

```math
\\begin{align}
W_{ik} &= \\begin{cases}
w^{(f)}_{k,i} & \\text{if } i \\in \\mathcal{C}_k\\,,\\\\
0 & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``W_{ik}``: Entry of the weight matrix of fold ``f``, the weight of sub-portfolio ``k`` on asset ``i``.
  - ``w^{(f)}_{k,i}``: Weight of sub-portfolio ``k`` on asset ``i`` in fold ``f``.
  - ``\\mathcal{C}_k``: Assets of sub-portfolio ``k``. It is every asset for a [`FullUniverse`](@ref), and cluster ``k`` for a [`ClusterUniverse`](@ref).

The clusters of a [`ClusterUniverse`](@ref) partition the assets, so a column is the real weight of its sub-portfolio over every asset.

# Arguments

  - `predictions`: One [`MultiPeriodPredictionResult`](@ref) for each sub-portfolio.
  - `u`: Sub-portfolio enumeration, a [`SubPortfolioUniverse`](@ref).
  - `f`: Fold index.
  - `na`: Number of real assets.

# Returns

  - The `assets × sub-portfolios` weight matrix of fold `f`.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`rebuild_asset_panel`](@ref)
  - [`collapse_asset_panel`](@ref)
"""
function fold_weight_matrix(predictions::VecMPredRes, ::FullUniverse, f::Integer,
                            na::Integer)
    ws = [prediction.pred[f].res.w for prediction in predictions]
    W = Matrix{mapreduce(eltype, promote_type, ws)}(undef, na, length(ws))
    @inbounds for (i, w) in enumerate(ws)
        W[:, i] = w
    end
    return W
end
function fold_weight_matrix(predictions::VecMPredRes, u::ClusterUniverse, f::Integer,
                            na::Integer)
    ws = [prediction.pred[f].res.w for prediction in predictions]
    W = zeros(mapreduce(eltype, promote_type, ws), na, length(ws))
    @inbounds for (i, (w, cl)) in enumerate(zip(ws, u.cls))
        W[cl, i] = w
    end
    return W
end
"""
    fold_asset_panel(pnl::AssetPanel, nx, wi, anchor) -> AssetPanel

Collapse the original [`AssetPanel`](@ref) onto the synthetic assets of one fold, with an observation axis.

The collapse is [`collapse_asset_panel`](@ref) of the original panel with the weights of the fold. The anchor of the fold depends on the shape of the panel:

  - A static panel has no observation axis, so its anchor is the number of observations of the fold. The method repeats its one collapsed panel over them. The collapse depends on the weights of the fold, so it is constant in one fold and changes in the next, and the outer problem sees a time-varying panel.
  - For a time-varying panel, the method first cuts the panel to the rows of the fold in the original clock, and the collapse keeps that observation axis. The anchor is those rows, and [`fold_row_indices`](@ref) finds them from the timestamps of the fold.

# Algorithm

 1. For a static panel, collapse it, giving `c`, and lift every field of `c` to the `anchor` observations of the fold with [`panel_field_lift`](@ref). Both universe masks are all `true`.
 2. For a time-varying panel, view the rows `anchor` with [`port_opt_view`](@ref) and collapse that view.

[`rebuild_asset_panel`](@ref) returns before this method when the returns data has no panel.

# Arguments

  - `pnl`: The original Asset Panel, not sliced.
  - `nx`: The asset names of the returns data, or `nothing`. Only the square case reads them.
  - `wi`: The weights of the fold, `assets × sub-portfolios`.
  - `anchor`: The number of observations of the fold for a static panel, its rows for a time-varying one.

# Returns

  - A time-varying Asset Panel over the observations of the fold.

# Related

  - [`collapse_asset_panel`](@ref)
  - [`fold_weight_matrix`](@ref)
  - [`fold_feature_anchors`](@ref)
  - [`rebuild_asset_panel`](@ref)
  - [`panel_field_lift`](@ref)
"""
function fold_asset_panel(pnl::AssetPanel, nx::Option{<:VecStr}, wi::MatNum, anchor)
    if panel_is_static(pnl)
        c = collapse_asset_panel(pnl, wi, nx)
        n = Int(anchor)
        na = panel_axes(c)[end]
        return AssetPanel(; pf = [panel_field_lift(f, n) for f in c.pf],
                          amsk = trues(n, na), emsk = trues(n, na))
    end
    return collapse_asset_panel(port_opt_view(pnl, anchor, :, nx), wi, nx)
end
"""
    panel_field_stack(fs::AbstractVector{<:NumericPanelField}) -> NumericPanelField
    panel_field_stack(fs::AbstractVector{<:TensorPanelField}) -> TensorPanelField

Stack the collapses of one Panel Field over the folds, along the observation axis.

Every fold collapses the same source field onto the same synthetic assets, so the fold results agree on every axis except the observations, and the stack concatenates them there. The observed masks stack with the values. [`collapse_panel_field`](@ref) turns a categorical field into a tensor field, so a fold result is never categorical.

# Algorithm

 1. Concatenate the values of every fold along the observation axis.
 2. Concatenate the observed masks in the same way, or keep `nothing` when the field carries none.
 3. Build the field again with its keyword constructor, which runs every check again.

# Arguments

  - `fs`: One collapsed Panel Field for each fold, in fold order.

# Returns

  - The stacked Panel Field.

# Related

  - [`rebuild_asset_panel`](@ref)
  - [`fold_asset_panel`](@ref)
  - [`AbstractPanelField`](@ref)
"""
function panel_field_stack(fs::AbstractVector{<:NumericPanelField})
    return NumericPanelField(; name = fs[1].name, vals = vcat((f.vals for f in fs)...),
                             omsk = if isnothing(fs[1].omsk)
                                 nothing
                             else
                                 vcat((f.omsk for f in fs)...)
                             end)
end
function panel_field_stack(fs::AbstractVector{<:TensorPanelField})
    return TensorPanelField(; name = fs[1].name, axis = fs[1].axis, labels = fs[1].labels,
                            groups = fs[1].groups,
                            vals = cat((f.vals for f in fs)...; dims = 1),
                            omsk = if isnothing(fs[1].omsk)
                                nothing
                            else
                                cat((f.omsk for f in fs)...; dims = 1)
                            end)
end
"""
    fold_feature_anchors(rd, pred)

Give each fold the anchor that [`fold_asset_panel`](@ref) needs: the number of observations for a static panel, and the rows for a time-varying one.

Only a time-varying panel needs rows, so only that shape asks the returns data for timestamps. A static panel runs on the fold sizes alone.

# Arguments

  - `rd`: The original [`ReturnsResult`](@ref).
  - `pred`: The per-fold [`PredictionResult`](@ref) objects of one sub-portfolio.

# Returns

  - One anchor for each fold: an `Integer` for a static panel or no panel, a vector of row indices for a time-varying one.

# Related

  - [`fold_asset_panel`](@ref)
  - [`fold_row_indices`](@ref)
"""
function fold_feature_anchors(rd::ReturnsResult, pred::VecPredRes)
    return if !isnothing(rd.pnl) && !panel_is_static(rd.pnl)
        fold_row_indices(rd, pred)
    else
        [length(p.rd.X) for p in pred]
    end
end
"""
    rebuild_asset_panel(rd, predictions, u, pred1)

Compute the [`AssetPanel`](@ref) of the outer problem on the cross-validated path.

For each fold, the method makes the same [`collapse_asset_panel`](@ref) call that [`prepare_outer_rd`](@ref) makes on the fold-less path, with the same asset names and the same original panel, and the weights of that fold. Thus `cv`, which controls execution only, does not change what the outer optimiser measures.

The stacked panel is time-varying, and both of its universe masks are all `true`. The fold panels cover disjoint windows of one synthetic universe, and each synthetic asset exists in every window.

# Algorithm

 1. Return `nothing` when `rd` carries no panel.
 2. For each fold, build its weight matrix with [`fold_weight_matrix`](@ref) and collapse the original panel with [`fold_asset_panel`](@ref) at the anchor from [`fold_feature_anchors`](@ref), giving `ps`.
 3. Stack each Panel Field over the folds with [`panel_field_stack`](@ref), giving `pf`.
 4. Count the observations of all folds, giving `nobs`, and build the panel with all-`true` masks of size `nobs × sub-portfolios`.

# Arguments

  - `rd`: The original [`ReturnsResult`](@ref). The method collapses its panel without slicing it first.
  - `predictions`: One [`MultiPeriodPredictionResult`](@ref) for each sub-portfolio.
  - `u`: Sub-portfolio enumeration, a [`SubPortfolioUniverse`](@ref).
  - `pred1`: The folds of the first sub-portfolio, from [`assert_fold_alignment`](@ref). Every sub-portfolio agrees with them, so they set the fold boundaries.

# Returns

  - `pnl::Option{AssetPanel}`: The Asset Panel over the synthetic assets, or `nothing` when `rd` carries none.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`rebuild_returns_result`](@ref)
  - [`prepare_outer_rd`](@ref)
  - [`fold_asset_panel`](@ref)
  - [`fold_feature_anchors`](@ref)
  - [`panel_field_stack`](@ref)
"""
function rebuild_asset_panel(rd::ReturnsResult, predictions::VecMPredRes,
                             u::SubPortfolioUniverse, pred1::VecPredRes)
    # A local, so the check below narrows it; a second read of the field would not be.
    pnl = rd.pnl
    if isnothing(pnl)
        return nothing
    end
    na = size(rd.X, 2)
    ps = [fold_asset_panel(pnl, rd.nx, fold_weight_matrix(predictions, u, f, na), anchor)
          for (f, anchor) in enumerate(fold_feature_anchors(rd, pred1))]
    #! A panel with no Panel Field is the ingestion layer's shape, and an untyped
    #! comprehension over no field answers a `Vector{Any}` the panel's constructor refuses,
    #! so the comprehension is typed: it answers the same vector empty or full.
    pf = AbstractPanelField[panel_field_stack(concrete_typed_array_if_abstract([p.pf[k]
                                                                                for p in ps]))
                            for k in eachindex(ps[1].pf)]
    nobs = sum(p -> size(p.amsk, 1), ps)
    return AssetPanel(; pf = pf, amsk = trues(nobs, size(ps[1].amsk, 2)),
                      emsk = trues(nobs, size(ps[1].amsk, 2)))
end
"""
    rebuild_returns_result(rd, predictions, u)

Build the returns data of the outer problem from the cross-validation predictions of the sub-portfolios.

Column `k` of the result is the out-of-sample prediction of sub-portfolio `k`, and its rows are the rows of the folds, in fold order. `u` is the sub-portfolio enumeration: a [`ClusterUniverse`](@ref) for [`NestedClustered`](@ref) and a [`FullUniverse`](@ref) for [`Stacking`](@ref). It states where the weights of a fold sit on the asset axis.

`u` is positional and has no default. With a full-universe default, a two-argument call still runs. That is correct for [`Stacking`](@ref), but for [`NestedClustered`](@ref) it writes the weights of every cluster to the wrong rows and gives a wrong Asset Panel with no error.

The folds carry no Asset Panel. The method collapses the original `rd.pnl` again for each fold, with the same [`collapse_asset_panel`](@ref) call as [`prepare_outer_rd`](@ref) makes on the fold-less path, and stacks the fold results along the observation axis (see [`rebuild_asset_panel`](@ref)). The stack has the `observations × assets × features` shape of a time-varying panel, and a [`FeatureDistance`](@ref) with its default [`LastObservation`](@ref) reads the collapse of the last fold. The inner solves still see the panel of their own cluster. The collapse of the original panel also serves a square panel under [`NestedClustered`](@ref). Its folds see the returns of one cluster each, so their own panels have different feature axes and do not stack.

# Algorithm

 1. Copy the returns `X`, and `B` and `iv` when present, of the first prediction. Wrap its `ivpa` in a vector when present. Each prediction keeps the `ivpa` of its last fold as one number.
 2. Check the folds with [`assert_fold_alignment`](@ref), giving `pred1`.
 3. Append the returns, `iv` and `B` of every other prediction, and push its `ivpa`.
 4. Reshape the returns to `observations × sub-portfolios`, giving `X`.
 5. Check that the folds cover `size(X, 1)` observations, `nobs`.
 6. Build the Asset Panel with [`rebuild_asset_panel`](@ref), giving `pnl`.
 7. Reshape `B` and `iv` to `observations × sub-portfolios`, and name the benchmark columns `_b1`, `_b2`, ….
 8. Build the [`ReturnsResult`](@ref), with the asset names `_1`, `_2`, … and the factors and timestamps of the first prediction.

The method reads `predictions` and does not change them, so two calls on one vector give the same result.

# Arguments

  - `rd`: The original [`ReturnsResult`](@ref).
  - `predictions`: One [`MultiPeriodPredictionResult`](@ref) for each sub-portfolio.
  - `u`: Sub-portfolio enumeration, a [`SubPortfolioUniverse`](@ref).

# Validation

  - The checks of [`assert_fold_alignment`](@ref).
  - The folds of the first sub-portfolio cover as many observations as the stacked returns have rows, else `DimensionMismatch`.

# Returns

  - The returns data of the outer problem, one synthetic asset for each sub-portfolio.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`NestedClustered`](@ref)
  - [`Stacking`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`rebuild_asset_panel`](@ref)
  - [`assert_fold_alignment`](@ref)
  - [`prepare_outer_rd`](@ref)
"""
function rebuild_returns_result(rd::ReturnsResult, predictions::VecMPredRes,
                                u::SubPortfolioUniverse)
    N = length(predictions)
    nb = rd.nb
    B_flag = !isnothing(rd.B)
    iv_flag = !isnothing(rd.iv)
    ivpa_flag = !isnothing(rd.ivpa)
    rd1 = predictions[1].mrd
    # Copies, not the first prediction's own buffers: the loop below grows these, and
    # appending into `predictions[1].mrd` would leave the predictions mutated — a second
    # call on the same vector would then assemble a result of the wrong height. `ivpa` is
    # per-sub-portfolio rather than per-observation, so it is *wrapped*, not copied: each
    # fold already collapsed it to one number and `MultiPeriodPredictionResult` kept the
    # last fold's, so this builds the length-`N` vector from `N` scalars.
    X = copy(rd1.X)
    B = B_flag ? copy(rd1.B) : nothing
    iv = iv_flag ? copy(rd1.iv) : nothing
    ivpa = ivpa_flag ? [rd1.ivpa] : nothing
    pred1 = assert_fold_alignment(predictions)
    @inbounds for i in 2:N
        rdi = predictions[i].mrd
        append!(X, rdi.X)
        if iv_flag
            append!(iv, rdi.iv)
        end
        if ivpa_flag
            push!(ivpa, rdi.ivpa)
        end
        if B_flag
            append!(B, rdi.B)
        end
    end
    X = reshape(X, :, N)
    # The stacked rows are the fold rows, in fold order. `reshape` above has assumed it
    # since before the feature matrix existed; the recompute below depends on it too.
    # This count is the folds' own, not the model-wide `:T` of any one sub-portfolio.
    nobs = sum(p -> length(p.rd.X), pred1)
    @argcheck(nobs == size(X, 1),
              DimensionMismatch("the stacked sub-portfolio returns must have one row per cross-validated observation, but the folds cover $(nobs) observations and the stacked returns have $(size(X, 1))"))
    pnl = rebuild_asset_panel(rd, predictions, u, pred1)
    if B_flag
        B = reshape(B, :, N)
        nb = ["_b$(i)" for i in 1:N]
    end
    iv = iv_flag ? reshape(iv, :, N) : nothing
    return ReturnsResult(; nx = ["_$i" for i in 1:N], X = X, nf = rd1.nf, F = rd1.F,
                         nb = nb, B = B, ts = rd1.ts, iv = iv, ivpa = ivpa, pnl = pnl)
end
"""
    sub_portfolio_predictions(::Type{T}, opti, u, rd, cv, ex) where {T}

Cross-validate every sub-portfolio in parallel, over the same returns data.

Every sub-portfolio runs the same cross-validation over the same returns data, which is the condition that [`assert_fold_alignment`](@ref) checks.

# Algorithm

 1. Allocate one slot of type `T` for each of the [`sub_portfolio_count`](@ref) sub-portfolios, giving `predictions`.
 2. For each sub-portfolio `i`, in parallel under `ex`, give it its own scheme with [`sub_portfolio_cv`](@ref) and write the result of [`sub_portfolio_predict`](@ref) to `predictions[i]`.

# Arguments

  - `T`: Element type of the prediction vector. It is [`MultiPeriodPredictionResult`](@ref) on the non-combinatorial path and [`PopulationPredictionResult`](@ref) on the combinatorial one.
  - `opti`: The inner optimiser field of the meta-optimiser.
  - `u`: Sub-portfolio enumeration, a [`SubPortfolioUniverse`](@ref).
  - `rd`: Returns data.
  - `cv`: Cross-validation scheme.
  - `ex`: FLoops executor that controls parallelism.

# Returns

  - One prediction result for each sub-portfolio.

# Related

  - [`predict_outer_returns`](@ref)
  - [`sub_portfolio_predict`](@ref)
  - [`sub_portfolio_cv`](@ref)
"""
function sub_portfolio_predictions(::Type{T}, opti, u::SubPortfolioUniverse,
                                   rd::ReturnsResult, cv,
                                   ex::FLoops.Transducers.Executor) where {T}
    predictions = Vector{T}(undef, sub_portfolio_count(u, opti))
    FLoops.@floop ex for i in eachindex(predictions)
        predictions[i] = sub_portfolio_predict(u, opti, i, rd, sub_portfolio_cv(cv), ex)
    end
    return predictions
end
"""
    predict_outer_returns(cv::Option{<:OptimisationCrossValidation}, opt,
                          u::SubPortfolioUniverse, rd::ReturnsResult,
                          pr::AbstractPriorResult, fees::Option{<:Fees}, wi::MatNum,
                          resi::VecOpt)
    predict_outer_returns(cv::OptimisationCrossValidation{<:NonCombOptCV}, opt,
                          u::SubPortfolioUniverse, rd::ReturnsResult,
                          pr::AbstractPriorResult, fees::Option{<:Fees}, wi::MatNum,
                          resi::VecOpt)
    predict_outer_returns(cv::OptimisationCrossValidation{<:CombinatorialCrossValidation},
                          opt, u::SubPortfolioUniverse, rd::ReturnsResult,
                          pr::AbstractPriorResult, fees::Option{<:Fees}, wi::MatNum,
                          resi::VecOpt)

Predict the returns of the sub-portfolios of a meta-optimiser as the [`ReturnsResult`](@ref) of the outer problem.

One module serves every meta-optimiser that owns an outer optimiser. The sub-portfolio enumeration is the only part that changes between them, and it arrives as `u`. [`NestedClustered`](@ref) passes a [`ClusterUniverse`](@ref) and [`Stacking`](@ref) a [`FullUniverse`](@ref). No method reads the type of the meta-optimiser.

The methods dispatch on `cv`, which selects the prediction, so a custom cross-validation scheme is a method on the first argument:

  - Without folds, the column of each sub-portfolio is the net return of its own solve in `resi`, on the Prior Result and the Fees viewed onto it with [`sub_portfolio_view`](@ref).
  - Non-combinatorial cross-validation predicts each sub-portfolio out of sample and stacks the folds with [`rebuild_returns_result`](@ref).
  - Combinatorial cross-validation does the same, and first the `scorer` of the scheme selects one path from the population of each sub-portfolio. The default scorer is [`NearestQuantilePrediction`](@ref).

`wi` holds the weights of the inner optimisers alone. A Combination Weight acts at the combination after the outer solve, so no method applies it here (see [`combination_weights`](@ref)).

# Mathematical definition

Without folds:

```math
\\begin{align}
R_{tk} &= \\boldsymbol{x}_t^\\intercal \\mathbf{W}_{\\cdot k}\\,.
\\end{align}
```

Where:

  - ``R_{tk}``: Return of synthetic asset ``k`` at observation ``t``, before Fees.
  - $(math_dict[:x_t_obs])
  - $(math_dict[:W_inner])

With Fees, [`calc_net_returns`](@ref) subtracts the fee of sub-portfolio ``k`` from its column.

# Algorithm

Without folds:

 1. Prepare the returns data of the outer problem with [`prepare_outer_rd`](@ref), giving `nb`, `B`, `iv`, `ivpa`, `pnl` and the buffer `X`.
 2. For each inner result `res` in `resi`, write its net returns with [`calc_net_returns`](@ref) to column `i` of `X`, on the Prior Result and Fees viewed onto sub-portfolio `i`.
 3. Build the [`ReturnsResult`](@ref), with the asset names `_1`, `_2`, … and the factors and timestamps of `rd`.

With cross-validation:

 1. Cross-validate every sub-portfolio with [`sub_portfolio_predictions`](@ref) under the scheme `cv.cv`, giving `predictions`.
 2. On the combinatorial path, select one path of each population with `cv.scorer`, or with [`NearestQuantilePrediction`](@ref) when it is `nothing`.
 3. Stack the predictions with [`rebuild_returns_result`](@ref).

# Arguments

  - `cv`: The cross-validation scheme of the meta-optimiser.
  - `opt`: The meta-optimiser. The cross-validated methods read its `opti` and `ex` fields.
  - `u`: Sub-portfolio enumeration.
  - `rd`: Returns data.
  - `pr`: Prior Result over the whole universe.
  - `fees`: Fees over the whole universe.
  - `wi`: Inner weights, `assets × sub-portfolios`.
  - `resi`: The results of the inner optimisations.

# Returns

  - The [`ReturnsResult`](@ref) of the outer problem, one synthetic asset for each sub-portfolio.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`prepare_outer_rd`](@ref)
  - [`rebuild_returns_result`](@ref)
  - [`sub_portfolio_predictions`](@ref)
  - [`NestedClustered`](@ref)
  - [`Stacking`](@ref)
"""
function predict_outer_returns(::Option{<:OptimisationCrossValidation}, ::Any,
                               u::SubPortfolioUniverse, rd::ReturnsResult,
                               pr::AbstractPriorResult, fees::Option{<:Fees}, wi::MatNum,
                               resi::VecOpt)
    nb, B, iv, ivpa, pnl, X = prepare_outer_rd(rd, wi)
    for (i, res) in enumerate(resi)
        X[:, i] = calc_net_returns(res, sub_portfolio_view(u, pr, i),
                                   sub_portfolio_view(u, fees, i))
    end
    return ReturnsResult(; nx = ["_$i" for i in 1:size(wi, 2)], X = X, nf = rd.nf, F = rd.F,
                         nb = nb, B = B, ts = rd.ts, iv = iv, ivpa = ivpa, pnl = pnl)
end
function predict_outer_returns(cv::OptimisationCrossValidation{<:NonCombOptCV}, opt,
                               u::SubPortfolioUniverse, rd::ReturnsResult,
                               ::AbstractPriorResult, ::Option{<:Fees}, ::MatNum, ::VecOpt)
    predictions = sub_portfolio_predictions(MultiPeriodPredictionResult, opt.opti, u, rd,
                                            cv.cv, opt.ex)
    return rebuild_returns_result(rd, predictions, u)
end
function predict_outer_returns(cv::OptimisationCrossValidation{<:CombinatorialCrossValidation},
                               opt, u::SubPortfolioUniverse, rd::ReturnsResult,
                               ::AbstractPriorResult, ::Option{<:Fees}, ::MatNum, ::VecOpt)
    predictions = sub_portfolio_predictions(PopulationPredictionResult, opt.opti, u, rd,
                                            cv.cv, opt.ex)
    scorer = isnothing(cv.scorer) ? NearestQuantilePrediction() : cv.scorer
    return rebuild_returns_result(rd, [scorer(prediction) for prediction in predictions], u)
end
public SubPortfolioUniverse, sub_portfolio_count, sub_portfolio_predict, sub_portfolio_view,
       fold_weight_matrix
