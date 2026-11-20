"""
$(DocStringExtensions.TYPEDEF)

Bounds the magnitude of each weight by a multiple of the score of its asset, on the returns of each fit.

[`weight_bounds_constraints`](@ref) resolves the estimator in each fit. It scores the assets of the fit with [`asset_scores`](@ref), multiplies each score by its entry of `val`, and joins the result with the static bounds of `wb`. The tighter bound wins at each asset. Each fit reads the scores of its own window, so a fold of a cross-validation reads the scores of its training rows, and the last fit reads the scores of the whole window.

Two common bounds are of this form.

  - A participation limit. The score is the median traded amount of the asset, `PanelFieldScorer(; fields = ["adj_close", "adj_volume"])`, and `val = k / V`, where `V` is the notional of the portfolio. A position then needs at most `k` days of trade to enter or to leave.
  - A bound relative to size. The score is the market capitalisation, and `val` is `m` divided by the capitalisation of the universe. No weight is then more than `m` times the capitalisation weight of its asset.

# Mathematical definition

```math
\\begin{align}
c_{i} &= v_{i} s_{i}\\,,\\\\
\\max(l_{i},\\, -c_{i}) &\\leq w_{i} \\leq \\min(u_{i},\\, c_{i})\\,.
\\end{align}
```

Where:

  - ``s_{i}``: The score of asset ``i`` on the returns of the fit.
  - ``v_{i}``: The entry of `val` of asset ``i``.
  - ``c_{i}``: The score bound of asset ``i``. An asset whose score is not finite takes `nonfinite`, or ``+\\infty`` when `nonfinite` is `nothing`. An infinite ``v_{i}`` gives ``c_{i} = +\\infty``.
  - ``l_{i}``, ``u_{i}``: The static bounds of asset ``i`` from `wb`.
  - $(math_dict[:w_port])

With `symmetric = false`, the lower side is ``l_{i}`` alone.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ScoreWeightBounds(;
        score::BaseRM_AsSc,
        val::VectorEstValType = 1,
        dval::Option{<:Number} = nothing,
        wb::Option{<:WbE_Wb} = WeightBounds(),
        symmetric::Bool = true,
        nonfinite::Option{<:Number} = 0
    ) -> ScoreWeightBounds

Keywords correspond to the struct's fields. An asset that a pair, a dictionary or a vector of pairs in `val` does not name takes `dval`, or `one(datatype)` when `dval` is `nothing`.

## Validation

  - `score` can score one asset, through [`assert_scoreable`](@ref). An `ArgumentError` is thrown otherwise.
  - `val` is not empty, finite and non-negative, through [`assert_nonempty_nonneg_finite_val`](@ref). An `IsEmptyError` or a `DomainError` is thrown otherwise. An algorithmic rule is not checked.
  - If `dval` is not `nothing`: `dval >= 0`, through [`assert_nonneg`](@ref). This rule admits `+Inf`, which gives the assets that `val` does not name no bound from the score. A `DomainError` is thrown otherwise.
  - If `nonfinite` is not `nothing`: `nonfinite >= 0`, through [`assert_nonneg`](@ref). A `DomainError` is thrown otherwise.

A score bound is a magnitude, so the resolution refuses a negative score. The static bounds of `wb` have no values until the fit, so the resolution also checks them against the score bound. [`weight_bounds_constraints`](@ref) states both checks.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `val`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `wb`: Viewed through [`port_opt_view`](@ref).

The view keeps `score` unchanged. The score reads the returns of the fit, and the fit views them by the same index.

# Examples

```jldoctest
julia> ScoreWeightBounds(; score = PanelFieldScorer(; fields = \"volume\"), val = 0.5)
ScoreWeightBounds
      score ┼ PanelFieldScorer
            │    fields ┼ String: \"volume\"
            │      stat ┼ MedianValue
            │           │   w ┴ nothing
            │       bib ┼ nothing
            │   min_obs ┴ Int64: 1
        val ┼ Float64: 0.5
       dval ┼ nothing
         wb ┼ WeightBounds
            │   lb ┼ Float64: 0.0
            │   ub ┴ Float64: 1.0
  symmetric ┼ Bool: true
  nonfinite ┴ Int64: 0
```

# Related

  - [`AbstractWeightBoundsEstimator`](@ref)
  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints`](@ref)
  - [`asset_scores`](@ref)
  - [`PanelFieldScorer`](@ref)
  - [`AbstractAssetScorer`](@ref)
"""
@propagatable @concrete struct ScoreWeightBounds <: AbstractWeightBoundsEstimator
    """
    $(field_dict[:swb_score])
    """
    score
    """
    $(field_dict[:swb_val])
    """
    @vprop val
    """
    $(field_dict[:dval])
    """
    dval
    """
    $(field_dict[:swb_wb])
    """
    @vprop wb
    """
    $(field_dict[:swb_symmetric])
    """
    symmetric
    """
    $(field_dict[:swb_nonfinite])
    """
    nonfinite
    function ScoreWeightBounds(score::BaseRM_AsSc, val::VectorEstValType,
                               dval::Option{<:Number}, wb::Option{<:WbE_Wb},
                               symmetric::Bool,
                               nonfinite::Option{<:Number})::ScoreWeightBounds
        assert_scoreable(score)
        assert_nonempty_nonneg_finite_val(val, :val)
        if !isnothing(dval)
            assert_nonneg(dval, :dval)
        end
        if !isnothing(nonfinite)
            assert_nonneg(nonfinite, :nonfinite)
        end
        return new{typeof(score), typeof(val), typeof(dval), typeof(wb), typeof(symmetric),
                   typeof(nonfinite)}(score, val, dval, wb, symmetric, nonfinite)
    end
end
function ScoreWeightBounds(; score::BaseRM_AsSc, val::VectorEstValType = 1,
                           dval::Option{<:Number} = nothing,
                           wb::Option{<:WbE_Wb} = WeightBounds(), symmetric::Bool = true,
                           nonfinite::Option{<:Number} = 0)::ScoreWeightBounds
    return ScoreWeightBounds(score, val, dval, wb, symmetric, nonfinite)
end
"""
    score_cap(v::Number, s::Number, nf::Number) -> Number

Return the score bound of one asset, ``c = v s``.

# Algorithm

 1. Compute `c = v * s`. The bound takes the type of this product.
 2. If `s` is not finite, return `nf` in the type of `c`.
 3. If `v` is infinite, return `v` in the type of `c`. The product `Inf * 0` is `NaN`, and an infinite multiple states no bound.
 4. Otherwise return `c`.

# Arguments

  - `v`: The multiple of the asset, from the `val` of a [`ScoreWeightBounds`](@ref).
  - `s`: The score of the asset.
  - `nf`: The bound of an asset whose score is not finite.

# Returns

  - `c::Number`: The score bound of the asset.

# Examples

```jldoctest
julia> PortfolioOptimisers.score_cap(0.5, 2.0, 0)
1.0

julia> PortfolioOptimisers.score_cap(0.5, NaN, 0)
0.0
```

# Related

  - [`ScoreWeightBounds`](@ref)
  - [`score_bounds`](@ref)
"""
function score_cap(v::Number, s::Number, nf::Number)
    c = v * s
    if !isfinite(s)
        return convert(typeof(c), nf)
    elseif isinf(v)
        return convert(typeof(c), v)
    end
    return c
end
"""
    score_bounds(wb::WeightBounds, c::VecNum, symmetric::Bool) -> WeightBounds

Join the resolved static bounds `wb` with the score bounds `c`.

# Algorithm

 1. Check that no entry of `c` is negative, that no lower bound of `wb` is above its entry of `c`, and, when `symmetric` is `true`, that no upper bound of `wb` is below the negative of its entry of `c`.
 2. Take the lower side `max.(wb.lb, -c)` when `symmetric` is `true`, and `wb.lb` otherwise.
 3. Take the upper side `min.(wb.ub, c)`.
 4. Build the [`WeightBounds`](@ref) of the two sides.

# Arguments

  - `wb`: The static bounds, one entry per asset on each side.
  - `c`: The score bounds, one entry per asset.
  - `symmetric`: If `true`, the score bounds both sides of the weight.

# Validation

  - `all(c .>= 0)`. A score bound is a magnitude, so a negative score has no bound. A `DomainError` that names the count of the assets is thrown otherwise.
  - `all(wb.lb .<= c)`. A lower bound above the score bound admits no weight. A `DomainError` that names the count of the assets is thrown otherwise.
  - If `symmetric`: `all(wb.ub .>= -c)`. An upper bound below the negative of the score bound admits no weight. A `DomainError` that names the count of the assets is thrown otherwise.

# Returns

  - `wb::WeightBounds`: The joined bounds.

# Examples

```jldoctest
julia> PortfolioOptimisers.score_bounds(WeightBounds([0.0, 0.0], [1.0, 0.2]), [0.5, 0.25], true)
WeightBounds
  lb ┼ Vector{Float64}: [0.0, 0.0]
  ub ┴ Vector{Float64}: [0.5, 0.2]
```

# Related

  - [`ScoreWeightBounds`](@ref)
  - [`score_cap`](@ref)
"""
function score_bounds(wb::WeightBounds, c::VecNum, symmetric::Bool)::WeightBounds
    @argcheck(all(x -> !(x < zero(x)), c),
              DomainError(c,
                          "the score bound is negative at $(count(x -> x < zero(x), c)) assets. A score bound is a magnitude, `|w| <= c`, so the score of every asset must be non-negative. Use a score that is non-negative, or a `val` of zero at the assets with a negative score."))
    @argcheck(all(map((l, x) -> l <= x, wb.lb, c)),
              DomainError((wb.lb, c),
                          "the lower bound of `wb` is above the score bound at $(count(map((l, x) -> !(l <= x), wb.lb, c))) assets, so no weight satisfies both. Lower the static lower bound of these assets, or raise their `val`."))
    @argcheck(!symmetric || all(map((u, x) -> -x <= u, wb.ub, c)),
              DomainError((wb.ub, c),
                          "the upper bound of `wb` is below the negative score bound at $(count(map((u, x) -> !(-x <= u), wb.ub, c))) assets, so no weight satisfies both. Raise the static upper bound of these assets, raise their `val`, or use `symmetric = false`."))
    lb = symmetric ? max.(wb.lb, -c) : wb.lb
    return WeightBounds(; lb = lb, ub = min.(wb.ub, c))
end
"""
    name_keyed(wb::ScoreWeightBounds) -> Bool

Tell whether a [`ScoreWeightBounds`](@ref) resolves a name over `sets`.

A number or a vector of numbers in `val` names no asset, so it resolves without `sets`. A pair, a dictionary, a vector of pairs or a value algorithm in `val` reads the universe of `sets`. The estimator also needs `sets` when its static bounds `wb` do.

# Returns

  - `::Bool`: `true` when `val` is not a number or a vector of numbers, or when `wb.wb` is keyed by name.

# Related

  - [`name_keyed`](@ref)
  - [`ScoreWeightBounds`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
function name_keyed(wb::ScoreWeightBounds)
    return !isa(wb.val, Num_VecNum) || name_keyed(wb.wb)
end
"""
    weight_bounds_constraints(wb::ScoreWeightBounds, sets::Option{<:UniverseSets};
                              rd::Option{<:ReturnsResult} = nothing, N::Integer = 0,
                              strict::Bool = false, datatype::DataType = Float64, kwargs...)

Resolve a [`ScoreWeightBounds`](@ref) on the returns of a fit.

# Algorithm

 1. Score the assets of `rd` with [`asset_scores`](@ref), giving `s`. A zero `N` takes the length of `s`.
 2. Resolve `wb.wb` with `weight_bounds_constraints` on the same arguments, then expand it to `N` entries on each side.
 3. Resolve `wb.val`, giving `v`. With `sets`, [`estimator_to_val`](@ref) resolves it against the universe, and an asset that `wb.val` does not name takes `wb.dval`, or `one(datatype)` when `wb.dval` is `nothing`. With no sets, `wb.val` names no asset, so the number or the vector of numbers is `v`.
 4. Compute the score bound of each asset with [`score_cap`](@ref), giving `c`. An asset whose score is not finite takes `wb.nonfinite`, or `+Inf` when `wb.nonfinite` is `nothing`.
 5. Join the static bounds with `c` through [`score_bounds`](@ref).

# Arguments

  - `wb`: The estimator.
  - `sets`: [`UniverseSets`](@ref) of the assets of the fit, or `nothing` when the estimator is not keyed by name ([`name_keyed`](@ref)).
  - `rd`: The returns of the fit, in the order of the assets of `sets`.
  - `N`: The number of assets. A zero takes the number of assets of `rd`.
  - $(arg_dict[:strict_opt])
  - `datatype`: Number type of the resolved bounds, and of the fill value `one(datatype)`.
  - $(arg_dict[:ignkwargs])

# Validation

  - `rd` is not `nothing`. An [`IsNothingError`](@ref) is thrown otherwise. The scores need the returns of the fit.
  - `sets` is not `nothing` when the estimator is keyed by name ([`name_keyed`](@ref)). An [`IsNothingError`](@ref) is thrown otherwise.
  - `length(s) == N`. A `DimensionMismatch` is thrown otherwise.
  - With no sets, a vector of numbers in `wb.val` holds `N` entries. A `DimensionMismatch` is thrown otherwise.
  - The checks of [`score_bounds`](@ref): no score bound is negative, no static lower bound is above its score bound, and, when `wb.symmetric` is `true`, no static upper bound is below the negative of its score bound. A `DomainError` is thrown otherwise.

# Returns

  - `wb::WeightBounds`: The resolved bounds, one entry per asset on each side, in the asset order of `sets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"volume\",
                                            vals = [1.0 0.5 0.25; 2.0 1.0 0.5; 0.5 0.25 NaN],
                                            alg = ForwardPanelFill())]; amsk = trues(3, 3),
                         emsk = trues(3, 3));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\", \"C\"], X = zeros(3, 3), pnl = pnl);

julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> wb = ScoreWeightBounds(; score = PanelFieldScorer(; fields = \"volume\", min_obs = 3),
                              val = 0.5);

julia> weight_bounds_constraints(wb, sets; rd = rd)
WeightBounds
  lb ┼ Vector{Float64}: [0.0, 0.0, 0.0]
  ub ┴ Vector{Float64}: [0.5, 0.25, 0.0]
```

The third volume of `C` is a fill and not a trade, so `C` holds two volumes, fewer than `min_obs`, and has no score. The default `nonfinite = 0` then admits no position in `C`.

# Related

  - [`ScoreWeightBounds`](@ref)
  - [`AbstractWeightBoundsEstimator`](@ref)
  - [`asset_scores`](@ref)
  - [`estimator_to_val`](@ref)
  - [`score_cap`](@ref)
  - [`score_bounds`](@ref)
"""
function weight_bounds_constraints(wb::ScoreWeightBounds, sets::Option{<:UniverseSets};
                                   rd::Option{<:ReturnsResult} = nothing, N::Integer = 0,
                                   strict::Bool = false, datatype::DataType = Float64,
                                   kwargs...)::WeightBounds
    @argcheck(!isnothing(rd),
              IsNothingError("a `ScoreWeightBounds` scores the assets on the returns of the fit, and this route carries no returns. Pass them as `weight_bounds_constraints(wb, sets; rd = rd)`, or put the estimator in the `wb` field of an optimiser, which passes the returns of each fit."))
    @argcheck(!(isnothing(sets) && name_keyed(wb)),
              IsNothingError("sets cannot be nothing when a `ScoreWeightBounds` is keyed by name: its `val` or its static bounds `wb` name assets, and only a UniverseSets resolves a name"))
    s = asset_scores(wb.score, rd)
    N = ifelse(iszero(N), length(s), N)
    @argcheck(length(s) == N,
              DimensionMismatch("the score holds $(length(s)) assets, and the bounds must cover $N assets"))
    swb = weight_bounds_constraints(wb.wb, sets; rd = rd, N = N, strict = strict,
                                    datatype = datatype)
    swb = weight_bounds_constraints(swb; N = N, datatype = datatype)
    v = if isnothing(sets)
        # The guard above leaves a number or a vector of numbers here.
        @argcheck(isa(wb.val, Number) || length(wb.val) == N,
                  DimensionMismatch("val holds $(length(wb.val)) multiples, and the bounds must cover $N assets"))
        wb.val
    else
        estimator_to_val(wb.val, sets, ifelse(isnothing(wb.dval), one(datatype), wb.dval);
                         datatype = datatype, strict = strict)
    end
    nf = ifelse(isnothing(wb.nonfinite), convert(datatype, Inf), wb.nonfinite)
    return score_bounds(swb, score_cap.(v, s, nf), wb.symmetric)
end

export ScoreWeightBounds
