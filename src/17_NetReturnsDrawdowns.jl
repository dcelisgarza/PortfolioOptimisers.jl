"""
    calc_net_returns(w::VecNum, X::MatNum, args...)
    calc_net_returns(w::VecNum, X::MatNum, fees::Fees)
    calc_net_returns(w::VecVecNum, X::MatNum, args...)

Compute the net portfolio returns. If `fees` is not `nothing`, it deducts the computed fees from the gross returns.

The fee is one scalar and it is charged in **every** period. [`calc_fees`](@ref) contracts the whole weight vector into a single number, and that number is subtracted from every row of ``\\mathbf{X}\\boldsymbol{w}``, so a `T`-row matrix charges it `T` times. On a three-period matrix with ``\\boldsymbol{w} = [0.6,\\, -0.4,\\, 0,\\, 0.25]`` and `l = 0.002`, `s = 0.003`, `fl = 0.01`, `fs = 0.02`, the fee measured `0.0429`, and each of the three net returns sat `0.0429` below its gross value. That is why the [`Fees`](@ref) rates must be stated per period of `X`.

The per asset returns sum to this series. `vec(sum(calc_net_asset_returns(w, X, fees); dims = 2))` reproduces `calc_net_returns(w, X, fees)`, because [`calc_asset_fees`](@ref) splits over the assets what [`calc_fees`](@ref) contracts into a scalar. The two sides add in a different order, so the identity holds to rounding and not to `==`: on the weights above the largest difference measured `6.9e-18`.

**This verb is the plain product, and a non-finite entry poisons its whole observation.** `0 * NaN` is `NaN`, so a zero weight does not save the row: one `NaN` in `X[t, i]` makes `val[t]` non-finite whatever `w[i]` holds. The verb takes no finiteness check, because it is the funnel of the library and a scan here is paid at each of its call sites on every evaluation. A gapped panel is scored through [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref), which reduces the window to the Investable Mask and filters the Held Gaps once with [`filter_held_gaps`](@ref) before it reaches this verb.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{R}(\\mathbf{X},\\, \\boldsymbol{w}) &= \\mathbf{X} \\boldsymbol{w} \\ominus F_{\\text{t}}(\\boldsymbol{w})
\\end{align}
```

Where:

  - ``\\boldsymbol{R}(\\mathbf{X},\\, \\boldsymbol{w})``: `T × 1` vector of portfolio net returns.
  - ``\\mathbf{X}``: `T × N` matrix of asset returns (observations × assets).
  - ``\\boldsymbol{w}``: `N × 1` vector of portfolio weights.
  - ``F_{\\text{t}}(\\boldsymbol{w})``: Total fees computed using [`calc_fees`](@ref).
  - ``\\ominus``: Elementwise (Hadamard) subtraction.

# Algorithm

 1. Contract `X` with `w`, giving `X * w`, the `T × 1` gross portfolio return series.
 2. On the `args...` method, return that series unchanged. The method reads none of its trailing arguments, so a `nothing` `fees` reaches it and charges nothing rather than charging a zero fee.
 3. On the `fees::Fees` method, hand the series to [`charge_fees`](@ref), which subtracts the per period charge from every entry, and the one-off charge on the clock `fees.fa` names.
 4. On the `w::VecVecNum` method, apply steps 1 to 3 to each weight vector `wi` of `w`, and collect one return series per weight vector.

# Arguments

  - `w`: Portfolio weights, or a vector of portfolio weight vectors.
  - `X`: Asset return matrix (observations × assets).
  - `fees`: [`Fees`](@ref) structure.
  - `args...`: Additional arguments (ignored).

# Returns

  - `val::VecNum`: Portfolio net returns, for a `w::VecNum`.
  - `val::Vector{<:VecNum}`: One net return series per weight vector, for a `w::VecVecNum`.

# Examples

```jldoctest
julia> calc_net_returns([0.5, 0.5], [0.01 0.02; 0.03 0.04])
2-element Vector{Float64}:
 0.015
 0.035
```

# Related

  - [`VecNum`](@ref)
  - [`VecVecNum`](@ref)
  - [`MatNum`](@ref)
  - [`calc_net_asset_returns`](@ref): The per asset split of this series. Its rows sum to this one.
  - [`calc_fees`](@ref): Computes the one scalar that step 3 subtracts.
  - [`Fees`](@ref)
"""
function calc_net_returns(w::VecNum, X::MatNum, args...)
    return X * w
end
function calc_net_returns(w::VecNum, X::MatNum, fees::Fees)
    return charge_fees(X * w, w, fees)
end
function calc_net_returns(w::VecVecNum, X::MatNum, args...)
    return [calc_net_returns(wi, X, args...) for wi in w]
end
"""
    charge_fees(r::VecNum, w::VecNum, fees::Option{<:Fees})

Subtract a fee from a portfolio return series, on the clock the fee states.

`l`, `s` and `tn` are rates per period, so they charge on every observation. `fl` and `fs` are currency amounts charged one time for the whole holding period, so the clock decides where in the series they land, and `fees.fa` names that clock. The length of the holding period is the length of the series, which this verb hands to [`calc_fees`](@ref).

# Algorithm

 1. A `nothing` `fees` returns `r` unchanged. It charges no fee rather than a zero fee.
 2. Read the pair `(amortised, one_time)` of [`calc_fees`](@ref), over the length of `r`. This site knows the series it charges, so it hands that length in.
 3. Subtract `amortised` from every observation.
 4. Subtract `one_time` from the first observation alone. Under an [`AmortisedFees`](@ref) it is zero, because step 2 spread that whole cost into `amortised`, and `iszero` gates the pass away.

Both clocks charge the same total when the horizon is the length of `r`. They give a different drawdown, because the first charges the whole cost on one observation and the second charges a fraction of it on each.

# Arguments

  - `r`: Gross portfolio return series.
  - `w`: Portfolio weights.
  - `fees`: [`Fees`](@ref) structure, or `nothing`.

# Returns

  - `val::VecNum`: The net return series.

# Related

  - [`Fees`](@ref)
  - [`AmortisedFees`](@ref)
  - [`VecNum`](@ref)
  - [`calc_net_returns`](@ref)
  - [`calc_periodic_fees`](@ref)
  - [`calc_one_off_fees`](@ref)
  - [`charge_asset_fees`](@ref)
"""
function charge_fees(r::VecNum, ::VecNum, ::Nothing)
    return r
end
function charge_fees(r::VecNum, w::VecNum, fees::Fees)
    amortised, one_time = calc_fees(w, length(r), fees)
    val = r .- amortised
    if !charges_nothing(one_time) && !iszero(one_time) && !isempty(val)
        val[1] -= one_time
    end
    return val
end
"""
    charges_nothing(v)

Report whether a fee term charges nothing at all.

The verb that prices an unset liquidation carrier returns an **empty** vector rather than a vector of zeros, because it holds no length to build one from: the axis is the complement of the Investable Mask, and a [`Fees`](@ref) does not carry the mask. A `nothing` reaches the same sites from the caller that has already taken a step. Both mean one thing, so the sites that ask reach one predicate rather than repeating the pair.

# Arguments

  - `v`: A fee term, a vector or `nothing`.

# Returns

  - `val::Bool`: `true` when the term charges nothing.

# Related

  - [`charge_asset_fees`](@ref)
  - [`charge_fee_axis!`](@ref)
  - [`AmortisedFees`](@ref)
"""
function charges_nothing(v)
    return isnothing(v) || isempty(v)
end
"""
    charge_asset_fees(R::MatNum, w::VecNum, fees::Option{<:Fees}, imsk::Option{<:BitVector} = nothing)

Subtract a per asset fee from a per asset return matrix, on the clock the fee states and in the columns it was priced on.

The per asset twin of [`charge_fees`](@ref). Its row sums are the series that verb returns, up to the order of summation.

A reduced [`Fees`](@ref) lives on **two** axes: the five per asset fields were sliced to the Investable Mask at the door, and the two liquidation carriers to its complement. `R` and `w` span the caller's own universe, so `imsk` is what reunites them, and each axis is charged in the columns it owns through [`charge_fee_axis!`](@ref). A liquidated asset earns no return, so its column of `R` is zero and holds its charge alone: the charge is neither smeared over the assets that stayed nor carried in a matrix of its own.

# Algorithm

 1. A `nothing` `fees` returns `R` unchanged. It charges no fee rather than a zero fee.
 2. Read the pair of pairs `((am_i, am_l), (ot_i, ot_l))` of [`calc_asset_fees`](@ref), over the row count of `R`. The weights it is handed are the investable ones, because that is the axis the five per asset fields were sliced to; the carriers read `w` for its element type alone.
 3. On a `nothing` `imsk`, charge the investable axis over the whole matrix, and refuse a `fees` that carries a liquidation: no mask says where the exits are, so the charge has nowhere to land, and dropping it would understate the return.
 4. On a `BitVector` `imsk`, charge the investable axis in the `imsk` columns and the liquidation axis in the complement's, each by the same two steps: the per period vector on every observation, and the one-off vector on the first alone. Under an [`AmortisedFees`](@ref) the one-off vector is zero, because step 2 spread that cost into the per period one.

# Arguments

  - `R`: Gross per asset return matrix (observations × assets), on the caller's universe.
  - `w`: Portfolio weights, on the same universe as `R`.
  - `fees`: [`Fees`](@ref) structure, or `nothing`.
  - `imsk`: The Investable Mask, or `nothing` when no asset left.

# Validation

  - `imsk` spans the columns of `R`, else a `DimensionMismatch` naming both widths.
  - Each charge spans the axis it is charged on, through [`assert_fee_axis_width`](@ref).
  - A `nothing` `imsk` meets no liquidation carrier, else an `ArgumentError` naming the mask.

# Returns

  - `val::MatNum`: The net per asset return matrix, on the caller's universe.

# Related

  - [`Fees`](@ref)
  - [`AmortisedFees`](@ref)
  - [`MatNum`](@ref)
  - [`calc_net_asset_returns`](@ref)
  - [`charge_fee_axis!`](@ref)
  - [`assert_fee_axis_width`](@ref)
  - [`calc_asset_periodic_fees`](@ref)
  - [`calc_asset_one_off_fees`](@ref)
  - [`charge_fees`](@ref)
"""
function charge_asset_fees(R::MatNum, ::VecNum, ::Nothing, ::Option{<:BitVector} = nothing)
    return R
end
function charge_asset_fees(R::MatNum, w::VecNum, fees::Fees, ::Nothing = nothing)
    (am_i, am_l), (ot_i, ot_l) = calc_asset_fees(w, size(R, 1), fees)
    # An unset term is `nothing` under an amortising clock and empty otherwise, and both mean
    # the same thing here: no charge was priced on the complement, so none has to land there.
    @argcheck(charges_nothing(am_l) && charges_nothing(ot_l),
              ArgumentError("`fees` carries a forced liquidation, and no Investable Mask says which columns of the $(size(R, 2))-asset matrix the exits are; the charge has nowhere to land.\nPass the mask as the fourth argument, `calc_net_asset_returns(w, X, fees, imsk)`, or clear `fees.lq` and `fees.flq`."))
    return charge_fee_axis!(R .- transpose(am_i), nothing, ot_i)
end
function charge_asset_fees(R::MatNum, w::VecNum, fees::Fees, imsk::BitVector)
    @argcheck(size(R, 2) == length(imsk),
              DimensionMismatch("the investable mask spans $(length(imsk)) assets, but the return matrix holds $(size(R, 2)) columns; the mask and the matrix must state the same universe"))
    # The five per asset fields were sliced to the mask at the door, so the weights they are
    # priced against are the investable ones. The two carriers read `w` for its element type
    # alone and hold their own previous weights, so this slice reaches them harmlessly.
    (am_i, am_l), (ot_i, ot_l) = calc_asset_fees(view(w, imsk), size(R, 1), fees)
    assert_fee_axis_width(am_i, ot_i, count(imsk), "investable")
    lmsk = .!imsk
    assert_fee_axis_width(am_l, ot_l, count(lmsk), "liquidation")
    val = copy(R)
    # Each axis is charged in its own columns, by the **same two steps**: the per period
    # charge on every observation, and the one-off charge at the observation the clock names.
    # The investable fields were sliced to `imsk` at the door and the two carriers to its
    # complement, so the mask is what puts each charge back where it was priced.
    charge_fee_axis!(view(val, :, imsk), am_i, ot_i)
    # A liquidated asset earns no return, so its column is zero and holds the charge alone.
    # Its one-off amount lands at the index `fl` and `fs` land on, which is what puts both
    # axes on one clock and lets the row sums reproduce the series under every `fees.fa`.
    charge_fee_axis!(view(val, :, lmsk), am_l, ot_l)
    return val
end
"""
    assert_fee_axis_width(am, ot, n::Integer, axis::String)

Refuse a per asset charge whose width is not the width of the axis it is charged on.

[`charge_asset_fees`](@ref) splits a matrix by the Investable Mask and charges each half with the vector priced for it. A vector that does not span its half would broadcast into the wrong columns, or raise a `DimensionMismatch` naming neither the mask nor the fee, so both vectors of both axes are measured here first. An empty vector is the answer when a term was never set, so it passes: [`charge_fee_axis!`](@ref) skips the step it belongs to.

# Arguments

  - `am`: Per period charge of the axis, or `nothing`.
  - `ot`: One-off charge of the axis.
  - `n`: Number of columns the axis owns.
  - `axis`: Name of the axis, for the message.

# Validation

  - Each of `am` and `ot` is empty or spans `n`, else a `DimensionMismatch` naming both widths.

# Related

  - [`charge_asset_fees`](@ref)
  - [`charge_fee_axis!`](@ref)
"""
function assert_fee_axis_width(am, ot, n::Integer, axis::String)
    for (name, v) in (("per period", am), ("one-off", ot))
        @argcheck(charges_nothing(v) || length(v) == n,
                  DimensionMismatch("the $(name) charge of the $(axis) axis spans $(length(v)) assets, but that axis owns $(n) columns of the return matrix; the fee and the mask must come from the same reduction"))
    end
    return nothing
end
"""
    charge_fee_axis!(A::AbstractMatrix, am, ot)

Charge one axis of a per asset return matrix on the two clocks a fee states, in place.

The step [`charge_asset_fees`](@ref) takes once per axis: the per period vector `am` is subtracted from every row, and the one-off vector `ot` from the first row alone. `A` is a view of the columns the axis owns, so the verb never learns which axis it is charging, and the investable fields and the two liquidation carriers take the same two steps.

The one-off charge is one vector written into one row, never an array the size of the matrix: a charge made once is stored once, and under an [`AmortisedFees`](@ref) it is zero and no row is written at all.

**Each step is skipped by its own vector, not by the other's.** An empty vector is what an unset term returns, and the two terms of an axis are set independently: a `Fees` may carry `flq` and no `lq`, so a one-off charge is owed on an axis whose per period charge is empty. A step gated on the wrong vector would drop that charge in silence. A `nothing` `am` skips the per period step alone, for the caller that has already taken it.

# Arguments

  - `A`: The columns of the matrix this axis owns, charged in place.
  - `am`: Per period charge of the axis, one entry per column, or `nothing`.
  - `ot`: One-off charge of the axis, one entry per column. Zero under an [`AmortisedFees`](@ref), which spread it into `am`.

# Returns

  - `A::AbstractMatrix`: The same matrix, charged.

# Related

  - [`charge_asset_fees`](@ref)
  - [`charge_fees`](@ref)
  - [`calc_asset_fees`](@ref)
  - [`AmortisedFees`](@ref)
"""
function charge_fee_axis!(A::AbstractMatrix, am, ot)
    if !charges_nothing(am)
        A .-= transpose(am)
    end
    if !charges_nothing(ot) && !iszero(ot) && size(A, 1) > 0
        view(A, 1, :) .-= ot
    end
    return A
end
"""
    investable_reduction(X::MatNum, w, fees::Option{<:Fees}, strict::Bool)
    investable_reduction(rd::AbstractReturnsResult, w, fees::Option{<:Fees}, strict::Bool)
    investable_reduction(pr::AbstractPriorResult, w, fees::Option{<:Fees}, strict::Bool)
    investable_reduction(imsk::Nothing, pr::AbstractPriorResult, w, fees::Option{<:Fees}, strict::Bool)
    investable_reduction(imsk::BitVector, pr::AbstractPriorResult, w, fees::Option{<:Fees}, strict::Bool)

Reduce a prior result, the weights scored against it and the fees charged on them to the Investable Mask.

This is ADR 0115's rule at the **value-level** door. An optimiser reduces once at its entry, so no optimiser meets a gap. A caller who scores a weight vector against a prior result by hand does meet one: a non-investable asset carries `NaN` in `mu`, on the diagonal of `sigma` and down its column of `pr.X`, so `dot(w, pr.sigma, w)` and `pr.X * w` are `NaN` at **any** weight, the optimiser's own zero included. [`expected_risk`](@ref), [`expected_return`](@ref), [`risk_contribution`](@ref) and [`factor_risk_contribution`](@ref) reduce here instead, and a per-asset answer expands back through [`expand_investable_weights`](@ref).

Every block of the prior is reduced together by the [`port_opt_view`](@ref) method the prior's owner already writes, so a new block cannot be forgotten, and the reduced `pr.X` carries no dead column. The fees travel with the weights, because a [`Fees`](@ref) whose rates are one number per asset is indexed by the same axis and would otherwise meet a shorter weight vector.

A held non-investable asset is a holding the prior cannot value. It takes the library's strictness policy through [`strict_diagnostic`](@ref): a warning names the assets and their weights are dropped, or an `ArgumentError` names them under `strict`.

A bare returns matrix and a [`ReturnsResult`](@ref) carry no moments, so no mask exists to derive and they pass through. That is what lets one door state the reduction once and dispatch decide whether it happens.

# Algorithm

 1. Return `nothing` and the three arguments unchanged when the carrier is a matrix or a returns result.
 2. Derive the Investable Mask once with [`investable_mask`](@ref).
 3. Return `nothing` and the three arguments unchanged when every asset is investable.
 4. Otherwise report the held non-investable assets through [`strict_diagnostic`](@ref).
 5. Return the mask, a [`port_opt_view`](@ref) of the prior and of the fees at `findall(imsk)`, and the view of the weights at the mask.

# Arguments

  - `pr`: Prior result, [`ReturnsResult`](@ref), or asset returns matrix.
  - `w`: Portfolio weights, a population of them, or a weight path (observations × assets).
  - `fees`: [`Fees`](@ref) the figure is charged against, or `nothing`.
  - `strict`: Whether a held non-investable asset raises rather than warns.

# Validation

  - Every asset the weights hold is investable, else a warning naming the assets is emitted, or an `ArgumentError` naming them is raised under `strict`.

# Returns

  - `(imsk, pr, w, fees)`: The Investable Mask and the three reduced to it, or `nothing` and the three unchanged.

# Related

  - [`investable_mask`](@ref)
  - [`held_non_investable`](@ref)
  - [`investable_weights_view`](@ref)
  - [`expand_investable_weights`](@ref)
  - [`port_opt_view`](@ref)
  - [`strict_diagnostic`](@ref)
"""
function investable_reduction(X::MatNum, w::Union{<:VecNum, <:VecVecNum, <:MatNum},
                              fees::Option{<:Fees}, ::Bool)
    return nothing, X, w, fees
end
function investable_reduction(rd::AbstractReturnsResult,
                              w::Union{<:VecNum, <:VecVecNum, <:MatNum},
                              fees::Option{<:Fees}, ::Bool)
    return nothing, rd, w, fees
end
function investable_reduction(pr::AbstractPriorResult,
                              w::Union{<:VecNum, <:VecVecNum, <:MatNum},
                              fees::Option{<:Fees}, strict::Bool)
    return investable_reduction(investable_mask(pr), pr, w, fees, strict)
end
function investable_reduction(::Nothing, pr::AbstractPriorResult,
                              w::Union{<:VecNum, <:VecVecNum, <:MatNum},
                              fees::Option{<:Fees}, ::Bool)
    return nothing, pr, w, fees
end
function investable_reduction(imsk::BitVector, pr::AbstractPriorResult,
                              w::Union{<:VecNum, <:VecVecNum, <:MatNum},
                              fees::Option{<:Fees}, strict::Bool)
    held = held_non_investable(imsk, w)
    if !isempty(held)
        strict_diagnostic("a value-level verb cannot score a holding the prior could not estimate. Assets $(held) are not investable, and the weights hold them. Their weights are dropped, so the figure describes the portfolio without them. Pass `strict = true` to refuse instead, reduce the weights to the investable universe, or refit the prior over a history that covers these assets.",
                          strict)
    end
    idx = findall(imsk)
    # The fee is viewed at `pr.X`, the prior's **unreduced** returns matrix, because its two
    # liquidation carriers live on the complement of the mask and the view derives that
    # complement from the full width. Every other argument here takes the index alone.
    return imsk, port_opt_view(pr, idx), investable_weights_view(imsk, w),
           port_opt_view(fees, idx, pr.X)
end
"""
    investable_returns_view(imsk::Nothing, rd::AbstractReturnsResult)
    investable_returns_view(imsk::BitVector, rd::AbstractReturnsResult)

Take the view of a returns result at the Investable Mask.

The pairing of [`investable_reduction`](@ref) with the returns data a value-level verb takes beside the prior, so a factor regression that fits its own loadings fits them over the live assets alone.

# Arguments

  - `imsk`: The Investable Mask, or `nothing`.
  - $(arg_dict[:rd])

# Returns

  - `rd::AbstractReturnsResult`: The returns result reduced to the investable assets, or unchanged.

# Related

  - [`investable_reduction`](@ref)
  - [`port_opt_view`](@ref)
"""
function investable_returns_view(::Nothing, rd::AbstractReturnsResult)
    return rd
end
function investable_returns_view(imsk::BitVector, rd::AbstractReturnsResult)
    return port_opt_view(rd, findall(imsk))
end
"""
    held_gap_pairs(w::VecNum, X::MatNum)
    held_gap_pairs(w::VecVecNum, X::MatNum)
    held_gap_pairs(W::MatNum, X::MatNum)

Find the Held Gaps of a window.

A **Held Gap** is an `(observation, asset)` pair at which the portfolio's weight is non-zero and the asset's return is missing. It arises where the universe changes after the fit: an asset that was investable on the training window and delists inside the test window. A pair with a **zero** weight contributes nothing whatever it holds, and is never a Held Gap.

The weight argument's type is the picker, as it is for [`calc_net_returns`](@ref). A [`VecNum`](@ref) is one target weight vector, and it weighs every observation, so a whole column is held or none of it is. A [`MatNum`](@ref) is a weight path, and each pair is judged by the weight held through its own observation. A [`VecVecNum`](@ref) is a population, and a column is held when any member holds it.

# Arguments

  - `w`: Portfolio weights, a population of them, or a weight path (observations × assets).
  - `X`: Asset returns, `observations × assets`.

# Returns

  - `held::Vector{Tuple{Int, Int}}`: The `(observation, asset)` pairs, empty when there are none.

# Related

  - [`filter_held_gaps`](@ref)
  - [`calc_net_returns`](@ref)
  - [`attribution_net_returns`](@ref)
"""
function held_gap_pairs(w::VecNum, X::MatNum)
    return [(t, i) for i in axes(X, 2) if !iszero(w[i])
            for t in axes(X, 1) if !isfinite(X[t, i])]
end
function held_gap_pairs(w::VecVecNum, X::MatNum)
    return [(t, i) for i in axes(X, 2) if any(wi -> !iszero(wi[i]), w)
            for t in axes(X, 1) if !isfinite(X[t, i])]
end
function held_gap_pairs(W::MatNum, X::MatNum)
    return [(t, i) for i in axes(X, 2)
            for t in axes(X, 1) if !iszero(W[t, i]) && !isfinite(X[t, i])]
end
"""
    filter_held_gaps(w::Union{<:VecNum, <:VecVecNum, <:MatNum}, X::MatNum, strict::Bool;
                     nx::Option{<:VecStr} = nothing)

Return the window with every non-finite entry replaced by zero, and name the Held Gaps.

A point-in-time panel carries a `NaN` at every `(observation, asset)` pair where the asset is inactive: before it lists, and after it delists. `0 * NaN` is `NaN`, so a zero weight does not save the product `X * w`, and one delisting poisons every observation after it. This is the one place a fold cleans the window it is about to score, and it runs **once**, over the investable columns alone, before the series is formed and before a Weight Drift compounds.

A **Held Gap** takes the library's strictness policy through [`strict_diagnostic`](@ref): a warning names the pairs and the pair contributes zero, or an `ArgumentError` names them under `strict`. A zero weight at a gap is silent. Nothing is renormalised, so the missing weight sits in cash on that observation, which is the one reading that invents no trade the weights never stated.

The funnel [`calc_net_returns`](@ref) stays the plain product. A scan there is paid at each of its call sites on every evaluation; a scan here is paid once, on a window the fold already multiplies once.

# Algorithm

 1. Return `X` itself when every entry of `X` is finite.
 2. Otherwise find the Held Gaps with [`held_gap_pairs`](@ref) and report them through [`strict_diagnostic`](@ref), naming the assets by `nx` when it is given.
 3. Return a copy of `X` with every non-finite entry replaced by zero.

A fold hands this verb the window **viewed at the Investable Mask**, so a column index of `X` is a reduced one and names nothing to the caller. The fold therefore passes the view's `nx`, and the message names the asset rather than a position the caller cannot look up.

# Arguments

  - `w`: Portfolio weights, a population of them, or a weight path (observations × assets).
  - `X`: Asset returns, `observations × assets`.
  - `strict`: Whether a non-finite return at a held pair raises rather than warns.
  - `nx`: Optional asset names of the columns of `X`. When given, the message names the held assets by them; otherwise it reports their column indices.

# Validation

  - Every held pair of `X` is finite, else a warning naming the pairs is emitted, or an `ArgumentError` naming them is raised under `strict`.

# Returns

  - `X::MatNum`: The window with no non-finite entry.

# Related

  - [`held_gap_pairs`](@ref)
  - [`calc_net_returns`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
"""
function filter_held_gaps(w::Union{<:VecNum, <:VecVecNum, <:MatNum}, X::MatNum,
                          strict::Bool; nx::Option{<:VecStr} = nothing)
    if all(isfinite, X)
        return X
    end
    held = held_gap_pairs(w, X)
    if !isempty(held)
        strict_diagnostic(held_gap_msg(held, nx), strict)
    end
    return map(x -> isfinite(x) ? x : zero(x), X)
end
"""
    held_gap_msg(held::AbstractVector{<:Tuple{Integer, Integer}}, nx::Option{<:VecStr} = nothing)

Write the message a Held Gap raises.

The message names the assets, the count of pairs and the first observation, so a caller can find the delisting that made them, and it states the consequence: the pair contributes zero and the missing weight sits in cash. The assets are named by `nx` when it is given, because the window a fold filters is viewed at the Investable Mask and a column index of it is a reduced one; with no names the column indices are reported.

# Arguments

  - `held`: The `(observation, asset)` pairs [`held_gap_pairs`](@ref) found.
  - `nx`: Optional asset names of the window's columns.

# Returns

  - `msg::String`: The message.

# Related

  - [`filter_held_gaps`](@ref)
  - [`held_gap_pairs`](@ref)
  - [`strict_diagnostic`](@ref)
"""
function held_gap_msg(held::AbstractVector{<:Tuple{Integer, Integer}},
                      nx::Option{<:VecStr} = nothing)
    cols = unique(last.(held))
    assets = isnothing(nx) ? cols : nx[cols]
    return "a portfolio cannot earn a return an asset did not have. Assets $(assets) carry a non-finite return at $(length(held)) held (observation, asset) pair(s), the first at observation $(first(held)[1]). Those pairs contribute zero, so the weight of a missing asset sits in cash on that observation and the series understates the portfolio by whatever it would have earned. Pass `strict = true` to refuse instead, zero the weights over the observations the asset is inactive, or pass a weight history."
end
"""
    calc_net_returns(w::MatNum, X::MatNum, args...)

Compute the net portfolio returns of a weight path.

`w` is a `T × N` weight path: row `t` holds the weights the portfolio carried **through** observation `t`. [`weight_path`](@ref) is the verb that makes one, under a Weight Drift or at constant weights, so this method reads a drifted window the way the [`VecNum`](@ref) methods above read a constant one.

**The weight argument's type is the picker.** A [`VecNum`](@ref) is one target weight vector, and it weighs every observation. A [`MatNum`](@ref) is a path, and it weighs each observation by the row that belongs to it. A scorer that forwards its weights to this verb therefore reads a path with no change of its own: [`performance_summary`](@ref) is bound [`ArrNum`](@ref) and admits both today.

This method states no arithmetic of its own. It sums the per asset split of the same path along the asset axis, so the two cannot drift apart, and the fee it charges is the one [`calc_net_asset_returns`](@ref) charges. At constant weights every row of the path is the same vector, so the answer is the one the [`VecNum`](@ref) methods give. The two sides add in a different order, so that identity holds to rounding and not to `==`.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{R}(\\mathbf{X},\\, \\mathbf{U}) &= \\left(\\mathbf{X} \\odot \\mathbf{U} \\ominus \\boldsymbol{F}_{\\text{t}}(\\mathbf{U}_{1,\\cdot})^{\\intercal}\\right) \\boldsymbol{1}
\\end{align}
```

Where:

  - ``\\boldsymbol{R}(\\mathbf{X},\\, \\mathbf{U})``: `T × 1` vector of portfolio net returns.
  - ``\\mathbf{X}``: `T × N` matrix of asset returns (observations × assets).
  - ``\\mathbf{U}``: `T × N` weight path, whose row ``t`` holds the weights carried through observation ``t``.
  - ``\\mathbf{U}_{1,\\cdot}``: First row of the path, which is the target weights.
  - ``\\boldsymbol{F}_{\\text{t}}(\\boldsymbol{w})``: `N × 1` per asset vector of total portfolio fees computed using [`calc_fees`](@ref).
  - ``\\boldsymbol{1}``: `N × 1` vector of ones, which sums along the asset axis.
  - ``\\odot``: Elementwise (Hadamard) multiplication.
  - ``\\ominus``: Elementwise (Hadamard) subtraction.

# Algorithm

 1. Split the path over the assets with [`calc_net_asset_returns`](@ref), which scales each observation by the weights held through it and charges the fee from the path's first row.
 2. Sum that `T × N` matrix along its asset axis, giving the `T × 1` portfolio net return series.

# Validation

  - `size(w) == size(X)`, else the broadcast inside [`calc_net_asset_returns`](@ref) raises a `DimensionMismatch`. A fold's path has the size of the fold's asset returns by construction, and [`assert_held_weights_shape`](@ref) checks a stored one.

# Arguments

  - `w`: Weight path (observations × assets).
  - `X`: Asset return matrix (observations × assets).
  - `fees`: [`Fees`](@ref) structure.
  - `args...`: Additional arguments (ignored).

# Returns

  - `val::VecNum`: Portfolio net returns.

# Examples

```jldoctest
julia> calc_net_returns([0.5 0.5; 0.6 0.4], [0.01 0.02; 0.03 0.04])
2-element Vector{Float64}:
 0.015
 0.034
```

# Related

  - [`MatNum`](@ref)
  - [`ArrNum`](@ref)
  - [`weight_path`](@ref): Makes the path this method reads.
  - [`SelfFinancingDrift`](@ref)
  - [`calc_net_asset_returns`](@ref): The per asset split this method sums.
  - [`performance_summary`](@ref): Scores the series this method forms.
  - [`expected_risk`](@ref): Scores a path through this method on a [`NetReturnsInput`](@ref) measure.
  - [`Fees`](@ref)
"""
function calc_net_returns(w::MatNum, X::MatNum, args...)
    return vec(sum(calc_net_asset_returns(w, X, args...); dims = 2))
end
"""
    calc_net_asset_returns(w::VecNum, X::MatNum, args...)
    calc_net_asset_returns(w::VecNum, X::MatNum, fees::Fees, imsk::Option{<:BitVector} = nothing)

Compute the per asset net portfolio returns. If `fees` is not `nothing`, it deducts the computed fees from the gross returns.

The rows sum to the portfolio series. `vec(sum(calc_net_asset_returns(w, X, fees); dims = 2))` reproduces [`calc_net_returns(w, X, fees)`](@ref), because [`calc_asset_fees`](@ref) splits over the assets what [`calc_fees`](@ref) contracts into a scalar. The two sides add in a different order, so the identity holds to rounding and not to `==`: on ``\\boldsymbol{w} = [0.6,\\, -0.4,\\, 0,\\, 0.25]`` with all four rate fields set, the largest difference measured `6.9e-18`.

Each per asset fee is charged in **every** period, as it is for [`calc_net_returns`](@ref). The `N × 1` fee vector is subtracted from every row of ``\\mathbf{X} \\odot \\boldsymbol{w}^{\\intercal}``, so a `T`-row matrix charges it `T` times.

These are the constant-weight methods: the one vector `w` weighs every observation. The `w::MatNum` methods below read a weight path instead, one row of weights per observation, which is what a fold scored under a Weight Drift held.

**This verb is the plain product, and a non-finite entry poisons its own cell.** `0 * NaN` is `NaN`, so a zero weight does not save the entry, and the row it sits in no longer sums to a finite number. The verb takes no finiteness check, for the reason [`calc_net_returns`](@ref) takes none. A gapped panel is scored through [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref), which filters the Held Gaps once with [`filter_held_gaps`](@ref) before it reaches this verb.

# Mathematical definition

```math
\\begin{align}
\\mathbf{R}(\\mathbf{X},\\, \\boldsymbol{w}) &= \\mathbf{X} \\odot \\boldsymbol{w}^{\\intercal} \\ominus \\boldsymbol{F}_{\\text{t}}(\\boldsymbol{w})^{\\intercal}
\\end{align}
```

Where:

  - ``\\mathbf{R}(\\mathbf{X},\\, \\boldsymbol{w})``: `T × N` matrix of per asset portfolio net returns.
  - ``\\mathbf{X}``: `T × N` matrix of asset returns (observations × assets).
  - ``\\boldsymbol{w}``: `N × 1` vector of portfolio weights.
  - ``\\boldsymbol{F}_{\\text{t}}(\\boldsymbol{w})``: `N × 1` per asset vector of total portfolio fees computed using [`calc_fees`](@ref).
  - ``\\odot``: Elementwise (Hadamard) multiplication.
  - ``\\ominus``: Elementwise (Hadamard) subtraction.

# Algorithm

 1. Scale each column of `X` by its weight, giving `X ⊙ transpose(w)`, the `T × N` matrix of gross per asset contributions.
 2. On the `args...` method, return that matrix unchanged. The method reads none of its trailing arguments, so a `nothing` `fees` reaches it and charges nothing rather than charging a zero fee.
 3. On the `fees::Fees` method, hand the matrix to [`charge_asset_fees`](@ref), which subtracts the per asset per period charge from every row, and the one-off charge on the clock `fees.fa` names. `imsk` says which columns the five per asset fields were priced on and which the two liquidation carriers were, so each charge lands in the columns it was priced on; a `nothing` `imsk` charges the investable axis alone and refuses a fee that carries a liquidation.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset return matrix (observations × assets).
  - `fees`: [`Fees`](@ref) structure.
  - `imsk`: The Investable Mask, or `nothing` when no asset left.
  - `args...`: Additional arguments (ignored).

# Returns

  - `ret::MatNum`: Per asset portfolio net returns, on the caller's universe.

# Examples

```jldoctest
julia> calc_net_asset_returns([0.5, 0.5], [0.01 0.02; 0.03 0.04])
2×2 Matrix{Float64}:
 0.005  0.01
 0.015  0.02
```

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
  - [`calc_net_returns`](@ref): The portfolio series this matrix sums to along `dims = 2`.
  - [`calc_asset_fees`](@ref): Computes the per asset vector that step 3 subtracts.
  - [`calc_fees`](@ref)
  - [`Fees`](@ref)
"""
function calc_net_asset_returns(w::VecNum, X::MatNum, args...)
    return X ⊙ transpose(w)
end
function calc_net_asset_returns(w::VecNum, X::MatNum, fees::Fees,
                                imsk::Option{<:BitVector} = nothing)
    return charge_asset_fees(X ⊙ transpose(w), w, fees, imsk)
end
"""
    calc_net_asset_returns(w::MatNum, X::MatNum, args...)
    calc_net_asset_returns(w::MatNum, X::MatNum, fees::Fees, imsk::Option{<:BitVector} = nothing)

Compute the per asset net portfolio returns of a weight path.

`w` is a `T × N` weight path: row `t` holds the weights the portfolio carried **through** observation `t`. [`weight_path`](@ref) is the verb that makes one, under a Weight Drift or at constant weights, so this method reads a drifted window the way the [`VecNum`](@ref) methods above read a constant one.

The rows sum to the portfolio series of that same path. The wealth ratio of an observation is the weights held through it contracted with that observation's asset returns, so `vec(sum(calc_net_asset_returns(U, X, fees); dims = 2))` reproduces the series [`calc_net_returns`](@ref) forms from the same fold under the same drift. The two sides add in a different order, so the identity holds to rounding and not to `==`.

The fee is charged from the **first** row of the path. That row is the target weights, because nothing has drifted when the window opens, and it is the vector the portfolio series charges its own fee from. The same `N × 1` vector is therefore subtracted from every row here as there.

# Mathematical definition

```math
\\begin{align}
\\mathbf{R}(\\mathbf{X},\\, \\mathbf{U}) &= \\mathbf{X} \\odot \\mathbf{U} \\ominus \\boldsymbol{F}_{\\text{t}}(\\mathbf{U}_{1,\\cdot})^{\\intercal}
\\end{align}
```

Where:

  - ``\\mathbf{R}(\\mathbf{X},\\, \\mathbf{U})``: `T × N` matrix of per asset portfolio net returns.
  - ``\\mathbf{X}``: `T × N` matrix of asset returns (observations × assets).
  - ``\\mathbf{U}``: `T × N` weight path, whose row ``t`` holds the weights carried through observation ``t``.
  - ``\\mathbf{U}_{1,\\cdot}``: First row of the path, which is the target weights.
  - ``\\boldsymbol{F}_{\\text{t}}(\\boldsymbol{w})``: `N × 1` per asset vector of total portfolio fees computed using [`calc_fees`](@ref).
  - ``\\odot``: Elementwise (Hadamard) multiplication.
  - ``\\ominus``: Elementwise (Hadamard) subtraction.

# Algorithm

 1. Scale each observation of `X` by the weights held through it, giving `X ⊙ w`, the `T × N` matrix of gross per asset contributions.
 2. On the `args...` method, return that matrix unchanged. The method reads none of its trailing arguments, so a `nothing` `fees` reaches it and charges nothing rather than charging a zero fee.
 3. On the `fees::Fees` method, hand the matrix and the path's first row to [`charge_asset_fees`](@ref), which subtracts the per asset per period charge from every row, and the one-off charge on the clock `fees.fa` names. `imsk` says which columns each axis of the fee was priced on, so each charge lands in its own; a `nothing` `imsk` charges the investable axis alone and refuses a fee that carries a liquidation.

# Validation

  - `size(w) == size(X)`, else the broadcast raises a `DimensionMismatch`. A fold's path has the size of the fold's asset returns by construction, and [`assert_held_weights_shape`](@ref) checks a stored one.

# Arguments

  - `w`: Weight path (observations × assets).
  - `X`: Asset return matrix (observations × assets).
  - `fees`: [`Fees`](@ref) structure.
  - `imsk`: The Investable Mask, or `nothing` when no asset left.
  - `args...`: Additional arguments (ignored).

# Returns

  - `ret::MatNum`: Per asset portfolio net returns, on the caller's universe.

# Examples

```jldoctest
julia> calc_net_asset_returns([0.5 0.5; 0.6 0.4], [0.01 0.02; 0.03 0.04])
2×2 Matrix{Float64}:
 0.005  0.01
 0.018  0.016
```

# Related

  - [`MatNum`](@ref)
  - [`weight_path`](@ref): Makes the path this method reads.
  - [`SelfFinancingDrift`](@ref)
  - [`calc_net_returns`](@ref): The portfolio series this matrix sums to along `dims = 2`.
  - [`calc_asset_fees`](@ref): Computes the per asset vector that step 3 subtracts.
  - [`Fees`](@ref)
"""
function calc_net_asset_returns(w::MatNum, X::MatNum, args...)
    return X ⊙ w
end
function calc_net_asset_returns(w::MatNum, X::MatNum, fees::Fees,
                                imsk::Option{<:BitVector} = nothing)
    return charge_asset_fees(X ⊙ w, view(w, 1, :), fees, imsk)
end
"""
    calc_turnover(w::AbstractMatrix{<:Number})
    calc_turnover(w::AbstractVector{<:AbstractVector{<:Number}})

Compute the one-way turnover of a weight path.

The turnover of an observation is the sum of the absolute weight changes the rebalance into it costs. It is the value-level reading of the quantity [`Turnover`](@ref) bounds and [`set_turnover_fees!`](@ref) charges, and it is what [`plot_turnover`](@ref) draws.

The first observation has no predecessor, so its turnover is `NaN` rather than zero: a path that opens a book from cash trades its whole gross, and a path that inherits one trades nothing, and the series cannot tell the two apart. The answer therefore has one entry per observation of `w`, and the caller drops or keeps the leading gap.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TO}_{t} &= \\sum_{i=1}^{N} \\left\\lvert w_{t,i} - w_{t-1,i} \\right\\rvert\\,, \\qquad t = 2,\\, \\dots,\\, T\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TO}_{t}``: Turnover of observation ``t``.
  - ``w_{t,i}``: Weight of asset ``i`` at observation ``t``.
  - $(math_dict[:T])
  - $(math_dict[:N])

# Algorithm

 1. Allocate `tn`, one entry per observation of `w`, and write `NaN` into every entry.
 2. For each observation `t` after the first, sum the absolute differences of the two consecutive weight vectors, giving `tn[t]`.

# Arguments

  - `w`: Weight path. A matrix carries one observation per row, and a vector of vectors carries one observation per element.

# Returns

  - `tn::VecNum`: Turnover, one entry per observation of `w`, `NaN` at the first.

# Examples

```jldoctest
julia> calc_turnover([0.5 0.5; 0.25 0.75])
2-element Vector{Float64}:
 NaN
   0.5
```

# Related

  - [`Turnover`](@ref)
  - [`plot_turnover`](@ref)
  - [`forecast_portfolio`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_turnover(w::AbstractMatrix{<:Number})
    Tf = real(eltype(w))
    T = size(w, 1)
    tn = fill(Tf(NaN), T)
    for t in 2:T
        s = zero(Tf)
        for i in axes(w, 2)
            s += abs(w[t, i] - w[t - 1, i])
        end
        tn[t] = s
    end
    return tn
end
function calc_turnover(w::AbstractVector{<:AbstractVector{<:Number}})
    Tf = real(eltype(eltype(w)))
    T = length(w)
    tn = fill(Tf(NaN), T)
    for t in 2:T
        tn[t] = sum(abs, w[t] .- w[t - 1])
    end
    return tn
end
"""
$(DocStringExtensions.TYPEDEF)

Supertype for the algorithms that let a portfolio's weights drift with its own returns over the observations it is scored on.

A cross-validation scheme carries this family in its `wd` field. `nothing` scores every observation against the same weight vector, which is the library's original behaviour and stays its default. [`SelfFinancingDrift`](@ref) is the family's one leaf.

# Related

  - [`SelfFinancingDrift`](@ref)
  - [`weight_path`](@ref)
  - [`held_weights`](@ref)
  - [`calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs)`](@ref)
  - [`AbstractAlgorithm`](@ref)
"""
abstract type AbstractWeightDrift <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Grows each position at its own asset return and holds no trade in between, so the weights drift and the series is the wealth ratio of the drifted holdings.

The portfolio is self-financing over the window: no capital enters it and none leaves it, and the implicit cash position ``1 - \\sum_j w_j`` earns zero. The recursion therefore holds for a long-short book and for a partly invested one.

# Examples

```jldoctest
julia> SelfFinancingDrift()
SelfFinancingDrift()
```

# Related

  - [`AbstractWeightDrift`](@ref)
  - [`weight_path`](@ref)
  - [`held_weights`](@ref)
  - [`calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs)`](@ref)
"""
struct SelfFinancingDrift <: AbstractWeightDrift end
"""
    drift_position_values(wd::AbstractWeightDrift, w::VecNum, X::MatNum)

Compute the value of each position at each observation of a drifted window.

Each position starts at its weight and compounds at its own asset return, so row `t` holds what one unit of initial capital has become in each asset by the end of observation `t`. The matrix is the quantity every other drift verb reads, and it is a cumulative product, not a per step renormalisation.

# Mathematical definition

```math
\\begin{align}
P_{t j} &= w_{j} \\prod\\limits_{s=1}^{t} \\left(1 + X_{s j}\\right)
\\end{align}
```

Where:

  - ``P_{t j}``: Value of the position in asset ``j`` at the end of observation ``t``, per unit of initial capital.
  - ``X_{s j}``: Return of asset ``j`` at observation ``s``.
  - $(math_dict[:w_port])
  - $(math_dict[:T])
  - $(math_dict[:N])

# Algorithm

 1. Compound the asset returns with [`relative_cumulative_returns`](@ref), giving the `T × N` matrix of growth factors of each asset up to each observation.
 2. Scale column `j` of that matrix by `w[j]`, giving the position values.

# Arguments

  - `wd`: Weight drift algorithm.
  - `w`: Portfolio weights.
  - `X`: Asset return matrix (observations × assets).

# Returns

  - `P::MatNum`: `T × N` matrix of position values.

# Related

  - [`AbstractWeightDrift`](@ref)
  - [`SelfFinancingDrift`](@ref)
  - [`drift_wealth`](@ref): Contracts this matrix into the wealth of each observation.
  - [`relative_cumulative_returns`](@ref): The helper that step 1 reaches.
  - [`weight_path`](@ref)
"""
function drift_position_values(::AbstractWeightDrift, w::VecNum, X::MatNum)
    return relative_cumulative_returns(X) ⊙ transpose(w)
end
"""
    drift_wealth(P::MatNum, w::VecNum)

Compute the wealth of each observation of a drifted window from its position values.

The cash position is what the weights leave uninvested, `1 - sum(w)`, and it earns zero, so it enters every observation's wealth unchanged. A fully invested book holds none of it, and a levered or short book holds a negative one.

# Mathematical definition

```math
\\begin{align}
V_{t} &= \\sum\\limits_{j=1}^{N} P_{t j} + \\left(1 - \\sum\\limits_{j=1}^{N} w_{j}\\right)
\\end{align}
```

Where:

  - ``V_{t}``: Wealth at the end of observation ``t``, per unit of initial capital.
  - ``P_{t j}``: Value of the position in asset ``j`` at the end of observation ``t``.
  - $(math_dict[:w_port])
  - $(math_dict[:N])

# Algorithm

 1. Sum `P` along its asset axis, giving the invested wealth of each observation.
 2. Add the cash position `1 - sum(w)` to every entry.

# Arguments

  - `P`: `T × N` matrix of position values, from [`drift_position_values`](@ref).
  - `w`: Portfolio weights.

# Returns

  - `V::VecNum`: `T × 1` vector of wealth, one entry per observation.

# Related

  - [`drift_position_values`](@ref): The matrix this verb contracts.
  - [`drift_returns`](@ref): Reads this vector as a series of returns.
  - [`assert_positive_wealth`](@ref): The domain check this vector must pass.
  - [`weight_path`](@ref)
"""
function drift_wealth(P::MatNum, w::VecNum)
    return vec(sum(P; dims = 2)) .+ (one(eltype(P)) - sum(w))
end
"""
    drift_returns(V::VecNum)

Read a wealth vector as the return series of the drifted window.

The return of an observation is the ratio of its wealth to the wealth before it, and the wealth before the first observation is the initial capital of one. The series is therefore the wealth ratio of the drifted holdings, and it is **not** the dot product of the held weights with the asset returns: the two differ by up to 809 ulp, and the wealth ratio is what the reference arithmetic computes.

# Mathematical definition

```math
\\begin{align}
R_{t} &= \\frac{V_{t}}{V_{t-1}} - 1\\,, \\qquad V_{0} = 1
\\end{align}
```

Where:

  - ``R_{t}``: Return of observation ``t``.
  - ``V_{t}``: Wealth at the end of observation ``t``, per unit of initial capital.
  - $(math_dict[:T])

# Algorithm

 1. Prepend the initial capital of one to `V`, and drop its last entry, giving the wealth before each observation.
 2. Divide `V` by that vector elementwise, and subtract one.

# Arguments

  - `V`: `T × 1` vector of wealth, from [`drift_wealth`](@ref).

# Returns

  - `R::VecNum`: `T × 1` vector of drifted returns.

# Related

  - [`drift_wealth`](@ref): The vector this verb reads.
  - [`calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs)`](@ref): Charges the fee against this series.
  - [`cumulative_returns`](@ref)
"""
function drift_returns(V::VecNum)
    return V ./ vcat(one(eltype(V)), @view(V[1:(end - 1)])) .- one(eltype(V))
end
"""
    non_positive_wealth_index(V::VecNum)

Find the first observation whose wealth is not positive.

A `NaN` wealth fails `> 0` and is found, so an exactly zero wealth and the non-finite values it makes downstream are both caught here.

# Algorithm

 1. Return the index of the first entry of `V` that does not satisfy `x > 0`, and `nothing` when every entry does.

# Arguments

  - `V`: `T × 1` vector of wealth, from [`drift_wealth`](@ref).

# Returns

  - `i::Option{<:Integer}`: Index of the first non-positive wealth, or `nothing`.

# Related

  - [`drift_wealth`](@ref)
  - [`assert_positive_wealth`](@ref): The check that raises on this index.
  - [`NonPositiveWealthError`](@ref)
  - [`Option`](@ref)
"""
function non_positive_wealth_index(V::VecNum)
    return findfirst(x -> !(x > zero(x)), V)
end
"""
    assert_positive_wealth(V::VecNum, obs = nothing, member = nothing)

Check that every observation of a drifted window has a positive wealth.

The check runs before any return is formed, so a ruined window gives no partial series. `obs` names the failing observation in the message: a vector of labels names it by its label, a vector of integers names it by its panel row, and `nothing` names it by its row inside the window.

# Algorithm

 1. Find the first non-positive wealth with [`non_positive_wealth_index`](@ref). Return `nothing` when there is none.
 2. Build the phrase that names the failing observation, reading `obs`.
 3. Raise a [`NonPositiveWealthError`](@ref) that states the condition, the failing wealth, that phrase, and the member when one is named.

# Arguments

  - `V`: `T × 1` vector of wealth, from [`drift_wealth`](@ref).
  - `obs`: Observation labels of the window, panel rows of the window, or `nothing`.
  - `member`: Index of the population member the wealth belongs to, or `nothing` for a single weight vector.

# Validation

  - `V`: `all(>(0), V)`, else a [`NonPositiveWealthError`](@ref) is raised.

# Returns

  - `nothing`.

# Related

  - [`non_positive_wealth_index`](@ref): The finder step 1 reaches.
  - [`NonPositiveWealthError`](@ref)
  - [`drift_wealth`](@ref)
  - [`weight_path`](@ref)
"""
function assert_positive_wealth(V::VecNum, obs = nothing, member = nothing)::Nothing
    i = non_positive_wealth_index(V)
    if isnothing(i)
        return nothing
    end
    where_at = if isnothing(obs)
        "row $(i) of the window"
    elseif isa(obs[i], Integer)
        "panel row $(obs[i])"
    else
        "observation $(obs[i])"
    end
    whose = isnothing(member) ? "the wealth" : "the wealth of member $(member)"
    return throw(NonPositiveWealthError("the drifted wealth must satisfy `all(>(0), wealth)`, but $(whose) is $(V[i]) at $(where_at)"))
end
"""
    drifted_weight_path(P::MatNum, V::VecNum, w::VecNum)

Read the weights held through each observation of a drifted window, from one pass of the drift.

The weights held through observation `t` are the position values after observation `t - 1`, quoted against the wealth after observation `t - 1`. The first row is the target weights, because nothing has drifted yet when the window opens. [`weight_path`](@ref) is the verb that makes `P` and `V` itself; this one reads a pass a caller already holds, so a caller that needs the path **and** the held weights pays one pass rather than two.

# Mathematical definition

```math
\\begin{align}
u_{1 j} &= w_{j}\\,,\\\\
u_{t j} &= \\frac{P_{t-1, j}}{V_{t-1}}\\,, \\qquad t > 1\\,.
\\end{align}
```

Where:

  - ``P`` is the matrix of position values.
  - ``V`` is the wealth of each observation.
  - ``w`` is the vector of target weights.

# Algorithm

 1. Write the target weights into the first row.
 2. Write each later row as the position values of the observation before it, divided by the wealth of that observation.

# Arguments

  - `P`: Position values of the window, from [`drift_position_values`](@ref).
  - `V`: Wealth of each observation, from [`drift_wealth`](@ref).
  - `w`: Target weights of the window.

# Returns

  - `MatNum`: The weight path, `observations × assets`.

# Related

  - [`weight_path`](@ref)
  - [`drift_position_values`](@ref)
  - [`drift_wealth`](@ref)
  - [`drifted_held_weights`](@ref)
"""
function drifted_weight_path(P::MatNum, V::VecNum, w::VecNum)
    U = similar(P)
    U[1, :] .= w
    if size(P, 1) > 1
        U[2:end, :] .= @view(P[1:(end - 1), :]) ./ @view(V[1:(end - 1)])
    end
    return U
end
"""
    drifted_held_weights(P::MatNum, V::VecNum)

Read the weights held after the last observation of a drifted window, from one pass of the drift.

These are the weights a chain carries forward. They are one step beyond the last row of the weight path: that row holds the weights the window opened its last observation with, and these hold the weights it closed that observation with.

# Mathematical definition

```math
\\begin{align}
u_{T+1, j} &= \\frac{P_{T j}}{V_{T}}\\,.
\\end{align}
```

Where:

  - ``P`` is the matrix of position values.
  - ``V`` is the wealth of each observation.
  - ``T`` is the number of observations of the window.

# Algorithm

 1. Divide the last row of the position values by the last wealth.

# Arguments

  - `P`: Position values of the window, from [`drift_position_values`](@ref).
  - `V`: Wealth of each observation, from [`drift_wealth`](@ref).

# Returns

  - `VecNum`: The weights held after the last observation.

# Related

  - [`held_weights`](@ref)
  - [`drift_position_values`](@ref)
  - [`drift_wealth`](@ref)
  - [`drifted_weight_path`](@ref)
"""
function drifted_held_weights(P::MatNum, V::VecNum)
    return @view(P[end, :]) ./ V[end]
end
"""
    weight_path(wd::AbstractWeightDrift, w::VecNum, X::MatNum, obs = nothing)

Compute the weights held at each observation of a drifted window.

Row `t` holds the weights the book carries **through** observation `t`, so the first row is the target weights and every later row is the previous observation's position values deflated by its wealth. The rows and the deflated cash position sum to one at every observation.

# Mathematical definition

```math
\\begin{align}
U_{1 j} &= w_{j}\\\\
U_{t j} &= \\frac{P_{t-1,\\, j}}{V_{t-1}}\\,, \\qquad t > 1
\\end{align}
```

The equivalent closed form is the self-financing recursion, which a reader may recognise from the literature:

```math
\\begin{align}
\\boldsymbol{u}_{t+1} &= \\frac{\\boldsymbol{u}_{t} \\odot \\left(1 + \\boldsymbol{x}_{t}\\right)}{1 + \\boldsymbol{u}_{t} \\cdot \\boldsymbol{x}_{t}}
\\end{align}
```

The two forms agree in exact arithmetic and differ in floating point: the cumulative product is what the code computes, and the recursion was measured up to 824 ulp away from it.

Where:

  - ``U_{t j}``: Weight held in asset ``j`` through observation ``t``.
  - ``P_{t j}``: Value of the position in asset ``j`` at the end of observation ``t``.
  - ``V_{t}``: Wealth at the end of observation ``t``.
  - ``\\boldsymbol{u}_{t}``: `N × 1` vector of weights held through observation ``t``.
  - $(math_dict[:x_t_obs])
  - $(math_dict[:w_port])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - ``\\odot``: Elementwise (Hadamard) multiplication.

# Algorithm

 1. Compute the position values with [`drift_position_values`](@ref) and the wealth with [`drift_wealth`](@ref).
 2. Check the wealth with [`assert_positive_wealth`](@ref).
 3. Write the target weights into the first row.
 4. Write each earlier observation's position values, deflated by that observation's wealth, into the row after it.

# Arguments

  - `wd`: Weight drift algorithm.
  - `w`: Portfolio weights.
  - `X`: Asset return matrix (observations × assets).
  - `obs`: Observation labels of the window, panel rows of the window, or `nothing`. Read only by the message of step 2.

# Validation

  - The wealth of every observation is positive, else a [`NonPositiveWealthError`](@ref) is raised.

# Returns

  - `U::MatNum`: `T × N` matrix of held weights, one row per observation.

# Examples

```jldoctest
julia> PortfolioOptimisers.weight_path(SelfFinancingDrift(), [0.5, 0.5], [0.1 -0.1; 0.2 0.0])
2×2 Matrix{Float64}:
 0.5   0.5
 0.55  0.45
```

# Related

  - [`AbstractWeightDrift`](@ref)
  - [`SelfFinancingDrift`](@ref)
  - [`held_weights`](@ref): The weights held after the last observation.
  - [`drift_position_values`](@ref)
  - [`drift_wealth`](@ref)
  - [`assert_positive_wealth`](@ref)
"""
function weight_path(wd::AbstractWeightDrift, w::VecNum, X::MatNum, obs = nothing)
    P = drift_position_values(wd, w, X)
    V = drift_wealth(P, w)
    assert_positive_wealth(V, obs)
    return drifted_weight_path(P, V, w)
end
"""
    held_weights(wd::AbstractWeightDrift, w::VecNum, X::MatNum, obs = nothing)

Compute the weights held after the last observation of a drifted window.

These are the weights a fund carries into the next rebalance, so they are the base a trade is measured against. They are one step beyond the last row of [`weight_path`](@ref), which holds the weights carried **through** the last observation.

# Mathematical definition

```math
\\begin{align}
h_{j} &= \\frac{P_{T j}}{V_{T}}
\\end{align}
```

Where:

  - ``h_{j}``: Weight held in asset ``j`` after the last observation.
  - ``P_{t j}``: Value of the position in asset ``j`` at the end of observation ``t``.
  - ``V_{t}``: Wealth at the end of observation ``t``.
  - $(math_dict[:T])
  - $(math_dict[:N])

# Algorithm

 1. Compute the position values with [`drift_position_values`](@ref) and the wealth with [`drift_wealth`](@ref).
 2. Check the wealth with [`assert_positive_wealth`](@ref).
 3. Divide the last row of the position values by the last wealth.

# Arguments

  - `wd`: Weight drift algorithm.
  - `w`: Portfolio weights.
  - `X`: Asset return matrix (observations × assets).
  - `obs`: Observation labels of the window, panel rows of the window, or `nothing`. Read only by the message of step 2.

# Validation

  - The wealth of every observation is positive, else a [`NonPositiveWealthError`](@ref) is raised.

# Returns

  - `h::VecNum`: `N × 1` vector of held weights.

# Examples

```jldoctest
julia> PortfolioOptimisers.held_weights(SelfFinancingDrift(), [0.5, 0.5], [0.1 -0.1; 0.2 0.0])
2-element Vector{Float64}:
 0.5945945945945945
 0.4054054054054054
```

# Related

  - [`AbstractWeightDrift`](@ref)
  - [`SelfFinancingDrift`](@ref)
  - [`weight_path`](@ref): The weights held through each observation.
  - [`drift_position_values`](@ref)
  - [`drift_wealth`](@ref)
  - [`assert_positive_wealth`](@ref)
"""
function held_weights(wd::AbstractWeightDrift, w::VecNum, X::MatNum, obs = nothing)
    P = drift_position_values(wd, w, X)
    V = drift_wealth(P, w)
    assert_positive_wealth(V, obs)
    return drifted_held_weights(P, V)
end
"""
    calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs = nothing)
    calc_net_returns(w::VecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs = nothing)
    calc_net_returns(w::VecNum, X::MatNum, fees, wd::Nothing, args...)
    calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::Nothing, args...)

Compute the net portfolio returns of a window, reading the weight drift `wd`.

A `nothing` `wd` scores every observation against the same weight vector and reproduces [`calc_net_returns(w::VecNum, X::MatNum, args...)`](@ref) exactly. An [`AbstractWeightDrift`](@ref) lets each position grow at its own return, and the series becomes the wealth ratio of the drifted holdings. The fee is charged as it is on the undrifted series: [`calc_fees`](@ref) contracts the target weights into one scalar, and that scalar is subtracted from every observation.

A vector of weight vectors is a population, and a ruined member does not stop the run. The member's series is filled with `NaN`, one warning names every member that fell, and the run raises only when no member survives. A single weight vector is a population of one, so it raises.

# Mathematical definition

```math
\\begin{align}
R_{t} &= \\frac{V_{t}}{V_{t-1}} - 1 - F_{\\text{t}}(\\boldsymbol{w})\\,, \\qquad V_{0} = 1\\\\
V_{t} &= \\sum\\limits_{j=1}^{N} w_{j} \\prod\\limits_{s=1}^{t} \\left(1 + X_{s j}\\right) + \\left(1 - \\sum\\limits_{j=1}^{N} w_{j}\\right)
\\end{align}
```

Where:

  - ``R_{t}``: Net return of observation ``t``.
  - ``V_{t}``: Wealth at the end of observation ``t``, per unit of initial capital.
  - ``X_{s j}``: Return of asset ``j`` at observation ``s``.
  - ``F_{\\text{t}}(\\boldsymbol{w})``: Total fees computed using [`calc_fees`](@ref).
  - $(math_dict[:w_port])
  - $(math_dict[:T])
  - $(math_dict[:N])

# Algorithm

 1. On a `nothing` `wd`, delegate to [`calc_net_returns(w::VecNum, X::MatNum, args...)`](@ref), which charges the fee and reads no drift.
 2. On an [`AbstractWeightDrift`](@ref), give a `NaN` series when `w` is not finite: a failed solve has no weights to drift, and its series is `NaN` as the undrifted one is, with no fee charged.
 3. Otherwise compute the position values with [`drift_position_values`](@ref) and the wealth with [`drift_wealth`](@ref).
 4. Check the wealth with [`assert_positive_wealth`](@ref).
 5. Read the wealth as a return series with [`drift_returns`](@ref), and subtract the one fee scalar from every observation.
 6. On a population, run steps 2 to 5 for each member. A non-finite member's series is `NaN` and the member is not ruined. Fill a ruined member's series with `NaN`, warn once naming every member that fell, and raise when every member fell.

# Arguments

  - `w`: Portfolio weights, or a vector of portfolio weight vectors.
  - `X`: Asset return matrix (observations × assets).
  - `fees`: [`Fees`](@ref) structure, or `nothing`.
  - `wd`: Weight drift algorithm, or `nothing`.
  - `obs`: Observation labels of the window, panel rows of the window, or `nothing`. Read only by the message of step 3.
  - `args...`: Additional arguments (ignored).

# Validation

  - The wealth of every observation is positive, else a [`NonPositiveWealthError`](@ref) is raised. On a population, the raise happens only when no member survives. A non-finite `w` is not checked: it is a failed solve, and its series is `NaN`.

# Returns

  - `val::VecNum`: Portfolio net returns, for a `w::VecNum`.
  - `val::Vector{<:VecNum}`: One net return series per weight vector, for a `w::VecVecNum`.

# Examples

```jldoctest
julia> calc_net_returns([0.5, 0.5], [0.1 -0.1; 0.2 0.0], nothing, SelfFinancingDrift())
2-element Vector{Float64}:
 0.0
 0.1100000000000001
```

# Related

  - [`AbstractWeightDrift`](@ref)
  - [`SelfFinancingDrift`](@ref)
  - [`calc_net_returns(w::VecNum, X::MatNum, args...)`](@ref): The undrifted series a `nothing` `wd` reaches.
  - [`drift_returns`](@ref)
  - [`assert_positive_wealth`](@ref)
  - [`NonPositiveWealthError`](@ref)
  - [`VecVecNum`](@ref)
  - [`Fees`](@ref)
"""
function calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift,
                          obs = nothing)
    Tr = promote_type(eltype(X), eltype(first(w)))
    ret = Vector{Vector{Tr}}(undef, length(w))
    ruined = Int[]
    for (i, wi) in pairs(w)
        if !all(isfinite, wi)
            ret[i] = fill(convert(Tr, NaN), size(X, 1))
            continue
        end
        P = drift_position_values(wd, wi, X)
        V = drift_wealth(P, wi)
        if isnothing(non_positive_wealth_index(V))
            ret[i] = charge_fees(drift_returns(V), wi, fees)
        else
            push!(ruined, i)
            ret[i] = fill(convert(Tr, NaN), size(X, 1))
        end
    end
    if length(ruined) == length(w)
        assert_positive_wealth(drift_wealth(drift_position_values(wd, first(w), X),
                                            first(w)), obs, 1)
    elseif !isempty(ruined)
        @warn "the drifted wealth of $(length(ruined)) of $(length(w)) population member(s) is not positive, so their series are `NaN` and their members are dropped: $(ruined)"
    end
    return ret
end
function calc_net_returns(w::VecNum, X::MatNum, fees, wd::AbstractWeightDrift,
                          obs = nothing)
    if !all(isfinite, w)
        return fill(convert(promote_type(eltype(X), eltype(w)), NaN), size(X, 1))
    end
    P = drift_position_values(wd, w, X)
    V = drift_wealth(P, w)
    assert_positive_wealth(V, obs)
    return charge_fees(drift_returns(V), w, fees)
end
function calc_net_returns(w::VecNum, X::MatNum, fees, ::Nothing, args...)
    return calc_net_returns(w, X, fees)
end
function calc_net_returns(w::VecVecNum, X::MatNum, fees, ::Nothing, args...)
    return calc_net_returns(w, X, fees)
end
"""
    relative_cumulative_returns(X; dims = 1)

Compute the relative cumulative returns from a return matrix.

Internal helper that computes cumulative returns as `cumprod(1 .+ X; dims=dims)`, returning the cumulative portfolio value relative to the starting value.

The series is quoted against an initial capital of one, so an entry below one is a loss and an entry above one is a gain. [`relative_drawdown_arr`](@ref) reads that convention through its `init = one(eltype(X))`.

# Algorithm

 1. Add one to every entry of `X`, giving the per period growth factors.
 2. Take the running product of those factors along `dims`, giving the cumulative portfolio value relative to the starting value.

# Arguments

  - `X`: Return matrix.
  - $(arg_dict[:dims])

# Returns

  - `ret::ArrNum`: Relative cumulative return matrix, same shape as `X`.

# Related

  - [`ArrNum`](@ref)
  - [`absolute_cumulative_returns`](@ref): The additive sibling, whose series starts at zero.
  - [`cumulative_returns`](@ref): The public entry point that reaches this helper when `compound` is `true`.
  - [`relative_drawdown_arr`](@ref)
"""
function relative_cumulative_returns(X::ArrNum; dims::Int = 1)
    return cumprod(one(eltype(X)) .+ X; dims = dims)
end
"""
    absolute_cumulative_returns(X; dims = 1)

Compute the absolute cumulative returns from a return matrix.

Internal helper that computes `cumsum(X; dims=dims)`, returning the cumulative sum of portfolio returns.

The series is quoted against an initial capital of zero, so an entry below zero is a loss and an entry above zero is a gain. [`absolute_drawdown_arr`](@ref) reads that convention through its `init = zero(eltype(X))`.

# Algorithm

 1. Take the running sum of `X` along `dims`, giving the cumulative portfolio return.

# Arguments

  - `X`: Return matrix.
  - $(arg_dict[:dims])

# Returns

  - `ret::ArrNum`: Cumulative return matrix, same shape as `X`.

# Related

  - [`ArrNum`](@ref)
  - [`relative_cumulative_returns`](@ref): The multiplicative sibling, whose series starts at one.
  - [`cumulative_returns`](@ref): The public entry point that reaches this helper when `compound` is `false`.
  - [`absolute_drawdown_arr`](@ref)
"""
function absolute_cumulative_returns(X::ArrNum; dims::Int = 1)
    return cumsum(X; dims = dims)
end
"""
    cumulative_returns(X::ArrNum, compound::Bool = false; dims::Int = 1)

Compute simple or compounded cumulative returns along a specified dimension.

`cumulative_returns` computes the cumulative returns for an array of asset or portfolio returns. By default, it computes simple cumulative returns using `cumsum`. If `compound` is `true`, it computes compounded cumulative returns using `cumprod(one(eltype(X)) .+ X)`.

**The Precomputed-returns contract: the series the caller hands this verb must be finite.** It takes no finiteness check, because every internal caller hands it the output of [`calc_net_returns`](@ref) and a scan on a long series would be paid by all of them. One non-finite entry poisons every entry after it, because both accumulations carry it forward. A caller who holds a gapped series drops the gaps first with `x[isfinite.(x)]`, which is the reference implementation's own drop-per-column answer, and a caller who holds a gapped panel scores it through [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref) instead.

# Mathematical definition

## Portfolio cumulative returns

```math
\\begin{align}
\\boldsymbol{CR}_{a}(\\boldsymbol{X}) &= \\left\\{j \\in [1,\\,T]\\, |\\, \\boldsymbol{CR}_{a,\\, j}\\right\\} \\\\
CR_{a,\\, j}(\\boldsymbol{X}) &= \\sum\\limits_{i=1}^{j} X_{i} \\\\
\\boldsymbol{CR}_{r}(\\boldsymbol{X}) &= \\left\\{j \\in [1,\\,T]\\, |\\, \\boldsymbol{CR}_{r,\\, j}\\right\\} \\\\
CR_{r,\\, j}(\\boldsymbol{X}) &= \\prod\\limits_{i=1}^{j} (1 + X_{i})
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.
  - ``CR_{a,\\, j}(\\boldsymbol{X})``: Simple cumulative portfolio returns at period `j`.
  - ``CR_{r,\\, j}(\\boldsymbol{X})``: Compound cumulative portfolio returns at period `j`.
  - ``\\boldsymbol{CR}_{a}``: `T × 1` vector of simple cumulative portfolio returns.
  - ``\\boldsymbol{CR}_{r}``: `T × 1` vector of compound cumulative portfolio returns.

## Per asset cumulative returns

The same definitions apply as above, but for each individual asset in the returns matrix ``\\mathbf{X}`` instead of the portfolio return ``\\mathbf{X} \\boldsymbol{w}``.

# Algorithm

 1. Read `compound`, which selects one of two helpers.
 2. When `compound` is `false`, reach [`absolute_cumulative_returns`](@ref), giving the running sum of `X` along `dims`.
 3. When `compound` is `true`, reach [`relative_cumulative_returns`](@ref), giving the running product of `one(eltype(X)) .+ X` along `dims`.

# Arguments

  - `X`: Array of asset or portfolio returns (vector or matrix).
  - `compound`: If `true`, computes compounded cumulative returns; otherwise, computes simple cumulative returns.
  - $(arg_dict[:dims])

# Returns

  - `ret::ArrNum`: Array of cumulative returns, same shape as `X`.

# Examples

```jldoctest
julia> cumulative_returns([0.01, 0.02, -0.01])
3-element Vector{Float64}:
 0.01
 0.03
 0.02

julia> cumulative_returns([0.01, 0.02, -0.01], true)
3-element Vector{Float64}:
 1.01
 1.0302
 1.019898
```

# Related

  - [`ArrNum`](@ref)
  - [`absolute_cumulative_returns`](@ref): The helper that step 2 reaches.
  - [`relative_cumulative_returns`](@ref): The helper that step 3 reaches.
  - [`drawdowns`](@ref)
"""
function cumulative_returns(X::ArrNum, compound::Bool = false; dims::Int = 1)
    return if !compound
        absolute_cumulative_returns(X; dims = dims)
    else
        relative_cumulative_returns(X; dims = dims)
    end
end
"""
    absolute_drawdown_arr(X::ArrNum; cX::Bool = false, dims::Int = 1) -> ArrNum

Compute the absolute drawdown array for a matrix of cumulative (or raw) returns.

Each element represents the drawdown from the running peak along the specified dimension.

**The running peak starts at the initial capital, not at the first observation.** `accumulate(max, ...)` is seeded with `init = zero(eltype(X))`, which is the value an additive cumulative series starts from, so a series that is under water from the first period reports a negative drawdown there. `absolute_drawdown_arr([-0.1, 0.05])` returns `[-0.1, -0.05]` and not `[0.0, 0.0]`. That `init` is the whole convention, and this function holds one of its two definitions.

# Algorithm

 1. Read `cX`, which says whether `X` is already cumulative.
 2. When `cX` is `false`, reach [`absolute_cumulative_returns`](@ref) along `dims`, giving the cumulative series `cX`. When `cX` is `true`, take `X` as that series unchanged.
 3. Take the running maximum of `cX` along `dims` with `init = zero(eltype(X))`, giving the running peak. The `init` starts the peak at the initial capital of zero.
 4. Subtract the running peak from `cX`, giving the drawdown array.

# Arguments

  - `X::ArrNum`: Returns array (or cumulative-returns array if `cX = true`).
  - `cX::Bool = false`: If `true`, treat `X` as already cumulative returns.
  - $(arg_dict[:dims])

# Returns

  - `dd::ArrNum`: Drawdown array of the same shape as `X`. Every entry is zero or negative.

# Related

  - [`ArrNum`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`relative_drawdown_arr`](@ref): The compound sibling, seeded with `init = one(eltype(X))`.
  - [`drawdowns`](@ref): The public entry point that reaches this function when `compound` is `false`.
  - [`absolute_cumulative_returns`](@ref): The helper that step 2 reaches.
  - [`cumulative_returns`](@ref)
"""
function absolute_drawdown_arr(X::ArrNum; cX::Bool = false, dims::Int = 1)
    cX = !cX ? absolute_cumulative_returns(X; dims = dims) : X
    return cX - accumulate(max, cX; dims = dims, init = zero(eltype(X)))
end
"""
    relative_drawdown_arr(X::ArrNum; cX::Bool = false, dims::Int = 1) -> ArrNum

Compute the relative drawdown array for a matrix of cumulative (or raw) compounded returns.

Each element represents the relative drawdown from the running peak along the specified dimension.

**The running peak starts at the initial capital, not at the first observation.** `accumulate(max, ...)` is seeded with `init = one(eltype(X))`, which is the value a compound cumulative series starts from, so a series that is under water from the first period reports a negative drawdown there. `relative_drawdown_arr([-0.1, 0.05])` returns `[-0.09999999999999998, -0.05499999999999994]` and not `[0.0, 0.0]`. That `init` is the whole convention, and this function holds one of its two definitions.

The two conventions agree to first order on a small return. On `[-1e-6, 5e-7]` the additive and the compound answers differ by `5.0e-13`.

# Algorithm

 1. Read `cX`, which says whether `X` is already cumulative.
 2. When `cX` is `false`, reach [`relative_cumulative_returns`](@ref) along `dims`, giving the cumulative series `cX`. When `cX` is `true`, take `X` as that series unchanged.
 3. Take the running maximum of `cX` along `dims` with `init = one(eltype(X))`, giving the running peak. The `init` starts the peak at the initial capital of one.
 4. Divide `cX` by the running peak and subtract one, giving the relative drawdown array.

# Arguments

  - `X::ArrNum`: Returns array (or cumulative-returns array if `cX = true`).
  - `cX::Bool = false`: If `true`, treat `X` as already cumulative compounded returns.
  - $(arg_dict[:dims])

# Returns

  - `dd::ArrNum`: Relative drawdown array of the same shape as `X`. Every entry is zero or negative.

# Related

  - [`ArrNum`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`absolute_drawdown_arr`](@ref): The additive sibling, seeded with `init = zero(eltype(X))`.
  - [`drawdowns`](@ref): The public entry point that reaches this function when `compound` is `true`.
  - [`relative_cumulative_returns`](@ref): The helper that step 2 reaches.
  - [`cumulative_returns`](@ref)
"""
function relative_drawdown_arr(X::ArrNum; cX::Bool = false, dims::Int = 1)
    cX = !cX ? relative_cumulative_returns(X; dims = dims) : X
    return cX ./ accumulate(max, cX; dims = dims, init = one(eltype(X))) .- one(eltype(X))
end
"""
    drawdowns(X::ArrNum, compound::Bool = false; cX::Bool = false, dims::Int = 1)

Compute simple or compounded drawdowns along a specified dimension.

**The running peak starts at the initial capital, not at the first observation**, so a series that is under water from the first period reports a negative drawdown there. `drawdowns([-0.1, 0.05])` returns `[-0.1, -0.05]` and not `[0.0, 0.0]`. `drawdowns` dispatches to [`absolute_drawdown_arr`](@ref) or to [`relative_drawdown_arr`](@ref), which hold the single definition of the peak: the `init` of their `accumulate(max, ...)` is the initial capital, `zero(eltype(X))` for the additive path and `one(eltype(X))` for the compound one.

The two paths agree to first order on a small return. On `[-1e-6, 5e-7]` the additive and the compound answers differ by `5.0e-13`.

**The Precomputed-returns contract: the series the caller hands this verb must be finite.** It takes no finiteness check, for the reason [`cumulative_returns`](@ref) takes none. One non-finite entry poisons the running peak and every drawdown after it. A caller who holds a gapped series drops the gaps first with `x[isfinite.(x)]`, and a caller who holds a gapped panel scores it through [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref) instead.

# Mathematical definition

## Portfolio drawdowns

```math
\\begin{align}
\\boldsymbol{DD}_{a}(\\boldsymbol{X}) &= \\left\\{j \\in [1,\\,T] \\, |\\, \\mathrm{DD_{a}}(\\boldsymbol{X},\\, j)\\right\\}\\\\
DD_{a}(\\boldsymbol{X},\\, j) &= \\sum\\limits_{i=1}^{j} X_{i} - \\underset{t \\in [1,\\, j]}{\\max}\\left( \\sum\\limits_{i=1}^{t} X_{i} \\right)\\\\
\\boldsymbol{DD}_{r}(\\boldsymbol{X}) &= \\left\\{j \\in [1,\\,T] \\, |\\, \\mathrm{DD_{r}}(\\boldsymbol{X},\\, j)\\right\\}\\\\
DD_{r}(\\boldsymbol{X},\\, j) &= \\dfrac{\\prod\\limits_{i=1}^{j}\\left(1 + X_{i}\\right)}{\\underset{t \\in [1,\\, j]}{\\max}\\left( \\prod\\limits_{i=1}^{t}\\left(1 + X_{i}\\right) \\right)} - 1
\\end{align}
```

Where:

  - ``\\boldsymbol{DD}_{a}(\\boldsymbol{X})``: `T × 1` vector of simple drawdowns.
  - ``DD_{a}(\\boldsymbol{X},\\, j)``: Simple drawdown at period `j`.
  - ``\\boldsymbol{DD}_{r}(\\boldsymbol{X})``: `T × 1` vector of compound drawdowns.
  - ``DD_{r}(\\boldsymbol{X},\\, j)``: Compound drawdown at period `j`.
  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.

## Per asset portfolio drawdowns

The same definitions apply as above, but for each individual asset in the returns matrix ``\\mathbf{X}`` instead of the portfolio return ``\\mathbf{X} \\boldsymbol{w}``.

# Algorithm

 1. Read `compound`, which selects one of two builders.
 2. When `compound` is `false`, reach [`absolute_drawdown_arr`](@ref), forwarding `cX` and `dims`.
 3. When `compound` is `true`, reach [`relative_drawdown_arr`](@ref), forwarding `cX` and `dims`.
 4. The builder that step 2 or step 3 reaches cumulates `X` along `dims` first, unless `cX` is `true`, in which case it takes `X` as the cumulative series it already is.
 5. That builder then subtracts the running peak, seeded at the initial capital.

# Arguments

  - `X`: Array of asset or portfolio returns (vector or matrix).
  - `compound`: If `true`, computes compounded drawdowns; otherwise, computes simple drawdowns.
  - `cX`: If `true`, treats `X` as cumulative returns; otherwise, computes cumulative returns first.
  - $(arg_dict[:dims])

# Returns

  - `dd::ArrNum`: Array of drawdowns, same shape as `X`. Every entry is zero or negative.

# Examples

```jldoctest
julia> drawdowns([0.01, 0.02, -0.01])
3-element Vector{Float64}:
  0.0
  0.0
 -0.009999999999999998

julia> drawdowns([0.01, 0.02, -0.01], true)
3-element Vector{Float64}:
  0.0
  0.0
 -0.010000000000000009

julia> drawdowns([-0.1, 0.05])
2-element Vector{Float64}:
 -0.1
 -0.05
```

# Related

  - [`ArrNum`](@ref)
  - [`cumulative_returns`](@ref)
  - [`absolute_drawdown_arr`](@ref): The builder that step 2 reaches, and one of the two homes of the running-peak convention.
  - [`relative_drawdown_arr`](@ref): The builder that step 3 reaches, and the other home of the running-peak convention.
"""
function drawdowns(X::ArrNum, compound::Bool = false; cX::Bool = false, dims::Int = 1)
    return if !compound
        absolute_drawdown_arr(X; cX = cX, dims = dims)
    else
        relative_drawdown_arr(X; cX = cX, dims = dims)
    end
end

"""
    AbstractPreviousWeightsSource <: AbstractAlgorithm

Abstract supertype of the Previous-Weights Source family.

A Previous-Weights Source names the weights [`fold_loop`](@ref) threads from a fold into the fold that follows it. `nothing` threads the target weights of the previous fold, which is the library's original behaviour, and [`DriftedWeights`](@ref) is the family's one leaf.

The two walk-forward schemes carry this family in their `pws` field, bound to `Option{<:AbstractPreviousWeightsSource}`. A scheme whose folds are not a timeline carries no such field: no fold of it has a fold behind it, so it has no previous weights of any kind to thread.

# Interfaces

A subtype names a source of previous weights and declares no method of its own. [`previous_weights`](@ref) reads it.

# Related

  - [`DriftedWeights`](@ref)
  - [`previous_weights`](@ref)
  - [`AbstractWeightDrift`](@ref)
  - [`IndexWalkForward`](@ref)
  - [`DateWalkForward`](@ref)
"""
abstract type AbstractPreviousWeightsSource <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Thread the weights a fold **held** after its last observation into the fold that follows it.

The fold loop threads the target weights of the previous fold by default. Those are the weights the optimiser chose, so a turnover, a tracking or a fee estimator then measures the change in the decision. This source threads the weights the portfolio held after the last observation of that fold instead, so the same estimators measure the trades a fund places.

`wd` names the Weight Drift the held weights are computed under, and it is read **only** when the scheme's own `wd` is `nothing`. A scheme that drifts its return series drifts its held weights the same way, because one drift runs per fold and [`HeldWeightsResult`](@ref) records the form that ran.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DriftedWeights(; wd::AbstractWeightDrift = SelfFinancingDrift()) -> DriftedWeights

Keyword arguments correspond to the struct's fields.

# Examples

```jldoctest
julia> DriftedWeights()
DriftedWeights
  wd ┴ SelfFinancingDrift()
```

# Related

  - [`AbstractPreviousWeightsSource`](@ref)
  - [`SelfFinancingDrift`](@ref)
  - [`HeldWeightsResult`](@ref)
  - [`previous_weights`](@ref)
  - [`fold_loop`](@ref)
"""
@concrete struct DriftedWeights <: AbstractPreviousWeightsSource
    """
    $(field_dict[:pws_wd])
    """
    wd
    function DriftedWeights(wd::AbstractWeightDrift)
        return new{typeof(wd)}(wd)
    end
end
function DriftedWeights(; wd::AbstractWeightDrift = SelfFinancingDrift())::DriftedWeights
    return DriftedWeights(wd)
end
"""
$(DocStringExtensions.TYPEDEF)

Records what a fold actually held, so a reader can recover the weight path of that fold.

The record is present on a [`PredictionResult`](@ref) only when a drift ran, and a reader dispatches on its absence rather than testing for it. It stores the asset returns the fold was scored over, the weights the drift started from, the weights held after the last observation, and the Weight Drift that produced them. It stores the weight path itself only when the scheme's `store_weight_path` is `true`; otherwise [`weight_path`](@ref) rebuilds the path from the stored members on demand.

`w0` is the first row of the path. On a solved fold it is the fold's target weights; on a failed fold under a drift it is the previous weights the fold was handed, because a fold that could not rebalance holds what it held. The record carries it so a rebuild never reads the fold's `res.w`, which is `NaN` on a failed fold.

`wd` records the form that **ran**, not the setting that asked for it. A rebuild that guesses the form is a second answer rather than a rebuild, so the form travels with the numbers it made.

Under a population result the weights are one vector per member, so `w` is a vector of vectors and `U`, when it is stored, is one matrix per member. A member whose drifted wealth is not positive carries `NaN` in both, and its entry in the fold's return code is an [`OptimisationFailure`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HeldWeightsResult(;
        X::MatNum,
        U::Option{<:MatNum_VecMatNum},
        w0::VecNum_VecVecNum,
        w::VecNum_VecVecNum,
        wd::AbstractWeightDrift
    ) -> HeldWeightsResult

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(X)`.
  - A stored `U` has the size of `X`, one matrix per member under a population.
  - `w0` and `w` each have one entry per column of `X`, one vector per member under a population, and `w0` has the shape of `w`.

# Related

  - [`PredictionResult`](@ref)
  - [`weight_path`](@ref)
  - [`held_weights`](@ref)
  - [`AbstractWeightDrift`](@ref)
  - [`SelfFinancingDrift`](@ref)
"""
@concrete struct HeldWeightsResult <: AbstractResult
    """
    $(field_dict[:hw_X])
    """
    X
    """
    $(field_dict[:hw_U])
    """
    U
    """
    Weights the drift started from: the first row of the path.
    """
    w0
    """
    $(field_dict[:hw_w])
    """
    w
    """
    $(field_dict[:hw_wd])
    """
    wd
    function HeldWeightsResult(X::MatNum, U::Option{<:MatNum_VecMatNum},
                               w0::VecNum_VecVecNum, w::VecNum_VecVecNum,
                               wd::AbstractWeightDrift)
        assert_nonempty(X, :X)
        assert_held_weights_shape(U, X, w)
        assert_held_weights_shape(nothing, X, w0)
        assert_held_start_shape(w0, w)
        return new{typeof(X), typeof(U), typeof(w0), typeof(w), typeof(wd)}(X, U, w0, w, wd)
    end
end
function HeldWeightsResult(; X::MatNum, U::Option{<:MatNum_VecMatNum}, w0::VecNum_VecVecNum,
                           w::VecNum_VecVecNum, wd::AbstractWeightDrift)::HeldWeightsResult
    return HeldWeightsResult(X, U, w0, w, wd)
end
"""
    assert_held_weights_shape(U::Nothing, X::MatNum, w::VecNum)
    assert_held_weights_shape(U::Nothing, X::MatNum, w::VecVecNum)
    assert_held_weights_shape(U::MatNum, X::MatNum, w::VecNum)
    assert_held_weights_shape(U::VecMatNum, X::MatNum, w::VecVecNum)

Check that the members of a [`HeldWeightsResult`](@ref) agree on their shape.

The method is chosen by the pair `(U, w)`, so a stored path under a single weight vector and a stored path under a population are separate checks, and an absent path checks only the weights. A pair that does not match at all, such as a matrix path beside a population's weights, reaches no method and raises a `MethodError`, which names both shapes.

# Algorithm

 1. Check that `w` holds one weight for each column of `X`, member by member under a population.
 2. Check that a stored `U` has the size of `X`, member by member under a population.

# Arguments

  - `U`: Weight path, or `nothing`.
  - `X`: Asset returns of the fold.
  - `w`: Held weights after the last observation.

# Returns

  - `nothing`.

# Related

  - [`HeldWeightsResult`](@ref)
  - [`weight_path`](@ref)
"""
function assert_held_weights_shape(::Nothing, X::MatNum, w::VecNum)::Nothing
    @argcheck(length(w) == size(X, 2),
              DimensionMismatch("`length(w) == size(X, 2)` must hold.\nlength(w) => $(length(w))\nsize(X, 2) => $(size(X, 2))"))
    return nothing
end
function assert_held_weights_shape(::Nothing, X::MatNum, w::VecVecNum)::Nothing
    for wi in w
        assert_held_weights_shape(nothing, X, wi)
    end
    return nothing
end
function assert_held_weights_shape(U::MatNum, X::MatNum, w::VecNum)::Nothing
    assert_held_weights_shape(nothing, X, w)
    @argcheck(size(U) == size(X),
              DimensionMismatch("`size(U) == size(X)` must hold.\nsize(U) => $(size(U))\nsize(X) => $(size(X))"))
    return nothing
end
function assert_held_weights_shape(U::VecMatNum, X::MatNum, w::VecVecNum)::Nothing
    @argcheck(length(U) == length(w),
              DimensionMismatch("`length(U) == length(w)` must hold.\nlength(U) => $(length(U))\nlength(w) => $(length(w))"))
    for (Ui, wi) in zip(U, w)
        assert_held_weights_shape(Ui, X, wi)
    end
    return nothing
end
"""
    assert_held_start_shape(w0::VecNum, w::VecNum)
    assert_held_start_shape(w0::VecVecNum, w::VecVecNum)

Check that the start weights of a [`HeldWeightsResult`](@ref) have the shape of its held weights.

A single weight vector starts from a single vector, and a population starts from one vector per member. A pair that mixes the two reaches no method and raises a `MethodError`, which names both shapes.

# Algorithm

 1. Under a single weight vector, nothing further: [`assert_held_weights_shape`](@ref) already checked each against the columns of `X`.
 2. Under a population, check that `w0` has one vector per member of `w`.

# Arguments

  - `w0`: Weights the drift started from.
  - `w`: Held weights after the last observation.

# Returns

  - `nothing`.

# Related

  - [`HeldWeightsResult`](@ref)
  - [`assert_held_weights_shape`](@ref)
"""
function assert_held_start_shape(::VecNum, ::VecNum)::Nothing
    return nothing
end
function assert_held_start_shape(w0::VecVecNum, w::VecVecNum)::Nothing
    @argcheck(length(w0) == length(w),
              DimensionMismatch("`length(w0) == length(w)` must hold.\nlength(w0) => $(length(w0))\nlength(w) => $(length(w))"))
    return nothing
end
"""
    weight_path(hw::Nothing, w::VecNum, X::MatNum)
    weight_path(hw::Nothing, w::VecVecNum, X::MatNum)
    weight_path(hw::HeldWeightsResult, w::VecNum_VecVecNum, args...)

Read the weight path of a fold, from the record the fold carries.

A fold that carries no record held its target weights on every one of its observations, so its path is the constant path of `w`. A fold that carries one gives the stored path when `store_weight_path` asked for it, and rebuilds the path from the record's own start weights, asset returns and Weight Drift when it did not. The rebuild is bit-identical to the store, because the record carries the form that ran and the weights it started from.

# Algorithm

 1. On `nothing`, repeat `w` over the rows of `X`.
 2. On a record, read [`rebuild_weight_path`](@ref), which gives the stored path or rebuilds it.

# Arguments

  - `hw`: Held-weights record of the fold, or `nothing`.
  - `w`: Target weights of the fold, which are the first row of the path when there is no record. A record carries its own start weights `w0`, so a record ignores this argument: on a failed fold the target is `NaN` and the path is not.
  - `X`: Asset returns of the fold. A record carries its own, so a record ignores this argument.

# Returns

  - `MatNum_VecMatNum`: The weight path, `observations × assets`, one matrix per member under a population.

# Related

  - [`HeldWeightsResult`](@ref)
  - [`rebuild_weight_path`](@ref)
  - [`weight_path(wd::AbstractWeightDrift, w::VecNum, X::MatNum, obs)`](@ref)
  - [`PredictionResult`](@ref)
"""
function weight_path(::Nothing, w::VecNum, X::MatNum)
    return repeat(transpose(w), size(X, 1))
end
function weight_path(::Nothing, w::VecVecNum, X::MatNum)
    return [repeat(transpose(wi), size(X, 1)) for wi in w]
end
function weight_path(hw::HeldWeightsResult, ::VecNum_VecVecNum, args...)
    return rebuild_weight_path(hw.U, hw.wd, hw.w0, hw.X)
end
"""
    rebuild_weight_path(U::MatNum_VecMatNum, wd::AbstractWeightDrift, w::VecNum_VecVecNum, X::MatNum)
    rebuild_weight_path(U::Nothing, wd::AbstractWeightDrift, w::VecNum, X::MatNum)
    rebuild_weight_path(U::Nothing, wd::AbstractWeightDrift, w::VecVecNum, X::MatNum)

Give a stored weight path, or rebuild one that was not stored.

The stored path is read by dispatch on `U`, so [`weight_path`](@ref) tests nothing. A rebuild reruns the drift that made the record over the record's own asset returns, which is what makes it bit-identical to the store.

# Algorithm

 1. On a stored `U`, return it.
 2. On `nothing` under a single weight vector, run the drift over `X` and give its path. A non-finite start vector, which is a failed fold with nothing to hold, gives a `NaN` path, as the store would.
 3. On `nothing` under a population, run the drift member by member, and fill a member whose wealth is not positive, or whose start vector is not finite, with `NaN`. That is what the store holds for such a member, so the rebuild stays bit-identical to it. [`held_weights_result`](@ref) already raised on a population every member of which is ruined, so no record reaching this verb holds one.

# Arguments

  - `U`: Stored weight path, or `nothing`.
  - `wd`: Weight drift that made the record.
  - `w`: Weights the drift started from, `w0` of the record.
  - `X`: Asset returns of the fold.

# Returns

  - `MatNum_VecMatNum`: The weight path.

# Related

  - [`weight_path`](@ref)
  - [`HeldWeightsResult`](@ref)
"""
function rebuild_weight_path(U::MatNum_VecMatNum, ::AbstractWeightDrift, ::VecNum_VecVecNum,
                             ::MatNum)
    return U
end
function rebuild_weight_path(::Nothing, wd::AbstractWeightDrift, w::VecNum, X::MatNum)
    if !all(isfinite, w)
        return fill(convert(promote_type(eltype(X), eltype(w)), NaN), size(X))
    end
    return weight_path(wd, w, X)
end
function rebuild_weight_path(::Nothing, wd::AbstractWeightDrift, w::VecVecNum, X::MatNum)
    Tw = promote_type(eltype(X), eltype(first(w)))
    Us = Vector{Matrix{Tw}}(undef, length(w))
    for (i, wi) in pairs(w)
        if !all(isfinite, wi)
            Us[i] = fill(convert(Tw, NaN), size(X))
            continue
        end
        P = drift_position_values(wd, wi, X)
        V = drift_wealth(P, wi)
        Us[i] = if isnothing(non_positive_wealth_index(V))
            drifted_weight_path(P, V, wi)
        else
            fill(convert(Tw, NaN), size(X))
        end
    end
    return Us
end
"""
    held_weights_drift(wd::Nothing, pws::Nothing)
    held_weights_drift(wd::AbstractWeightDrift, pws::Any)
    held_weights_drift(wd::Nothing, pws::AbstractPreviousWeightsSource)

Resolve the one Weight Drift a fold runs, from the two switches of its scheme.

The two switches are independent, and either one alone asks for a drift. A scheme that drifts its return series drifts its held weights the same way. A scheme that drifts nothing but threads drifted weights runs the form the [`DriftedWeights`](@ref) source carries, because the series stays at the target weights and the holdings still move.

# Algorithm

 1. Neither switch is set: no drift runs, and the fold carries no [`HeldWeightsResult`](@ref).
 2. `wd` is set: it is the form that runs, whatever `pws` is.
 3. Only `pws` is set: the form it carries is the form that runs.

# Arguments

  - `wd`: Weight drift of the scheme, or `nothing`.
  - `pws`: Previous-weights source of the scheme, or `nothing`.

# Returns

  - `Option{<:AbstractWeightDrift}`: The form that runs, or `nothing`.

# Related

  - [`AbstractWeightDrift`](@ref)
  - [`DriftedWeights`](@ref)
  - [`held_weights_result`](@ref)
  - [`HeldWeightsResult`](@ref)
"""
function held_weights_drift(::Nothing, ::Nothing)
    return nothing
end
function held_weights_drift(wd::AbstractWeightDrift, ::Any)
    return wd
end
function held_weights_drift(::Nothing, pws::AbstractPreviousWeightsSource)
    return pws.wd
end
"""
    nan_held_member(X::MatNum, store_weight_path::Bool, ::Type{Tw})

The `NaN` held weights and, under the flag, the `NaN` weight path of a member that drifted nothing.

A member that failed with nothing to hold, or whose drifted wealth is not positive, records `NaN` at every asset and every observation. The two builders of a [`HeldWeightsResult`](@ref) read this one place for that shape, so the store and the rebuild stay bit-identical.

# Arguments

  - `X`: Asset returns of the fold.
  - `store_weight_path`: If `true`, give the `NaN` path as well; else `nothing` in its place.
  - `Tw`: Element type of the record.

# Returns

  - `(w, U)`: The `NaN` held weights, and the `NaN` path or `nothing`.

# Related

  - [`held_weights_member`](@ref)
  - [`held_weights_result`](@ref)
"""
function nan_held_member(X::MatNum, store_weight_path::Bool, ::Type{Tw}) where {Tw}
    return fill(convert(Tw, NaN), size(X, 2)),
           store_weight_path ? fill(convert(Tw, NaN), size(X)) : nothing
end
"""
    held_weights_member(wd::AbstractWeightDrift, wi::VecNum, X::MatNum, store_weight_path::Bool, ::Type{Tw})

Drift one member of a population, and say whether the drift ruined it.

# Algorithm

 1. A non-finite start vector drifts nothing: give [`nan_held_member`](@ref)'s shape and `false`, because a failure is not a ruin.
 2. Otherwise drift it. A wealth that is not positive gives the same shape and `true`.
 3. A surviving member gives its held weights, its path under the flag, and `false`.

# Arguments

  - `wd`: Weight drift that runs.
  - `wi`: Start weights of the member.
  - `X`: Asset returns of the fold.
  - `store_weight_path`: If `true`, give the path as well; else `nothing` in its place.
  - `Tw`: Element type of the record.

# Returns

  - `(w, U, ruined)`: The held weights, the path or `nothing`, and whether the member is ruined.

# Related

  - [`held_weights_result`](@ref)
  - [`nan_held_member`](@ref)
  - [`non_positive_wealth_index`](@ref)
"""
function held_weights_member(wd::AbstractWeightDrift, wi::VecNum, X::MatNum,
                             store_weight_path::Bool, ::Type{Tw}) where {Tw}
    if !all(isfinite, wi)
        return nan_held_member(X, store_weight_path, Tw)..., false
    end
    P = drift_position_values(wd, wi, X)
    V = drift_wealth(P, wi)
    if !isnothing(non_positive_wealth_index(V))
        return nan_held_member(X, store_weight_path, Tw)..., true
    end
    return drifted_held_weights(P, V),
           store_weight_path ? drifted_weight_path(P, V, wi) : nothing, false
end
"""
    held_weights_result(wd::Nothing, w::VecNum_VecVecNum, X::MatNum, store_weight_path::Bool, obs = nothing)
    held_weights_result(wd::AbstractWeightDrift, w::VecNum, X::MatNum, store_weight_path::Bool, obs = nothing)
    held_weights_result(wd::AbstractWeightDrift, w::VecVecNum, X::MatNum, store_weight_path::Bool, obs = nothing)

Build the [`HeldWeightsResult`](@ref) of a fold, and name the members the drift ruined.

The verb runs the drift once and reads three things off that one pass: the weights held after the last observation, the weight path when the caller asked for it, and the members whose wealth is not positive. A single weight vector is a population of one, so a ruined single vector raises rather than reporting a ruined member.

A non-finite start vector is a fold, or a member, that failed with no previous weights to hold, and it is not ruined: nothing drifts, the record carries `NaN`, and the member keeps the failure code its own optimisation gave it. [`held_start_weights`](@ref) is what hands a failed fold its previous weights instead, when it has any.

# Algorithm

 1. With no drift, give `(nothing, nothing)`. The fold held its target weights, so there is nothing to record.
 2. Over one weight vector, drift it, check the wealth with [`assert_positive_wealth`](@ref), and record the held weights and, under the flag, the path. A non-finite vector drifts nothing and records `NaN`.
 3. Over a population, drift each member. A ruined member records `NaN` and is named in the second value; a non-finite member records `NaN` and is not named. Raise when every member is ruined.

# Arguments

  - `wd`: Weight drift that runs, from [`held_weights_drift`](@ref), or `nothing`.
  - `w`: Weights the drift starts from, `w0` of the record: the fold's target weights, or the previous weights a failed fold holds.
  - `X`: Asset returns of the fold.
  - `store_weight_path`: If `true`, store the weight path on the record.
  - `obs`: Observation labels the wealth message names, see [`assert_positive_wealth`](@ref).

# Returns

  - `(hw, ruined)`: The record, or `nothing`; and the indices of the ruined members, or `nothing`.

# Related

  - [`HeldWeightsResult`](@ref)
  - [`held_weights_drift`](@ref)
  - [`mark_ruined_members`](@ref)
  - [`assert_positive_wealth`](@ref)
  - [`drifted_weight_path`](@ref)
  - [`drifted_held_weights`](@ref)
"""
function held_weights_result(::Nothing, ::VecNum_VecVecNum, ::MatNum, ::Bool, args...)
    return nothing, nothing
end
function held_weights_result(wd::AbstractWeightDrift, w::VecNum, X::MatNum,
                             store_weight_path::Bool, obs = nothing)
    if !all(isfinite, w)
        (wn, U) = nan_held_member(X, store_weight_path, promote_type(eltype(X), eltype(w)))
        return HeldWeightsResult(X, U, w, wn, wd), nothing
    end
    P = drift_position_values(wd, w, X)
    V = drift_wealth(P, w)
    assert_positive_wealth(V, obs)
    U = store_weight_path ? drifted_weight_path(P, V, w) : nothing
    return HeldWeightsResult(X, U, w, drifted_held_weights(P, V), wd), nothing
end
function held_weights_result(wd::AbstractWeightDrift, w::VecVecNum, X::MatNum,
                             store_weight_path::Bool, obs = nothing)
    Tw = promote_type(eltype(X), eltype(first(w)))
    ws = Vector{Vector{Tw}}(undef, length(w))
    Us = Vector{Matrix{Tw}}(undef, length(w))
    ruined = Int[]
    for (i, wi) in pairs(w)
        (ws[i], Ui, dead) = held_weights_member(wd, wi, X, store_weight_path, Tw)
        if dead
            push!(ruined, i)
        end
        if !isnothing(Ui)
            Us[i] = Ui
        end
    end
    if length(ruined) == length(w)
        assert_positive_wealth(drift_wealth(drift_position_values(wd, first(w), X),
                                            first(w)), obs, first(ruined))
    end
    U = store_weight_path ? Us : nothing
    return HeldWeightsResult(X, U, w, ws, wd), ruined
end
"""
    drift_observations(ts::Nothing, test_idx)
    drift_observations(ts::Any, test_idx)

Name the observations of a fold, for the message a non-positive wealth raises.

A fold that carries timestamps names its observations by their labels. A fold that carries none names them by their absolute rows of the panel, which the fold's own test indices are.

# Algorithm

 1. Give the timestamps when the fold carries them.
 2. Give the test indices when it does not.

# Arguments

  - `ts`: Timestamps of the fold, or `nothing`.
  - `test_idx`: Absolute panel rows of the fold.

# Returns

  - The labels [`assert_positive_wealth`](@ref) reads.

# Related

  - [`assert_positive_wealth`](@ref)
  - [`NonPositiveWealthError`](@ref)
"""
function drift_observations(::Nothing, test_idx)
    return test_idx
end
function drift_observations(ts, ::Any)
    return ts
end
"""
    expand_investable_columns(imsk::BitVector, A::MatNum)

Expand a per asset history of the investable universe back onto the full asset universe.

The column of a non-investable asset is filled with zero, which is what [`filter_held_gaps`](@ref) writes over its whole `NaN` column anyway. A drift over a zero column at a zero weight moves no wealth, so the expanded history rebuilds the same weight path the reduced one did.

# Arguments

  - `imsk`: The Investable Mask, `true` at every asset whose prior moments were finite.
  - `A`: A history over the investable assets, `observations × investable assets`.

# Returns

  - `A::MatNum`: The history over the full asset universe.

# Related

  - [`expand_held_weights`](@ref)
  - [`expand_investable_weights`](@ref)
  - [`filter_held_gaps`](@ref)
"""
function expand_investable_columns(imsk::BitVector, A::MatNum)
    @argcheck(count(imsk) == size(A, 2),
              DimensionMismatch("the investable mask keeps $(count(imsk)) of $(length(imsk)) assets, but the history holds $(size(A, 2)) columns; the mask and the history must come from the same reduction"))
    out = zeros(eltype(A), size(A, 1), length(imsk))
    out[:, imsk] = A
    return out
end
"""
    expand_held_weights(imsk::Nothing, hw)
    expand_held_weights(imsk::BitVector, hw::Nothing)
    expand_held_weights(imsk::BitVector, hw::HeldWeightsResult)

Expand a fold's Held Weights record back onto the full asset universe.

A fold reduces its test window to the Investable Mask before it reads it, so the record it builds lives on the investable assets alone. The next fold's turnover reads `hw.w`, and it states the previous weights over the caller's own universe, so the record is expanded once here, on the way out of [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref). Every member of the record is expanded together, so a reader that rebuilds the weight path from `hw.X` gets the path the fold ran.

# Algorithm

 1. Return the record unchanged when the mask is `nothing`, so a universe with nothing to exclude keeps the path it took before the mask existed.
 2. Return `nothing` when the fold built no record.
 3. Otherwise expand the asset returns, the stored weight path and the held weights, and rebuild the record around them.

# Arguments

  - `imsk`: The Investable Mask, or `nothing`.
  - `hw`: The fold's Held Weights record, or `nothing`.

# Returns

  - `hw::Option{<:HeldWeightsResult}`: The record over the full asset universe.

# Related

  - [`HeldWeightsResult`](@ref)
  - [`expand_investable_columns`](@ref)
  - [`expand_held_member`](@ref)
  - [`held_weights_result`](@ref)
"""
function expand_held_weights(::Nothing, hw)
    return hw
end
function expand_held_weights(::BitVector, ::Nothing)
    return nothing
end
function expand_held_weights(imsk::BitVector, hw::HeldWeightsResult)
    return HeldWeightsResult(; X = expand_investable_columns(imsk, hw.X),
                             U = expand_held_member(imsk, hw.U),
                             w0 = expand_held_member(imsk, hw.w0),
                             w = expand_held_member(imsk, hw.w), wd = hw.wd)
end
"""
    expand_held_member(imsk::BitVector, x::Nothing)
    expand_held_member(imsk::BitVector, x::VecNum)
    expand_held_member(imsk::BitVector, x::MatNum)
    expand_held_member(imsk::BitVector, x::VecVecNum)
    expand_held_member(imsk::BitVector, x::VecMatNum)

Expand one member of a Held Weights record back onto the full asset universe.

The member's type is the picker, so [`expand_held_weights`](@ref) states no branch of its own. A weight vector expands through [`expand_investable_weights`](@ref) and a weight path through [`expand_investable_columns`](@ref); a population expands member by member; an absent path stays absent.

# Arguments

  - `imsk`: The Investable Mask, `true` at every asset whose prior moments were finite.
  - `x`: Held weights, a weight path, a population of either, or `nothing`.

# Returns

  - The member over the full asset universe.

# Related

  - [`expand_held_weights`](@ref)
  - [`expand_investable_columns`](@ref)
  - [`expand_investable_weights`](@ref)
"""
function expand_held_member(::BitVector, ::Nothing)
    return nothing
end
function expand_held_member(imsk::BitVector, x::VecNum)
    return expand_investable_weights(imsk, x)
end
function expand_held_member(imsk::BitVector, x::MatNum)
    return expand_investable_columns(imsk, x)
end
function expand_held_member(imsk::BitVector, x::VecVecNum)
    return [expand_investable_weights(imsk, xi) for xi in x]
end
function expand_held_member(imsk::BitVector, x::VecMatNum)
    return [expand_investable_columns(imsk, xi) for xi in x]
end
export calc_net_returns, calc_net_asset_returns, calc_turnover, cumulative_returns,
       drawdowns, SelfFinancingDrift, DriftedWeights, HeldWeightsResult
