"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for estimators that turn continuous portfolio weights into share counts.

A finite allocation estimator buys share counts with a stated cash budget at stated prices. The library has two. [`DiscreteAllocation`](@ref) solves a mixed-integer programme, and [`GreedyAllocation`](@ref) buys in two greedy passes. [martin2021](@cite) states only that the package it describes converts optimal weights into a discrete allocation, in its Methods on page 3. It states neither algorithm. [`optimise`](@ref) takes either one with a [`FiniteAllocationInput`](@ref) as its second argument, and allocates the long and the short side of the portfolio as two separate sub-problems.

# Related

  - [`OptimisationEstimator`](@ref)
  - [`DiscreteAllocation`](@ref)
  - [`GreedyAllocation`](@ref)
  - [`FiniteAllocationInput`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)

# References

  - $(ref_dict[:martin2021]) Methods, page 3. $(ref_dict[:no_original_source])
"""
abstract type FiniteAllocationOptimisationEstimator <: OptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the results of a finite allocation.

Every subtype has the fields `shares`, `cost`, `w`, `cash` and `fees`, and its last field is `fb`. The generic [`factory`](@ref) method rebuilds a subtype from its fields and replaces the last one alone, so a new subtype must keep `fb` last.

# Related

  - [`OptimisationResult`](@ref)
  - [`DiscreteAllocationResult`](@ref)
  - [`GreedyAllocationResult`](@ref)
  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`factory`](@ref)
"""
abstract type FiniteAllocationOptimisationResult <: OptimisationResult end
"""
    const FOptE_FOpt = Union{FiniteAllocationOptimisationEstimator,
                             FiniteAllocationOptimisationResult}

Alias for a finite allocation estimator or a finite allocation result.

It is the type of the `fb` field of a finite allocation estimator. The fallback is an estimator that [`optimise`](@ref) runs when the first one fails, or a precomputed result that `optimise` returns as it is.

# Related

  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
"""
const FOptE_FOpt = Union{FiniteAllocationOptimisationEstimator,
                         FiniteAllocationOptimisationResult}
"""
    const FOptE_FOpt_FbChain = Union{FOptE_FOpt, FbChain}

Alias for the values that the `fb` field of a finite allocation result can hold.

[`optimise`](@ref) writes `nothing` when the estimator that the caller gave answered, and the [`FbChain`](@ref) of the failed attempts when a fallback answered. The alias also admits a [`FOptE_FOpt`](@ref), so [`factory`](@ref) can put a fallback estimator or result in the field.

# Related

  - [`FOptE_FOpt`](@ref)
  - [`FbChain`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
"""
const FOptE_FOpt_FbChain = Union{FOptE_FOpt, FbChain}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that give each side of a finite allocation its cash.

A finite allocation solves the short side first and the long side second. A collateral algorithm states how much cash the short side can spend, and how much cash the long side can spend after the short side trades. [`FiniteAllocationInput`](@ref) holds one in its `ca` field, so every allocator of a fallback chain reads the same rule.

All concrete subtypes must subtype `AbstractCollateralAlgorithm`.

# Interfaces

To implement a new collateral algorithm, subtype `AbstractCollateralAlgorithm` and make it callable with these two methods:

  - `(ca::MyCollateral)(w::VecNum, prices::VecNum, cash::Number) -> Number`: Returns the cash of the short side, before it trades.
  - `(ca::MyCollateral)(w::VecNum, prices::VecNum, cash::Number, smoney::Number, sfee::Number) -> Number`: Returns the cash of the long side, after the short side trades.

Where:

  - `w`: Target portfolio weights over the full universe. A negative weight is on the short side.
  - `prices`: Asset prices, in the order of `w`. A `NaN` price marks an Unpriced Asset, and its weight in `w` is zero.
  - `cash`: Cash of the allocation, the `cash` of [`FiniteAllocationInput`](@ref).
  - `smoney`: Money of the shares that the short side sold, a non-negative number.
  - `sfee`: Fee that the short side paid over the whole horizon.

Each method must return a non-negative number. The allocators do not check it. A plain function with the same two methods is also a valid `ca`, as [`CollateralAlg_Func`](@ref) states.

# Examples

```jldoctest
julia> struct MyNoShortCollateral <: PortfolioOptimisers.AbstractCollateralAlgorithm end

julia> (::MyNoShortCollateral)(w, prices, cash) = zero(cash)

julia> (::MyNoShortCollateral)(w, prices, cash, smoney, sfee) = cash * sum(x -> max(x, zero(x)), w)

julia> fai = FiniteAllocationInput(; w = [0.7, 0.5, -0.2], prices = [10.0, 20.0, 5.0],
                                   cash = 1000.0, ca = MyNoShortCollateral());

julia> optimise(GreedyAllocation(), fai).shares
3-element view(::Matrix{Float64}, :, 1) with eltype Float64:
 70.0
 25.0
 -0.0

julia> no_short(w, prices, cash) = zero(cash);

julia> no_short(w, prices, cash, smoney, sfee) = cash * sum(x -> max(x, zero(x)), w);

julia> optimise(GreedyAllocation(),
                FiniteAllocationInput(; w = [0.7, 0.5, -0.2], prices = [10.0, 20.0, 5.0],
                                      cash = 1000.0, ca = no_short)).shares
3-element view(::Matrix{Float64}, :, 1) with eltype Float64:
 70.0
 25.0
 -0.0
```

# Related

  - [`ProceedsCollateral`](@ref)
  - [`CashCollateral`](@ref)
  - [`CollateralAlg_Func`](@ref)
  - [`FiniteAllocationInput`](@ref)
  - [`setup_alloc_optim`](@ref)
"""
abstract type AbstractCollateralAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Pays for the long side of a finite allocation with the proceeds of its short sales.

A short sale gives cash, and the long side can spend it. The long side receives its target, less the part of the short target that the short side did not sell, less the fee of the short side. So the net money of the book, long less short with both fees, is at most ``C_{\\text{tot}}\\, b``, whatever the short side sells. It is the default `ca` of [`FiniteAllocationInput`](@ref).

# Mathematical definition

```math
\\begin{align}
C_S^{\\prime} &= C_S\\,, \\\\
C_L^{\\prime} &= \\max\\left(0,\\, C_L - C_S + m_S - F_S\\right)\\,.
\\end{align}
```

``C_L - C_S = C_{\\text{tot}}\\, b`` is the net money of the book. When the short side sells its whole target and pays no fee, ``m_S = C_S`` and ``F_S = 0``, so ``C_L^{\\prime} = C_L``. Each unit of target that the short side does not sell is a unit of cash that the long side does not get. The floor at zero applies to a book with ``b < 0`` whose short side sells less than ``-C_{\\text{tot}}\\, b``. Its long side then buys nothing.

Where:

  - $(math_dict[:C_S_prime_alloc])
  - $(math_dict[:C_L_prime_alloc])
  - $(math_dict[:C_S_alloc])
  - $(math_dict[:C_L_alloc])
  - $(math_dict[:C_tot_alloc])
  - $(math_dict[:b_alloc])
  - $(math_dict[:m_S_alloc])
  - $(math_dict[:F_S_alloc])

# Functor

    (ca::ProceedsCollateral)(w::VecNum, prices::VecNum, cash::Number)
    (ca::ProceedsCollateral)(w::VecNum, prices::VecNum, cash::Number, smoney::Number,
                             sfee::Number)

The first method returns ``C_S^{\\prime}``, with `cash` as ``C_{\\text{tot}}``. The second method returns ``C_L^{\\prime}``, with `smoney` as ``m_S`` and `sfee` as ``F_S``. Neither method reads `prices`.

# Examples

```jldoctest
julia> ca = ProceedsCollateral()
ProceedsCollateral()

julia> ca([1.2, -0.5], [1.0, 1000.0], 100.0)
50.0

julia> ca([1.2, -0.5], [1.0, 1000.0], 100.0, 0.0, 0.0)
70.0
```

# Related

  - [`AbstractCollateralAlgorithm`](@ref)
  - [`CashCollateral`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
struct ProceedsCollateral <: AbstractCollateralAlgorithm end
function (::ProceedsCollateral)(w::VecNum, ::VecNum, cash::Number)
    return cash * sum(x -> max(-x, zero(x)), w)
end
function (::ProceedsCollateral)(w::VecNum, ::VecNum, cash::Number, smoney::Number,
                                sfee::Number)
    # `C_L - C_S` and not `C b`: the difference of the two side targets is exact where the
    # sum of the weights is not, as `100 * (0.9 - 0.8)` is below 10.
    lcash = cash * sum(x -> max(x, zero(x)), w) - cash * sum(x -> max(-x, zero(x)), w) +
            smoney - sfee
    return max(zero(lcash), lcash)
end
"""
$(DocStringExtensions.TYPEDEF)

Caps the money that a finite allocation ties up at a collateral amount.

A short sale gives no cash that the long side can spend, and the short position ties up collateral equal to its money. The long money, the short money and the fees of the two sides are together at most ``K``. The long side takes the collateral that the short side did not use, so it buys more than its target when the short side sells less than its target and ``K`` is large enough. With `amount = nothing`, ``K`` is the cash of the allocation, so the book never ties up more than its cash.

# Mathematical definition

```math
\\begin{align}
C_S^{\\prime} &= \\min\\left(C_S,\\, K\\right)\\,, \\\\
C_L^{\\prime} &= \\max\\left(0,\\, \\min\\left(C_{\\text{tot}} \\left(b_L + b_S\\right),\\, K\\right) - m_S - F_S\\right)\\,.
\\end{align}
```

When ``K \\geq C_{\\text{tot}} (b_L + b_S)``, ``C_L^{\\prime} = C_L + C_S - m_S - F_S``, the long target plus the collateral that the short side did not use. When ``K`` is smaller, ``K`` binds the whole book.

Where:

  - $(math_dict[:C_S_prime_alloc])
  - $(math_dict[:C_L_prime_alloc])
  - $(math_dict[:C_S_alloc])
  - $(math_dict[:C_L_alloc])
  - ``K``: Collateral, `amount`, or ``C_{\\text{tot}}`` when `amount` is `nothing`.
  - $(math_dict[:C_tot_alloc])
  - $(math_dict[:b_L_alloc])
  - $(math_dict[:b_S_alloc])
  - $(math_dict[:m_S_alloc])
  - $(math_dict[:F_S_alloc])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CashCollateral(;
        amount::Option{<:Number} = nothing
    ) -> CashCollateral

Keywords correspond to the struct's fields.

## Validation

  - `amount > 0` when `amount` is not `nothing`, else a `DomainError`.

# Functor

    (ca::CashCollateral)(w::VecNum, prices::VecNum, cash::Number)
    (ca::CashCollateral)(w::VecNum, prices::VecNum, cash::Number, smoney::Number,
                         sfee::Number)

The first method returns ``C_S^{\\prime}``, with `cash` as ``C_{\\text{tot}}``. The second method returns ``C_L^{\\prime}``, with `smoney` as ``m_S`` and `sfee` as ``F_S``. Neither method reads `prices`.

# Examples

```jldoctest
julia> ca = CashCollateral()
CashCollateral
  amount ┴ nothing

julia> ca([1.2, -0.5], [1.0, 1000.0], 100.0)
50.0

julia> ca([1.2, -0.5], [1.0, 1000.0], 100.0, 0.0, 0.0)
100.0
```

# Related

  - [`AbstractCollateralAlgorithm`](@ref)
  - [`ProceedsCollateral`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
@concrete struct CashCollateral <: AbstractCollateralAlgorithm
    """
    Collateral amount, the most money that the book can tie up. `nothing` means the cash of the allocation.
    """
    amount
    function CashCollateral(amount::Option{<:Number})
        if !isnothing(amount)
            @argcheck(amount > zero(amount), DomainError(amount, "amount must be > 0"))
        end
        return new{typeof(amount)}(amount)
    end
end
function CashCollateral(; amount::Option{<:Number} = nothing)::CashCollateral
    return CashCollateral(amount)
end
function (ca::CashCollateral)(w::VecNum, ::VecNum, cash::Number)
    return min(cash * sum(x -> max(-x, zero(x)), w), something(ca.amount, cash))
end
function (ca::CashCollateral)(w::VecNum, ::VecNum, cash::Number, smoney::Number,
                              sfee::Number)
    lcash = min(cash * sum(abs, w), something(ca.amount, cash)) - smoney - sfee
    return max(zero(lcash), lcash)
end
export ProceedsCollateral, CashCollateral
"""
    const CollateralAlg_Func = Union{AbstractCollateralAlgorithm, Function}

Alias for a rule that gives each side of a finite allocation its cash: a rule of the family, or a plain function with the two methods of the family.

It is the bound of the `ca` field of [`FiniteAllocationInput`](@ref). [`AbstractCollateralAlgorithm`](@ref) states the two methods.

# Related

  - [`AbstractCollateralAlgorithm`](@ref)
  - [`UnpricedAlg_Func`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
const CollateralAlg_Func = Union{AbstractCollateralAlgorithm, Function}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that move the target weight of an Unpriced Asset to other assets of a finite allocation.

An Unpriced Asset has a `NaN` price on the allocation date, because it has no bar on that date. An example is an asset that was delisted after the date of the weights. No allocator can buy such an asset. With `upw = nothing`, [`FiniteAllocationInput`](@ref) reads the weight of an Unpriced Asset as a zero weight, so the allocation does not spend it. A rule of this family moves that weight to priced assets of the same side, so the long and the short budget of the book do not change. The rule is the `upw` field of the input, not a field of the allocator, so every allocator of a fallback chain reads one rule.

# Interfaces

To implement a new rule, subtype `AbstractUnpricedWeightAlgorithm` and make it callable with this method:

  - `(upw::MyRule)(w::VecNum, umsk::AbstractVector{Bool}) -> VecNum`: Returns the new target weights.

Where:

  - `w`: Target portfolio weights over the full universe. A negative weight is on the short side.
  - `umsk`: Mask of the Unpriced Assets, in the order of `w`.

The returned vector must have the length of `w`, and a zero at every Unpriced Asset. The allocators check both conditions, and throw an `ArgumentError` when one fails. A plain function with the same signature is also a valid `upw`. [`unpriced_weights`](@ref) states the zero-weight reading of `nothing`, which a rule can call to start from.

# Examples

```jldoctest
julia> struct MyFirstAssetRule <: PortfolioOptimisers.AbstractUnpricedWeightAlgorithm end

julia> function (::MyFirstAssetRule)(w, umsk)
           wt = PortfolioOptimisers.unpriced_weights(nothing, w, umsk)
           wt[1] += sum(w[umsk])
           return wt
       end

julia> fai = FiniteAllocationInput(; w = [0.5, 0.3, 0.2], prices = [10.0, 20.0, NaN],
                                   cash = 1000.0, upw = MyFirstAssetRule());

julia> optimise(GreedyAllocation(), fai).shares
3-element view(::Matrix{Float64}, :, 1) with eltype Float64:
 70.0
 15.0
  0.0
```

# Related

  - [`ProportionalRedistribution`](@ref)
  - [`EvenRedistribution`](@ref)
  - [`AssetRedistribution`](@ref)
  - [`SideRedistribution`](@ref)
  - [`UnpricedAlg_Func`](@ref)
  - [`FiniteAllocationInput`](@ref)
  - [`unpriced_weights`](@ref)
"""
abstract type AbstractUnpricedWeightAlgorithm <: AbstractAlgorithm end
"""
    const UnpricedAlg_Func = Union{AbstractUnpricedWeightAlgorithm, Function}

Alias for a rule that moves the target weight of an Unpriced Asset: a rule of the family, or a plain function with the signature of the family.

It is the bound of the `upw` field of [`FiniteAllocationInput`](@ref), in the form `Option{<:UnpricedAlg_Func}`.

# Related

  - [`AbstractUnpricedWeightAlgorithm`](@ref)
  - [`CollateralAlg_Func`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
const UnpricedAlg_Func = Union{AbstractUnpricedWeightAlgorithm, Function}
"""
    unpriced_side_weight(w::VecNum, umsk::AbstractVector{Bool}, long::Bool)

Find the priced assets of one side of a book, and the target weight of the Unpriced Assets of that side.

A zero weight is on the long side, as in [`setup_alloc_optim`](@ref). The rules of [`AbstractUnpricedWeightAlgorithm`](@ref) call this method once for each side, so the weight of a side goes to assets of that side only.

# Mathematical definition

```math
d_\\sigma = \\sum_{i \\in \\mathcal{U} \\cap \\mathcal{S}_\\sigma} w_i\\,.
```

Where:

  - ``d_\\sigma``: Target weight of the Unpriced Assets of the side ``\\sigma``. It is negative on the short side.
  - ``\\mathcal{U}``: Set of the Unpriced Assets, where `umsk` is `true`.
  - ``\\mathcal{S}_\\sigma``: Set of the assets of the side, ``w_i \\geq 0`` for the long side and ``w_i < 0`` for the short side.
  - $(math_dict[:w_port])

# Arguments

  - `w::VecNum`: Target portfolio weights over the full universe.
  - `umsk::AbstractVector{Bool}`: Mask of the Unpriced Assets.
  - `long::Bool`: `true` for the long side.

# Returns

  - `r::BitVector`: Mask of the priced assets of the side, ``\\mathcal{S}_\\sigma \\setminus \\mathcal{U}``.
  - `d::Number`: The weight ``d_\\sigma``.

# Related

  - [`ProportionalRedistribution`](@ref)
  - [`EvenRedistribution`](@ref)
  - [`AssetRedistribution`](@ref)
"""
function unpriced_side_weight(w::VecNum, umsk::AbstractVector{Bool}, long::Bool)
    side = long ? w .>= zero(eltype(w)) : w .< zero(eltype(w))
    d = sum(view(w, side .& umsk); init = zero(eltype(w)))
    return side .& .!umsk, d
end
"""
$(DocStringExtensions.TYPEDEF)

Moves the target weight of each Unpriced Asset to the priced assets of its side, in proportion to their target weights.

The rule keeps the ratios of the target weights of the priced assets of each side. A priced asset with a zero target weight gets no weight. A side that has no priced asset with a nonzero target weight keeps the weight of its Unpriced Assets as a zero weight, as `upw = nothing` does.

# Mathematical definition

```math
\\tilde{w}_i = \\begin{cases} 0 & i \\in \\mathcal{U}\\,,\\\\ w_i \\left(1 + \\dfrac{d_\\sigma}{s_\\sigma}\\right) & i \\in \\mathcal{R}_\\sigma\\,,\\\\ w_i & \\text{otherwise}\\,, \\end{cases} \\qquad s_\\sigma = \\sum_{j \\in \\mathcal{R}_\\sigma} w_j\\,.
```

``\\mathcal{R}_\\sigma`` is the set of the priced assets of the side ``\\sigma`` with a nonzero target weight. ``s_\\sigma`` and ``d_\\sigma`` have the sign of the side, so the factor is larger than one on both sides, and the sum of the weights of each side does not change.

Where:

  - ``\\tilde{\\boldsymbol{w}}``: New target weights.
  - $(math_dict[:w_port])
  - ``\\mathcal{U}``: Set of the Unpriced Assets.
  - ``d_\\sigma``: Target weight of the Unpriced Assets of the side, from [`unpriced_side_weight`](@ref).
  - ``s_\\sigma``: Target weight of the priced assets of the side.

# Functor

    (upw::ProportionalRedistribution)(w::VecNum, umsk::AbstractVector{Bool})

Returns ``\\tilde{\\boldsymbol{w}}``, with `umsk` as the mask of ``\\mathcal{U}``.

# Examples

```jldoctest
julia> ProportionalRedistribution()([0.5, 0.3, 0.2], [false, false, true])
3-element Vector{Float64}:
 0.625
 0.375
 0.0
```

# Related

  - [`AbstractUnpricedWeightAlgorithm`](@ref)
  - [`EvenRedistribution`](@ref)
  - [`AssetRedistribution`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
struct ProportionalRedistribution <: AbstractUnpricedWeightAlgorithm end
function (::ProportionalRedistribution)(w::VecNum, umsk::AbstractVector{Bool})
    wt = unpriced_weights(nothing, w, umsk)
    for long in (true, false)
        r, d = unpriced_side_weight(w, umsk, long)
        r .&= .!iszero.(w)
        s = sum(view(w, r); init = zero(eltype(w)))
        if !iszero(d) && !iszero(s)
            view(wt, r) .*= (s + d) / s
        end
    end
    return wt
end
"""
$(DocStringExtensions.TYPEDEF)

Moves the target weight of each Unpriced Asset to the priced assets of its side, in equal parts.

Each priced asset of the side with a nonzero target weight gets the same part. A priced asset with a zero target weight gets no part, because a zero weight marks an asset that the optimisation did not hold, such as an asset that left its universe. A side that has no priced asset with a nonzero target weight keeps the weight of its Unpriced Assets as a zero weight, as `upw = nothing` does.

# Mathematical definition

```math
\\tilde{w}_i = \\begin{cases} 0 & i \\in \\mathcal{U}\\,,\\\\ w_i + \\dfrac{d_\\sigma}{\\lvert \\mathcal{R}_\\sigma \\rvert} & i \\in \\mathcal{R}_\\sigma\\,,\\\\ w_i & \\text{otherwise}\\,. \\end{cases}
```

``\\mathcal{R}_\\sigma`` is the set of the priced assets of the side ``\\sigma`` with a nonzero target weight, and ``\\lvert \\mathcal{R}_\\sigma \\rvert`` is its size.

Where:

  - ``\\tilde{\\boldsymbol{w}}``: New target weights.
  - $(math_dict[:w_port])
  - ``\\mathcal{U}``: Set of the Unpriced Assets.
  - ``d_\\sigma``: Target weight of the Unpriced Assets of the side, from [`unpriced_side_weight`](@ref).

# Functor

    (upw::EvenRedistribution)(w::VecNum, umsk::AbstractVector{Bool})

Returns ``\\tilde{\\boldsymbol{w}}``, with `umsk` as the mask of ``\\mathcal{U}``.

# Examples

```jldoctest
julia> EvenRedistribution()([0.5, 0.3, 0.2, 0.0], [false, false, true, false])
4-element Vector{Float64}:
 0.6
 0.4
 0.0
 0.0
```

# Related

  - [`AbstractUnpricedWeightAlgorithm`](@ref)
  - [`ProportionalRedistribution`](@ref)
  - [`AssetRedistribution`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
struct EvenRedistribution <: AbstractUnpricedWeightAlgorithm end
function (::EvenRedistribution)(w::VecNum, umsk::AbstractVector{Bool})
    wt = unpriced_weights(nothing, w, umsk)
    for long in (true, false)
        r, d = unpriced_side_weight(w, umsk, long)
        r .&= .!iszero.(w)
        if !iszero(d) && any(r)
            view(wt, r) .+= d / count(r)
        end
    end
    return wt
end
"""
$(DocStringExtensions.TYPEDEF)

Moves the target weight of each Unpriced Asset to the assets that `idx` names, in equal parts.

The weight of a side goes to the priced assets of `idx` on that side. An asset of `idx` with a zero target weight is on the long side, as in [`setup_alloc_optim`](@ref), so it can take the weight of the long side, for example a fund that stands in for cash. A side that has no priced asset of `idx` keeps the weight of its Unpriced Assets as a zero weight, as `upw = nothing` does. So a long-short book names one asset of each side to move the weight of both sides.

# Mathematical definition

```math
\\tilde{w}_i = \\begin{cases} 0 & i \\in \\mathcal{U}\\,,\\\\ w_i + \\dfrac{d_\\sigma}{\\lvert \\mathcal{R}_\\sigma \\rvert} & i \\in \\mathcal{R}_\\sigma\\,,\\\\ w_i & \\text{otherwise}\\,. \\end{cases}
```

``\\mathcal{R}_\\sigma`` is the set of the priced assets of `idx` on the side ``\\sigma``, and ``\\lvert \\mathcal{R}_\\sigma \\rvert`` is its size.

Where:

  - ``\\tilde{\\boldsymbol{w}}``: New target weights.
  - $(math_dict[:w_port])
  - ``\\mathcal{U}``: Set of the Unpriced Assets.
  - ``d_\\sigma``: Target weight of the Unpriced Assets of the side, from [`unpriced_side_weight`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AssetRedistribution(;
        idx::Union{<:Integer, <:VecInt}
    ) -> AssetRedistribution

Keywords correspond to the struct's fields.

## Validation

  - `idx` is not empty, else an [`IsEmptyError`](@ref).
  - Every entry of `idx` is positive, else a `DomainError`.
  - The functor throws a `BoundsError` when an entry of `idx` is larger than the length of `w`.

# Functor

    (upw::AssetRedistribution)(w::VecNum, umsk::AbstractVector{Bool})

Returns ``\\tilde{\\boldsymbol{w}}``, with `umsk` as the mask of ``\\mathcal{U}``.

# Examples

```jldoctest
julia> AssetRedistribution(; idx = 4)([0.5, 0.3, 0.2, 0.0], [false, false, true, false])
4-element Vector{Float64}:
 0.5
 0.3
 0.0
 0.2
```

# Related

  - [`AbstractUnpricedWeightAlgorithm`](@ref)
  - [`ProportionalRedistribution`](@ref)
  - [`EvenRedistribution`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
@concrete struct AssetRedistribution <: AbstractUnpricedWeightAlgorithm
    """
    Index, or vector of indices, of the assets that take the weight of the Unpriced Assets, in the order of `w`.
    """
    idx
    function AssetRedistribution(idx::Union{<:Integer, <:VecInt})
        @argcheck(!isempty(idx), IsEmptyError("idx cannot be empty"))
        @argcheck(all(>(0), idx), DomainError(idx, "every entry of idx must be > 0"))
        return new{typeof(idx)}(idx)
    end
end
function AssetRedistribution(; idx::Union{<:Integer, <:VecInt})::AssetRedistribution
    return AssetRedistribution(idx)
end
function (upw::AssetRedistribution)(w::VecNum, umsk::AbstractVector{Bool})
    wt = unpriced_weights(nothing, w, umsk)
    tmsk = falses(length(w))
    foreach(i -> tmsk[i] = true, upw.idx)
    for long in (true, false)
        r, d = unpriced_side_weight(w, umsk, long)
        r .&= tmsk
        if !iszero(d) && any(r)
            view(wt, r) .+= d / count(r)
        end
    end
    return wt
end
"""
$(DocStringExtensions.TYPEDEF)

Moves the target weight of the Unpriced Assets of the long side with one rule, and of the short side with another.

A short position that the book cannot open often needs no replacement, while a long position does. This rule composes two rules of the family, one for each side. `nothing` on a side reads the weight of the Unpriced Assets of that side as a zero weight. A zero weight is on the long side, as in [`setup_alloc_optim`](@ref).

The rule reads only the entries of its own side from each rule. The built-in rules move weight inside each side, so for them the result is exact. A plain function that moves weight from one side to the other gives a result that this rule does not define, because the rule of each side loses the entries of the other side.

# Mathematical definition

```math
\\tilde{w}_i = \\begin{cases} \\tilde{w}^{L}_i & w_i \\geq 0\\,,\\\\ \\tilde{w}^{S}_i & w_i < 0\\,. \\end{cases}
```

Where:

  - ``\\tilde{\\boldsymbol{w}}``: New target weights.
  - $(math_dict[:w_port])
  - ``\\tilde{\\boldsymbol{w}}^{L}``, ``\\tilde{\\boldsymbol{w}}^{S}``: New target weights of the rules `l` and `s`, from [`unpriced_weights`](@ref), which also checks each one.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SideRedistribution(;
        l::Option{<:UnpricedAlg_Func} = nothing,
        s::Option{<:UnpricedAlg_Func} = nothing
    ) -> SideRedistribution

Keywords correspond to the struct's fields. With both rules `nothing`, the rule is equal to `upw = nothing`.

## Validation

  - Neither `l` nor `s` is a `SideRedistribution`, else an `ArgumentError`. The rule of a side gives only the entries of that side, so a nested rule adds nothing.

# Functor

    (upw::SideRedistribution)(w::VecNum, umsk::AbstractVector{Bool})

Returns ``\\tilde{\\boldsymbol{w}}``, with `umsk` as the mask of the Unpriced Assets.

# Examples

```jldoctest
julia> SideRedistribution(; l = ProportionalRedistribution())([0.6, 0.3, 0.3, -0.1, -0.1],
                                                              [false, false, true, false, true])
5-element Vector{Float64}:
  0.8
  0.4
  0.0
 -0.1
  0.0
```

# Related

  - [`AbstractUnpricedWeightAlgorithm`](@ref)
  - [`ProportionalRedistribution`](@ref)
  - [`EvenRedistribution`](@ref)
  - [`AssetRedistribution`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
@concrete struct SideRedistribution <: AbstractUnpricedWeightAlgorithm
    """
    Rule for the weight of the Unpriced Assets of the long side, or `nothing` for a zero weight.
    """
    l
    """
    Rule for the weight of the Unpriced Assets of the short side, or `nothing` for a zero weight.
    """
    s
    function SideRedistribution(l::Option{<:UnpricedAlg_Func},
                                s::Option{<:UnpricedAlg_Func})
        @argcheck(!isa(l, SideRedistribution) && !isa(s, SideRedistribution),
                  ArgumentError("a SideRedistribution cannot hold a SideRedistribution: each side reads only its own entries, so the nested rule adds nothing"))
        return new{typeof(l), typeof(s)}(l, s)
    end
end
function SideRedistribution(; l::Option{<:UnpricedAlg_Func} = nothing,
                            s::Option{<:UnpricedAlg_Func} = nothing)::SideRedistribution
    return SideRedistribution(l, s)
end
function (upw::SideRedistribution)(w::VecNum, umsk::AbstractVector{Bool})
    lw = unpriced_weights(upw.l, w, umsk)
    sw = unpriced_weights(upw.s, w, umsk)
    return ifelse.(w .>= zero(eltype(w)), lw, sw)
end
"""
    unpriced_weights(::Nothing, w::VecNum, umsk::AbstractVector{Bool})
    unpriced_weights(upw::UnpricedAlg_Func, w::VecNum, umsk::AbstractVector{Bool})

Give the target weights that a finite allocation buys when some of its assets are Unpriced Assets.

The method on `nothing` sets the weight of each Unpriced Asset to zero, and changes no other weight. The allocation then reads an Unpriced Asset as an asset with a zero target weight, so it does not spend its weight. The method on a rule calls the rule, and checks the vector that it returns. [`prepare_finite_allocation`](@ref) calls this method only when the prices hold a `NaN`.

The method on `nothing` returns a vector of `float_if_integer(eltype(w))`, so a rule that starts from it can add a fraction to integer weights.

# Validation

  - The vector of a rule has the length of `w`, and a zero at every Unpriced Asset, else an `ArgumentError`.

# Arguments

  - `upw`: The rule of the `upw` field of [`FiniteAllocationInput`](@ref), or `nothing`.
  - `w::VecNum`: Target portfolio weights over the full universe.
  - `umsk::AbstractVector{Bool}`: Mask of the Unpriced Assets, in the order of `w`.

# Returns

  - `wt::VecNum`: The new target weights.

# Related

  - [`AbstractUnpricedWeightAlgorithm`](@ref)
  - [`UnpricedAlg_Func`](@ref)
  - [`prepare_finite_allocation`](@ref)
"""
function unpriced_weights(::Nothing, w::VecNum, umsk::AbstractVector{Bool})
    wt = zeros(float_if_integer(eltype(w)), length(w))
    pmsk = .!umsk
    wt[pmsk] = view(w, pmsk)
    return wt
end
function unpriced_weights(upw::UnpricedAlg_Func, w::VecNum, umsk::AbstractVector{Bool})
    wt = upw(w, umsk)
    @argcheck(length(wt) == length(w) && all(iszero, view(wt, umsk)),
              ArgumentError("the rule `upw` must return a weight vector of length $(length(w)) with a zero at every unpriced asset $(findall(umsk))"))
    return wt
end
export ProportionalRedistribution, EvenRedistribution, AssetRedistribution,
       SideRedistribution
public unpriced_weights
"""
$(DocStringExtensions.TYPEDEF)

Holds the target weights, prices and cash that a finite allocation reads.

[`optimise`](@ref) takes it as the second argument, with a [`DiscreteAllocation`](@ref) or a [`GreedyAllocation`](@ref). It also holds the cash before the trade, an optional horizon, fee and Investable Mask, and the collateral algorithm that gives each side of the book its cash. It subtypes [`AbstractEstimator`](@ref) and not [`OptimisationResult`](@ref), because it is the input of an allocation. The methods that dispatch on a result, such as the plots and the result [`factory`](@ref), cannot read its fields.

An optimisation over a reduced universe solves on its Investable Mask, and expands the weights back to the full universe. Its result then has a full-length `w` and a fee on two reduced axes. `imsk` records the reduction. [`allocation_side_fees`](@ref) calls [`lift_fees`](@ref) to put the fee back on the axis of `w` and `prices`, so the allocation charges the forced exit of an asset that left the universe on the money that the exit sold. The fee records its mask in its own `imsk` field too, and the constructor makes the two agree through [`mark_fees`](@ref). A stated `imsk` marks a fee that has no mask. A marked fee gives its mask when `imsk` is `nothing`. The constructor refuses two masks that differ.

An Unpriced Asset has a `NaN` price, the price of an asset with no bar on the allocation date. Both allocators solve each side over its priced assets only, so they buy no Unpriced Asset, and its shares, cost and realised weight are zero. Its price reaches no other number of the result. `upw` states what its target weight becomes. With `upw = nothing`, the weight is read as a zero weight. A rule of [`AbstractUnpricedWeightAlgorithm`](@ref) moves it to priced assets of the same side. A held Unpriced Asset ends the trade with no money, so its turnover fee charges the sale of the whole previous position, as [`allocation_side_fees`](@ref) states.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FiniteAllocationInput(;
        w::VecNum,
        prices::VecNum,
        cash::Number = 1e6,
        prev_cash::Number = cash,
        horizon::Option{<:Number} = nothing,
        fees::Option{<:Fees} = nothing,
        imsk::Option{<:BitVector} = nothing,
        ca::CollateralAlg_Func = ProceedsCollateral(),
        upw::Option{<:UnpricedAlg_Func} = nothing
    ) -> FiniteAllocationInput

Keywords correspond to the struct's fields.

    FiniteAllocationInput(
        res::NonFiniteAllocationOptimisationResult;
        prices::VecNum,
        cash::Number = 1e6,
        prev_cash::Number = cash,
        w::Option{<:VecNum} = nothing,
        horizon::Option{<:Number} = nothing,
        fees::Option{<:Fees} = nothing,
        imsk::Option{<:BitVector} = nothing,
        ca::CollateralAlg_Func = ProceedsCollateral(),
        upw::Option{<:UnpricedAlg_Func} = nothing
    ) -> FiniteAllocationInput

Reads from a fitted optimisation every value that an allocation can take from it, so the caller states only the prices and the cash. A stated keyword takes priority. A value that the result does not carry becomes `nothing`, so one call serves every optimisation family:

  - `w` from `res.w`, which every optimisation result carries on the full universe.
  - `fees` through [`extract_fees`](@ref), from the `fees` property when the result has one.
  - `imsk` through [`result_investable_mask`](@ref), which gives `nothing` when the optimisation did not reduce its universe.
  - `horizon` through [`allocation_horizon`](@ref), the number of observations of the result's prior. An equal weighted optimisation without a prior gives none. The constructor then needs a stated horizon, but only when the input also has a fee.

## Validation

  - `!isempty(w)` and `!isempty(prices)`, else an [`IsEmptyError`](@ref).
  - `length(w) == length(prices)`, else a `DimensionMismatch`.
  - No entry of `prices` is infinite, else a `DomainError`. An absent price is `NaN`, as the ingestion layer spells it.
  - `cash > 0`, else a `DomainError`.
  - `prev_cash >= 0`, else a `DomainError`.
  - `horizon` is not `nothing` when `fees` is not `nothing`, else an [`IsNothingError`](@ref).
  - The mask of `fees` is `nothing` or equal to `imsk`. [`mark_fees`](@ref) throws an `ArgumentError` otherwise.
  - `imsk`, stated or read from the fee, has `length(imsk) == length(w)`, else a `DimensionMismatch`, and `any(imsk)`, else an [`IsEmptyError`](@ref).

# Examples

```jldoctest
julia> FiniteAllocationInput(; w = [0.6, 0.4], prices = [10.0, 20.0], cash = 1000.0)
FiniteAllocationInput
          w ┼ Vector{Float64}: [0.6, 0.4]
     prices ┼ Vector{Float64}: [10.0, 20.0]
       cash ┼ Float64: 1000.0
  prev_cash ┼ Float64: 1000.0
    horizon ┼ nothing
       fees ┼ nothing
       imsk ┼ nothing
         ca ┼ ProceedsCollateral()
        upw ┴ nothing
```

# Related

  - [`DiscreteAllocation`](@ref)
  - [`GreedyAllocation`](@ref)
  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`allocation_horizon`](@ref)
  - [`allocation_side_fees`](@ref)
  - [`extract_fees`](@ref)
  - [`lift_fees`](@ref)
  - [`mark_fees`](@ref)
  - [`result_investable_mask`](@ref)
  - [`setup_alloc_optim`](@ref)
  - [`AbstractCollateralAlgorithm`](@ref)
  - [`AbstractUnpricedWeightAlgorithm`](@ref)
  - [`prepare_finite_allocation`](@ref)
  - [`optimise`](@ref)
"""
@concrete struct FiniteAllocationInput <: AbstractEstimator
    """
    Target portfolio weights, the continuous weights that the allocation turns into share counts.
    """
    w
    """
    Asset prices, in the order of `w`.
    """
    prices
    """
    Cash of the allocation, which the long and the short side share.
    """
    cash
    """
    Cash that the portfolio held before the trade. The money of a position before the trade is `prev_cash` times its previous weight, which `fees.tn.w` and `fees.lq.w` hold. The default is `cash`.
    """
    prev_cash
    """
    Horizon of the fee, in periods. `l`, `s`, `tn` and `lq` are rates per period, so the allocation charges each of them `horizon` times. `fl`, `fs` and `flq` are charged one time. It can be `nothing` only when `fees` is `nothing`.
    """
    horizon
    """
    Fee of the allocation, or `nothing`. A fee that an optimisation reduced to its Investable Mask is on two reduced axes, and its own `imsk` records the mask that puts it back on the axis of `w`.
    """
    fees
    """
    Investable Mask of the optimisation that gave `w`, `true` for each asset that it traded. `nothing` means that the weights and the fee are on the full universe. The constructor makes it agree with the mask of the fee through [`mark_fees`](@ref).
    """
    imsk
    """
    Collateral algorithm, which gives the short side its cash, and the long side its cash after the short side trades. A rule of [`AbstractCollateralAlgorithm`](@ref), or a plain function with its two methods. The default [`ProceedsCollateral`](@ref) pays for the long side with the proceeds of the short sales.
    """
    ca
    """
    Rule for the target weight of an Unpriced Asset, an asset whose price is not finite. `nothing` reads the weight as a zero weight, so the allocation does not spend it. A rule of [`AbstractUnpricedWeightAlgorithm`](@ref), or a plain function with its signature, moves the weight to priced assets of the same side.
    """
    upw
    function FiniteAllocationInput(w::VecNum, prices::VecNum, cash::Number,
                                   prev_cash::Number, horizon::Option{<:Number},
                                   fees::Option{<:Fees}, imsk::Option{<:BitVector},
                                   ca::CollateralAlg_Func, upw::Option{<:UnpricedAlg_Func})
        @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
        @argcheck(!isempty(prices), IsEmptyError("prices cannot be empty"))
        @argcheck(length(w) == length(prices),
                  DimensionMismatch("w ($(length(w))) must match prices ($(length(prices)))"))
        @argcheck(!any(isinf, prices),
                  DomainError(prices,
                              "an infinite value is neither a price nor the marker of an absence: spell an absent price `NaN`"))
        @argcheck(cash > zero(cash), DomainError(cash, "cash must be > 0"))
        @argcheck(prev_cash >= zero(prev_cash),
                  DomainError(prev_cash, "prev_cash must be >= 0"))
        if !isnothing(fees)
            @argcheck(!isnothing(horizon),
                      IsNothingError("horizon cannot be nothing when fees are provided"))
        end
        fees, imsk = mark_fees(fees, imsk)
        if !isnothing(imsk)
            @argcheck(length(imsk) == length(w),
                      DimensionMismatch("imsk ($(length(imsk))) must match w ($(length(w)))"))
            @argcheck(any(imsk),
                      IsEmptyError("imsk must keep at least one asset, and it keeps none"))
        end
        return new{typeof(w), typeof(prices), typeof(cash), typeof(prev_cash),
                   typeof(horizon), typeof(fees), typeof(imsk), typeof(ca), typeof(upw)}(w,
                                                                                         prices,
                                                                                         cash,
                                                                                         prev_cash,
                                                                                         horizon,
                                                                                         fees,
                                                                                         imsk,
                                                                                         ca,
                                                                                         upw)
    end
end
function FiniteAllocationInput(; w::VecNum, prices::VecNum, cash::Number = 1e6,
                               prev_cash::Number = cash,
                               horizon::Option{<:Number} = nothing,
                               fees::Option{<:Fees} = nothing,
                               imsk::Option{<:BitVector} = nothing,
                               ca::CollateralAlg_Func = ProceedsCollateral(),
                               upw::Option{<:UnpricedAlg_Func} = nothing)::FiniteAllocationInput
    return FiniteAllocationInput(w, prices, cash, prev_cash, horizon, fees, imsk, ca, upw)
end
function FiniteAllocationInput(res::NonFiniteAllocationOptimisationResult; prices::VecNum,
                               cash::Number = 1e6, prev_cash::Number = cash,
                               w::Option{<:VecNum} = nothing,
                               horizon::Option{<:Number} = nothing,
                               fees::Option{<:Fees} = nothing,
                               imsk::Option{<:BitVector} = nothing,
                               ca::CollateralAlg_Func = ProceedsCollateral(),
                               upw::Option{<:UnpricedAlg_Func} = nothing)::FiniteAllocationInput
    # A stated keyword wins over the result, and each reader answers `nothing` on a result
    # carrying no such object, so a family holding fewer of them takes the same call.
    return FiniteAllocationInput(isnothing(w) ? res.w : w, prices, cash, prev_cash,
                                 allocation_horizon(res, horizon), extract_fees(res, fees),
                                 isnothing(imsk) ? result_investable_mask(res) : imsk, ca,
                                 upw)
end
export FiniteAllocationInput
public AbstractCollateralAlgorithm, AbstractUnpricedWeightAlgorithm
"""
    allocation_horizon(res::NonFiniteAllocationOptimisationResult,
                       horizon::Option{<:Number} = nothing)

Read the horizon of a finite allocation from an optimisation result.

The horizon is the number of observations of the result's prior, which is the number of periods over which [`calc_total_fees`](@ref) charges the rates on the window that the fit saw. A stated `horizon` takes priority. The method reads the prior with `hasproperty` and not by dispatch, as [`extract_fees`](@ref) reads `fees`. The families that keep a prior keep it under the name `pr`, and a result that keeps none gives `nothing`. [`FiniteAllocationInput`](@ref) then needs a stated horizon, but only when the input also has a fee.

# Algorithm

 1. If `horizon` is not `nothing`, return it.
 2. If `res` has a `pr` property, and `pr` is not `nothing` and has an `X` property, return `size(pr.X, 1)`.
 3. Otherwise return `nothing`.

# Arguments

  - `res`: Fitted optimisation result, which can have a `pr` property.
  - `horizon`: Horizon that the caller states, or `nothing`.

# Returns

  - `horizon::Option{<:Number}`: The horizon in periods, or `nothing` when the method finds none.

# Related

  - [`FiniteAllocationInput`](@ref)
  - [`extract_fees`](@ref)
  - [`calc_total_fees`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function allocation_horizon(res::NonFiniteAllocationOptimisationResult,
                            horizon::Option{<:Number} = nothing)
    if isnothing(horizon) && hasproperty(res, :pr)
        pr = res.pr
        if !isnothing(pr) && hasproperty(pr, :X)
            horizon = size(pr.X, 1)
        end
    end
    return horizon
end
"""
    factory(res::FiniteAllocationOptimisationResult, fb::Option{<:FOptE_FOpt_FbChain})

Rebuild a finite allocation result with a new fallback record `fb`.

Every finite allocation result has `fb` as its last field, as the continuous results do. [`optimise`](@ref) calls this method with the [`FbChain`](@ref) of the failed attempts when a fallback answered. A concrete result type can define its own method when a rebuild needs more than a new `fb`.

# Algorithm

 1. Read every field of `res`, in order, into the tuple `flds`.
 2. Call the constructor of the type of `res`, without its type parameters, with every entry of `flds` except the last, and then `fb`.

# Related

  - [`FOptE_FOpt_FbChain`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
"""
function factory(res::FiniteAllocationOptimisationResult, fb::Option{<:FOptE_FOpt_FbChain})
    flds = ntuple(i -> getfield(res, i), Val(fieldcount(typeof(res))))
    return (typeof(res).name.wrapper)(Base.front(flds)..., fb)
end

"""
    setup_alloc_optim(w::VecNum)

Split a portfolio into its long and its short side.

Both finite allocators solve one sub-problem per side. A zero weight is on the long side. The method gives no side its cash. The `ca` of [`FiniteAllocationInput`](@ref) does that, through the two methods of [`AbstractCollateralAlgorithm`](@ref). The method charges no fee. Each sub-problem charges the fee of its own side on the money that it buys, through [`allocation_fee`](@ref) or [`set_allocation_fees!`](@ref).

# Mathematical definition

```math
\\begin{align}
b_L &= \\sum_{i:\\, w_i \\geq 0} w_i\\,, \\\\
b_S &= -\\sum_{i:\\, w_i < 0} w_i\\,.
\\end{align}
```

It follows that ``b = b_L - b_S``.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:N])
  - $(math_dict[:b_alloc])
  - $(math_dict[:b_L_alloc])
  - $(math_dict[:b_S_alloc])

# Algorithm

 1. Compute `lidx`, the mask `w .>= 0`.
 2. If every weight is non-negative, set `lbgt` to the sum of `w`, set `sbgt` to zero, and set `sidx` to an empty vector.
 3. Otherwise set `sidx = .!lidx`, and compute `lbgt` and `sbgt` over the two masks.

# Arguments

  - `w::VecNum`: Target portfolio weights over the full universe.

# Returns

  - `lbgt::Number`: Long budget ``b_L``.
  - `sbgt::Number`: Short budget ``b_S``, which is non-negative.
  - `lidx`: Mask of the long side, `w .>= 0`.
  - `sidx`: Mask of the short side. When every weight is non-negative, it is an empty vector of the element type of `w`. Both forms select no asset in that case.

# Related

  - [`AbstractCollateralAlgorithm`](@ref)
  - [`allocation_side_fees`](@ref)
  - [`finite_sub_allocation`](@ref)
  - [`finite_sub_allocation!`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
function setup_alloc_optim(w::VecNum)
    lidx = w .>= zero(eltype(w))
    if all(lidx)
        lbgt = sum(w)
        sbgt = zero(eltype(w))
        sidx = Vector{eltype(w)}(undef, 0)
    else
        sidx = .!lidx
        lbgt = sum(view(w, lidx))
        sbgt = -sum(view(w, sidx))
    end
    return lbgt, sbgt, lidx, sidx
end
"""
    prepare_finite_allocation(fai::FiniteAllocationInput)

Give the target weights, the two sides and the two charges that both finite allocators solve over.

Both allocators call this method first, so they read one definition of an Unpriced Asset, one rule for its weight, and one fee. A `NaN` price marks an Unpriced Asset, and the method removes each one from its side. So no sub-problem reads its price.

# Algorithm

 1. Compute `umsk`, the mask of the `NaN` prices.
 2. If `umsk` marks an asset, replace `w` with [`unpriced_weights`](@ref) of `fai.upw`.
 3. Split `w` into its two sides with [`setup_alloc_optim`](@ref). An Unpriced Asset now has a zero weight, so it is on the long side, and the method removes it from `lidx`.
 4. Split the fee into the charge of each side with [`allocation_side_fees`](@ref), which charges the Unpriced Assets as a constant of the long side.

# Arguments

  - `fai::FiniteAllocationInput`: The input of the allocation.

# Returns

  - `w::VecNum`: Target weights that the allocation buys, with a zero at every Unpriced Asset.
  - `lbgt::Number`: Long budget of `w`.
  - `sbgt::Number`: Short budget of `w`.
  - `lidx`: Mask of the priced assets of the long side.
  - `sidx`: Mask of the short side.
  - `lsf`: The long side's charge, or `nothing`.
  - `ssf`: The short side's charge, or `nothing`.

# Related

  - [`FiniteAllocationInput`](@ref)
  - [`AbstractUnpricedWeightAlgorithm`](@ref)
  - [`unpriced_weights`](@ref)
  - [`setup_alloc_optim`](@ref)
  - [`allocation_side_fees`](@ref)
"""
function prepare_finite_allocation(fai::FiniteAllocationInput)
    w = fai.w
    umsk = isnan.(fai.prices)
    if any(umsk)
        w = unpriced_weights(fai.upw, w, umsk)
    end
    lbgt, sbgt, lidx, sidx = setup_alloc_optim(w)
    lidx = lidx .& .!umsk
    lsf, ssf = allocation_side_fees(fai.fees, fai.horizon, fai.prev_cash, lidx, sidx, umsk)
    return w, lbgt, sbgt, lidx, sidx, lsf, ssf
end
"""
    allocation_turnover_money(::Nothing, ::Number, ::Any, ::Bool)
    allocation_turnover_money(tn::Turnover, prev_cash::Number, idx, short::Bool)

Give one side its turnover rate and the money that each of its positions held before the trade.

The allocator solves the short side on its negated weights, so the method negates the short side's previous money too. A position that is on the same side before and after the trade has a non-negative previous money. A position that changes side has a negative one, and the turnover term ``|\\boldsymbol{m} - \\boldsymbol{m}^{\\text{prev}}|`` of [`allocation_fee`](@ref) is then the money that closes the old position plus the money of the new one. A `nothing` `tn` gives `nothing` for both values.

# Mathematical definition

```math
\\boldsymbol{m}^{\\text{prev}} = s\\, C^{\\text{prev}}\\, \\boldsymbol{w}^{\\text{prev}}\\,.
```

Where:

  - $(math_dict[:m_prev_alloc])
  - ``s``: Sign of the side, ``1`` for the long side and ``-1`` for the short side.
  - $(math_dict[:C_prev_alloc])
  - ``\\boldsymbol{w}^{\\text{prev}}``: Previous weights of the assets of this side, `tn.w` viewed to `idx`.

# Arguments

  - `tn`: Turnover of the fee, or `nothing` when the fee has none.
  - `prev_cash::Number`: Cash that the portfolio held before the trade.
  - `idx`: Mask of this side.
  - `short::Bool`: `true` for the short side.

# Returns

  - `val`: Turnover rate of this side, or `nothing`.
  - `prev_money`: Previous money ``\\boldsymbol{m}^{\\text{prev}}`` of this side, or `nothing`.

# Related

  - [`allocation_side_fees`](@ref)
  - [`allocation_fee`](@ref)
  - [`Turnover`](@ref)
"""
function allocation_turnover_money(::Nothing, ::Number, ::Any, ::Bool)
    return nothing, nothing
end
function allocation_turnover_money(tn::Turnover, prev_cash::Number, idx, short::Bool)
    sgn = ifelse(short, -one(prev_cash), one(prev_cash))
    return nothing_scalar_array_view(tn.val, idx), sgn * prev_cash * view(tn.w, idx)
end
"""
    allocation_liquidation_fee(::Nothing, ::Number, ::Number)
    allocation_liquidation_fee(fees::Fees, T::Number, prev_cash::Number)

Charge the forced exit of every asset that left the universe, as one constant.

A [`Fees`](@ref) has two liquidation charges, `lq` and `flq`. `lq` is a rate and `flq` is an amount of money, and the `w` of each holds the previous weights of the positions that the optimisation had to sell. The allocator solves no share count for these assets, so the charge needs no variable and is a constant of the sub-problem. [`set_liquidation_fees!`](@ref) and [`set_fixed_liquidation_fees!`](@ref) add the same charge to the JuMP model as a constant.

The long sub-problem pays the whole charge. Every asset that left has a zero target weight, and [`setup_alloc_optim`](@ref) puts a zero weight on the long side. The total fee does not depend on the side that pays, but that side has less cash for shares, and a long-only book has no other side.

# Mathematical definition

```math
L = T\\, C^{\\text{prev}} \\left\\lvert \\boldsymbol{w}_{\\text{lq}} \\right\\rvert \\cdot \\boldsymbol{f}_{\\text{lq}} + 1\\left\\{\\boldsymbol{w}_{\\text{flq}} \\neq 0\\right\\} \\cdot \\boldsymbol{f}_{\\text{flq}}\\,.
```

This is ``T F_{\\text{lq}} + F_{\\text{flq}}`` of [`Fees`](@ref), with the rate term in money.

Where:

  - $(math_dict[:L_liq_alloc])
  - $(math_dict[:T_alloc])
  - $(math_dict[:C_prev_alloc])
  - ``\\boldsymbol{w}_{\\text{lq}}``, ``\\boldsymbol{w}_{\\text{flq}}``: Previous weights of the assets that left, `lq.w` and `flq.w`.
  - ``\\boldsymbol{f}_{\\text{lq}}``: Liquidation rate per period, `lq.val`. A scalar applies to every asset.
  - ``\\boldsymbol{f}_{\\text{flq}}``: Fixed liquidation amount per position, `flq.val`. A scalar applies to every asset.
  - ``1\\left\\{\\cdot\\right\\}``: Elementwise indicator, `1` where the condition holds and `0` elsewhere. ``\\boldsymbol{w}_{\\text{flq}} \\neq 0`` is `!isapprox(w, 0; kwargs...)` with the `kwargs` of the fee.

# Algorithm

 1. If `fees` is `nothing`, return zero, because no asset left the universe.
 2. Set `fee` to zero.
 3. If `lq` is not `nothing`, add `T` times the contraction of `lq.val` with `abs.(prev_cash * lq.w)`.
 4. If `flq` is not `nothing`, add [`calc_fixed_liquidation_fees`](@ref) of `flq`, which charges the long and the short positions that left.
 5. Return `fee`.

# Arguments

  - `fees`: Fee with the two liquidation charges, `lq` and `flq`, or `nothing`.
  - `T::Number`: Horizon, in periods.
  - `prev_cash::Number`: Cash that the portfolio held before the trade.

# Returns

  - `fee::Number`: The forced-exit charge ``L`` over the whole horizon.

# Related

  - [`allocation_side_fees`](@ref)
  - [`allocation_turnover_money`](@ref)
  - [`calc_fixed_liquidation_fees`](@ref)
  - [`calc_liquidation_fees`](@ref)
  - [`Fees`](@ref)
"""
function allocation_liquidation_fee(::Nothing, T::Number, prev_cash::Number)
    return zero(promote_type(typeof(T), typeof(prev_cash)))
end
function allocation_liquidation_fee(fees::Fees, T::Number, prev_cash::Number)
    fee = zero(promote_type(typeof(T), typeof(prev_cash)))
    lq = fees.lq
    if !isnothing(lq)
        # Per period: the rate contracted with the money the forced exit sold. `lq.w` is a
        # weight, so that money is `prev_cash * lq.w`, the rule `tn` is read under.
        fee += T * dot_scalar(lq.val, abs.(prev_cash * lq.w))
    end
    flq = fees.flq
    if !isnothing(flq)
        # One time: a currency amount, charged once per position the exit sold. Both sides
        # of the book are charged, which is what `calc_fixed_liquidation_fees` sums.
        fee += calc_fixed_liquidation_fees(flq, fees.kwargs)
    end
    return fee
end
"""
    allocation_side_fees(::Nothing, ::Option{<:Number}, ::Number, ::Any, ::Any, ::Any)
    allocation_side_fees(fees::Fees, T::Number, prev_cash::Number, lidx, sidx, umsk)

Split a fee into the charge of the long side and the charge of the short side.

Each sub-problem charges its own side. The long side takes `l` and `fl`, the short side takes `s` and `fs`, and both take the turnover rate and their previous money from [`allocation_turnover_money`](@ref). [`nothing_scalar_array_view`](@ref) views a vector rate to the side and passes a scalar rate through. A `nothing` `fees` gives `nothing` for both charges.

`lidx` and `sidx` are masks of the full universe, because they come from weights that an optimisation expanded back to it. A fee that the same optimisation reduced to its Investable Mask is on a shorter axis and records the mask in `imsk`, so the method lifts it with [`lift_fees`](@ref) before it takes a view. [`lift_fees`](@ref) returns a fee that has no mask unchanged.

The forced exit of [`allocation_liquidation_fee`](@ref) is a constant of the whole allocation. It is the `liq` entry of the long side's charge, and the short side's `liq` is zero, so both charges have the same entries.

An Unpriced Asset is on neither side, because no sub-problem can buy it. It ends the trade with no money, so its fee is a constant too, and the long side's `liq` adds it. The method charges it with [`allocation_fee`](@ref) on zero money, so it is the turnover term alone, ``T \\left\\lvert \\boldsymbol{m}^{\\text{prev}} \\right\\rvert \\cdot \\boldsymbol{f}_{\\text{Tn}}`` over the Unpriced Assets. The proportional and the fixed term of zero money are zero.

# Algorithm

 1. Lift `fees` onto the full universe with [`lift_fees`](@ref).
 2. Compute `ltn_val` and `lprev` for the long side, and `stn_val` and `sprev` for the short side, with [`allocation_turnover_money`](@ref).
 3. Compute `liq`, the forced-exit charge, with [`allocation_liquidation_fee`](@ref).
 4. Add to `liq` the [`allocation_fee`](@ref) of zero money in the Unpriced Assets, with their turnover rate and previous money from [`allocation_turnover_money`](@ref).
 5. Return the long side's charge, from `l`, `fl`, `ltn_val`, `lprev` and `liq`, and the short side's charge, from `s`, `fs`, `stn_val`, `sprev` and a zero `liq`.

# Arguments

  - `fees`: Fee to split, or `nothing`.
  - `T`: Horizon, in periods.
  - `prev_cash::Number`: Cash that the portfolio held before the trade.
  - `lidx`: Mask of the long side. It holds no Unpriced Asset.
  - `sidx`: Mask of the short side. It holds no Unpriced Asset.
  - `umsk`: Mask of the Unpriced Assets.

# Returns

  - `lsf`: The long side's charge, or `nothing`.
  - `ssf`: The short side's charge, or `nothing`.

Each charge is a named tuple of `T`, `prop`, `fixed`, `tn_val`, `prev_money` and `liq`.

# Related

  - [`allocation_liquidation_fee`](@ref)
  - [`allocation_turnover_money`](@ref)
  - [`allocation_fee`](@ref)
  - [`lift_fees`](@ref)
  - [`set_allocation_fees!`](@ref)
  - [`setup_alloc_optim`](@ref)
  - [`Fees`](@ref)
"""
function allocation_side_fees(::Nothing, ::Option{<:Number}, ::Number, ::Any, ::Any, ::Any)
    return nothing, nothing
end
function allocation_side_fees(fees::Fees, T::Number, prev_cash::Number, lidx, sidx, umsk)
    # `lidx` and `sidx` index the full universe, so the fee comes onto that axis first.
    fees = lift_fees(fees)
    ltn_val, lprev = allocation_turnover_money(fees.tn, prev_cash, lidx, false)
    stn_val, sprev = allocation_turnover_money(fees.tn, prev_cash, sidx, true)
    liq = allocation_liquidation_fee(fees, T, prev_cash)
    # An unpriced asset holds no money after the trade, whatever it held before. Its price
    # never enters: the charge reads zero money, so only the sale of the old position costs.
    utn_val, uprev = allocation_turnover_money(fees.tn, prev_cash, umsk, false)
    z = fill(zero(liq), count(umsk))
    liq = allocation_fee((T = T, prop = nothing, fixed = nothing, tn_val = utn_val,
                          prev_money = uprev, liq = liq), z, z)
    return ((T = T, prop = nothing_scalar_array_view(fees.l, lidx),
             fixed = nothing_scalar_array_view(fees.fl, lidx), tn_val = ltn_val,
             prev_money = lprev, liq = liq),
            (T = T, prop = nothing_scalar_array_view(fees.s, sidx),
             fixed = nothing_scalar_array_view(fees.fs, sidx), tn_val = stn_val,
             prev_money = sprev, liq = zero(liq)))
end
"""
    allocation_fee(::Nothing, ::VecNum, shares::VecNum)
    allocation_fee(sf::NamedTuple, p::VecNum, shares::VecNum)

Charge the whole fee of one side on a vector of share counts.

The method charges every term on the money in each position, which is exact for share counts. Both allocators report this number. A `nothing` `sf` charges zero.

A side that held money before the trade owes a turnover fee when it buys nothing, because a sale to zero is a trade. A book whose universe lost an asset owes the forced exit whatever it buys. For these reasons, both allocators charge this method on the empty book too. On the long side, ``L`` also holds the turnover fee of the Unpriced Assets, which [`allocation_side_fees`](@ref) charges on zero money.

# Mathematical definition

```math
F(\\boldsymbol{x}) = L + T \\left(\\boldsymbol{m} \\cdot \\boldsymbol{f}_{\\text{p}} + \\left\\lvert \\boldsymbol{m} - \\boldsymbol{m}^{\\text{prev}} \\right\\rvert \\cdot \\boldsymbol{f}_{\\text{Tn}}\\right) + 1\\left\\{\\boldsymbol{m} \\neq 0\\right\\} \\cdot \\boldsymbol{f}_{\\text{f}}\\,.
```

The sum of the two sides is the fee of [`Fees`](@ref) on the weights of the book. Sign the money of the short side negative, and set ``\\boldsymbol{w} = \\boldsymbol{m} / C^{\\text{prev}}``. The sum is then ``C^{\\text{prev}}\\, T F_{\\text{r}}(\\boldsymbol{w}) + F_{\\text{o}}(\\boldsymbol{w})``, which [`calc_periodic_fees`](@ref) and [`calc_one_off_fees`](@ref) compute.

Where:

  - $(math_dict[:F_side_fee])
  - ``\\boldsymbol{x}``: Share count vector of one side, `shares`.
  - $(math_dict[:p_prices])
  - $(math_dict[:m_money])
  - $(math_dict[:m_prev_alloc])
  - $(math_dict[:L_liq_alloc])
  - $(math_dict[:T_alloc])
  - ``\\boldsymbol{f}_{\\text{p}}``: Proportional rate per period of the side, `l` on the long side and `s` on the short side.
  - ``\\boldsymbol{f}_{\\text{Tn}}``: Turnover rate per period, `tn.val`.
  - ``\\boldsymbol{f}_{\\text{f}}``: Fixed amount per position of the side, `fl` on the long side and `fs` on the short side.
  - ``1\\left\\{\\cdot\\right\\}``: Elementwise indicator, `1` where the condition holds and `0` elsewhere.
  - $(math_dict[:C_prev_alloc])
  - ``F_{\\text{r}}``, ``F_{\\text{o}}``: Fee per period and one-off fee of [`Fees`](@ref).
  - ``\\boldsymbol{w}``: Signed weights of the book on the previous cash.

A scalar rate applies to every asset.

# Algorithm

 1. Compute `money = shares .* p`.
 2. Set `fee` to `sf.liq`.
 3. If `prop` is not `nothing`, add `T` times its contraction with `money`.
 4. If `tn_val` is not `nothing`, add `T` times its contraction with `abs.(money - prev_money)`.
 5. If `fixed` is not `nothing`, add its contraction with the indicator `.!iszero.(money)`.
 6. Return `fee`.

# Arguments

  - `sf`: One side's charge, from [`allocation_side_fees`](@ref), or `nothing`.
  - `p::VecNum`: Asset prices of this side.
  - `shares::VecNum`: Share count per asset.

# Returns

  - `fee::Number`: The fee of this side over the whole horizon.

# Related

  - [`allocation_liquidation_fee`](@ref)
  - [`allocation_side_fees`](@ref)
  - [`calc_periodic_fees`](@ref)
  - [`calc_one_off_fees`](@ref)
  - [`greedy_fee_delta`](@ref)
  - [`set_allocation_fees!`](@ref)
  - [`finite_sub_allocation`](@ref)
  - [`finite_sub_allocation!`](@ref)
"""
function allocation_fee(::Nothing, p::VecNum, shares::VecNum)
    return zero(promote_type(eltype(p), eltype(shares)))
end
function allocation_fee(sf::NamedTuple, p::VecNum, shares::VecNum)
    money = shares .* p
    # The forced exit is a constant of the allocation, so it is owed before a share is
    # bought and it does not move when one is.
    fee = sf.liq + zero(promote_type(eltype(p), eltype(shares)))
    prop = sf.prop
    if !isnothing(prop)
        fee += sf.T * dot_scalar(prop, money)
    end
    tn_val = sf.tn_val
    if !isnothing(tn_val)
        fee += sf.T * dot_scalar(tn_val, abs.(money - sf.prev_money))
    end
    fixed = sf.fixed
    if !isnothing(fixed)
        fee += dot_scalar(fixed, .!iszero.(money))
    end
    return fee
end
"""
    permute_side_fees(::Nothing, ::Any)
    permute_side_fees(sf::NamedTuple, idx)

Put one side's charge in the order that `idx` gives.

[`finite_sub_allocation!`](@ref) sorts its assets by descending target weight, so the rates and the previous money must take the same order. [`nothing_scalar_array_view`](@ref) views a vector entry to `idx` and passes a scalar entry through. `T` and `liq` are constants of the side, so they do not change. The fee does not change either. [`allocation_fee`](@ref) of the permuted charge, prices and shares is equal to the fee in the first order.

# Arguments

  - `sf`: One side's charge, from [`allocation_side_fees`](@ref), or `nothing`.
  - `idx`: The new order.

# Returns

  - `sf`: The charge in the order of `idx`, or `nothing`.

# Related

  - [`allocation_side_fees`](@ref)
  - [`finite_sub_allocation!`](@ref)
"""
function permute_side_fees(::Nothing, ::Any)
    return nothing
end
function permute_side_fees(sf::NamedTuple, idx)
    return (T = sf.T, prop = nothing_scalar_array_view(sf.prop, idx),
            fixed = nothing_scalar_array_view(sf.fixed, idx),
            tn_val = nothing_scalar_array_view(sf.tn_val, idx),
            prev_money = nothing_scalar_array_view(sf.prev_money, idx), liq = sf.liq)
end
