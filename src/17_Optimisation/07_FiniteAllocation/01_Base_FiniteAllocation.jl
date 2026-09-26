"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for estimators that turn continuous portfolio weights into share counts.

A finite allocation estimator buys share counts with a stated cash budget at stated prices. The library has two. [`DiscreteAllocation`](@ref) solves a mixed-integer programme, and [`GreedyAllocation`](@ref) buys in two greedy passes. [`optimise`](@ref) takes either one with a [`FiniteAllocationInput`](@ref) as its second argument, and allocates the long and the short side of the portfolio as two separate sub-problems.

# Related

  - [`OptimisationEstimator`](@ref)
  - [`DiscreteAllocation`](@ref)
  - [`GreedyAllocation`](@ref)
  - [`FiniteAllocationInput`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)

# References

  - $(ref_dict[:martin2021])
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
    const FOptE_FOpt = Union{<:FiniteAllocationOptimisationEstimator,
                             <:FiniteAllocationOptimisationResult}

Alias for a finite allocation estimator or a finite allocation result.

It is the type of the `fb` field of a finite allocation estimator. The fallback is an estimator that [`optimise`](@ref) runs when the first one fails, or a precomputed result that `optimise` returns as it is.

# Related

  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
"""
const FOptE_FOpt = Union{<:FiniteAllocationOptimisationEstimator,
                         <:FiniteAllocationOptimisationResult}
"""
    const FOptE_FOpt_FbChain = Union{<:FOptE_FOpt, <:FbChain}

Alias for the values that the `fb` field of a finite allocation result can hold.

[`optimise`](@ref) writes `nothing` when the estimator that the caller gave answered, and the [`FbChain`](@ref) of the failed attempts when a fallback answered. The alias also admits a [`FOptE_FOpt`](@ref), so [`factory`](@ref) can put a fallback estimator or result in the field.

# Related

  - [`FOptE_FOpt`](@ref)
  - [`FbChain`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
"""
const FOptE_FOpt_FbChain = Union{<:FOptE_FOpt, <:FbChain}
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
  - `prices`: Asset prices, in the order of `w`.
  - `cash`: Cash of the allocation, the `cash` of [`FiniteAllocationInput`](@ref).
  - `smoney`: Money of the shares that the short side sold, a non-negative number.
  - `sfee`: Fee that the short side paid over the whole horizon.

Each method must return a non-negative number. The allocators do not check it.

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
```

# Related

  - [`ProceedsCollateral`](@ref)
  - [`CashCollateral`](@ref)
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
$(DocStringExtensions.TYPEDEF)

Holds the target weights, prices and cash that a finite allocation reads.

[`optimise`](@ref) takes it as the second argument, with a [`DiscreteAllocation`](@ref) or a [`GreedyAllocation`](@ref). It also holds the cash before the trade, an optional horizon, fee and Investable Mask, and the collateral algorithm that gives each side of the book its cash. It subtypes [`AbstractEstimator`](@ref) and not [`OptimisationResult`](@ref), because it is the input of an allocation. The methods that dispatch on a result, such as the plots and the result [`factory`](@ref), cannot read its fields.

An optimisation over a reduced universe solves on its Investable Mask, and expands the weights back to the full universe. Its result then has a full-length `w` and a fee on two reduced axes. `imsk` records the reduction. [`allocation_side_fees`](@ref) calls [`lift_fees`](@ref) to put the fee back on the axis of `w` and `prices`, so the allocation charges the forced exit of an asset that left the universe on the money that the exit sold. The fee records its mask in its own `imsk` field too, and the constructor makes the two agree through [`mark_fees`](@ref). A stated `imsk` marks a fee that has no mask. A marked fee gives its mask when `imsk` is `nothing`. The constructor refuses two masks that differ.

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
        ca::AbstractCollateralAlgorithm = ProceedsCollateral()
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
        ca::AbstractCollateralAlgorithm = ProceedsCollateral()
    ) -> FiniteAllocationInput

Reads from a fitted optimisation every value that an allocation can take from it, so the caller states only the prices and the cash. A stated keyword takes priority. A value that the result does not carry becomes `nothing`, so one call serves every optimisation family:

  - `w` from `res.w`, which every optimisation result carries on the full universe.
  - `fees` through [`extract_fees`](@ref), from the `fees` property when the result has one.
  - `imsk` through [`result_investable_mask`](@ref), which gives `nothing` when the optimisation did not reduce its universe.
  - `horizon` through [`allocation_horizon`](@ref), the number of observations of the result's prior. An equal weighted optimisation without a prior gives none. The constructor then needs a stated horizon, but only when the input also has a fee.

## Validation

  - `!isempty(w)` and `!isempty(prices)`, else an [`IsEmptyError`](@ref).
  - `length(w) == length(prices)`, else a `DimensionMismatch`.
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
         ca ┴ ProceedsCollateral()
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
    Collateral algorithm, which gives the short side its cash, and the long side its cash after the short side trades. The default [`ProceedsCollateral`](@ref) pays for the long side with the proceeds of the short sales.
    """
    ca
    function FiniteAllocationInput(w::VecNum, prices::VecNum, cash::Number,
                                   prev_cash::Number, horizon::Option{<:Number},
                                   fees::Option{<:Fees}, imsk::Option{<:BitVector},
                                   ca::AbstractCollateralAlgorithm)
        @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
        @argcheck(!isempty(prices), IsEmptyError("prices cannot be empty"))
        @argcheck(length(w) == length(prices),
                  DimensionMismatch("w ($(length(w))) must match prices ($(length(prices)))"))
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
                   typeof(horizon), typeof(fees), typeof(imsk), typeof(ca)}(w, prices, cash,
                                                                            prev_cash,
                                                                            horizon, fees,
                                                                            imsk, ca)
    end
end
function FiniteAllocationInput(; w::VecNum, prices::VecNum, cash::Number = 1e6,
                               prev_cash::Number = cash,
                               horizon::Option{<:Number} = nothing,
                               fees::Option{<:Fees} = nothing,
                               imsk::Option{<:BitVector} = nothing,
                               ca::AbstractCollateralAlgorithm = ProceedsCollateral())::FiniteAllocationInput
    return FiniteAllocationInput(w, prices, cash, prev_cash, horizon, fees, imsk, ca)
end
function FiniteAllocationInput(res::NonFiniteAllocationOptimisationResult; prices::VecNum,
                               cash::Number = 1e6, prev_cash::Number = cash,
                               w::Option{<:VecNum} = nothing,
                               horizon::Option{<:Number} = nothing,
                               fees::Option{<:Fees} = nothing,
                               imsk::Option{<:BitVector} = nothing,
                               ca::AbstractCollateralAlgorithm = ProceedsCollateral())::FiniteAllocationInput
    # A stated keyword wins over the result, and each reader answers `nothing` on a result
    # carrying no such object, so a family holding fewer of them takes the same call.
    return FiniteAllocationInput(isnothing(w) ? res.w : w, prices, cash, prev_cash,
                                 allocation_horizon(res, horizon), extract_fees(res, fees),
                                 isnothing(imsk) ? result_investable_mask(res) : imsk, ca)
end
export FiniteAllocationInput
public AbstractCollateralAlgorithm
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

A [`Fees`](@ref) has two liquidation carriers. `lq` is a rate and `flq` is an amount of money, and the `w` of each holds the previous weights of the positions that the optimisation had to sell. The allocator solves no share count for these assets, so the charge needs no variable and is a constant of the sub-problem. [`set_liquidation_fees!`](@ref) and [`set_fixed_liquidation_fees!`](@ref) add the same charge to the JuMP model as a constant.

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

  - `fees`: Fee with the two liquidation carriers, or `nothing`.
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
    allocation_side_fees(::Nothing, ::Option{<:Number}, ::Number, ::Any, ::Any)
    allocation_side_fees(fees::Fees, T::Number, prev_cash::Number, lidx, sidx)

Split a fee into the charge of the long side and the charge of the short side.

Each sub-problem charges its own side. The long side takes `l` and `fl`, the short side takes `s` and `fs`, and both take the turnover rate and their previous money from [`allocation_turnover_money`](@ref). [`nothing_scalar_array_view`](@ref) views a vector rate to the side and passes a scalar rate through. A `nothing` `fees` gives `nothing` for both charges.

`lidx` and `sidx` are masks of the full universe, because they come from weights that an optimisation expanded back to it. A fee that the same optimisation reduced to its Investable Mask is on a shorter axis and records the mask in `imsk`, so the method lifts it with [`lift_fees`](@ref) before it takes a view. [`lift_fees`](@ref) returns a fee that has no mask unchanged.

The forced exit of [`allocation_liquidation_fee`](@ref) is a constant of the whole allocation. It is the `liq` entry of the long side's charge, and the short side's `liq` is zero, so both charges have the same entries.

# Algorithm

 1. Lift `fees` onto the full universe with [`lift_fees`](@ref).
 2. Compute `ltn_val` and `lprev` for the long side, and `stn_val` and `sprev` for the short side, with [`allocation_turnover_money`](@ref).
 3. Compute `liq`, the forced-exit charge, with [`allocation_liquidation_fee`](@ref).
 4. Return the long side's charge, from `l`, `fl`, `ltn_val`, `lprev` and `liq`, and the short side's charge, from `s`, `fs`, `stn_val`, `sprev` and a zero `liq`.

# Arguments

  - `fees`: Fee to split, or `nothing`.
  - `T`: Horizon, in periods.
  - `prev_cash::Number`: Cash that the portfolio held before the trade.
  - `lidx`: Mask of the long side.
  - `sidx`: Mask of the short side.

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
function allocation_side_fees(::Nothing, ::Option{<:Number}, ::Number, ::Any, ::Any)
    return nothing, nothing
end
function allocation_side_fees(fees::Fees, T::Number, prev_cash::Number, lidx, sidx)
    # `lidx` and `sidx` index the full universe, so the fee comes onto that axis first.
    fees = lift_fees(fees)
    ltn_val, lprev = allocation_turnover_money(fees.tn, prev_cash, lidx, false)
    stn_val, sprev = allocation_turnover_money(fees.tn, prev_cash, sidx, true)
    liq = allocation_liquidation_fee(fees, T, prev_cash)
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

A side that held money before the trade owes a turnover fee when it buys nothing, because a sale to zero is a trade. A book whose universe lost an asset owes the forced exit whatever it buys. For these reasons, both allocators charge this method on the empty book too.

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
