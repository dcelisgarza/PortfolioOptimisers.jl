"""
    abstract type AbstractPositionReading <: AbstractAlgorithm end

Abstract supertype of the readings of a position of a [`ConcentrationLimit`](@ref) or of a [`PositionConstraint`](@ref).

A reading tells the rule how a short position counts. A long-only book has no short position, so [`GrossPosition`](@ref), [`NetPosition`](@ref) and [`LongPosition`](@ref) give the weights there, and [`ShortPosition`](@ref) gives zeros. The name avoids the word "exposure", because the factor exposures use it.

| reading                 | expression | a short of -30 % of X          | A1 +8 %, A2 -8 % of one issuer |
|:----------------------- |:---------- |:------------------------------ |:------------------------------ |
| [`GrossPosition`](@ref) | `lw + sw`  | counts as 30 %                 | 16 %                           |
| [`NetPosition`](@ref)   | `w`        | below every concentration rule | 0 %                            |
| [`LongPosition`](@ref)  | `lw`       | not seen by the rule           | 8 %                            |
| [`ShortPosition`](@ref) | `sw`       | counts as 30 %                 | 8 %                            |

# Interfaces

To make a new reading, subtype `AbstractPositionReading` and implement the method below.

## Reading of the weights

  - `PortfolioOptimisers.position_reading(pos::AbstractPositionReading, w::AbstractVector, lw::AbstractVector, sw::AbstractVector) -> AbstractVector`: The reading of each asset, from the weights `w` and their long and short parts `lw` and `sw`.

One method serves every caller. The model gives the variables of its parts. A long-only model has no parts, and the library passes the weights as `lw` and zeros as `sw`. [`concentration_margins`](@ref) gives the parts of a weight vector, `max.(w, 0)` and `max.(-w, 0)`. There is no fallback.

# Related

  - [`GrossPosition`](@ref)
  - [`NetPosition`](@ref)
  - [`LongPosition`](@ref)
  - [`ShortPosition`](@ref)
  - [`ConcentrationLimit`](@ref)
  - [`PositionConstraint`](@ref)
"""
abstract type AbstractPositionReading <: AbstractAlgorithm end
"""
    struct GrossPosition <: AbstractPositionReading end

Reads a position as the sum of its long and its short part, so a short position counts at its size.

This is the default reading of a [`ConcentrationLimit`](@ref) and of a [`PositionConstraintEstimator`](@ref), and the safe one for a cap: a short position cannot hide a long position of the same issuer. The model reads the parts `lw` and `sw` of the weights. They give `lw + sw >= |w|`, so the reading is never below the gross position. When the short budget `sbgt` pins the sum of the parts and `xbgt` is off, the parts can be larger than `|w|`, and then the rule is stricter than the gross position.

A [`RiskBudgeting`](@ref) head declares its own parts of the weights, and the reading uses them.

# Related

  - [`AbstractPositionReading`](@ref)
  - [`NetPosition`](@ref)
  - [`LongPosition`](@ref)
"""
struct GrossPosition <: AbstractPositionReading end
"""
    struct NetPosition <: AbstractPositionReading end

Reads a position as its net weight, so a short position lowers the position of its issuer.

A short position is below every threshold, so the rule does not see it. Two share classes of one issuer at +8 % and -8 % give a position of zero.

# Related

  - [`AbstractPositionReading`](@ref)
  - [`GrossPosition`](@ref)
  - [`LongPosition`](@ref)
"""
struct NetPosition <: AbstractPositionReading end
"""
    struct LongPosition <: AbstractPositionReading end

Reads a position as its long part, so the rule does not see a short position.

The model reads the long part `lw` of the weights. It gives `lw >= max(w, 0)`, so the reading is never below the long position. When the budgets pin the sum of the long part and `xbgt` is off, `lw` can be larger than `max(w, 0)`, and then the rule is stricter than the long position.

# Related

  - [`AbstractPositionReading`](@ref)
  - [`GrossPosition`](@ref)
  - [`NetPosition`](@ref)
"""
struct LongPosition <: AbstractPositionReading end
"""
    struct ShortPosition <: AbstractPositionReading end

Reads a position as the size of its short part, so the rule does not see a long position.

The reading is non-negative, with the sign of the short budget `sbgt`: a weight of -30 % reads as 30 %. The model reads the short part `sw` of the weights. It gives `sw >= max(-w, 0)`, so the reading is never below the short position. When the budgets pin the sum of the short part and `xbgt` is off, `sw` can be larger than `max(-w, 0)`, and then a cap is stricter than the short position. A long-only model has no short part, so the reading is zero there.

# Related

  - [`AbstractPositionReading`](@ref)
  - [`GrossPosition`](@ref)
  - [`LongPosition`](@ref)
"""
struct ShortPosition <: AbstractPositionReading end
"""
    abstract type AbstractConcentrationAlgorithm <: AbstractAlgorithm end

Abstract supertype of the forms of a [`ConcentrationLimit`](@ref) in the model.

# Interfaces

To make a new form, subtype `AbstractConcentrationAlgorithm` and implement the method below.

## Rows of the rule

  - `PortfolioOptimisers.set_concentration_rows!(model::JuMP.Model, alg::AbstractConcentrationAlgorithm, sp::ConcentrationMIPSpace, x::AbstractVector, wb::WeightBounds, ss::Option{<:Number}) -> Nothing`: Add the rows of the rule `sp.cl` on the positions `x`. Register each entry with `state_set!(model, Symbol(""), mip_key(sp, name), val)`, so that two rules do not collide. A form with a binary declares it and its gate with `declare_mip_gate!`, so that its rows hold under a ratio objective too.

There is no fallback. [`concentration_margins`](@ref) reads the rule itself whatever the form, so a new form needs no method there.

# Related

  - [`ExactConcentration`](@ref)
  - [`SumLargestConcentration`](@ref)
  - [`ConcentrationLimit`](@ref)
"""
abstract type AbstractConcentrationAlgorithm <: AbstractAlgorithm end
"""
    struct ExactConcentration <: AbstractConcentrationAlgorithm end

States a [`ConcentrationLimit`](@ref) exactly, with one binary for each position.

The binary says that the position is over the threshold `t`. The feasible set is the rule itself, see the mathematical definition of [`ConcentrationLimit`](@ref). It is the default form. It needs a solver that takes integer variables.

The rule holds to the tolerance of the solver. A solver takes a binary at ``\\epsilon`` as zero when ``\\epsilon`` is under its integrality tolerance. Under a constant budget, a position with such a binary can be at ``t + (c - t) \\epsilon``. Under a ratio objective or a [`RiskBudgeting`](@ref) head, the gate multiplies ``\\epsilon`` by the big-M constant `ss` of the block, see [`declare_mip_gate!`](@ref). So the position can be at ``t + (c - t) M_s \\epsilon / k``. With ``M_s = 1000``, ``\\epsilon = 10^{-6}`` and ``k = 1``, the excess is 0.005 % of the book under 5/10/40. The sum of the rule is a step, so a check with no tolerance then counts the whole position. Use one of these remedies:

  - A smaller constant, such as `ss = 10` in the block. It must stay above the `k` of the solution.
  - A smaller integrality tolerance in the settings of the solver.
  - A threshold `t` a little under the limit of the mandate.
  - A `tol` at the size of the excess in [`concentration_margins`](@ref), for a check that accepts the excess.

# Related

  - [`AbstractConcentrationAlgorithm`](@ref)
  - [`SumLargestConcentration`](@ref)
"""
struct ExactConcentration <: AbstractConcentrationAlgorithm end
"""
    struct SumLargestConcentration <: AbstractConcentrationAlgorithm end

States a [`ConcentrationLimit`](@ref) as a linear form with no integer variable: the sum of the `k` largest positions is at most `s`, with `k = ceil(s / t)`, and each position is at most `c`.

The form is safe: a book that meets it meets the rule. A book with more than `k` positions over `t` has a sum of its `k` largest positions over `k t >= s`. So at most `k` positions are over `t`. They are among the `k` largest positions, so their sum is at most `s`. The form reads the positive part of each position, so a negative net position adds nothing.

The form is stricter than the rule. With 5/10/40, `k = 8`. A book of four positions at 10 % and six at 5 % meets the rule, because only the four positions at 10 % are over 5 %, and their sum is 40 %. Its eight largest positions sum to 50 %, so the form refuses it.

| book               | the rule | this form        | the sum of the 4 largest |
|:------------------ |:-------- |:---------------- |:------------------------ |
| 8 x 9 %            | fails    | fails            | passes (not safe)        |
| 4 x 10 % + 6 x 5 % | passes   | fails (stricter) | passes                   |
| 20 x 5 %           | passes   | passes           | passes                   |

The form "the sum of the 4 largest positions is at most 40 %" is not safe. A book of eight positions at 9 % meets it, but the sum of its positions over 5 % is 72 %.

Use the form for a solver with no integer variables, for a large universe, or for a loop of folds with many solves.

# Related

  - [`AbstractConcentrationAlgorithm`](@ref)
  - [`ExactConcentration`](@ref)
  - [`sum_largest_count`](@ref)
"""
struct SumLargestConcentration <: AbstractConcentrationAlgorithm end
"""
    sum_largest_count(t::Number, s::Number) -> Int

Return the smallest integer `k` with `k t >= s`, the count of positions of [`SumLargestConcentration`](@ref).

A division in floating point can give `ceil(s / t)` one too large: `0.14 / 0.02` is `7.000000000000001`, so `ceil` gives 8 where 7 is enough. A count that is too large keeps the form safe, but makes it stricter than it must be. So the function lowers the count while one less still gives `k t >= s`.

# Algorithm

 1. Set `k = ceil(Int, s / t)`.
 2. While `k > 1` and `(k - 1) t >= s`, lower `k` by one.
 3. Return `k`.

# Arguments

  - `t`: The threshold of a large position.
  - `s`: The limit of the sum.

# Returns

  - `k::Int`: The count of positions.

# Examples

```jldoctest
julia> PortfolioOptimisers.sum_largest_count(0.05, 0.4)
8

julia> PortfolioOptimisers.sum_largest_count(0.05, 0.42)
9

julia> PortfolioOptimisers.sum_largest_count(0.02, 0.14)
7
```

# Related

  - [`SumLargestConcentration`](@ref)
"""
function sum_largest_count(t::Number, s::Number)::Int
    k = ceil(Int, s / t)
    while k > 1 && (k - 1) * t >= s
        k -= 1
    end
    return k
end
"""
    assert_concentration_limit(t::Number, c::Number, s::Number)

Check the three numbers of a concentration rule.

[`ConcentrationLimitEstimator`](@ref) and [`ConcentrationLimit`](@ref) call it, because the numbers alone decide whether a rule has a meaning.

# Validation

  - `t`, `c` and `s` are finite and `> 0`, through [`assert_nonempty_gt0_finite_val`](@ref).
  - `c > t`. A cap at or below the threshold leaves no position over the threshold, so the sum limit has no meaning. A `DomainError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`ConcentrationLimitEstimator`](@ref)
  - [`ConcentrationLimit`](@ref)
"""
function assert_concentration_limit(t::Number, c::Number, s::Number)::Nothing
    assert_nonempty_gt0_finite_val(t, :t)
    assert_nonempty_gt0_finite_val(c, :c)
    assert_nonempty_gt0_finite_val(s, :s)
    @argcheck(c > t,
              DomainError((t, c),
                          "the cap `c` must be greater than the threshold `t`, or no position can be over `t`"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Configures a concentration rule, whose positions can be written as issuer names that resolve against a universe.

A concentration rule caps each position at `c`, and limits the sum of the positions over the threshold `t` to `s`. The UCITS rule, the "5/10/40 rule" of Art. 52(2) of Directive 2009/65/EC, is the default: a fund holds at most 10 % of its assets in one issuer, and the issuers over 5 % sum to at most 40 %. The weight bounds give the cap alone. The limit of the sum depends on which positions are over `t`, so no linear constraint gives it.

[`concentration_limit_constraints`](@ref) resolves the estimator to a [`ConcentrationLimit`](@ref). It reads the positions from `set`. With `set = nothing`, each asset is its own position. An [`AssetSetsMatrixEstimator`](@ref) groups the assets by a key of `sets.dict`, such as the issuer of each asset. Then the cap is on the issuer, which is what the directive states, and two share classes of one issuer are one position.

The exceptions of Art. 52(3) to 52(5) of the directive are more rules, not more fields of one rule. The field `on` of [`AssetSetsMatrixEstimator`](@ref) gives each rule its issuers. A vector of rules goes in the slot `conc` of [`ConstraintBlock`](@ref), and the group constraints give the limit of a government issuer.

```julia
sets = UniverseSets(;
                    dict = Dict(\"nx\" => [\"C1\", \"C2\", \"K1\", \"G1\"],
                                \"nx_issuer\" => [\"Acme\", \"Acme\", \"BankX\", \"DE\"],
                                \"ucits_standard\" => [\"C1\", \"C2\"],
                                \"ucits_covered\" => [\"K1\"]))
conc = [ConcentrationLimitEstimator(;
                                    set = AssetSetsMatrixEstimator(; val = \"nx_issuer\",
                                                                   on = \"ucits_standard\")),
        ConcentrationLimitEstimator(; c = 0.25, s = 0.8,
                                    set = AssetSetsMatrixEstimator(; val = \"nx_issuer\",
                                                                   on = \"ucits_covered\"))]
lcse = LinearConstraintEstimator(; val = \"G1 <= 0.35\")
```

The clustering and the naive optimisers have no slot for the rule, as they have none for a cardinality or a threshold. [`concentration_margins`](@ref) checks the weights of any optimiser against the rule.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConcentrationLimitEstimator(;
        t::Number = 0.05,
        c::Number = 0.1,
        s::Number = 0.4,
        set::Option{<:MatNum_ASetMatE} = nothing,
        pos::AbstractPositionReading = GrossPosition(),
        alg::AbstractConcentrationAlgorithm = ExactConcentration()
    ) -> ConcentrationLimitEstimator

Keywords correspond to the struct's fields.

## Validation

  - The rules of [`assert_concentration_limit`](@ref) for `t`, `c` and `s`.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `set`: A membership matrix keeps the columns of the selected assets. An [`AssetSetsMatrixEstimator`](@ref) stays unchanged, and it resolves against the universe of the view.

# Examples

```jldoctest
julia> ConcentrationLimitEstimator(; set = AssetSetsMatrixEstimator(; val = \"nx_issuer\"))
ConcentrationLimitEstimator
    t ┼ Float64: 0.05
    c ┼ Float64: 0.1
    s ┼ Float64: 0.4
  set ┼ AssetSetsMatrixEstimator
      │   val ┼ String: "nx_issuer"
      │    on ┴ nothing
  pos ┼ GrossPosition()
  alg ┴ ExactConcentration()
```

# Related

  - [`ConcentrationLimit`](@ref)
  - [`concentration_limit_constraints`](@ref)
  - [`concentration_margins`](@ref)
  - [`AssetSetsMatrixEstimator`](@ref)
  - [`AbstractPositionReading`](@ref)
  - [`AbstractConcentrationAlgorithm`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
"""
@propagatable @concrete struct ConcentrationLimitEstimator <: AbstractConstraintEstimator
    """
    $(field_dict[:cnl_t])
    """
    t
    """
    $(field_dict[:cnl_c])
    """
    c
    """
    $(field_dict[:cnl_s])
    """
    s
    """
    $(field_dict[:cnl_set])
    """
    @vprop set
    """
    $(field_dict[:cnl_pos])
    """
    pos
    """
    $(field_dict[:cnl_alg])
    """
    alg
    function ConcentrationLimitEstimator(t::Number, c::Number, s::Number,
                                         set::Option{<:MatNum_ASetMatE},
                                         pos::AbstractPositionReading,
                                         alg::AbstractConcentrationAlgorithm)::ConcentrationLimitEstimator
        assert_concentration_limit(t, c, s)
        return new{typeof(t), typeof(c), typeof(s), typeof(set), typeof(pos), typeof(alg)}(t,
                                                                                           c,
                                                                                           s,
                                                                                           set,
                                                                                           pos,
                                                                                           alg)
    end
end
function ConcentrationLimitEstimator(; t::Number = 0.05, c::Number = 0.1, s::Number = 0.4,
                                     set::Option{<:MatNum_ASetMatE} = nothing,
                                     pos::AbstractPositionReading = GrossPosition(),
                                     alg::AbstractConcentrationAlgorithm = ExactConcentration())::ConcentrationLimitEstimator
    return ConcentrationLimitEstimator(t, c, s, set, pos, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Caps each position at `c`, and limits the sum of the positions over the threshold `t` to `s`.

This is the resolved form of [`ConcentrationLimitEstimator`](@ref), whose docstring states the rule and the directive that it comes from. The positions are the rows of `set`, or the assets when `set` is `nothing`. The slot `conc` of [`ConstraintBlock`](@ref) takes one rule or a vector of rules, and [`set_concentration_constraints!`](@ref) adds each rule to the model.

# Mathematical definition

The form [`ExactConcentration`](@ref) states the rule with one binary for each position:

```math
\\begin{align}
x_j &\\leq t + (c - t)\\, z_j\\,, \\\\
y_j &\\geq x_j - c\\,(1 - z_j)\\,,\\quad y_j \\geq 0\\,, \\\\
\\sum_{j=1}^{M} y_j &\\leq s\\,,\\quad \\boldsymbol{z} \\in \\{0, 1\\}^{M}\\,.
\\end{align}
```

Where:

  - ``x_j``: Position ``j``, row ``j`` of `set` times the reading of the weights that `pos` gives.
  - ``z_j``: Binary that says that position ``j`` is over ``t``.
  - ``y_j``: The part of position ``j`` that counts towards the sum.
  - ``t``, ``c``, ``s``: The threshold, the cap and the limit of the sum.
  - ``M``: The number of positions.

When ``z_j = 0``, the position is at most ``t``, and ``y_j`` can be zero. When ``z_j = 1``, the position is at most ``c``, and ``y_j \\geq x_j``. So the feasible set is the rule itself. A solver has no reason to set ``z_j = 1`` for a small position, because that only adds to the sum. The binary is not the held bit of [`HeldIndicators`](@ref) or [`LongShortIndicators`](@ref), because "held" and "over ``t``" are two different facts.

The form [`SumLargestConcentration`](@ref) states a stricter rule with no binary.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConcentrationLimit(;
        t::Number = 0.05,
        c::Number = 0.1,
        s::Number = 0.4,
        set::Option{<:MatNum} = nothing,
        pos::AbstractPositionReading = GrossPosition(),
        alg::AbstractConcentrationAlgorithm = ExactConcentration()
    ) -> ConcentrationLimit

Keywords correspond to the struct's fields.

## Validation

  - The rules of [`assert_concentration_limit`](@ref) for `t`, `c` and `s`.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `set`: The membership matrix keeps the columns of the selected assets.

# Examples

```jldoctest
julia> ConcentrationLimit(; set = [1 1 0; 0 0 1])
ConcentrationLimit
    t ┼ Float64: 0.05
    c ┼ Float64: 0.1
    s ┼ Float64: 0.4
  set ┼ 2×3 Matrix{Int64}
  pos ┼ GrossPosition()
  alg ┴ ExactConcentration()
```

# Related

  - [`ConcentrationLimitEstimator`](@ref)
  - [`concentration_limit_constraints`](@ref)
  - [`set_concentration_constraints!`](@ref)
  - [`concentration_margins`](@ref)
  - [`AbstractConstraintResult`](@ref)
"""
@propagatable @concrete struct ConcentrationLimit <: AbstractConstraintResult
    """
    $(field_dict[:cnl_t])
    """
    t
    """
    $(field_dict[:cnl_c])
    """
    c
    """
    $(field_dict[:cnl_s])
    """
    s
    """
    $(field_dict[:cnl_res_set])
    """
    @vprop set
    """
    $(field_dict[:cnl_pos])
    """
    pos
    """
    $(field_dict[:cnl_alg])
    """
    alg
    function ConcentrationLimit(t::Number, c::Number, s::Number, set::Option{<:MatNum},
                                pos::AbstractPositionReading,
                                alg::AbstractConcentrationAlgorithm)::ConcentrationLimit
        assert_concentration_limit(t, c, s)
        return new{typeof(t), typeof(c), typeof(s), typeof(set), typeof(pos), typeof(alg)}(t,
                                                                                           c,
                                                                                           s,
                                                                                           set,
                                                                                           pos,
                                                                                           alg)
    end
end
function ConcentrationLimit(; t::Number = 0.05, c::Number = 0.1, s::Number = 0.4,
                            set::Option{<:MatNum} = nothing,
                            pos::AbstractPositionReading = GrossPosition(),
                            alg::AbstractConcentrationAlgorithm = ExactConcentration())::ConcentrationLimit
    return ConcentrationLimit(t, c, s, set, pos, alg)
end
"""
    const CnlE_Cnl = Union{ConcentrationLimit, ConcentrationLimitEstimator}

Alias for a concentration rule, resolved or not.

A caller can write a rule in either form, so the slot `conc` accepts both. [`concentration_limit_constraints`](@ref) gives a [`ConcentrationLimit`](@ref) for either form.

# Related

  - [`ConcentrationLimit`](@ref)
  - [`ConcentrationLimitEstimator`](@ref)
  - [`VecCnlE_Cnl`](@ref)
"""
const CnlE_Cnl = Union{ConcentrationLimit, ConcentrationLimitEstimator}
"""
    const VecCnlE_Cnl = AbstractVector{<:CnlE_Cnl}

Alias for a vector of concentration rules, resolved or not.

A mandate can hold more than one rule, such as a 5/10/40 rule over the standard issuers and a 5/25/80 rule over the issuers of covered bonds. The model adds each rule on its own, and the rules intersect.

# Related

  - [`CnlE_Cnl`](@ref)
  - [`CnlE_Cnl_VecCnlE_Cnl`](@ref)
"""
const VecCnlE_Cnl = AbstractVector{<:CnlE_Cnl}
"""
    const CnlE_Cnl_VecCnlE_Cnl = Union{CnlE_Cnl, VecCnlE_Cnl}

Alias for one concentration rule or a vector of rules, resolved or not. The slot `conc` of [`ConstraintBlock`](@ref) takes it.

# Related

  - [`CnlE_Cnl`](@ref)
  - [`VecCnlE_Cnl`](@ref)
  - [`Cnl_VecCnl`](@ref)
"""
const CnlE_Cnl_VecCnlE_Cnl = Union{CnlE_Cnl, VecCnlE_Cnl}
"""
    const Cnl_VecCnl = Union{ConcentrationLimit, AbstractVector{<:ConcentrationLimit}}

Alias for one resolved concentration rule or a vector of resolved rules.

This is the resolved form of [`CnlE_Cnl_VecCnlE_Cnl`](@ref). The field `conc` of [`ResolvedConstraintBlock`](@ref) and the builder [`set_concentration_constraints!`](@ref) take it, so they refuse an estimator.

# Related

  - [`ConcentrationLimit`](@ref)
  - [`CnlE_Cnl_VecCnlE_Cnl`](@ref)
"""
const Cnl_VecCnl = Union{ConcentrationLimit, AbstractVector{<:ConcentrationLimit}}
"""
    concentration_limit_constraints(cl::Option{<:ConcentrationLimit}, args...; kwargs...)
    concentration_limit_constraints(cl::ConcentrationLimitEstimator,
                                    sets::Option{<:UniverseSets}; strict::Bool = false,
                                    kwargs...)
    concentration_limit_constraints(cl::VecCnlE_Cnl, sets::Option{<:UniverseSets};
                                    kwargs...)

Resolve a concentration rule, or a vector of rules, against a universe.

The method for a [`ConcentrationLimit`](@ref) or `nothing` returns its input unchanged. The method for a [`ConcentrationLimitEstimator`](@ref) resolves its `set` with [`asset_sets_matrix`](@ref), and keeps the other fields. The method for a vector resolves each entry and keeps the order.

The block resolves the rule at stage 1, [`resolve_constraint_block_names`](@ref), before the reduction to the Investable Mask. So the matrix has one column for each asset of the caller, and the view at the mask keeps the columns of the investable assets, as it does for `smtx`.

# Algorithm

 1. Resolve `cl.set` with `asset_sets_matrix(cl.set, sets; strict = strict)`. `nothing` and a matrix stay unchanged.
 2. Return a [`ConcentrationLimit`](@ref) with the resolved matrix and the other fields of `cl`.

# Arguments

  - `cl`: The rule, or a vector of rules.
  - `sets`: A [`UniverseSets`](@ref) that holds the asset universe and the groups. A rule whose `set` holds no estimator does not read it, so it can be `nothing`.
  - $(arg_dict[:strict])
  - `args...`, `kwargs...`: More arguments, which the method for a resolved rule ignores, and which the method for a vector gives to the method of each entry.

# Validation

  - `sets` is not `nothing` when `cl.set` is an [`AssetSetsMatrixEstimator`](@ref). An [`IsNothingError`](@ref) is thrown otherwise.
  - The rules of [`asset_sets_matrix`](@ref) for an [`AssetSetsMatrixEstimator`](@ref) in `cl.set`.

# Returns

  - `cl::Option{<:Cnl_VecCnl}`: The resolved rule, or one resolved rule for each entry of a vector.

# Examples

```jldoctest
julia> sets = UniverseSets(;
                           dict = Dict(\"nx\" => [\"A1\", \"A2\", \"B\"], \"nx_issuer\" => [\"A\", \"A\", \"B\"]));

julia> cl = ConcentrationLimitEstimator(; set = AssetSetsMatrixEstimator(; val = \"nx_issuer\"));

julia> concentration_limit_constraints(cl, sets).set
2×3 transpose(::BitMatrix) with eltype Bool:
 1  1  0
 0  0  1
```

# Related

  - [`ConcentrationLimitEstimator`](@ref)
  - [`ConcentrationLimit`](@ref)
  - [`asset_sets_matrix`](@ref)
  - [`resolve_constraint_block_names`](@ref)
"""
function concentration_limit_constraints(cl::Option{<:ConcentrationLimit}, args...;
                                         kwargs...)::Option{<:ConcentrationLimit}
    return cl
end
function concentration_limit_constraints(cl::ConcentrationLimitEstimator,
                                         sets::Option{<:UniverseSets}; strict::Bool = false,
                                         kwargs...)::ConcentrationLimit
    @argcheck(!(isnothing(sets) && name_keyed(cl)),
              IsNothingError("sets cannot be nothing when the `set` of a concentration rule is an `AssetSetsMatrixEstimator`: it names a key of `sets.dict`, and only a UniverseSets resolves it"))
    return ConcentrationLimit(; t = cl.t, c = cl.c, s = cl.s,
                              set = asset_sets_matrix(cl.set, sets; strict = strict),
                              pos = cl.pos, alg = cl.alg)
end
function concentration_limit_constraints(cl::VecCnlE_Cnl, sets::Option{<:UniverseSets};
                                         kwargs...)
    return [concentration_limit_constraints(cli, sets; kwargs...) for cli in cl]
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the margins of a weight vector against one concentration rule.

[`concentration_margins`](@ref) makes it. A margin `<= 0` passes, so a book meets the rule when every entry of `cap` is `<= 0` and `total <= 0`. A positive margin is the excess over the rule. A position counts towards the sum only when it is over the threshold by more than the keyword `tol` of that function. The check reads the rule itself, the form of [`ExactConcentration`](@ref), whatever the field `alg` of the rule holds.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConcentrationMargins(;
        x::VecNum,
        cap::VecNum,
        total::Number
    ) -> ConcentrationMargins

Keywords correspond to the struct's fields.

## Validation

  - `length(x) == length(cap)`.

# Related

  - [`concentration_margins`](@ref)
  - [`ConcentrationLimit`](@ref)
"""
@concrete struct ConcentrationMargins <: AbstractResult
    """
    $(field_dict[:cnm_x])
    """
    x
    """
    $(field_dict[:cnm_cap])
    """
    cap
    """
    $(field_dict[:cnm_total])
    """
    total
    function ConcentrationMargins(x::VecNum, cap::VecNum,
                                  total::Number)::ConcentrationMargins
        @argcheck(length(x) == length(cap),
                  DimensionMismatch("x and cap must have the same length:\nlength(x) => $(length(x))\nlength(cap) => $(length(cap))"))
        return new{typeof(x), typeof(cap), typeof(total)}(x, cap, total)
    end
end
function ConcentrationMargins(; x::VecNum, cap::VecNum, total::Number)::ConcentrationMargins
    return ConcentrationMargins(x, cap, total)
end
"""
    position_reading(pos::AbstractPositionReading, w::AbstractVector, lw::AbstractVector,
                     sw::AbstractVector)

Return the reading of each asset that `pos` gives, from the weights and their long and short parts.

The builders [`mip_wx!`](@ref) and [`linear_row_reading!`](@ref) give the variables of the parts of the model. [`concentration_margins`](@ref) gives the smallest parts of a weight vector, `max.(w, 0)` and `max.(-w, 0)`. Then the gross reading is `|w|`, and the long reading is the positive part of `w`.

# Algorithm

The method that Julia selects is the algorithm:

  - [`GrossPosition`](@ref): `lw + sw`.
  - [`NetPosition`](@ref): `w`.
  - [`LongPosition`](@ref): `lw`.
  - [`ShortPosition`](@ref): `sw`.

# Arguments

  - `pos`: The reading.
  - `w`: The weights.
  - `lw`, `sw`: The long and the short parts of the weights.

# Returns

  - `v::AbstractVector`: The reading of each asset.

# Related

  - [`concentration_margins`](@ref)
  - [`ConcentrationMIPSpace`](@ref)
  - [`linear_row_reading!`](@ref)
  - [`long_only_reading`](@ref)
  - [`AbstractPositionReading`](@ref)
"""
function position_reading(::GrossPosition, ::AbstractVector, lw::AbstractVector,
                          sw::AbstractVector)
    return lw + sw
end
function position_reading(::NetPosition, w::AbstractVector, ::AbstractVector,
                          ::AbstractVector)
    return w
end
function position_reading(::LongPosition, ::AbstractVector, lw::AbstractVector,
                          ::AbstractVector)
    return lw
end
function position_reading(::ShortPosition, ::AbstractVector, ::AbstractVector,
                          sw::AbstractVector)
    return sw
end
"""
    long_only_reading(pos::AbstractPositionReading, w::AbstractVector)
    long_only_reading(pos::Union{GrossPosition, NetPosition, LongPosition}, w::AbstractVector)

Return the reading of weights that have no short part.

# Algorithm

The method that Julia selects is the algorithm:

  - A [`GrossPosition`](@ref), a [`NetPosition`](@ref) or a [`LongPosition`](@ref): the weights `w`.
  - Any other reading: [`position_reading`](@ref) with `w` as the long part and zeros as the short part. The zero of a variable is an affine expression, so a row of zeros is a row of the model.

# Arguments

  - `pos`: The reading.
  - `w`: The weights.

# Returns

  - `v::AbstractVector`: The reading of each asset.

# Related

  - [`position_reading`](@ref)
  - [`block_position_reading!`](@ref)
  - [`mip_wx!`](@ref)
"""
function long_only_reading(pos::AbstractPositionReading, w::AbstractVector)
    return position_reading(pos, w, w, zero.(w))
end
function long_only_reading(::Union{GrossPosition, NetPosition, LongPosition},
                           w::AbstractVector)
    return w
end
"""
    concentration_positions(set::Option{<:MatNum}, v::AbstractVector)

Return the positions of a rule from the reading `v` of each asset: `v` itself when `set` is `nothing`, and `set * v` otherwise.

[`concentration_margins`](@ref) gives it a vector of numbers, and the builder [`mip_wx!`](@ref) gives it a vector of model expressions.

# Arguments

  - `set`: The membership matrix of the rule, or `nothing`.
  - `v`: The reading of each asset.

# Returns

  - `x::AbstractVector`: The position of each row of the rule.

# Related

  - [`concentration_margins`](@ref)
  - [`ConcentrationMIPSpace`](@ref)
"""
function concentration_positions(::Nothing, v::AbstractVector)
    return v
end
function concentration_positions(set::MatNum, v::AbstractVector)
    return set * v
end
"""
    concentration_margins(cl::ConcentrationLimit, w::VecNum;
                          tol::Number = weights_bound_tolerance(eltype(w)), kwargs...)
    concentration_margins(cl::ConcentrationLimitEstimator, w::VecNum;
                          sets::Option{<:UniverseSets} = nothing, strict::Bool = false,
                          kwargs...)
    concentration_margins(cl::VecCnlE_Cnl, w::VecNum; kwargs...)

Check a weight vector against a concentration rule, and return its margins.

The check serves the mandate check of a book from any optimiser. The clustering and the naive optimisers have no slot for the rule, so their weights can break it. A rule holds when every margin is `<= 0`. The precedent of a check on weights is [`weights_meet_bounds`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathrm{cap}_j &= x_j - c\\,, \\\\
\\mathrm{total} &= \\sum_{j\\,:\\,x_j > t + \\tau} x_j - s\\,.
\\end{align}
```

Where:

  - ``x_j``: Position ``j``, row ``j`` of `set` times the reading of the weights that [`position_reading`](@ref) gives for `pos`.
  - ``t``, ``c``, ``s``: The threshold, the cap and the limit of the sum.
  - ``\\tau``: The tolerance `tol`.

The sum is a step: a position that passes ``t + \\tau`` adds its whole size, and not its excess over ``t``. So the check of a book of a solver needs a `tol` at the size of the tolerance of the solver. [`ExactConcentration`](@ref) states the excess that a mixed-integer solver can give, and how to lower it. `tol` does not move the margins of the cap and of the sum, so compare them with a tolerance too.

# Algorithm

 1. Resolve an estimator with [`concentration_limit_constraints`](@ref) against `sets`.
 2. Read the weights with [`position_reading`](@ref), with the parts `max.(w, 0)` and `max.(-w, 0)`, and make the positions with [`concentration_positions`](@ref).
 3. Return the margins of the formulas above. The method for a vector returns one [`ConcentrationMargins`](@ref) for each rule, in order.

# Arguments

  - `cl`: The rule, or a vector of rules.
  - `w`: The weights, one for each asset of the universe of the rule.
  - `tol`: The excess over `t` up to which a position does not count towards the sum. The default is the tolerance of [`weights_bound_tolerance`](@ref).
  - `sets`: A [`UniverseSets`](@ref) against which an estimator resolves, or `nothing`.
  - $(arg_dict[:strict])
  - `kwargs...`: The method for a resolved rule ignores them. The method for an estimator gives them to the method for its resolved rule, and the method for a vector gives them to each entry.

# Validation

  - The rules of [`concentration_limit_constraints`](@ref) for an estimator.
  - The width of `cl.set` is `length(w)`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `cm::ConcentrationMargins`: The margins of one rule, or one for each rule of a vector.

# Examples

Six positions at 10 % sum to 60 %, so the book is 20 % over the limit of the sum, and no position is over the cap.

```jldoctest
julia> w = [fill(0.1, 6); fill(0.04, 10)];

julia> cm = concentration_margins(ConcentrationLimit(), w);

julia> maximum(cm.cap) <= 0, round(cm.total; digits = 10)
(true, 0.2)
```

# Related

  - [`ConcentrationMargins`](@ref)
  - [`ConcentrationLimit`](@ref)
  - [`ConcentrationLimitEstimator`](@ref)
  - [`weights_meet_bounds`](@ref)
"""
function concentration_margins(cl::ConcentrationLimit, w::VecNum;
                               tol::Number = weights_bound_tolerance(eltype(w)),
                               kwargs...)::ConcentrationMargins
    z = zero(eltype(w))
    v = position_reading(cl.pos, w, max.(w, z), max.(-w, z))
    x = concentration_positions(cl.set, v)
    total = sum(xi for xi in x if xi > cl.t + tol; init = zero(eltype(x))) - cl.s
    return ConcentrationMargins(; x = x, cap = x .- cl.c, total = total)
end
function concentration_margins(cl::ConcentrationLimitEstimator, w::VecNum;
                               sets::Option{<:UniverseSets} = nothing, strict::Bool = false,
                               kwargs...)::ConcentrationMargins
    return concentration_margins(concentration_limit_constraints(cl, sets; strict = strict),
                                 w; kwargs...)
end
function concentration_margins(cl::VecCnlE_Cnl, w::VecNum; kwargs...)
    return [concentration_margins(cli, w; kwargs...) for cli in cl]
end

export GrossPosition, NetPosition, LongPosition, ShortPosition, ExactConcentration,
       SumLargestConcentration, ConcentrationLimitEstimator, ConcentrationLimit,
       ConcentrationMargins, concentration_limit_constraints, concentration_margins
public AbstractPositionReading, AbstractConcentrationAlgorithm, position_reading
