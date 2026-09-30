"""
    assert_panel_terms(x::AbstractString, sym::Sym_Str) -> nothing
    assert_panel_terms(x::AbstractVector{<:Pair{<:AbstractString, <:Real}}, sym::Sym_Str) -> nothing

Check that a Panel Field term is well formed: one non-empty name, or a non-empty vector of `name => coefficient` pairs with non-empty names and finite coefficients.

A term is what the numerator or the denominator of a [`PanelFieldRatio`](@ref) holds. The check runs once, in the constructor, so the read through [`descriptor_field_values`](@ref) can assume a well-formed term.

# Arguments

  - `x`: The term to check.
  - `sym`: Symbolic name of the term, displayed in the error messages.

# Validation

  - A name is not empty. Raises an [`IsEmptyError`](@ref).
  - A vector of pairs is not empty, no name in it is empty, and every coefficient is finite. Raises an [`IsEmptyError`](@ref) or a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`PanelFieldRatio`](@ref)
  - [`panel_term_names`](@ref)
  - [`descriptor_field_values`](@ref)
"""
function assert_panel_terms(x::AbstractString, sym::Sym_Str)::Nothing
    @argcheck(!isempty(x),
              IsEmptyError("$sym names a Panel Field, so it cannot be the empty string"))
    return nothing
end
function assert_panel_terms(x::AbstractVector{<:Pair{<:AbstractString, <:Real}},
                            sym::Sym_Str)::Nothing
    @argcheck(!isempty(x),
              IsEmptyError("$sym is a combination of Panel Fields, so it needs at least one `name => coefficient` term"))
    for (k, (name, c)) in enumerate(x)
        @argcheck(!isempty(name),
                  IsEmptyError("term $k of $sym names a Panel Field, so its name cannot be the empty string"))
        assert_finite(c, "the coefficient of term $k of $sym")
    end
    return nothing
end
"""
    panel_term_names(x::Nothing) -> Vector{String}
    panel_term_names(x::AbstractString) -> Vector{String}
    panel_term_names(x::AbstractVector{<:Pair{<:AbstractString, <:Real}}) -> Vector{String}
    panel_term_names(x::AbstractVector{<:AbstractString}) -> Vector{String}

Return the Panel Field names a term reads, in order.

`nothing` names the returns of an [`EWVolumeRatio`](@ref), which are not a Panel Field, so it reads none. A vector of names is the product side of an [`EWVolumeRatio`](@ref), and it reads each name.

# Arguments

  - `x`: `nothing`, a Panel Field name, a vector of `name => coefficient` pairs, or a vector of names.

# Returns

  - `names::Vector{String}`: One entry per Panel Field the term reads.

# Examples

```jldoctest
julia> PortfolioOptimisers.panel_term_names(\"sales_ttm\")
1-element Vector{String}:
 \"sales_ttm\"

julia> PortfolioOptimisers.panel_term_names([\"sales_ttm\" => 1, \"cost_of_revenue_ttm\" => -1])
2-element Vector{String}:
 \"sales_ttm\"
 \"cost_of_revenue_ttm\"
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`assert_panel_terms`](@ref)
"""
function panel_term_names(::Nothing)::Vector{String}
    return String[]
end
function panel_term_names(x::AbstractString)::Vector{String}
    return [String(x)]
end
function panel_term_names(x::AbstractVector{<:Pair{<:AbstractString, <:Real}})::Vector{String}
    return map(p -> String(first(p)), x)
end
function panel_term_names(x::AbstractVector{<:AbstractString})::Vector{String}
    return map(String, x)
end
"""
    assert_panel_guard_names(names::Nothing, known::VecStr, sym::Sym_Str) -> nothing
    assert_panel_guard_names(names::VecStr, known::VecStr, sym::Sym_Str) -> nothing

Check that every Panel Field a guard names is one the ratio reads.

A guard on a Panel Field the ratio never reads would be checked against nothing, and a typo in a guard would then pass in silence. The check runs in the constructors of [`PanelFieldRatio`](@ref), [`EWVolumeRatio`](@ref) and [`DaysToCover`](@ref).

# Arguments

  - `names`: The guard's Panel Field names, or `nothing` when the guard is off.
  - `known`: The Panel Field names the numerator and the denominator read.
  - `sym`: Symbolic name of the guard, displayed in the error messages.

# Validation

  - `!isempty(names)`. Raises an [`IsEmptyError`](@ref).
  - Every entry of `names` is in `known`. Raises an `ArgumentError` carrying a [`did_you_mean`](@ref) suggestion.

# Returns

  - `nothing`.

# Related

  - [`PanelFieldRatio`](@ref)
  - [`panel_term_names`](@ref)
"""
function assert_panel_guard_names(::Nothing, ::VecStr, ::Sym_Str)::Nothing
    return nothing
end
function assert_panel_guard_names(names::VecStr, known::VecStr, sym::Sym_Str)::Nothing
    @argcheck(!isempty(names),
              IsEmptyError("$sym cannot be empty: pass nothing to turn the guard off"))
    for name in names
        @argcheck(name in known,
                  ArgumentError("$sym names the Panel Field \"$name\", which neither the numerator nor the denominator reads$(did_you_mean(name, known)). The ratio reads: $(join(known, ", "))"))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Divides one Panel Field, or a combination of Panel Fields, by another at every observation.

This is the archetype of every point-in-time ratio Descriptor: a book-to-price, a return on assets, a market leverage. Each named ratio of the library is a constructor function that fixes the Panel Fields this type reads, so `BookToPrice()` prints as a `PanelFieldRatio`, and every one of them accepts a keyword that renames a field. A numerator or a denominator is one Panel Field name, or a vector of `name => coefficient` pairs read as their sum, which is how a gross profit or a total capital enters.

# Mathematical definition

```math
\\begin{align}
u_{t,i} &= \\sum_{k} c_{k}\\, z^{(k)}_{t,i}\\,,\\qquad v_{t,i} = \\sum_{l} c'_{l}\\, y^{(l)}_{t,i}\\\\
d_{t,i} &= \\begin{cases} u_{t,i} / v_{t,i} & \\text{if } v_{t,i} > 0 \\text{ and every guarded field is positive} \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - ``z^{(k)}_{t,i}``, ``c_{k}``: The numerator's Panel Fields and their coefficients. A single name is one field with coefficient one.
  - ``y^{(l)}_{t,i}``, ``c'_{l}``: The denominator's Panel Fields and their coefficients.

A cell that is not observed in any field it reads, or that is not active, is `NaN`.

The three guards separate a data error from a ratio that is not defined. A field that is positive by construction, a price, a market capitalisation, a share count or a total of assets, goes in `gt0`, and a field that is non-negative by construction, a dividend or a sales figure, goes in `nonneg`: a value that breaks the sign is a data error, and the ratio refuses it. A denominator that valid data can make zero or negative, a book equity or an enterprise value, carries no guard: the ratio is not defined for that firm at that observation, and the cell is `NaN`. A named constructor sets the guards of its default fields, and this type with no guard gives `NaN` in every such cell.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PanelFieldRatio(;
        num::Union{<:AbstractString, <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
        den::Union{<:AbstractString, <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
        nonneg::Option{<:VecStr} = nothing,
        pos::Option{<:VecStr} = nothing,
        gt0::Option{<:VecStr} = nothing
    ) -> PanelFieldRatio

Keywords correspond to the struct's fields.

## Validation

  - `num` and `den` are well formed, see [`assert_panel_terms`](@ref).
  - Every name in `nonneg`, `pos` and `gt0` is a Panel Field that `num` or `den` reads, see [`assert_panel_guard_names`](@ref).

# Examples

```jldoctest
julia> PanelFieldRatio(; num = \"book_equity\", den = \"market_cap\")
PanelFieldRatio
     num ┼ String: \"book_equity\"
     den ┼ String: \"market_cap\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ nothing

julia> PanelFieldRatio(; num = [\"sales_ttm\" => 1, \"cost_of_revenue_ttm\" => -1], den = \"sales_ttm\")
PanelFieldRatio
     num ┼ Vector{Pair{String, Int64}}: [\"sales_ttm\" => 1, \"cost_of_revenue_ttm\" => -1]
     den ┼ String: \"sales_ttm\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ nothing
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`positive_divide`](@ref)
  - [`PanelFieldLog`](@ref)
  - [`Passthrough`](@ref)
  - [`BookToPrice`](@ref)
  - [`GrossMargin`](@ref)
  - [`MarketLeverage`](@ref)
"""
@concrete struct PanelFieldRatio <: AbstractDescriptorEstimator
    """
    The numerator: the name of one Panel Field, or a vector of `name => coefficient` pairs read as their sum.
    """
    num
    """
    The denominator, in the same form. The ratio is `NaN` wherever it is not strictly positive.
    """
    den
    """
    $(field_dict[:nonneg_pnl])
    """
    nonneg
    """
    Names of Panel Fields that must be strictly positive for the ratio to be defined, beyond the denominator itself, or `nothing`. The ratio is `NaN` where one of them is not.
    """
    pos
    """
    $(field_dict[:gt0_pnl])
    """
    gt0
    function PanelFieldRatio(num::Union{<:AbstractString,
                                        <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                             den::Union{<:AbstractString,
                                        <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                             nonneg::Option{<:VecStr}, pos::Option{<:VecStr},
                             gt0::Option{<:VecStr})
        assert_panel_terms(num, :num)
        assert_panel_terms(den, :den)
        known = vcat(panel_term_names(num), panel_term_names(den))
        assert_panel_guard_names(nonneg, known, :nonneg)
        assert_panel_guard_names(pos, known, :pos)
        assert_panel_guard_names(gt0, known, :gt0)
        return new{typeof(num), typeof(den), typeof(nonneg), typeof(pos), typeof(gt0)}(num,
                                                                                       den,
                                                                                       nonneg,
                                                                                       pos,
                                                                                       gt0)
    end
end
function PanelFieldRatio(;
                         num::Union{<:AbstractString,
                                    <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                         den::Union{<:AbstractString,
                                    <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                         nonneg::Option{<:VecStr} = nothing,
                         pos::Option{<:VecStr} = nothing,
                         gt0::Option{<:VecStr} = nothing)::PanelFieldRatio
    return PanelFieldRatio(num, den, nonneg, pos, gt0)
end
"""
$(DocStringExtensions.TYPEDEF)

Takes the natural logarithm of one Panel Field at every observation.

The size Descriptor of an equity factor model is the logarithm of the market capitalisation, which tames the right skew of the raw capitalisation. The logarithm is `NaN` wherever the Panel Field is not strictly positive. Under `gt0 = true` a value at or below zero is a data error instead, and the Descriptor refuses it: [`LogMarketCap`](@ref) sets it, because a market capitalisation is positive by construction.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PanelFieldLog(; field::AbstractString, gt0::Bool = false) -> PanelFieldLog

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(field)`.

# Examples

```jldoctest
julia> PanelFieldLog(; field = \"market_cap\")
PanelFieldLog
  field ┼ String: \"market_cap\"
    gt0 ┴ Bool: false
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`LogMarketCap`](@ref)
  - [`PanelFieldRatio`](@ref)
  - [`Passthrough`](@ref)
"""
@concrete struct PanelFieldLog <: AbstractDescriptorEstimator
    """
    Name of the Panel Field whose logarithm is taken.
    """
    field
    """
    Whether the Panel Field must be strictly positive wherever it is observed and active. A value at or below zero then raises a `DomainError`; otherwise its logarithm is `NaN`.
    """
    gt0
    function PanelFieldLog(field::AbstractString, gt0::Bool)
        assert_panel_terms(field, :field)
        return new{typeof(field), typeof(gt0)}(field, gt0)
    end
end
function PanelFieldLog(; field::AbstractString, gt0::Bool = false)::PanelFieldLog
    return PanelFieldLog(field, gt0)
end
"""
$(DocStringExtensions.TYPEDEF)

Returns one numeric Panel Field unchanged, as a Descriptor.

A vendor field that is already a Descriptor, or a value computed upstream of the panel, enters a Factor Exposure through this type. The only change it makes is the two conventions every Descriptor follows: a cell the panel's fill policy touched is `NaN`, and an inactive cell is `NaN`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Passthrough(; field::AbstractString) -> Passthrough

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(field)`.

# Examples

```jldoctest
julia> Passthrough(; field = \"eps_ntm\")
Passthrough
  field ┴ String: \"eps_ntm\"
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`descriptor_field_values`](@ref)
  - [`PanelFieldRatio`](@ref)
  - [`PanelFieldLog`](@ref)
"""
@concrete struct Passthrough <: AbstractDescriptorEstimator
    """
    Name of the Panel Field to return.
    """
    field
    function Passthrough(field::AbstractString)
        assert_panel_terms(field, :field)
        return new{typeof(field)}(field)
    end
end
function Passthrough(; field::AbstractString)::Passthrough
    return Passthrough(field)
end
"""
    assert_panel_field_sign(rd::ReturnsResult, names::Nothing, strict::Bool) -> nothing
    assert_panel_field_sign(rd::ReturnsResult, names::VecStr, strict::Bool) -> nothing

Check the sign of the named Panel Fields on every cell that is observed and active.

A field that is non-negative by construction, a dividend or a sales figure, is checked with `strict = false`. A field that is positive by construction, a price, a market capitalisation, a share count or a total of assets, is checked with `strict = true`. A value that breaks the sign is a data error and not a state of the firm, so the check refuses it rather than write `NaN` into its cell. A cell outside the active mask never reaches the Descriptor, and a cell that is not observed reads back as `NaN`, so neither is checked.

# Algorithm

 1. Read each named Panel Field through [`descriptor_field_values`](@ref).
 2. Find the first cell that is active, observed and below zero, or at or below zero under `strict`. Throw, naming the Panel Field, the observation and the asset.

# Arguments

  - $(arg_dict[:rd])
  - `names`: The Panel Field names to check, or `nothing` for no check.
  - `strict`: `true` to refuse zero as well as a negative value.

# Validation

  - Every active, observed cell of every named Panel Field is `>= 0`, or `> 0` under `strict`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`PanelFieldRatio`](@ref)
  - [`PanelFieldLog`](@ref)
  - [`GrowthRate`](@ref)
  - [`ChangeToScale`](@ref)
  - [`ChangeInIntensity`](@ref)
  - [`EWVolumeRatio`](@ref)
  - [`DaysToCover`](@ref)
  - [`descriptor_field_values`](@ref)
"""
function assert_panel_field_sign(::ReturnsResult, ::Nothing, ::Bool)::Nothing
    return nothing
end
function assert_panel_field_sign(rd::ReturnsResult, names::VecStr, strict::Bool)::Nothing
    for name in names
        V = descriptor_field_values(rd, name)
        amsk = rd.pnl.amsk
        z = zero(eltype(V))
        k = findfirst(k -> amsk[k] && (strict ? V[k] <= z : V[k] < z), eachindex(V, amsk))
        sign = strict ? "strictly positive" : "non-negative"
        @argcheck(isnothing(k),
                  DomainError(isnothing(k) ? NaN : V[k],
                              "the Panel Field \"$name\" must be $sign wherever it is observed and active, and it is $(isnothing(k) ? NaN : V[k]) at observation $(isnothing(k) ? 0 : Tuple(CartesianIndices(V)[k])[1]) for asset $(isnothing(k) ? 0 : Tuple(CartesianIndices(V)[k])[2]). A value of this sign in this field is a data error, so clean the input rather than pass it through."))
    end
    return nothing
end
"""
    positive_panel_fields_fill!(D::AbstractMatrix{<:Real}, rd::ReturnsResult, names::Nothing) -> nothing
    positive_panel_fields_fill!(D::AbstractMatrix{<:Real}, rd::ReturnsResult, names::VecStr) -> nothing

Write `NaN` into every cell of a Descriptor where one of the named Panel Fields is not strictly positive, in place.

# Algorithm

 1. Read each named Panel Field through [`descriptor_field_values`](@ref).
 2. Write `NaN` into `D` wherever the field is zero, negative or `NaN`.

# Arguments

  - `D`: The Descriptor, `observations × assets`, changed in place.
  - $(arg_dict[:rd])
  - `names`: The Panel Field names that must be positive, or `nothing` for no fill.

# Returns

  - `nothing`. `D` carries the filled Descriptor.

# Related

  - [`PanelFieldRatio`](@ref)
  - [`positive_divide`](@ref)
  - [`descriptor_field_values`](@ref)
"""
function positive_panel_fields_fill!(::AbstractMatrix{<:Real}, ::ReturnsResult,
                                     ::Nothing)::Nothing
    return nothing
end
function positive_panel_fields_fill!(D::AbstractMatrix{<:Real}, rd::ReturnsResult,
                                     names::VecStr)::Nothing
    Tf = eltype(D)
    for name in names
        V = descriptor_field_values(rd, name)
        for k in CartesianIndices(D)
            if !(V[k] > zero(eltype(V)))
                D[k] = Tf(NaN)
            end
        end
    end
    return nothing
end
"""
    descriptor(de::PanelFieldRatio, rd::ReturnsResult) -> Matrix{<:Real}
    descriptor(de::PanelFieldLog, rd::ReturnsResult) -> Matrix{<:Real}
    descriptor(de::Passthrough, rd::ReturnsResult) -> Matrix{<:Real}

Compute a point-in-time Descriptor from the Panel Fields of a [`ReturnsResult`](@ref).

The three archetypes read the same way, through [`descriptor_field_values`](@ref), and end the same way, through [`descriptor_active_fill!`](@ref). They part on the arithmetic between the two.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`PanelFieldRatio`](@ref): check the `nonneg` and `gt0` guards, read the numerator and the denominator, divide through [`positive_divide`](@ref), and write `NaN` where a `pos` field is not positive.
 2. [`PanelFieldLog`](@ref): check the field under `gt0`, read it, and take its logarithm where it is strictly positive and `NaN` elsewhere.
 3. [`Passthrough`](@ref): read the field.

Every method then writes `NaN` into the inactive cells.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`descriptor_field_values`](@ref) for every Panel Field the estimator names.
  - The rule of [`assert_panel_field_sign`](@ref) for a [`PanelFieldRatio`](@ref) with a `nonneg` or a `gt0` guard, and for a [`PanelFieldLog`](@ref) under `gt0 = true`. A named constructor sets these guards, so `BookToPrice()` and `LogMarketCap()` refuse the zero market capitalisation of the example below.

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"book_equity\", vals = [2.0 3.0; 4.0 5.0]),
                          NumericPanelInput(; name = \"market_cap\", vals = [4.0 0.0; 8.0 10.0])];
                         amsk = [true true; false true], emsk = [true true; false true]);

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), pnl = pnl);

julia> descriptor(PanelFieldRatio(; num = \"book_equity\", den = \"market_cap\"), rd)
2×2 Matrix{Float64}:
   0.5  NaN
 NaN      0.5

julia> descriptor(PanelFieldLog(; field = \"market_cap\"), rd)
2×2 Matrix{Float64}:
   1.38629  NaN
 NaN          2.30259

julia> descriptor(Passthrough(; field = \"market_cap\"), rd)
2×2 Matrix{Float64}:
   4.0   0.0
 NaN    10.0
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`PanelFieldRatio`](@ref)
  - [`PanelFieldLog`](@ref)
  - [`Passthrough`](@ref)
  - [`descriptor_field_values`](@ref)
  - [`positive_divide`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function descriptor(de::PanelFieldRatio, rd::ReturnsResult)::Matrix{<:Real}
    assert_panel_field_sign(rd, de.nonneg, false)
    assert_panel_field_sign(rd, de.gt0, true)
    D = positive_divide.(descriptor_field_values(rd, de.num),
                         descriptor_field_values(rd, de.den))
    positive_panel_fields_fill!(D, rd, de.pos)
    descriptor_active_fill!(D, rd.pnl)
    return D
end
function descriptor(de::PanelFieldLog, rd::ReturnsResult)::Matrix{<:Real}
    if de.gt0
        assert_panel_field_sign(rd, [String(de.field)], true)
    end
    D = descriptor_field_values(rd, de.field)
    Tf = eltype(D)
    for k in eachindex(D)
        D[k] = D[k] > zero(Tf) ? log(D[k]) : Tf(NaN)
    end
    descriptor_active_fill!(D, rd.pnl)
    return D
end
function descriptor(de::Passthrough, rd::ReturnsResult)::Matrix{<:Real}
    D = descriptor_field_values(rd, de.field)
    descriptor_active_fill!(D, rd.pnl)
    return D
end
"""
    BookToPrice(; num::AbstractString = "book_equity",
                den::AbstractString = "market_cap") -> PanelFieldRatio

Book equity over market capitalisation, the value Descriptor.

The ratio is `book_equity / market_cap` at each observation. A market capitalisation at or below zero is a data error, and the estimator raises on one. A negative book equity is kept, because it carries information about the balance sheet. The aggregate form is used rather than the per-share form, because it cannot suffer a split-adjustment mismatch between its two sides.

# Arguments

  - `num`: Name of the book equity Panel Field.
  - `den`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `gt0 = [den]`.

# Examples

```jldoctest
julia> BookToPrice()
PanelFieldRatio
     num ┼ String: \"book_equity\"
     den ┼ String: \"market_cap\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"market_cap\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`SalesToPrice`](@ref)
  - [`CashFlowToPrice`](@ref)
"""
function BookToPrice(; num::AbstractString = "book_equity",
                     den::AbstractString = "market_cap")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, gt0 = [String(den)])
end
"""
    CashFlowToPrice(; num::AbstractString = "operating_cash_flow_ttm",
                    den::AbstractString = "market_cap") -> PanelFieldRatio

Trailing operating cash flow over market capitalisation, a value Descriptor.

The ratio is `operating_cash_flow_ttm / market_cap`. A market capitalisation at or below zero is a data error, and the estimator raises on one. An operating cash flow can be negative, so the Descriptor can too. It is less exposed to accrual accounting choices than an earnings ratio.

# Arguments

  - `num`: Name of the operating cash flow Panel Field.
  - `den`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `gt0 = [den]`.

# Examples

```jldoctest
julia> CashFlowToPrice()
PanelFieldRatio
     num ┼ String: \"operating_cash_flow_ttm\"
     den ┼ String: \"market_cap\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"market_cap\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`BookToPrice`](@ref)
  - [`CashFlowToAssets`](@ref)
"""
function CashFlowToPrice(; num::AbstractString = "operating_cash_flow_ttm",
                         den::AbstractString = "market_cap")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, gt0 = [String(den)])
end
"""
    SalesToPrice(; num::AbstractString = "sales_ttm",
                 den::AbstractString = "market_cap") -> PanelFieldRatio

Trailing sales over market capitalisation, a value Descriptor.

The ratio is `sales_ttm / market_cap`. A market capitalisation at or below zero is a data error, and the estimator raises on one. Sales are the least exposed of the fundamentals to accounting choices, and the ratio stays defined for a firm whose earnings or book equity are negative.

# Arguments

  - `num`: Name of the sales Panel Field.
  - `den`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `gt0 = [den]`.

# Examples

```jldoctest
julia> SalesToPrice()
PanelFieldRatio
     num ┼ String: \"sales_ttm\"
     den ┼ String: \"market_cap\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"market_cap\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`BookToPrice`](@ref)
  - [`SalesToEnterpriseValue`](@ref)
"""
function SalesToPrice(; num::AbstractString = "sales_ttm",
                      den::AbstractString = "market_cap")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, gt0 = [String(den)])
end
"""
    EarningsToPrice(; num::AbstractString = "net_income_ttm",
                    den::AbstractString = "market_cap") -> PanelFieldRatio

Trailing net income over market capitalisation, the earnings yield Descriptor.

The ratio is `net_income_ttm / market_cap`. A market capitalisation at or below zero is a data error, and the estimator raises on one. A loss makes it negative, which the price-to-earnings inverse would not survive.

# Arguments

  - `num`: Name of the net income Panel Field.
  - `den`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `gt0 = [den]`.

# Examples

```jldoctest
julia> EarningsToPrice()
PanelFieldRatio
     num ┼ String: \"net_income_ttm\"
     den ┼ String: \"market_cap\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"market_cap\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`ForwardEarningsToPrice`](@ref)
  - [`EbitdaToEnterpriseValue`](@ref)
"""
function EarningsToPrice(; num::AbstractString = "net_income_ttm",
                         den::AbstractString = "market_cap")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, gt0 = [String(den)])
end
"""
    ForwardEarningsToPrice(; num::AbstractString = "eps_ntm",
                           den::AbstractString = "adj_close") -> PanelFieldRatio

Forward earnings per share over the adjusted close, the forward earnings yield Descriptor.

The ratio is `eps_ntm / adj_close`. A price at or below zero is a data error, and the estimator raises on one. Both sides are per share, so both must be on one split-adjustment basis.

# Arguments

  - `num`: Name of the forward earnings per share Panel Field.
  - `den`: Name of the adjusted close Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `gt0 = [den]`.

# Examples

```jldoctest
julia> ForwardEarningsToPrice()
PanelFieldRatio
     num ┼ String: \"eps_ntm\"
     den ┼ String: \"adj_close\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"adj_close\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`EarningsToPrice`](@ref)
  - [`AnalystDispersionToPrice`](@ref)
"""
function ForwardEarningsToPrice(; num::AbstractString = "eps_ntm",
                                den::AbstractString = "adj_close")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, gt0 = [String(den)])
end
"""
    EbitdaToEnterpriseValue(; num::AbstractString = "ebitda_ttm",
                            den::AbstractString = "enterprise_value") -> PanelFieldRatio

Trailing EBITDA over enterprise value, an earnings yield Descriptor that is neutral to the capital structure.

The ratio is `ebitda_ttm / enterprise_value`, `NaN` where the enterprise value is not strictly positive. The enterprise value is a Panel Field the caller supplies, market capitalisation plus debt less cash, and this estimator does not rebuild it.

# Arguments

  - `num`: Name of the EBITDA Panel Field.
  - `den`: Name of the enterprise value Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed.

# Examples

```jldoctest
julia> EbitdaToEnterpriseValue()
PanelFieldRatio
     num ┼ String: \"ebitda_ttm\"
     den ┼ String: \"enterprise_value\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`EarningsToPrice`](@ref)
  - [`SalesToEnterpriseValue`](@ref)
"""
function EbitdaToEnterpriseValue(; num::AbstractString = "ebitda_ttm",
                                 den::AbstractString = "enterprise_value")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den)
end
"""
    DividendToPrice(; num::AbstractString = "dividends_ttm",
                    den::AbstractString = "market_cap") -> PanelFieldRatio

Trailing common dividends over market capitalisation, the dividend yield Descriptor.

The ratio is `dividends_ttm / market_cap`. A market capitalisation at or below zero is a data error, and the estimator raises on one. The dividends must be non-negative wherever they are observed: a negative dividend is a data error, and the estimator raises on one.

# Arguments

  - `num`: Name of the dividends Panel Field.
  - `den`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed, `nonneg = [num]` and `gt0 = [den]`.

# Examples

```jldoctest
julia> DividendToPrice()
PanelFieldRatio
     num ┼ String: \"dividends_ttm\"
     den ┼ String: \"market_cap\"
  nonneg ┼ Vector{String}: [\"dividends_ttm\"]
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"market_cap\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`ForwardDividendToPrice`](@ref)
  - [`ShareholderYield`](@ref)
"""
function DividendToPrice(; num::AbstractString = "dividends_ttm",
                         den::AbstractString = "market_cap")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, nonneg = [String(num)],
                           gt0 = [String(den)])
end
"""
    ForwardDividendToPrice(; num::AbstractString = "dps_ntm",
                           den::AbstractString = "adj_close") -> PanelFieldRatio

Forward dividends per share over the adjusted close, the forward dividend yield Descriptor.

The ratio is `dps_ntm / adj_close`. A price at or below zero is a data error, and the estimator raises on one. The dividends must be non-negative wherever they are observed, and the estimator raises on a negative one.

# Arguments

  - `num`: Name of the forward dividends per share Panel Field.
  - `den`: Name of the adjusted close Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed, `nonneg = [num]` and `gt0 = [den]`.

# Examples

```jldoctest
julia> ForwardDividendToPrice()
PanelFieldRatio
     num ┼ String: \"dps_ntm\"
     den ┼ String: \"adj_close\"
  nonneg ┼ Vector{String}: [\"dps_ntm\"]
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"adj_close\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`DividendToPrice`](@ref)
"""
function ForwardDividendToPrice(; num::AbstractString = "dps_ntm",
                                den::AbstractString = "adj_close")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, nonneg = [String(num)],
                           gt0 = [String(den)])
end
"""
    ShareholderYield(; dividends::AbstractString = "dividends_ttm",
                     buybacks::AbstractString = "net_buybacks_ttm",
                     den::AbstractString = "market_cap") -> PanelFieldRatio

Trailing dividends plus net buybacks over market capitalisation, the total payout Descriptor.

The ratio is `(dividends_ttm + net_buybacks_ttm) / market_cap`. A market capitalisation at or below zero is a data error, and the estimator raises on one. The dividends must be non-negative wherever they are observed, and the estimator raises on a negative one. Net buybacks can be negative, because a net issuance is one.

# Arguments

  - `dividends`: Name of the dividends Panel Field.
  - `buybacks`: Name of the net buybacks Panel Field.
  - `den`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with `num = [dividends => 1, buybacks => 1]`, `nonneg = [dividends]` and `gt0 = [den]`.

# Examples

```jldoctest
julia> ShareholderYield()
PanelFieldRatio
     num ┼ Vector{Pair{String, Int64}}: [\"dividends_ttm\" => 1, \"net_buybacks_ttm\" => 1]
     den ┼ String: \"market_cap\"
  nonneg ┼ Vector{String}: [\"dividends_ttm\"]
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"market_cap\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`DividendToPrice`](@ref)
"""
function ShareholderYield(; dividends::AbstractString = "dividends_ttm",
                          buybacks::AbstractString = "net_buybacks_ttm",
                          den::AbstractString = "market_cap")::PanelFieldRatio
    return PanelFieldRatio(; num = [String(dividends) => 1, String(buybacks) => 1],
                           den = den, nonneg = [String(dividends)], gt0 = [String(den)])
end
"""
    BookLeverage(; debt::AbstractString = "total_debt",
                 equity::AbstractString = "book_equity") -> PanelFieldRatio

Total debt over total book capital, the book leverage Descriptor.

The ratio is `total_debt / (total_debt + book_equity)`, `NaN` where the total capital is not strictly positive, which a negative book equity can make it. A negative debt is a data error, and the estimator raises on one. It is bounded in `[0, 1]` for a firm whose book equity is positive, which is why it is preferred to the debt-to-equity ratio it is a monotone function of. A negative book equity that leaves the total capital positive gives a ratio above one, which is a valid signal of distress.

# Arguments

  - `debt`: Name of the total debt Panel Field.
  - `equity`: Name of the book equity Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with `num = debt`, `den = [debt => 1, equity => 1]` and `nonneg = [debt]`.

# Examples

```jldoctest
julia> BookLeverage()
PanelFieldRatio
     num ┼ String: \"total_debt\"
     den ┼ Vector{Pair{String, Int64}}: [\"total_debt\" => 1, \"book_equity\" => 1]
  nonneg ┼ Vector{String}: [\"total_debt\"]
     pos ┼ nothing
     gt0 ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`MarketLeverage`](@ref)
  - [`DebtToAssets`](@ref)
"""
function BookLeverage(; debt::AbstractString = "total_debt",
                      equity::AbstractString = "book_equity")::PanelFieldRatio
    return PanelFieldRatio(; num = debt, den = [String(debt) => 1, String(equity) => 1],
                           nonneg = [String(debt)])
end
"""
    MarketLeverage(; debt::AbstractString = "total_debt",
                   mcap::AbstractString = "market_cap") -> PanelFieldRatio

Total debt over total market capital, the market leverage Descriptor.

The ratio is `total_debt / (total_debt + market_cap)`. A market capitalisation at or below zero or a negative debt is a data error, and the estimator raises on one, so the total capital is always positive. It reprices the equity leg of the capital structure every observation, where [`BookLeverage`](@ref) reads it from the balance sheet.

# Arguments

  - `debt`: Name of the total debt Panel Field.
  - `mcap`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with `num = debt`, `den = [debt => 1, mcap => 1]`, `nonneg = [debt]` and `gt0 = [mcap]`.

# Examples

```jldoctest
julia> MarketLeverage()
PanelFieldRatio
     num ┼ String: \"total_debt\"
     den ┼ Vector{Pair{String, Int64}}: [\"total_debt\" => 1, \"market_cap\" => 1]
  nonneg ┼ Vector{String}: [\"total_debt\"]
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"market_cap\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`BookLeverage`](@ref)
  - [`DebtToAssets`](@ref)
"""
function MarketLeverage(; debt::AbstractString = "total_debt",
                        mcap::AbstractString = "market_cap")::PanelFieldRatio
    return PanelFieldRatio(; num = debt, den = [String(debt) => 1, String(mcap) => 1],
                           nonneg = [String(debt)], gt0 = [String(mcap)])
end
"""
    DebtToAssets(; num::AbstractString = "total_debt",
                 den::AbstractString = "total_assets") -> PanelFieldRatio

Total debt over total assets, a leverage Descriptor.

The ratio is `total_debt / total_assets`. Total assets at or below zero are a data error, and the estimator raises on them.

# Arguments

  - `num`: Name of the total debt Panel Field.
  - `den`: Name of the total assets Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `gt0 = [den]`.

# Examples

```jldoctest
julia> DebtToAssets()
PanelFieldRatio
     num ┼ String: \"total_debt\"
     den ┼ String: \"total_assets\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"total_assets\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`BookLeverage`](@ref)
  - [`MarketLeverage`](@ref)
"""
function DebtToAssets(; num::AbstractString = "total_debt",
                      den::AbstractString = "total_assets")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, gt0 = [String(den)])
end
"""
    GrossProfitability(; sales::AbstractString = "sales_ttm",
                       cogs::AbstractString = "cost_of_revenue_ttm",
                       den::AbstractString = "total_assets") -> PanelFieldRatio

Gross profit over total assets, the gross profitability Descriptor.

The ratio is `(sales_ttm - cost_of_revenue_ttm) / total_assets`. Total assets at or below zero are a data error, and the estimator raises on them. Gross profit sits above the accounting choices that shape net income, which is what makes it the cleaner profitability signal.

# Arguments

  - `sales`: Name of the sales Panel Field.
  - `cogs`: Name of the cost of revenue Panel Field.
  - `den`: Name of the total assets Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with `num = [sales => 1, cogs => -1]` and `gt0 = [den]`.

# Examples

```jldoctest
julia> GrossProfitability()
PanelFieldRatio
     num ┼ Vector{Pair{String, Int64}}: [\"sales_ttm\" => 1, \"cost_of_revenue_ttm\" => -1]
     den ┼ String: \"total_assets\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"total_assets\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`GrossMargin`](@ref)
  - [`ReturnOnAssets`](@ref)
"""
function GrossProfitability(; sales::AbstractString = "sales_ttm",
                            cogs::AbstractString = "cost_of_revenue_ttm",
                            den::AbstractString = "total_assets")::PanelFieldRatio
    return PanelFieldRatio(; num = [String(sales) => 1, String(cogs) => -1], den = den,
                           gt0 = [String(den)])
end
"""
    GrossMargin(; sales::AbstractString = "sales_ttm",
                cogs::AbstractString = "cost_of_revenue_ttm") -> PanelFieldRatio

Gross profit over sales, the gross margin Descriptor.

The ratio is `(sales_ttm - cost_of_revenue_ttm) / sales_ttm`, `NaN` where the sales are zero. Negative sales are a data error, and the estimator raises on them.

# Arguments

  - `sales`: Name of the sales Panel Field.
  - `cogs`: Name of the cost of revenue Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with `num = [sales => 1, cogs => -1]`, `den = sales` and `nonneg = [sales]`.

# Examples

```jldoctest
julia> GrossMargin()
PanelFieldRatio
     num ┼ Vector{Pair{String, Int64}}: [\"sales_ttm\" => 1, \"cost_of_revenue_ttm\" => -1]
     den ┼ String: \"sales_ttm\"
  nonneg ┼ Vector{String}: [\"sales_ttm\"]
     pos ┼ nothing
     gt0 ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`GrossProfitability`](@ref)
"""
function GrossMargin(; sales::AbstractString = "sales_ttm",
                     cogs::AbstractString = "cost_of_revenue_ttm")::PanelFieldRatio
    return PanelFieldRatio(; num = [String(sales) => 1, String(cogs) => -1], den = sales,
                           nonneg = [String(sales)])
end
"""
    ReturnOnAssets(; num::AbstractString = "net_income_ttm",
                   den::AbstractString = "total_assets") -> PanelFieldRatio

Trailing net income over total assets, the return on assets Descriptor.

The ratio is `net_income_ttm / total_assets`. Total assets at or below zero are a data error, and the estimator raises on them.

# Arguments

  - `num`: Name of the net income Panel Field.
  - `den`: Name of the total assets Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `gt0 = [den]`.

# Examples

```jldoctest
julia> ReturnOnAssets()
PanelFieldRatio
     num ┼ String: \"net_income_ttm\"
     den ┼ String: \"total_assets\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"total_assets\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`ReturnOnEquity`](@ref)
  - [`GrossProfitability`](@ref)
"""
function ReturnOnAssets(; num::AbstractString = "net_income_ttm",
                        den::AbstractString = "total_assets")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, gt0 = [String(den)])
end
"""
    ReturnOnEquity(; num::AbstractString = "net_income_ttm",
                   den::AbstractString = "book_equity") -> PanelFieldRatio

Trailing net income over book equity, the return on equity Descriptor.

The ratio is `net_income_ttm / book_equity`, `NaN` where the book equity is not strictly positive. A negative book equity would flip the sign of the ratio, so it is `NaN` rather than a number that ranks a distressed firm as profitable.

# Arguments

  - `num`: Name of the net income Panel Field.
  - `den`: Name of the book equity Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed.

# Examples

```jldoctest
julia> ReturnOnEquity()
PanelFieldRatio
     num ┼ String: \"net_income_ttm\"
     den ┼ String: \"book_equity\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`ReturnOnAssets`](@ref)
"""
function ReturnOnEquity(; num::AbstractString = "net_income_ttm",
                        den::AbstractString = "book_equity")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den)
end
"""
    AssetTurnover(; num::AbstractString = "sales_ttm",
                  den::AbstractString = "total_assets") -> PanelFieldRatio

Trailing sales over total assets, the asset turnover Descriptor.

The ratio is `sales_ttm / total_assets`. Total assets at or below zero are a data error, and the estimator raises on them.

# Arguments

  - `num`: Name of the sales Panel Field.
  - `den`: Name of the total assets Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `gt0 = [den]`.

# Examples

```jldoctest
julia> AssetTurnover()
PanelFieldRatio
     num ┼ String: \"sales_ttm\"
     den ┼ String: \"total_assets\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"total_assets\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`GrossProfitability`](@ref)
"""
function AssetTurnover(; num::AbstractString = "sales_ttm",
                       den::AbstractString = "total_assets")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, gt0 = [String(den)])
end
"""
    CashFlowToAssets(; num::AbstractString = "operating_cash_flow_ttm",
                     den::AbstractString = "total_assets") -> PanelFieldRatio

Trailing operating cash flow over total assets, a profitability Descriptor.

The ratio is `operating_cash_flow_ttm / total_assets`. Total assets at or below zero are a data error, and the estimator raises on them.

# Arguments

  - `num`: Name of the operating cash flow Panel Field.
  - `den`: Name of the total assets Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `gt0 = [den]`.

# Examples

```jldoctest
julia> CashFlowToAssets()
PanelFieldRatio
     num ┼ String: \"operating_cash_flow_ttm\"
     den ┼ String: \"total_assets\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"total_assets\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`CashFlowToPrice`](@ref)
  - [`ReturnOnAssets`](@ref)
"""
function CashFlowToAssets(; num::AbstractString = "operating_cash_flow_ttm",
                          den::AbstractString = "total_assets")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, gt0 = [String(den)])
end
"""
    SalesToEnterpriseValue(; num::AbstractString = "sales_ttm",
                           den::AbstractString = "enterprise_value") -> PanelFieldRatio

Trailing sales over enterprise value, a profitability Descriptor that is neutral to the capital structure.

The ratio is `sales_ttm / enterprise_value`, `NaN` where the enterprise value is not strictly positive.

# Arguments

  - `num`: Name of the sales Panel Field.
  - `den`: Name of the enterprise value Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed.

# Examples

```jldoctest
julia> SalesToEnterpriseValue()
PanelFieldRatio
     num ┼ String: \"sales_ttm\"
     den ┼ String: \"enterprise_value\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`SalesToPrice`](@ref)
  - [`EbitdaToEnterpriseValue`](@ref)
"""
function SalesToEnterpriseValue(; num::AbstractString = "sales_ttm",
                                den::AbstractString = "enterprise_value")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den)
end
"""
    AccrualsCashFlow(; income::AbstractString = "net_income_ttm",
                     cash_flow::AbstractString = "operating_cash_flow_ttm",
                     den::AbstractString = "total_assets") -> PanelFieldRatio

Accruals over total assets, the earnings quality Descriptor.

The ratio is `(net_income_ttm - operating_cash_flow_ttm) / total_assets`. Total assets at or below zero are a data error, and the estimator raises on them. A large positive value says that the reported income ran ahead of the cash the business collected.

# Arguments

  - `income`: Name of the net income Panel Field.
  - `cash_flow`: Name of the operating cash flow Panel Field.
  - `den`: Name of the total assets Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with `num = [income => 1, cash_flow => -1]` and `gt0 = [den]`.

# Examples

```jldoctest
julia> AccrualsCashFlow()
PanelFieldRatio
     num ┼ Vector{Pair{String, Int64}}: [\"net_income_ttm\" => 1, \"operating_cash_flow_ttm\" => -1]
     den ┼ String: \"total_assets\"
  nonneg ┼ nothing
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"total_assets\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`AnalystDispersionToPrice`](@ref)
  - [`CashFlowToAssets`](@ref)
"""
function AccrualsCashFlow(; income::AbstractString = "net_income_ttm",
                          cash_flow::AbstractString = "operating_cash_flow_ttm",
                          den::AbstractString = "total_assets")::PanelFieldRatio
    return PanelFieldRatio(; num = [String(income) => 1, String(cash_flow) => -1],
                           den = den, gt0 = [String(den)])
end
"""
    AnalystDispersionToPrice(; num::AbstractString = "eps_ntm_std",
                             den::AbstractString = "adj_close") -> PanelFieldRatio

Dispersion of the forward earnings estimates over the adjusted close, an earnings quality Descriptor.

The ratio is `eps_ntm_std / adj_close`. A price at or below zero is a data error, and the estimator raises on one. A standard deviation is non-negative, so the estimator raises on a negative dispersion.

# Arguments

  - `num`: Name of the forward earnings dispersion Panel Field.
  - `den`: Name of the adjusted close Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed, `nonneg = [num]` and `gt0 = [den]`.

# Examples

```jldoctest
julia> AnalystDispersionToPrice()
PanelFieldRatio
     num ┼ String: \"eps_ntm_std\"
     den ┼ String: \"adj_close\"
  nonneg ┼ Vector{String}: [\"eps_ntm_std\"]
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"adj_close\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`ForwardEarningsToPrice`](@ref)
  - [`AccrualsCashFlow`](@ref)
"""
function AnalystDispersionToPrice(; num::AbstractString = "eps_ntm_std",
                                  den::AbstractString = "adj_close")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, nonneg = [String(num)],
                           gt0 = [String(den)])
end
"""
    LogMarketCap(; field::AbstractString = "market_cap") -> PanelFieldLog

Natural logarithm of the market capitalisation, the size Descriptor.

The value is `log(market_cap)`. A market capitalisation at or below zero is a data error, and the estimator raises on one. The logarithm tames the right skew of the raw capitalisation and gives the cross-section a stable scale.

# Arguments

  - `field`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldLog`: The estimator, with the Panel Field fixed and `gt0 = true`.

# Examples

```jldoctest
julia> LogMarketCap()
PanelFieldLog
  field ┼ String: \"market_cap\"
    gt0 ┴ Bool: true
```

# Related

  - [`PanelFieldLog`](@ref)
  - [`descriptor`](@ref)
"""
function LogMarketCap(; field::AbstractString = "market_cap")::PanelFieldLog
    return PanelFieldLog(; field = field, gt0 = true)
end
"""
    ShortInterest(; num::AbstractString = "short_interest",
                  den::AbstractString = "adj_shares_outstanding") -> PanelFieldRatio

Shares sold short over shares outstanding, the short interest Descriptor.

The ratio is `short_interest / adj_shares_outstanding`. A share count at or below zero is a data error, and the estimator raises on one. The short interest must be non-negative wherever it is observed, and the estimator raises on a negative one. Both sides are share counts, so both must be on one split-adjustment basis.

# Arguments

  - `num`: Name of the short interest Panel Field.
  - `den`: Name of the shares outstanding Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed, `nonneg = [num]` and `gt0 = [den]`.

# Examples

```jldoctest
julia> ShortInterest()
PanelFieldRatio
     num ┼ String: \"short_interest\"
     den ┼ String: \"adj_shares_outstanding\"
  nonneg ┼ Vector{String}: [\"short_interest\"]
     pos ┼ nothing
     gt0 ┴ Vector{String}: [\"adj_shares_outstanding\"]
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`DividendToPrice`](@ref)
"""
function ShortInterest(; num::AbstractString = "short_interest",
                       den::AbstractString = "adj_shares_outstanding")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, nonneg = [String(num)],
                           gt0 = [String(den)])
end

export PanelFieldRatio, PanelFieldLog, Passthrough, BookToPrice, CashFlowToPrice,
       SalesToPrice, EarningsToPrice, ForwardEarningsToPrice, EbitdaToEnterpriseValue,
       DividendToPrice, ForwardDividendToPrice, ShareholderYield, BookLeverage,
       MarketLeverage, DebtToAssets, GrossProfitability, GrossMargin, ReturnOnAssets,
       ReturnOnEquity, AssetTurnover, CashFlowToAssets, SalesToEnterpriseValue,
       AccrualsCashFlow, AnalystDispersionToPrice, LogMarketCap, ShortInterest
