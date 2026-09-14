"""
    assert_panel_terms(x::AbstractString, sym::Sym_Str) -> nothing
    assert_panel_terms(x::AbstractVector{<:Pair{<:AbstractString, <:Real}}, sym::Sym_Str) -> nothing

Check that a Panel Field term is well formed: one non-empty name, or a non-empty vector of `name => coefficient` pairs with non-empty names and finite coefficients.

A term is what the numerator or the denominator of a [`PanelFieldRatio`](@ref) holds. The check runs once, in the constructor, so the read through [`panel_field_values`](@ref) can assume a well-formed term.

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
  - [`panel_field_values`](@ref)
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
    panel_term_names(x::AbstractString) -> Vector{String}
    panel_term_names(x::AbstractVector{<:Pair{<:AbstractString, <:Real}}) -> Vector{String}

Return the Panel Field names a term reads, in order.

# Arguments

  - `x`: A Panel Field name, or a vector of `name => coefficient` pairs.

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
function panel_term_names(x::AbstractString)::Vector{String}
    return [String(x)]
end
function panel_term_names(x::AbstractVector{<:Pair{<:AbstractString, <:Real}})::Vector{String}
    return map(p -> String(first(p)), x)
end
"""
    assert_panel_guard_names(names::Nothing, known::VecStr, sym::Sym_Str) -> nothing
    assert_panel_guard_names(names::VecStr, known::VecStr, sym::Sym_Str) -> nothing

Check that every Panel Field a guard names is one the ratio reads.

A guard on a Panel Field the ratio never reads would be checked against nothing, and a typo in a guard would then pass in silence. The check runs in the constructor of [`PanelFieldRatio`](@ref).

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

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PanelFieldRatio(;
        num::Union{<:AbstractString, <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
        den::Union{<:AbstractString, <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
        nonneg::Option{<:VecStr} = nothing,
        pos::Option{<:VecStr} = nothing
    ) -> PanelFieldRatio

Keywords correspond to the struct's fields.

## Validation

  - `num` and `den` are well formed, see [`assert_panel_terms`](@ref).
  - Every name in `nonneg` and in `pos` is a Panel Field that `num` or `den` reads, see [`assert_panel_guard_names`](@ref).

# Examples

```jldoctest
julia> PanelFieldRatio(; num = \"book_equity\", den = \"market_cap\")
PanelFieldRatio
     num ┼ String: \"book_equity\"
     den ┼ String: \"market_cap\"
  nonneg ┼ nothing
     pos ┴ nothing

julia> PanelFieldRatio(; num = [\"sales_ttm\" => 1, \"cost_of_revenue_ttm\" => -1], den = \"sales_ttm\")
PanelFieldRatio
     num ┼ Vector{Pair{String, Int64}}: [\"sales_ttm\" => 1, \"cost_of_revenue_ttm\" => -1]
     den ┼ String: \"sales_ttm\"
  nonneg ┼ nothing
     pos ┴ nothing
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
    Names of Panel Fields that must be non-negative wherever they are observed and active, or `nothing`. A negative value raises a `DomainError`, because a dividend, a short interest or a dispersion below zero is a data error and not a signal.
    """
    nonneg
    """
    Names of Panel Fields that must be strictly positive for the ratio to be defined, beyond the denominator itself, or `nothing`. The ratio is `NaN` where one of them is not.
    """
    pos
    function PanelFieldRatio(num::Union{<:AbstractString,
                                        <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                             den::Union{<:AbstractString,
                                        <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                             nonneg::Option{<:VecStr}, pos::Option{<:VecStr})
        assert_panel_terms(num, :num)
        assert_panel_terms(den, :den)
        known = vcat(panel_term_names(num), panel_term_names(den))
        assert_panel_guard_names(nonneg, known, :nonneg)
        assert_panel_guard_names(pos, known, :pos)
        return new{typeof(num), typeof(den), typeof(nonneg), typeof(pos)}(num, den, nonneg,
                                                                          pos)
    end
end
function PanelFieldRatio(;
                         num::Union{<:AbstractString,
                                    <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                         den::Union{<:AbstractString,
                                    <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                         nonneg::Option{<:VecStr} = nothing,
                         pos::Option{<:VecStr} = nothing)::PanelFieldRatio
    return PanelFieldRatio(num, den, nonneg, pos)
end
"""
$(DocStringExtensions.TYPEDEF)

Takes the natural logarithm of one Panel Field at every observation.

The size Descriptor of an equity factor model is the logarithm of the market capitalisation, which tames the right skew of the raw capitalisation. The logarithm is `NaN` wherever the Panel Field is not strictly positive.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PanelFieldLog(; field::AbstractString) -> PanelFieldLog

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(field)`.

# Examples

```jldoctest
julia> PanelFieldLog(; field = \"market_cap\")
PanelFieldLog
  field ┴ String: \"market_cap\"
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
    function PanelFieldLog(field::AbstractString)
        assert_panel_terms(field, :field)
        return new{typeof(field)}(field)
    end
end
function PanelFieldLog(; field::AbstractString)::PanelFieldLog
    return PanelFieldLog(field)
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
  - [`panel_field_values`](@ref)
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
    assert_nonneg_panel_fields(rd::ReturnsResult, names::Nothing) -> nothing
    assert_nonneg_panel_fields(rd::ReturnsResult, names::VecStr) -> nothing

Check that the named Panel Fields are non-negative on every cell that is observed and active.

# Algorithm

 1. Read each named Panel Field through [`panel_field_values`](@ref).
 2. Find the first cell that is active, observed and negative. Throw, naming the Panel Field, the observation and the asset.

# Arguments

  - $(arg_dict[:rd])
  - `names`: The Panel Field names to check, or `nothing` for no check.

# Validation

  - Every active, observed cell of every named Panel Field is `>= 0`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`PanelFieldRatio`](@ref)
  - [`panel_field_values`](@ref)
"""
function assert_nonneg_panel_fields(::ReturnsResult, ::Nothing)::Nothing
    return nothing
end
function assert_nonneg_panel_fields(rd::ReturnsResult, names::VecStr)::Nothing
    for name in names
        V = panel_field_values(rd, name)
        amsk = rd.pnl.amsk
        k = findfirst(k -> amsk[k] && V[k] < zero(eltype(V)), eachindex(V, amsk))
        @argcheck(isnothing(k),
                  DomainError(isnothing(k) ? NaN : V[k],
                              "the Panel Field \"$name\" must be non-negative wherever it is observed and active, and it is negative at observation $(isnothing(k) ? 0 : Tuple(CartesianIndices(V)[k])[1]) for asset $(isnothing(k) ? 0 : Tuple(CartesianIndices(V)[k])[2]). A negative value in this field is a data error, so clean the input rather than pass it through."))
    end
    return nothing
end
"""
    positive_panel_fields_fill!(D::AbstractMatrix{<:Real}, rd::ReturnsResult, names::Nothing) -> nothing
    positive_panel_fields_fill!(D::AbstractMatrix{<:Real}, rd::ReturnsResult, names::VecStr) -> nothing

Write `NaN` into every cell of a Descriptor where one of the named Panel Fields is not strictly positive, in place.

# Algorithm

 1. Read each named Panel Field through [`panel_field_values`](@ref).
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
  - [`panel_field_values`](@ref)
"""
function positive_panel_fields_fill!(::AbstractMatrix{<:Real}, ::ReturnsResult,
                                     ::Nothing)::Nothing
    return nothing
end
function positive_panel_fields_fill!(D::AbstractMatrix{<:Real}, rd::ReturnsResult,
                                     names::VecStr)::Nothing
    Tf = eltype(D)
    for name in names
        V = panel_field_values(rd, name)
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

Compute a point-in-time Descriptor from the Panel Fields of a carrier.

The three archetypes read the same way, through [`panel_field_values`](@ref), and end the same way, through [`descriptor_active_fill!`](@ref). They part on the arithmetic between the two.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`PanelFieldRatio`](@ref): check the `nonneg` guard, read the numerator and the denominator, divide through [`positive_divide`](@ref), and write `NaN` where a `pos` field is not positive.
 2. [`PanelFieldLog`](@ref): read the field, and take its logarithm where it is strictly positive and `NaN` elsewhere.
 3. [`Passthrough`](@ref): read the field.

Every method then writes `NaN` into the inactive cells.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`panel_field_values`](@ref) for every Panel Field the estimator names.
  - The rule of [`assert_nonneg_panel_fields`](@ref) for a [`PanelFieldRatio`](@ref) with a `nonneg` guard.

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"book_equity\", vals = [2.0 3.0; 4.0 5.0]),
                          NumericPanelInput(; name = \"market_cap\", vals = [4.0 0.0; 8.0 10.0])];
                         amsk = [true true; false true], emsk = [true true; false true]);

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), pnl = pnl);

julia> descriptor(BookToPrice(), rd)
2×2 Matrix{Float64}:
   0.5  NaN
 NaN      0.5

julia> descriptor(LogMarketCap(), rd)
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
  - [`panel_field_values`](@ref)
  - [`positive_divide`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function descriptor(de::PanelFieldRatio, rd::ReturnsResult)::Matrix{<:Real}
    assert_nonneg_panel_fields(rd, de.nonneg)
    D = positive_divide.(panel_field_values(rd, de.num), panel_field_values(rd, de.den))
    positive_panel_fields_fill!(D, rd, de.pos)
    descriptor_active_fill!(D, rd.pnl)
    return D
end
function descriptor(de::PanelFieldLog, rd::ReturnsResult)::Matrix{<:Real}
    D = panel_field_values(rd, de.field)
    Tf = eltype(D)
    for k in eachindex(D)
        D[k] = D[k] > zero(Tf) ? log(D[k]) : Tf(NaN)
    end
    descriptor_active_fill!(D, rd.pnl)
    return D
end
function descriptor(de::Passthrough, rd::ReturnsResult)::Matrix{<:Real}
    D = panel_field_values(rd, de.field)
    descriptor_active_fill!(D, rd.pnl)
    return D
end
"""
    BookToPrice(; num::AbstractString = "book_equity",
                den::AbstractString = "market_cap") -> PanelFieldRatio

Book equity over market capitalisation, the value Descriptor.

The ratio is `book_equity / market_cap` at each observation, `NaN` where the market capitalisation is not strictly positive. A negative book equity is kept, because it carries information about the balance sheet. The aggregate form is used rather than the per-share form, because it cannot suffer a split-adjustment mismatch between its two sides.

# Arguments

  - `num`: Name of the book equity Panel Field.
  - `den`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed.

# Examples

```jldoctest
julia> BookToPrice()
PanelFieldRatio
     num ┼ String: \"book_equity\"
     den ┼ String: \"market_cap\"
  nonneg ┼ nothing
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`SalesToPrice`](@ref)
  - [`CashFlowToPrice`](@ref)
"""
function BookToPrice(; num::AbstractString = "book_equity",
                     den::AbstractString = "market_cap")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den)
end
"""
    CashFlowToPrice(; num::AbstractString = "operating_cash_flow_ttm",
                    den::AbstractString = "market_cap") -> PanelFieldRatio

Trailing operating cash flow over market capitalisation, a value Descriptor.

The ratio is `operating_cash_flow_ttm / market_cap`, `NaN` where the market capitalisation is not strictly positive. An operating cash flow can be negative, so the Descriptor can too. It is less exposed to accrual accounting choices than an earnings ratio.

# Arguments

  - `num`: Name of the operating cash flow Panel Field.
  - `den`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed.

# Examples

```jldoctest
julia> CashFlowToPrice()
PanelFieldRatio
     num ┼ String: \"operating_cash_flow_ttm\"
     den ┼ String: \"market_cap\"
  nonneg ┼ nothing
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`BookToPrice`](@ref)
  - [`CashFlowToAssets`](@ref)
"""
function CashFlowToPrice(; num::AbstractString = "operating_cash_flow_ttm",
                         den::AbstractString = "market_cap")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den)
end
"""
    SalesToPrice(; num::AbstractString = "sales_ttm",
                 den::AbstractString = "market_cap") -> PanelFieldRatio

Trailing sales over market capitalisation, a value Descriptor.

The ratio is `sales_ttm / market_cap`, `NaN` where the market capitalisation is not strictly positive. Sales are the least exposed of the fundamentals to accounting choices, and the ratio stays defined for a firm whose earnings or book equity are negative.

# Arguments

  - `num`: Name of the sales Panel Field.
  - `den`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed.

# Examples

```jldoctest
julia> SalesToPrice()
PanelFieldRatio
     num ┼ String: \"sales_ttm\"
     den ┼ String: \"market_cap\"
  nonneg ┼ nothing
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`BookToPrice`](@ref)
  - [`SalesToEnterpriseValue`](@ref)
"""
function SalesToPrice(; num::AbstractString = "sales_ttm",
                      den::AbstractString = "market_cap")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den)
end
"""
    EarningsToPrice(; num::AbstractString = "net_income_ttm",
                    den::AbstractString = "market_cap") -> PanelFieldRatio

Trailing net income over market capitalisation, the earnings yield Descriptor.

The ratio is `net_income_ttm / market_cap`, `NaN` where the market capitalisation is not strictly positive. A loss makes it negative, which the price-to-earnings inverse would not survive.

# Arguments

  - `num`: Name of the net income Panel Field.
  - `den`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed.

# Examples

```jldoctest
julia> EarningsToPrice()
PanelFieldRatio
     num ┼ String: \"net_income_ttm\"
     den ┼ String: \"market_cap\"
  nonneg ┼ nothing
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`ForwardEarningsToPrice`](@ref)
  - [`EbitdaToEnterpriseValue`](@ref)
"""
function EarningsToPrice(; num::AbstractString = "net_income_ttm",
                         den::AbstractString = "market_cap")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den)
end
"""
    ForwardEarningsToPrice(; num::AbstractString = "eps_ntm",
                           den::AbstractString = "adj_close") -> PanelFieldRatio

Forward earnings per share over the adjusted close, the forward earnings yield Descriptor.

The ratio is `eps_ntm / adj_close`, `NaN` where the price is not strictly positive. Both sides are per share, so both must be on one split-adjustment basis.

# Arguments

  - `num`: Name of the forward earnings per share Panel Field.
  - `den`: Name of the adjusted close Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed.

# Examples

```jldoctest
julia> ForwardEarningsToPrice()
PanelFieldRatio
     num ┼ String: \"eps_ntm\"
     den ┼ String: \"adj_close\"
  nonneg ┼ nothing
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`EarningsToPrice`](@ref)
  - [`AnalystDispersionToPrice`](@ref)
"""
function ForwardEarningsToPrice(; num::AbstractString = "eps_ntm",
                                den::AbstractString = "adj_close")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den)
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
     pos ┴ nothing
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

The ratio is `dividends_ttm / market_cap`, `NaN` where the market capitalisation is not strictly positive. The dividends must be non-negative wherever they are observed: a negative dividend is a data error, and the estimator raises on one.

# Arguments

  - `num`: Name of the dividends Panel Field.
  - `den`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `nonneg = [num]`.

# Examples

```jldoctest
julia> DividendToPrice()
PanelFieldRatio
     num ┼ String: \"dividends_ttm\"
     den ┼ String: \"market_cap\"
  nonneg ┼ Vector{String}: [\"dividends_ttm\"]
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`ForwardDividendToPrice`](@ref)
  - [`ShareholderYield`](@ref)
"""
function DividendToPrice(; num::AbstractString = "dividends_ttm",
                         den::AbstractString = "market_cap")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, nonneg = [String(num)])
end
"""
    ForwardDividendToPrice(; num::AbstractString = "dps_ntm",
                           den::AbstractString = "adj_close") -> PanelFieldRatio

Forward dividends per share over the adjusted close, the forward dividend yield Descriptor.

The ratio is `dps_ntm / adj_close`, `NaN` where the price is not strictly positive. The dividends must be non-negative wherever they are observed, and the estimator raises on a negative one.

# Arguments

  - `num`: Name of the forward dividends per share Panel Field.
  - `den`: Name of the adjusted close Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `nonneg = [num]`.

# Examples

```jldoctest
julia> ForwardDividendToPrice()
PanelFieldRatio
     num ┼ String: \"dps_ntm\"
     den ┼ String: \"adj_close\"
  nonneg ┼ Vector{String}: [\"dps_ntm\"]
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`DividendToPrice`](@ref)
"""
function ForwardDividendToPrice(; num::AbstractString = "dps_ntm",
                                den::AbstractString = "adj_close")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, nonneg = [String(num)])
end
"""
    ShareholderYield(; dividends::AbstractString = "dividends_ttm",
                     buybacks::AbstractString = "net_buybacks_ttm",
                     den::AbstractString = "market_cap") -> PanelFieldRatio

Trailing dividends plus net buybacks over market capitalisation, the total payout Descriptor.

The ratio is `(dividends_ttm + net_buybacks_ttm) / market_cap`, `NaN` where the market capitalisation is not strictly positive. The dividends must be non-negative wherever they are observed, and the estimator raises on a negative one. Net buybacks can be negative, because a net issuance is one.

# Arguments

  - `dividends`: Name of the dividends Panel Field.
  - `buybacks`: Name of the net buybacks Panel Field.
  - `den`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with `num = [dividends => 1, buybacks => 1]` and `nonneg = [dividends]`.

# Examples

```jldoctest
julia> ShareholderYield()
PanelFieldRatio
     num ┼ Vector{Pair{String, Int64}}: [\"dividends_ttm\" => 1, \"net_buybacks_ttm\" => 1]
     den ┼ String: \"market_cap\"
  nonneg ┼ Vector{String}: [\"dividends_ttm\"]
     pos ┴ nothing
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
                           den = den, nonneg = [String(dividends)])
end
"""
    BookLeverage(; debt::AbstractString = "total_debt",
                 equity::AbstractString = "book_equity") -> PanelFieldRatio

Total debt over total book capital, the book leverage Descriptor.

The ratio is `total_debt / (total_debt + book_equity)`, `NaN` where the total capital is not strictly positive. It is bounded in `[0, 1]` for a firm whose book equity is positive, which is why it is preferred to the debt-to-equity ratio it is a monotone function of. A negative book equity that leaves the total capital positive gives a ratio above one, which is a valid signal of distress.

# Arguments

  - `debt`: Name of the total debt Panel Field.
  - `equity`: Name of the book equity Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with `num = debt` and `den = [debt => 1, equity => 1]`.

# Examples

```jldoctest
julia> BookLeverage()
PanelFieldRatio
     num ┼ String: \"total_debt\"
     den ┼ Vector{Pair{String, Int64}}: [\"total_debt\" => 1, \"book_equity\" => 1]
  nonneg ┼ nothing
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`MarketLeverage`](@ref)
  - [`DebtToAssets`](@ref)
"""
function BookLeverage(; debt::AbstractString = "total_debt",
                      equity::AbstractString = "book_equity")::PanelFieldRatio
    return PanelFieldRatio(; num = debt, den = [String(debt) => 1, String(equity) => 1])
end
"""
    MarketLeverage(; debt::AbstractString = "total_debt",
                   mcap::AbstractString = "market_cap") -> PanelFieldRatio

Total debt over total market capital, the market leverage Descriptor.

The ratio is `total_debt / (total_debt + market_cap)`, `NaN` where the market capitalisation or the total capital is not strictly positive. It reprices the equity leg of the capital structure every observation, where [`BookLeverage`](@ref) reads it from the balance sheet.

# Arguments

  - `debt`: Name of the total debt Panel Field.
  - `mcap`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with `num = debt`, `den = [debt => 1, mcap => 1]` and `pos = [mcap]`.

# Examples

```jldoctest
julia> MarketLeverage()
PanelFieldRatio
     num ┼ String: \"total_debt\"
     den ┼ Vector{Pair{String, Int64}}: [\"total_debt\" => 1, \"market_cap\" => 1]
  nonneg ┼ nothing
     pos ┴ Vector{String}: [\"market_cap\"]
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
                           pos = [String(mcap)])
end
"""
    DebtToAssets(; num::AbstractString = "total_debt",
                 den::AbstractString = "total_assets") -> PanelFieldRatio

Total debt over total assets, a leverage Descriptor.

The ratio is `total_debt / total_assets`, `NaN` where the total assets are not strictly positive.

# Arguments

  - `num`: Name of the total debt Panel Field.
  - `den`: Name of the total assets Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed.

# Examples

```jldoctest
julia> DebtToAssets()
PanelFieldRatio
     num ┼ String: \"total_debt\"
     den ┼ String: \"total_assets\"
  nonneg ┼ nothing
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`BookLeverage`](@ref)
  - [`MarketLeverage`](@ref)
"""
function DebtToAssets(; num::AbstractString = "total_debt",
                      den::AbstractString = "total_assets")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den)
end
"""
    GrossProfitability(; sales::AbstractString = "sales_ttm",
                       cogs::AbstractString = "cost_of_revenue_ttm",
                       den::AbstractString = "total_assets") -> PanelFieldRatio

Gross profit over total assets, the gross profitability Descriptor.

The ratio is `(sales_ttm - cost_of_revenue_ttm) / total_assets`, `NaN` where the total assets are not strictly positive. Gross profit sits above the accounting choices that shape net income, which is what makes it the cleaner profitability signal.

# Arguments

  - `sales`: Name of the sales Panel Field.
  - `cogs`: Name of the cost of revenue Panel Field.
  - `den`: Name of the total assets Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with `num = [sales => 1, cogs => -1]`.

# Examples

```jldoctest
julia> GrossProfitability()
PanelFieldRatio
     num ┼ Vector{Pair{String, Int64}}: [\"sales_ttm\" => 1, \"cost_of_revenue_ttm\" => -1]
     den ┼ String: \"total_assets\"
  nonneg ┼ nothing
     pos ┴ nothing
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
    return PanelFieldRatio(; num = [String(sales) => 1, String(cogs) => -1], den = den)
end
"""
    GrossMargin(; sales::AbstractString = "sales_ttm",
                cogs::AbstractString = "cost_of_revenue_ttm") -> PanelFieldRatio

Gross profit over sales, the gross margin Descriptor.

The ratio is `(sales_ttm - cost_of_revenue_ttm) / sales_ttm`, `NaN` where the sales are not strictly positive.

# Arguments

  - `sales`: Name of the sales Panel Field.
  - `cogs`: Name of the cost of revenue Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with `num = [sales => 1, cogs => -1]` and `den = sales`.

# Examples

```jldoctest
julia> GrossMargin()
PanelFieldRatio
     num ┼ Vector{Pair{String, Int64}}: [\"sales_ttm\" => 1, \"cost_of_revenue_ttm\" => -1]
     den ┼ String: \"sales_ttm\"
  nonneg ┼ nothing
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`GrossProfitability`](@ref)
"""
function GrossMargin(; sales::AbstractString = "sales_ttm",
                     cogs::AbstractString = "cost_of_revenue_ttm")::PanelFieldRatio
    return PanelFieldRatio(; num = [String(sales) => 1, String(cogs) => -1], den = sales)
end
"""
    ReturnOnAssets(; num::AbstractString = "net_income_ttm",
                   den::AbstractString = "total_assets") -> PanelFieldRatio

Trailing net income over total assets, the return on assets Descriptor.

The ratio is `net_income_ttm / total_assets`, `NaN` where the total assets are not strictly positive.

# Arguments

  - `num`: Name of the net income Panel Field.
  - `den`: Name of the total assets Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed.

# Examples

```jldoctest
julia> ReturnOnAssets()
PanelFieldRatio
     num ┼ String: \"net_income_ttm\"
     den ┼ String: \"total_assets\"
  nonneg ┼ nothing
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`ReturnOnEquity`](@ref)
  - [`GrossProfitability`](@ref)
"""
function ReturnOnAssets(; num::AbstractString = "net_income_ttm",
                        den::AbstractString = "total_assets")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den)
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
     pos ┴ nothing
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

The ratio is `sales_ttm / total_assets`, `NaN` where the total assets are not strictly positive.

# Arguments

  - `num`: Name of the sales Panel Field.
  - `den`: Name of the total assets Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed.

# Examples

```jldoctest
julia> AssetTurnover()
PanelFieldRatio
     num ┼ String: \"sales_ttm\"
     den ┼ String: \"total_assets\"
  nonneg ┼ nothing
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`GrossProfitability`](@ref)
"""
function AssetTurnover(; num::AbstractString = "sales_ttm",
                       den::AbstractString = "total_assets")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den)
end
"""
    CashFlowToAssets(; num::AbstractString = "operating_cash_flow_ttm",
                     den::AbstractString = "total_assets") -> PanelFieldRatio

Trailing operating cash flow over total assets, a profitability Descriptor.

The ratio is `operating_cash_flow_ttm / total_assets`, `NaN` where the total assets are not strictly positive.

# Arguments

  - `num`: Name of the operating cash flow Panel Field.
  - `den`: Name of the total assets Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed.

# Examples

```jldoctest
julia> CashFlowToAssets()
PanelFieldRatio
     num ┼ String: \"operating_cash_flow_ttm\"
     den ┼ String: \"total_assets\"
  nonneg ┼ nothing
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`CashFlowToPrice`](@ref)
  - [`ReturnOnAssets`](@ref)
"""
function CashFlowToAssets(; num::AbstractString = "operating_cash_flow_ttm",
                          den::AbstractString = "total_assets")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den)
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
     pos ┴ nothing
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

The ratio is `(net_income_ttm - operating_cash_flow_ttm) / total_assets`, `NaN` where the total assets are not strictly positive. A large positive value says that the reported income ran ahead of the cash the business collected.

# Arguments

  - `income`: Name of the net income Panel Field.
  - `cash_flow`: Name of the operating cash flow Panel Field.
  - `den`: Name of the total assets Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with `num = [income => 1, cash_flow => -1]`.

# Examples

```jldoctest
julia> AccrualsCashFlow()
PanelFieldRatio
     num ┼ Vector{Pair{String, Int64}}: [\"net_income_ttm\" => 1, \"operating_cash_flow_ttm\" => -1]
     den ┼ String: \"total_assets\"
  nonneg ┼ nothing
     pos ┴ nothing
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
                           den = den)
end
"""
    AnalystDispersionToPrice(; num::AbstractString = "eps_ntm_std",
                             den::AbstractString = "adj_close") -> PanelFieldRatio

Dispersion of the forward earnings estimates over the adjusted close, an earnings quality Descriptor.

The ratio is `eps_ntm_std / adj_close`, `NaN` where the price is not strictly positive. A standard deviation is non-negative, so the estimator raises on a negative dispersion.

# Arguments

  - `num`: Name of the forward earnings dispersion Panel Field.
  - `den`: Name of the adjusted close Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `nonneg = [num]`.

# Examples

```jldoctest
julia> AnalystDispersionToPrice()
PanelFieldRatio
     num ┼ String: \"eps_ntm_std\"
     den ┼ String: \"adj_close\"
  nonneg ┼ Vector{String}: [\"eps_ntm_std\"]
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`ForwardEarningsToPrice`](@ref)
  - [`AccrualsCashFlow`](@ref)
"""
function AnalystDispersionToPrice(; num::AbstractString = "eps_ntm_std",
                                  den::AbstractString = "adj_close")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, nonneg = [String(num)])
end
"""
    LogMarketCap(; field::AbstractString = "market_cap") -> PanelFieldLog

Natural logarithm of the market capitalisation, the size Descriptor.

The value is `log(market_cap)`, `NaN` where the market capitalisation is not strictly positive. The logarithm tames the right skew of the raw capitalisation and gives the cross-section a stable scale.

# Arguments

  - `field`: Name of the market capitalisation Panel Field.

# Returns

  - `de::PanelFieldLog`: The estimator, with the Panel Field fixed.

# Examples

```jldoctest
julia> LogMarketCap()
PanelFieldLog
  field ┴ String: \"market_cap\"
```

# Related

  - [`PanelFieldLog`](@ref)
  - [`descriptor`](@ref)
"""
function LogMarketCap(; field::AbstractString = "market_cap")::PanelFieldLog
    return PanelFieldLog(; field = field)
end
"""
    ShortInterest(; num::AbstractString = "short_interest",
                  den::AbstractString = "adj_shares_outstanding") -> PanelFieldRatio

Shares sold short over shares outstanding, the short interest Descriptor.

The ratio is `short_interest / adj_shares_outstanding`, `NaN` where the share count is not strictly positive. The short interest must be non-negative wherever it is observed, and the estimator raises on a negative one. Both sides are share counts, so both must be on one split-adjustment basis.

# Arguments

  - `num`: Name of the short interest Panel Field.
  - `den`: Name of the shares outstanding Panel Field.

# Returns

  - `de::PanelFieldRatio`: The estimator, with the two Panel Fields fixed and `nonneg = [num]`.

# Examples

```jldoctest
julia> ShortInterest()
PanelFieldRatio
     num ┼ String: \"short_interest\"
     den ┼ String: \"adj_shares_outstanding\"
  nonneg ┼ Vector{String}: [\"short_interest\"]
     pos ┴ nothing
```

# Related

  - [`PanelFieldRatio`](@ref)
  - [`descriptor`](@ref)
  - [`DividendToPrice`](@ref)
"""
function ShortInterest(; num::AbstractString = "short_interest",
                       den::AbstractString = "adj_shares_outstanding")::PanelFieldRatio
    return PanelFieldRatio(; num = num, den = den, nonneg = [String(num)])
end

export PanelFieldRatio, PanelFieldLog, Passthrough, BookToPrice, CashFlowToPrice,
       SalesToPrice, EarningsToPrice, ForwardEarningsToPrice, EbitdaToEnterpriseValue,
       DividendToPrice, ForwardDividendToPrice, ShareholderYield, BookLeverage,
       MarketLeverage, DebtToAssets, GrossProfitability, GrossMargin, ReturnOnAssets,
       ReturnOnEquity, AssetTurnover, CashFlowToAssets, SalesToEnterpriseValue,
       AccrualsCashFlow, AnalystDispersionToPrice, LogMarketCap, ShortInterest
