"""
```julia
struct PartialLinearConstraint{T1, T2} <: AbstractConstraintResult
    A::T1
    B::T2
end
```

Container for a set of linear constraints (either equality or inequality) in the form `A * x = B` or `A * x ≤ B`.

`PartialLinearConstraint` stores the coefficient matrix `A` and right-hand side vector `B` for a group of linear constraints. This type is used internally by [`LinearConstraint`](@ref) to represent either the equality or inequality constraints in a portfolio optimisation problem.

# Fields

  - `A`: Coefficient matrix of the linear constraints (typically `AbstractMatrix`).
  - `B`: Right-hand side vector of the linear constraints (typically `AbstractVector`).

# Constructor

```julia
PartialLinearConstraint(; A::AbstractMatrix, B::AbstractVector)
```

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(A)`.
  - `!isempty(B)`.

# Examples

```jldoctest
julia> PartialLinearConstraint(; A = [1.0 2.0; 3.0 4.0], B = [5.0, 6.0])
PartialLinearConstraint
  A | 2×2 Matrix{Float64}
  B | Vector{Float64}: [5.0, 6.0]
```

# Related

  - [`LinearConstraint`](@ref)
  - [`LinearConstraintEstimator`](@ref)
"""
struct PartialLinearConstraint{T1, T2} <: AbstractConstraintResult
    A::T1
    B::T2
end
function PartialLinearConstraint(; A::AbstractMatrix, B::AbstractVector)
    @argcheck(!isempty(A) && !isempty(B),
              DimensionMismatch("`A` and `B` must be non-empty:\nisempty(A) => $(isempty(A))\nisempty(B) => $(isempty(B))"))
    return PartialLinearConstraint(A, B)
end

"""
```julia
struct LinearConstraint{T1, T2} <: AbstractConstraintResult
    ineq::T1
    eq::T2
end
```

Container for a set of linear constraints, separating inequality and equality constraints.

`LinearConstraint` holds both the inequality and equality constraints for a portfolio optimisation problem, each represented as a [`PartialLinearConstraint`](@ref). This type is used to encapsulate all linear constraints in a unified structure, enabling composable and modular constraint handling.

# Fields

  - `ineq`: Inequality constraints, as a [`PartialLinearConstraint`](@ref) or `nothing`.
  - `eq`: Equality constraints, as a [`PartialLinearConstraint`](@ref) or `nothing`.

# Constructor

```julia
LinearConstraint(; ineq::Union{Nothing, <:PartialLinearConstraint} = nothing,
                 eq::Union{Nothing, <:PartialLinearConstraint} = nothing)
```

Keyword arguments correspond to the fields above.

## Validation

  - `isnothing(ineq) ⊼ isnothing(eq)`, i.e. they cannot both be `nothing` at the same time.

# Examples

```jldoctest
julia> ineq = PartialLinearConstraint(; A = [1.0 2.0; 3.0 4.0], B = [5.0, 6.0]);

julia> eq = PartialLinearConstraint(; A = [7.0 8.0; 9.0 10.0], B = [11.0, 12.0]);

julia> LinearConstraint(; ineq = ineq, eq = eq)
LinearConstraint
  ineq | PartialLinearConstraint
       |   A | 2×2 Matrix{Float64}
       |   B | Vector{Float64}: [5.0, 6.0]
    eq | PartialLinearConstraint
       |   A | 2×2 Matrix{Float64}
       |   B | Vector{Float64}: [11.0, 12.0]
```

# Related

  - [`PartialLinearConstraint`](@ref)
  - [`LinearConstraintEstimator`](@ref)
"""
struct LinearConstraint{T1, T2} <: AbstractConstraintResult
    ineq::T1
    eq::T2
end
function LinearConstraint(; ineq::Union{Nothing, <:PartialLinearConstraint} = nothing,
                          eq::Union{Nothing, <:PartialLinearConstraint} = nothing)
    @argcheck(isnothing(ineq) ⊼ isnothing(eq),
              AssertionError("`ineq` and `eq` cannot both be `nothing`:\nisnothing(ineq) => $(isnothing(ineq))\nisnothing(eq) => $(isnothing(eq))"))
    return LinearConstraint(ineq, eq)
end
function Base.getproperty(obj::LinearConstraint, sym::Symbol)
    return if sym == :A_ineq
        isnothing(obj.ineq) ? nothing : obj.ineq.A
    elseif sym == :B_ineq
        isnothing(obj.ineq) ? nothing : obj.ineq.B
    elseif sym == :A_eq
        isnothing(obj.eq) ? nothing : obj.eq.A
    elseif sym == :B_eq
        isnothing(obj.eq) ? nothing : obj.eq.B
    else
        getfield(obj, sym)
    end
end

"""
```julia
abstract type AbstractParsingResult <: AbstractConstraintResult end
```

Abstract supertype for all equation parsing result types in PortfolioOptimisers.jl.

All concrete types representing parsing results should subtype `AbstractParsingResult`. This enables a consistent interface for handling different types of parsed constraint equations.

# Related

  - [`ParsingResult`](@ref)
  - [`RhoParsingResult`](@ref)
"""
abstract type AbstractParsingResult <: AbstractConstraintResult end

"""
```julia
struct ParsingResult{T1, T2, T3, T4, T5} <: AbstractParsingResult
    vars::T1
    coef::T2
    op::T3
    rhs::T4
    eqn::T5
end
```

Structured result for standard linear constraint equation parsing.

`ParsingResult` is the canonical output of [`parse_equation`](@ref) for standard linear constraints. It stores all information needed to construct constraint matrices for portfolio optimisation, including variable names, coefficients, the comparison operator, right-hand side value, and a formatted equation string.

# Fields

  - `vars`: Vector of variable names as strings.
  - `coef`: Vector of coefficients.
  - `op`: The comparison operator as a string.
  - `rhs`: The right-hand side value.
  - `eqn`: The formatted equation string.

# Related

  - [`AbstractParsingResult`](@ref)
  - [`parse_equation`](@ref)
  - [`RhoParsingResult`](@ref)
"""
struct ParsingResult{T1, T2, T3, T4, T5} <: AbstractParsingResult
    vars::T1
    coef::T2
    op::T3
    rhs::T4
    eqn::T5
end

"""
```julia
struct RhoParsingResult{T1, T2, T3, T4, T5, T6} <: AbstractParsingResult
    vars::T1
    coef::T2
    op::T3
    rhs::T4
    eqn::T5
    ij::T6
end
```

Structured result for correlation view constraint equation parsing.

`RhoParsingResult` is produced when parsing correlation view constraints, such as those used in entropy pooling prior models. It extends the standard [`ParsingResult`](@ref) by including an `ij` field, which stores the tuple of asset pairs (indices) relevant for the correlation view.

# Fields

  - `vars`: Vector of variable names as strings.
  - `coef`: Vector of coefficients.
  - `op`: The comparison operator as a string.
  - `rhs`: The right-hand side value.
  - `eqn`: The formatted equation string.
  - `ij`: Tuple or vector of asset index pairs for correlation views.

# Details

  - Produced by correlation view parsing routines, typically when the constraint involves asset pairs (e.g., `"(A, B) == 0.5"`).
  - The `ij` field enables downstream routines to map parsed correlation views to the appropriate entries in the correlation matrix.
  - Used internally for entropy pooling, Black-Litterman, and other advanced portfolio models that support correlation views.

# Related

  - [`AbstractParsingResult`](@ref)
  - [`ParsingResult`](@ref)
  - [`replace_prior_views`](@ref)
"""
struct RhoParsingResult{T1, T2, T3, T4, T5, T6} <: AbstractParsingResult
    vars::T1
    coef::T2
    op::T3
    rhs::T4
    eqn::T5
    ij::T6
end
Base.getindex(res::AbstractParsingResult, i) = i == 1 ? res : throw(BoundsError(res, i))

"""
```julia
struct AssetSets{T1, T2} <: AbstractEstimator
    key::T1
    dict::T2
end
```

Container for asset set and group information used in constraint generation.

`AssetSets` provides a unified interface for specifying the asset universe and any groupings or partitions of assets. It is used throughout constraint generation and estimator routines to expand group references, map group names to asset lists, and validate asset membership.

# Fields

  - `key`: The key in `dict` that identifies the primary list of assets (typically a `Symbol` or `AbstractString`).
  - `dict`: A dictionary mapping group names (or asset set names) to vectors of asset identifiers (typically `AbstractDict{<:Union{Symbol, AbstractString}, <:AbstractVector}`).

# Constructor

```julia
AssetSets(; key::Union{Symbol, AbstractString} = "nx",
          dict::AbstractDict{<:Union{Symbol, AbstractString}})
```

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(dict)`.
  - `haskey(dict, key)`.

# Examples

```jldoctest
julia> AssetSets(; key = :nx, dict = Dict(:nx => ["A", "B", "C"], :group1 => ["A", "B"]))
AssetSets
   key | Symbol: :nx
  dict | Dict{Symbol, Vector{String}}: Dict(:group1 => ["A", "B"], :nx => ["A", "B", "C"])
```

# Related

  - [`replace_group_by_assets`](@ref)
  - [`estimator_to_val`](@ref)
  - [`linear_constraints`](@ref)
"""
struct AssetSets{T1, T2} <: AbstractEstimator
    key::T1
    dict::T2
end
function AssetSets(; key::Union{Symbol, <:AbstractString} = "nx",
                   dict::AbstractDict{<:Union{Symbol, <:AbstractString}})
    @argcheck(!isempty(dict) && haskey(dict, key),
              AssertionError("The following conditions must be met:\n`dict` must be non-empty => !isempty(dict) = $(!isempty(dict))\n`dict` must contain `key` = $key, typeof(key) = $(typeof(key)) => haskey(dict, key) = $(haskey(dict, key))"))
    return AssetSets(key, dict)
end
function nothing_asset_sets_view(sets::AssetSets, i::AbstractVector)
    dict = Dict(k => v for (k, v) in sets.dict if k != sets.key)
    dict[sets.key] = view(sets.dict[sets.key], i)
    return AssetSets(; key = sets.key, dict = dict)
end
"""
```julia
nothing_asset_sets_view(::Nothing, ::Any)
```

No-op fallback for indexing `nothing` asset sets.
"""
function nothing_asset_sets_view(::Nothing, ::Any)
    return nothing
end

"""
Maps the assets in group `key` of `sdict` to their corresponding indices in `nx`, and sets the corresponding entries in the values array `arr` to `val`.
"""
function group_to_val!(nx::AbstractVector, sdict::AbstractDict, key::Any, val::Real,
                       dict::Union{<:AbstractDict, <:Pair{<:Any, <:Real},
                                   <:AbstractVector{<:Pair{<:Any, <:Real}}},
                       arr::AbstractArray, strict::Bool)
    assets = get(sdict, key, nothing)
    if isnothing(assets)
        if strict
            throw(ArgumentError("$(key) is not in $(keys(sdict)).\n$(dict)"))
        else
            @warn("$(key) is not in $(keys(sdict)).\n$(dict)")
        end
    else
        unique!(assets)
        arr[[findfirst(x -> x == asset, nx) for asset in assets]] .= val
    end
    return nothing
end

"""
sets the array of values from a dict/vector of pairs or a single pair, based on the asset sets provided.

  - `arr`: The array to be modified in-place.
  - `dict`: A dictionary or vector of pairs mapping asset or group names to values, or a single pair.
  - `sets`: The `AssetSets` containing the asset groups.
  - `strict`: If `true`, throws an error if a key in `dict` is not found in the asset sets; if `false`, issues a warning instead.
"""
function estimator_to_val!(arr::AbstractArray,
                           dict::Union{<:AbstractDict,
                                       <:AbstractVector{<:Pair{<:Any, <:Real}}},
                           sets::AssetSets; strict::Bool = false)
    nx = sets.dict[sets.key]
    for (key, val) in dict
        if key in nx
            arr[nx[key]] .= val
        else
            group_to_val!(nx, sets.dict, key, val, dict, arr, strict)
        end
    end
    return nothing
end
"""
sets the array of values from a single pair, based on the asset sets provided.

  - `arr`: The array to be modified in-place.
  - `dict`: A single pair mapping an asset or group name to a value.
  - `sets`: The `AssetSets` containing the asset groups.
  - `strict`: If `true`, throws an error if the key in `dict` is not found in the asset sets; if `false`, issues a warning instead.
"""
function estimator_to_val!(arr::AbstractArray, dict::Pair{<:Any, <:Real}, sets::AssetSets;
                           strict::Bool = false)
    nx = sets.dict[sets.key]
    key, val = dict
    if key in nx
        arr[nx[key]] .= val
    else
        group_to_val!(nx, sets.dict, key, val, dict, arr, strict)
    end
    return nothing
end
"""
generates the array of values from a dict/vector of pairs, a single pair, or returns the value as-is.

  - `dict`: A dictionary or vector of pairs mapping asset or group names to values, a single pair, or a numeric value.
  - `sets`: The `AssetSets` containing the asset groups.
  - `val`: The default value to fill in for assets/groups not specified in `dict`.
  - `strict`: If `true`, throws an error if a key in `dict` is not found in the asset sets; if `false`, issues a warning instead.
"""
function estimator_to_val(dict::Union{<:AbstractDict,
                                      <:AbstractVector{<:Pair{<:Any, <:Real}}},
                          sets::AssetSets, val::Real = 0.0; strict::Bool = false)
    nx = sets.dict[sets.key]
    arr = fill(val, length(nx))
    for (key, val) in dict
        if key in nx
            arr[findfirst(x -> x == key, nx)] = val
        else
            group_to_val!(nx, sets.dict, key, val, dict, arr, strict)
        end
    end
    return arr
end
"""
generates the array of values from a single pair, or returns the value as-is.

  - `dict`: A single pair mapping an asset or group name to a value, or a numeric value.
  - `sets`: The `AssetSets` containing the asset groups.
  - `val`: The default value to fill in for assets/groups not specified in `dict`.
  - `strict`: If `true`, throws an error if the key in `dict` is not found in the asset sets; if `false`, issues a warning instead.
"""
function estimator_to_val(dict::Pair{<:Any, <:Real}, sets::AssetSets, val::Real = 0.0;
                          strict::Bool = false)
    nx = sets.dict[sets.key]
    arr = fill(val, length(nx))
    key, val = dict
    if key in nx
        arr[findfirst(x -> x == key, nx)] = val
    else
        group_to_val!(nx, sets.dict, key, val, dict, arr, strict)
    end
    return arr
end
"""
fallback no-op
returns the value as-is
"""
function estimator_to_val(val::Union{Nothing, <:Real, <:AbstractVector{<:Real}}, args...;
                          kwargs...)
    return val
end

# Recursively evaluate numeric functions
function _eval_numeric_functions(expr)
    return if expr isa Expr
        if expr.head == :call
            fname = expr.args[1]
            # Only evaluate if all arguments are numeric
            args = [_eval_numeric_functions(arg) for arg in expr.args[2:end]]
            if all(x -> isa(x, Number), args)
                Base.invokelatest(getfield(Base, fname), args...)
            else
                Expr(:call, fname, args...)
            end
        else
            Expr(expr.head, map(_eval_numeric_functions, expr.args)...)
        end
    elseif expr isa Symbol && expr == :Inf
        Inf
    else
        expr
    end
end
# Collect terms: returns vector of (coefficient, variable::Union{String,Nothing})
function _collect_terms(expr)
    terms = []
    _collect_terms!(expr, 1.0, terms)
    return terms
end
function _collect_terms!(expr, coeff, terms)
    if expr isa Number
        push!(terms, (coeff * float(expr), nothing))
    elseif expr isa Symbol
        push!(terms, (coeff, string(expr)))
    elseif expr isa Expr
        if expr.head == :call && expr.args[1] == :*
            # Multiplication: find numeric and variable part
            a, b = expr.args[2], expr.args[3]
            if a isa Number
                _collect_terms!(b, coeff * float(a), terms)
            elseif b isa Number
                _collect_terms!(a, coeff * float(b), terms)
            else
                # e.g. x*y, treat as variable
                push!(terms, (coeff, string(expr)))
            end
        elseif expr.head == :call && expr.args[1] == :/
            a, b = expr.args[2], expr.args[3]
            if b isa Number
                _collect_terms!(a, coeff / float(b), terms)
            else
                # e.g. x/y, treat as variable
                push!(terms, (coeff, string(expr)))
            end
        elseif expr.head == :call && expr.args[1] == :+
            for i in 2:length(expr.args)
                # Collect terms from addition
                _collect_terms!(expr.args[i], coeff, terms)
            end
        elseif expr.head == :call && expr.args[1] == :-
            for i in 2:(length(expr.args) - 1)
                # Collect terms from addition
                _collect_terms!(expr.args[i], coeff, terms)
            end
            _collect_terms!(expr.args[length(expr.args)], -coeff, terms)
        else
            # treat as variable (e.g. sin(x))
            push!(terms, (coeff, string(expr)))
        end
    end
end
function _format_term(coeff, var)
    return if coeff == 1.0
        var
    elseif coeff == -1.0
        "-$var"
    else
        "$(coeff)*$var"
    end
end
function _rethrow_parse_error(::Any, side = :lhs)
    return nothing
end
function _rethrow_parse_error(::Nothing, side = :lhs)
    @warn("$(side) of equation is empy, assuming zero")
    return nothing
end
function _rethrow_parse_error(expr::Expr, side = :lhs)
    if expr.head == :incomplete
        throw(Meta.ParseError("$side is an incomplete expression.\n$expr"))
    end
    return nothing
end
function _parse_equation(lhs, opstr::AbstractString, rhs, datatype::DataType = Float64)
    # 3. Evaluate numeric functions on both sides
    lexpr = _eval_numeric_functions(lhs)
    _rethrow_parse_error(lexpr, :lhs)
    rexpr = _eval_numeric_functions(rhs)
    _rethrow_parse_error(rexpr, :rhs)

    # 4. Move all terms to LHS: lhs - rhs == 0
    diff_expr = :($lexpr - ($rexpr))

    # 5. Expand and collect like terms
    terms = _collect_terms(diff_expr)

    # 6. Separate variables and constant
    varmap = Dict{String, Float64}()
    constant::datatype = 0.0
    for (coeff, var) in terms
        if var === nothing
            constant += coeff
        else
            varmap[var] = get(varmap, var, 0.0) + coeff
        end
    end

    # 7. Move constant to RHS, variables to LHS
    variables = collect(keys(varmap))
    coefficients = [varmap[v] for v in variables]
    rhs_val = -constant

    # 8. Format the simplified expression
    lhs_str = join([_format_term(coeff, var)
                    for (coeff, var) in zip(coefficients, variables)], " + ")
    lhs_str = replace(lhs_str, "+ -" => "-", "  " => " ")
    rhs_str = string(rhs_val)
    formatted = strip("$lhs_str $opstr $rhs_str")

    return ParsingResult(variables, coefficients, opstr, rhs_val, formatted)
end
"""
```julia
parse_equation(eqn::AbstractString; ops1::Tuple = ("==", "<=", ">="),
               datatype::DataType = Float64, kwargs...)
```

Parse a linear constraint equation from a string into a structured [`ParsingResult`](@ref).

# Arguments

  - `eqn`: The equation string to parse. Must contain exactly one comparison operator from `ops1`.
  - `ops1`: Tuple of valid comparison operators as strings.
  - `datatype`: The numeric type to use for coefficients and right-hand side.
  - `kwargs...`: Additional keyword arguments, ignored.

# Validation

  - The equation must contain exactly one valid comparison operator from `ops1`.
  - Both sides of the equation must be valid Julia expressions.

# Details

  - The function first checks for invalid operator patterns (e.g., `"++"`).
  - It searches for the first occurrence of a valid comparison operator from `ops1` in the equation string. Errors if there are more than one or none.
  - The equation is split into left- and right-hand sides using the detected operator.
  - Both sides are parsed into Julia expressions using `Meta.parse`.
  - Numeric functions and constants (e.g., `Inf`) are recursively evaluated.
  - All terms are moved to the left-hand side and collected, separating coefficients and variables.
  - The constant term is moved to the right-hand side, and the equation is formatted for display.
  - The result is returned as a [`ParsingResult`](@ref) containing the collected information.

# Returns

  - `eqn::ParsingResult`: Structured parsing result.

# Examples

```jldoctest
julia> parse_equation("w_A + 2w_B <= 1")
PortfolioOptimisers.ParsingResult
  vars | Vector{String}: ["w_A", "w_B"]
  coef | Vector{Float64}: [1.0, 2.0]
    op | String: "<="
   rhs | Float64: 1.0
   eqn | SubString{String}: "w_A + 2.0*w_B <= 1.0"
```

# Related

  - [`ParsingResult`](@ref)
"""
function parse_equation(eqn::AbstractString; ops1::Tuple = ("==", "<=", ">="),
                        datatype::DataType = Float64, kwargs...)
    if occursin("++", eqn)
        throw(Meta.ParseError("Invalid operator '++' detected in equation."))
    end
    # 1. Identify the comparison operator
    op = findfirst(op -> occursin(op, eqn), ops1)
    if isnothing(op)
        error("Equation must contain a valid comparison operator $(join(ops1,", ")) .\n$(eqn)")
    end
    opstr = ops1[op]
    parts = split(eqn, opstr)
    if length(parts) != 2
        error("Equation must have exactly one comparison operator.\n$(eqn)")
    end
    lhs, rhs = strip.(parts)
    # 2. Parse both sides into Julia expressions
    lexpr = Meta.parse(lhs)
    _rethrow_parse_error(lexpr, :lhs)
    rexpr = Meta.parse(rhs)
    _rethrow_parse_error(rexpr, :rhs)
    return _parse_equation(lexpr, opstr, rexpr, datatype)
end
function parse_equation(expr::Expr; ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                        datatype::DataType = Float64, kwargs...)
    # 1. Identify the comparison operator in the expression
    if expr.head != :call || !(expr.args[1] in ops2[2:end])
        error("Expression must be a valid comparison $(join(ops2,", ")) .\n$expr")
    end
    opstr = string(expr.args[1])
    lhs, rhs = expr.args[2], expr.args[3]
    return _parse_equation(lhs, opstr, rhs, datatype)
end
function parse_equation(eqn::AbstractVector{<:Union{<:AbstractString, Expr}};
                        ops1::Tuple = ("==", "<=", ">="),
                        ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                        datatype::DataType = Float64)
    return parse_equation.(eqn; ops1 = ops1, ops2 = ops2, datatype = datatype)
end

"""
```julia
replace_group_by_assets(res::ParsingResult, sets::AssetSets; bl_flag::Bool = false,
                        prior_flag::Bool = false, rho_flag::Bool = false)
```

Expand group or special variable references in a [`ParsingResult`](@ref) to their corresponding asset names.

This function takes a [`ParsingResult`](@ref) containing variable names (which may include group names, `prior(...)` expressions, or correlation views like `(A, B)`), and replaces these with the actual asset names from the provided [`AssetSets`](@ref). It supports Black-Litterman-style group expansion, entropy pooling prior views, and correlation view parsing for advanced constraint generation.

# Arguments

  - `res`: A [`ParsingResult`](@ref) object containing variables and coefficients to be expanded.
  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.
  - `bl_flag`: If `true`, enables Black-Litterman-style group expansion.
  - `prior_flag`: If `true`, enables expansion of `prior(...)` expressions for entropy pooling.
  - `rho_flag`: If `true`, enables expansion of correlation views `(A, B)` for entropy pooling.

# Validation

    - `bl_flag` can only be `true` if both `prior_flag` and `rho_flag` are `false`.
    - `rho_flag` can only be `true` if `prior_flag` is also `true`.

# Details

  - Group names in `res.vars` are replaced by the corresponding asset names from `sets.dict`.
  - If `bl_flag` is `true`, coefficients for group references are divided equally among the assets in the group.
  - If `prior_flag` is `true`, expands `prior(asset)` or `prior(group)` expressions for entropy pooling.
  - If `rho_flag` is `true`, expands correlation view expressions `(A, B)` or `prior(A, B)` for entropy pooling, mapping them to asset pairs.
  - If a variable or group is not found in `sets.dict`, it is skipped.

# Returns

  - `ParsingResult`: A new [`ParsingResult`](@ref) with all group and special variable references expanded to asset names.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx", dict = Dict("nx" => ["A", "B", "C"], "group1" => ["A", "B"]));

julia> res = parse_equation("group1 + 2C == 1")
ParsingResult
  vars | Vector{String}: ["C", "group1"]
  coef | Vector{Float64}: [2.0, 1.0]
    op | String: "=="
   rhs | Float64: 1.0
   eqn | SubString{String}: "2.0*C + group1 == 1.0"

julia> replace_group_by_assets(res, sets)
ParsingResult
  vars | Vector{String}: ["C", "A", "B"]
  coef | Vector{Float64}: [2.0, 1.0, 1.0]
    op | String: "=="
   rhs | Float64: 1.0
   eqn | String: "2.0*C + 1.0*A + 1.0*B == 1.0"
```

# Related

  - [`AssetSets`](@ref)
  - [`ParsingResult`](@ref)
  - [`parse_equation`](@ref)
"""
function replace_group_by_assets(res::ParsingResult, sets::AssetSets, bl_flag::Bool = false,
                                 prior_flag::Bool = false, rho_flag::Bool = false)
    if bl_flag && (rho_flag || prior_flag)
        throw(ArgumentError("`bl_flag` can only be used if `prior_flag` and `rho_flag` are false."))
    end
    if rho_flag && !prior_flag
        throw(ArgumentError("`rho_flag` can only be used if `prior_flag` is also true."))
    end
    variables, coeffs = res.vars, res.coef
    variables_new = copy(variables)
    coeffs_new = copy(coeffs)
    variables_tmp = Vector{eltype(variables)}(undef, 0)
    coeffs_tmp = Vector{eltype(coeffs)}(undef, 0)
    idx_rm = Vector{Int}(undef, 0)
    prior_pattern = r"prior\(([^()]*)\)"
    corr_pattern = r"\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)"
    for (i, (v, c)) in enumerate(zip(variables, coeffs))
        m = match(prior_pattern, v)
        if isnothing(m)
            n = match(corr_pattern, v)
            if isnothing(n) && !rho_flag
                asset = get(sets.dict, v, nothing)
                if isnothing(asset)
                    continue
                end
                c = !bl_flag ? coeffs[i] : coeffs[i] / length(asset)
                append!(variables_tmp, asset)
                append!(coeffs_tmp, Iterators.repeated(c, length(asset)))
                push!(idx_rm, i)
            else
                if !(prior_flag && rho_flag)
                    throw(ArgumentError("`(a, b)` can only be used for rho_views in entropy pooling."))
                end
                if isnothing(n)
                    throw(ArgumentError("Correlation views can only be of the form `(a, b)`."))
                end
                asset1 = n.captures[1]
                asset2 = n.captures[2]
                asset1 = get(sets.dict, asset1, nothing)
                asset2 = get(sets.dict, asset2, nothing)
                if isnothing(asset1) && isnothing(asset2)
                    continue
                end
                @argcheck(!isnothing(asset1) &&
                          !isnothing(asset2) &&
                          length(asset1) == length(asset2),
                          AssertionError("The following conditions must be met:\n`asset1` must not be `nothing` => !isnothing(asset1) = $(!isnothing(asset1))\n`asset2` must not be `nothing` => !isnothing(asset2) = $(!isnothing(asset2))\nlength(asset1) == length(asset2) => $(length(asset1)) == $(length(asset2))"))
                push!(variables_tmp, "([$(join(asset1, ", "))], [$(join(asset2, ", "))])")
                push!(coeffs_tmp, coeffs[i])
                push!(idx_rm, i)
            end
        else
            if !prior_flag
                throw(ArgumentError("`prior(a)` can only be used in entropy pooling."))
            end
            n = match(corr_pattern, v)
            if isnothing(n) && !rho_flag
                asset = get(sets.dict, v[7:(end - 1)], nothing)
                if isnothing(asset)
                    continue
                end
                c = !bl_flag ? coeffs[i] : coeffs[i] / length(asset)
                append!(variables_tmp, ["prior($a)" for a in asset])
                append!(coeffs_tmp, Iterators.repeated(c, length(asset)))
                push!(idx_rm, i)
            else
                if !rho_flag
                    throw(ArgumentError("`prior(a, b)` can only be used for rho_views in entropy pooling."))
                end
                if isnothing(n)
                    throw(ArgumentError("Correlation views can only be of the form `(a, b)`."))
                end
                asset1 = n.captures[1]
                asset2 = n.captures[2]
                asset1 = get(sets.dict, asset1, nothing)
                asset2 = get(sets.dict, asset2, nothing)
                if isnothing(asset1) && isnothing(asset2)
                    continue
                end
                @argcheck(!isnothing(asset1) &&
                          !isnothing(asset2) &&
                          length(asset1) == length(asset2),
                          AssertionError("The following conditions must be met:\n`asset1` must not be `nothing` => !isnothing(asset1) = $(!isnothing(asset1))\n`asset2` must not be `nothing` => !isnothing(asset2) = $(!isnothing(asset2))\nlength(asset1) == length(asset2) => $(length(asset1)) == $(length(asset2))"))
                push!(variables_tmp,
                      "prior([$(join(asset1, ", "))], [$(join(asset2, ", "))])")
                push!(coeffs_tmp, coeffs[i])
                push!(idx_rm, i)
            end
        end
    end
    if isempty(variables_tmp)
        return res
    end
    deleteat!(variables_new, idx_rm)
    deleteat!(coeffs_new, idx_rm)
    append!(variables_new, variables_tmp)
    append!(coeffs_new, coeffs_tmp)
    eqn = replace(join(string.(coeffs_new) .* "*" .* variables_new, " + "), "+ -" => "-",
                  "  " => " ")
    return ParsingResult(variables_new, coeffs_new, res.op, res.rhs,
                         "$(eqn) $(res.op) $(res.rhs)")
end
function replace_group_by_assets(res::AbstractVector{<:ParsingResult}, sets::AssetSets,
                                 args...)
    return replace_group_by_assets.(res, sets, args...)
end
function get_linear_constraints(lcs::Union{<:ParsingResult,
                                           <:AbstractVector{<:ParsingResult}},
                                sets::AssetSets; datatype::DataType = Float64,
                                strict::Bool = false)
    if isa(lcs, AbstractVector)
        @argcheck(!isempty(lcs), IsEmptyError(non_empty_msg("lcs") * "."))
    end
    A_ineq = Vector{datatype}(undef, 0)
    B_ineq = Vector{datatype}(undef, 0)
    A_eq = Vector{datatype}(undef, 0)
    B_eq = Vector{datatype}(undef, 0)
    nx = sets.dict[sets.key]
    At = Vector{datatype}(undef, length(nx))
    for lc in lcs
        fill!(At, zero(eltype(At)))
        for (v, c) in zip(lc.vars, lc.coef)
            Ai = (nx .== v)
            if !any(isone, Ai)
                msg = "$(v) is not found in $(nx)."
                strict ? throw(ArgumentError(msg)) : @warn(msg)
                continue
            end
            At += Ai * c
        end
        @argcheck(any(x -> !iszero(x), At),
                  DomainError("At least one entry in At must be non-zero:\nany(x -> !iszero(x), At) => $(any(x -> !iszero(x), At))"))
        d = ifelse(lc.op == ">=", -1, 1)
        flag = d == -1 || lc.op == "<="
        A = At .* d
        B = lc.rhs * d
        if flag
            append!(A_ineq, A)
            append!(B_ineq, B)
        else
            append!(A_eq, A)
            append!(B_eq, B)
        end
    end
    ineq_flag = !isempty(A_ineq)
    eq_flag = !isempty(A_eq)
    ineq = nothing
    eq = nothing
    if ineq_flag
        A_ineq = transpose(reshape(A_ineq, length(nx), :))
        ineq = PartialLinearConstraint(; A = A_ineq, B = B_ineq)
    end
    if eq_flag
        A_eq = transpose(reshape(A_eq, length(nx), :))
        eq = PartialLinearConstraint(; A = A_eq, B = B_eq)
    end
    return if !ineq_flag && !eq_flag
        nothing
    else
        LinearConstraint(; ineq = ineq, eq = eq)
    end
end
function linear_constraints(lcs::Union{Nothing, LinearConstraint}, args...; kwargs...)
    return lcs
end
"""
```julia
linear_constraints(eqn::Union{<:AbstractString, Expr,
                              <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                   sets::AssetSets; ops1::Tuple = ("==", "<=", ">="),
                   ops2::Tuple = (:call, :(==), :(<=), :(>=)), datatype::DataType = Float64,
                   strict::Bool = false, bl_flag::Bool = false)
```

Parse and convert one or more linear constraint equations into a [`LinearConstraint`](@ref) object.

This function parses one or more constraint equations (as strings, expressions, or vectors thereof), replaces group or asset references using the provided [`AssetSets`](@ref), and constructs the corresponding constraint matrices. The result is a [`LinearConstraint`](@ref) object containing both equality and inequality constraints, suitable for use in portfolio optimisation routines.

# Arguments

  - `eqn`: A single constraint equation (as `AbstractString` or `Expr`), or a vector of such equations.
  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.
  - `ops1`: Tuple of valid comparison operators as strings (default: `("==", "<=", ">=")`).
  - `ops2`: Tuple of valid comparison operators as expression heads (default: `(:call, :(==), :(<=), :(>=))`).
  - `datatype`: Numeric type for coefficients and right-hand side (default: `Float64`).
  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning (default: `false`).
  - `bl_flag`: If `true`, enables Black-Litterman-style group expansion (default: `false`).

# Details

  - Each equation is parsed using [`parse_equation`](@ref), supporting both string and expression input.
  - Asset and group references in the equations are expanded using [`replace_group_by_assets`](@ref) and the provided `sets`.
  - The function separates equality and inequality constraints, assembling the corresponding matrices and right-hand side vectors.
  - Input validation is performed using `@argcheck` to ensure non-empty and consistent constraints.
  - Returns `nothing` if no valid constraints are found after parsing and expansion.

# Returns

  - `LinearConstraint`: An object containing the assembled equality and inequality constraints, or `nothing` if no constraints are present.

# Examples

```jldoctest
julia> sets = AssetSets(; key = :nx, dict = Dict(:nx => ["w_A", "w_B", "w_C"]));

julia> linear_constraints(["w_A + w_B == 1", "w_A >= 0.1"], sets)
LinearConstraint
  ineq | PartialLinearConstraint
       |   A | 1×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       |   B | Vector{Float64}: [-0.1]
    eq | PartialLinearConstraint
       |   A | 1×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       |   B | Vector{Float64}: [1.0]
```

# Related

  - [`parse_equation`](@ref)
  - [`replace_group_by_assets`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`LinearConstraint`](@ref)
  - [`AssetSets`](@ref)
"""
function linear_constraints(eqn::Union{<:AbstractString, Expr,
                                       <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                            sets::AssetSets; ops1::Tuple = ("==", "<=", ">="),
                            ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false)
    lcs = parse_equation(eqn; ops1 = ops1, ops2 = ops2, datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets, bl_flag)
    return get_linear_constraints(lcs, sets; datatype = datatype, strict = strict)
end
struct RiskBudgetResult{T1} <: AbstractConstraintResult
    val::T1
end
function RiskBudgetResult(; val::AbstractVector{<:Real})
    @argcheck(!isempty(val) && all(x -> x >= zero(x), val),
              AssertionError("`val` must be non-empty and all its entries must be non-negative:\n!isempty(val) => $(!isempty(val))\nall(x -> x >= zero(x), val) => $(all(x -> x >= zero(x), val))"))
    return RiskBudgetResult(val)
end
function risk_budget_view(::Nothing, args...)
    return nothing
end
function risk_budget_view(rb::RiskBudgetResult, i::AbstractVector)
    val = nothing_scalar_array_view(rb.val, i)
    return RiskBudgetResult(; val = val)
end
function risk_budget_constraints(::Nothing, args...; N::Real, datatype::DataType = Float64,
                                 kwargs...)
    iN = datatype(inv(N))
    return RiskBudgetResult(; val = range(; start = iN, stop = iN, length = N))
end
function risk_budget_constraints(rb::RiskBudgetResult, args...; kwargs...)
    return rb
end
function risk_budget_constraints(rb::Union{<:AbstractDict, <:Pair{<:Any, <:Real},
                                           <:AbstractVector{<:Pair{<:Any, <:Real}}},
                                 sets::AssetSets; N::Real = length(sets.dict[sets.key]),
                                 strict::Bool = false, datatype::DataType = Float64)
    val = estimator_to_val(rb, sets, inv(N); strict = strict, datatype = datatype)
    return RiskBudgetResult(; val = val / sum(val))
end
struct RiskBudgetEstimator{T1} <: AbstractConstraintEstimator
    val::T1
end
function RiskBudgetEstimator(;
                             val::Union{<:AbstractDict, <:Pair{<:Any, <:Real},
                                        <:AbstractVector{<:Union{<:Pair{<:Any, <:Real}}}})
    if isa(val, Union{<:AbstractDict, <:AbstractVector})
        @argcheck(!isempty(val), IsEmptyError(non_empty_msg("`val`") * "."))
        if isa(val, AbstractDict)
            @argcheck(all(x -> x >= zero(x), values(val)),
                      DomainError("All entries of `val` must be non-negative"))
        elseif isa(val, AbstractVector{<:Pair})
            @argcheck(all(x -> x >= zero(x), getproperty.(val, :second)),
                      DomainError("The numerical value of all entries of `val` must be non-negative"))
        end
    elseif isa(val, Pair)
        @argcheck(val.second >= zero(val.second),
                  DomainError("The numerical value of `val` must be non-negative:\nval.second => $(val.second)"))
    end
    return RiskBudgetEstimator(val)
end
function risk_budget_view(rb::RiskBudgetEstimator, ::Any)
    return rb
end
function risk_budget_constraints(rb::RiskBudgetEstimator, sets::AssetSets;
                                 strict::Bool = false, datatype::DataType = Float64,
                                 kwargs...)
    return risk_budget_constraints(rb.val, sets; strict = strict, datatype = datatype)
end
function asset_sets_matrix(smtx::Union{Symbol, <:AbstractString}, sets::AssetSets)
    @argcheck(haskey(sets.dict, smtx), KeyError("key $smtx not found in `sets.dict`"))
    all_sets = sets.dict[smtx]
    @argcheck(length(sets.dict[sets.key]) == length(all_sets),
              AssertionError("The following conditions must be met:\n`sets.dict` must contain key $smtx => haskey(sets.dict, smtx) = $(haskey(sets.dict, smtx))\nlengths of sets.dict[sets.key] and `all_sets` must be equal:\nlength(sets.dict[sets.key]) => length(sets.dict[$(sets.key)]) => $(length(sets.dict[sets.key]))\nlength(all_sets) => $(length(all_sets))"))
    unique_sets = unique(all_sets)
    A = BitMatrix(undef, length(all_sets), length(unique_sets))
    for (i, val) in pairs(unique_sets)
        A[:, i] = all_sets .== val
    end
    return transpose(A)
end
function asset_sets_matrix(smtx::Union{Nothing, <:AbstractMatrix}, args...)
    return smtx
end
function asset_sets_matrix_view(smtx::AbstractMatrix, i::AbstractVector; kwargs...)
    return view(smtx, :, i)
end

"""
```julia
struct LinearConstraintEstimator{T1} <: AbstractConstraintEstimator
    val::T1
end
```

Container for one or more linear constraint equations to be parsed and converted into constraint matrices.

# Fields

  - `val`: A single equation as an `AbstractString` or `Expr`, or a vector of such equations.

# Constructor

```julia
LinearConstraintEstimator(;
                          val::Union{<:AbstractString, Expr,
                                     <:AbstractVector{<:Union{<:AbstractString, Expr}}})
```

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(val)`.

# Examples

```jldoctest
julia> lce = LinearConstraintEstimator(; val = ["w_A + w_B == 1", "w_A >= 0.1"]);

julia> sets = AssetSets(; key = :nx, dict = Dict(:nx => ["w_A", "w_B"]));

julia> linear_constraints(lce, sets)
LinearConstraint
  ineq | PartialLinearConstraint
       |   A | 1×2 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       |   B | Vector{Float64}: [-0.1]
    eq | PartialLinearConstraint
       |   A | 1×2 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       |   B | Vector{Float64}: [1.0]
```

# Related

  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`parse_equation`](@ref)
  - [`linear_constraints`](@ref)
"""
struct LinearConstraintEstimator{T1} <: AbstractConstraintEstimator
    val::T1
end
function LinearConstraintEstimator(;
                                   val::Union{<:AbstractString, Expr,
                                              <:AbstractVector{<:Union{<:AbstractString,
                                                                       Expr}}})
    if isa(val, Union{<:AbstractString, <:AbstractVector})
        @argcheck(!isempty(val))
    end
    return LinearConstraintEstimator(val)
end
function linear_constraints(lcs::LinearConstraintEstimator, sets::AssetSets;
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false)
    return linear_constraints(lcs.val, sets; datatype = datatype, strict = strict,
                              bl_flag = bl_flag)
end
function linear_constraints(lcs::AbstractVector{<:LinearConstraintEstimator},
                            sets::AssetSets; datatype::DataType = Float64,
                            strict::Bool = false, bl_flag::Bool = false)
    return linear_constraints.(lcs.val, Ref(sets); datatype = datatype, strict = strict,
                               bl_flag = bl_flag)
end
function risk_budget_constraints(lcs::LinearConstraintEstimator, sets::AssetSets;
                                 datatype::DataType = Float64, strict::Bool = false)
    return risk_budget_constraints(lcs.val, sets; datatype = datatype, strict = strict)
end
struct AssetSetsMatrixEstimator{T1} <: AbstractConstraintEstimator
    val::T1
end
function AssetSetsMatrixEstimator(; val::Union{<:Symbol, <:AbstractString})
    if isa(val, AbstractString)
        @argcheck(!isempty(val))
    end
    return AssetSetsMatrixEstimator(val)
end
function asset_sets_matrix(smtx::AssetSetsMatrixEstimator, sets::AssetSets)
    return asset_sets_matrix(smtx.val, sets)
end
function asset_sets_matrix(smtx::AbstractVector{<:Union{Nothing, <:AbstractMatrix,
                                                        <:AssetSetsMatrixEstimator}},
                           sets::AssetSets)
    return asset_sets_matrix.(smtx, Ref(sets))
end
function asset_sets_matrix_view(smtx::Union{Nothing, AssetSetsMatrixEstimator}, ::Any;
                                kwargs...)
    return smtx
end
function asset_sets_matrix_view(smtx::AbstractVector{<:Union{Nothing, <:AbstractMatrix,
                                                             <:AssetSetsMatrixEstimator}},
                                i::AbstractVector; kwargs...)
    return asset_sets_matrix_view.(smtx, Ref(i); kwargs...)
end

export AssetSets, PartialLinearConstraint, LinearConstraint, LinearConstraintEstimator,
       AssetSetsMatrixEstimator, RiskBudgetResult, RiskBudgetEstimator, ParsingResult,
       RhoParsingResult, parse_equation, replace_group_by_assets, linear_constraints,
       asset_sets_matrix
