"""
    struct PartialLinearConstraint{T1, T2} <: AbstractConstraintResult
        A::T1
        B::T2
    end

Container for a set of linear constraints (either equality or inequality) in the form `A * x = B` or `A * x ≤ B`.

`PartialLinearConstraint` stores the coefficient matrix `A` and right-hand side vector `B` for a group of linear constraints. This type is used internally by [`LinearConstraint`](@ref) to represent either the equality or inequality constraints in a portfolio optimisation problem.

# Fields

  - `A`: Coefficient matrix of the linear constraints (typically `AbstractMatrix`).
  - `B`: Right-hand side vector of the linear constraints (typically `AbstractVector`).

# Constructor

    PartialLinearConstraint(; A::AbstractMatrix, B::AbstractVector)

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
    function PartialLinearConstraint(A::AbstractMatrix, B::AbstractVector)
        @argcheck(!isempty(A) && !isempty(B),
                  DimensionMismatch("`A` and `B` must be non-empty:\nisempty(A) => $(isempty(A))\nisempty(B) => $(isempty(B))"))
        return new{typeof(A), typeof(B)}(A, B)
    end
end
function PartialLinearConstraint(; A::AbstractMatrix, B::AbstractVector)
    return PartialLinearConstraint(A, B)
end
"""
    struct LinearConstraint{T1, T2} <: AbstractConstraintResult
        ineq::T1
        eq::T2
    end

Container for a set of linear constraints, separating inequality and equality constraints.

`LinearConstraint` holds both the inequality and equality constraints for a portfolio optimisation problem, each represented as a [`PartialLinearConstraint`](@ref). This type is used to encapsulate all linear constraints in a unified structure, enabling composable and modular constraint handling.

# Fields

  - `ineq`: Inequality constraints, as a [`PartialLinearConstraint`](@ref) or `nothing`.
  - `eq`: Equality constraints, as a [`PartialLinearConstraint`](@ref) or `nothing`.

# Constructor

    LinearConstraint(; ineq::Union{Nothing, <:PartialLinearConstraint} = nothing,
                     eq::Union{Nothing, <:PartialLinearConstraint} = nothing)

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
    function LinearConstraint(ineq::Union{Nothing, <:PartialLinearConstraint},
                              eq::Union{Nothing, <:PartialLinearConstraint})
        @argcheck(isnothing(ineq) ⊼ isnothing(eq),
                  AssertionError("`ineq` and `eq` cannot both be `nothing`:\nisnothing(ineq) => $(isnothing(ineq))\nisnothing(eq) => $(isnothing(eq))"))
        return new{typeof(ineq), typeof(eq)}(ineq, eq)
    end
end
function LinearConstraint(; ineq::Union{Nothing, <:PartialLinearConstraint} = nothing,
                          eq::Union{Nothing, <:PartialLinearConstraint} = nothing)
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
    abstract type AbstractParsingResult <: AbstractConstraintResult end

Abstract supertype for all equation parsing result types in PortfolioOptimisers.jl.

All concrete types representing parsing results should subtype `AbstractParsingResult`. This enables a consistent interface for handling different types of parsed constraint equations.

# Related

  - [`ParsingResult`](@ref)
  - [`RhoParsingResult`](@ref)
"""
abstract type AbstractParsingResult <: AbstractConstraintResult end
"""
    struct ParsingResult{T1, T2, T3, T4, T5} <: AbstractParsingResult
        vars::T1
        coef::T2
        op::T3
        rhs::T4
        eqn::T5
    end

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
    function ParsingResult(vars::AbstractVector{<:AbstractString},
                           coef::AbstractVector{<:Real}, op::AbstractString, rhs::Real,
                           eqn::AbstractString)
        @argcheck(length(vars) == length(coef),
                  DimensionMismatch("`vars` and `coef` must have the same length:\nlength(vars) => $(length(vars))\nlength(coef) => $(length(coef))"))
        return new{typeof(vars), typeof(coef), typeof(op), typeof(rhs), typeof(eqn)}(vars,
                                                                                     coef,
                                                                                     op,
                                                                                     rhs,
                                                                                     eqn)
    end
end
"""
    struct RhoParsingResult{T1, T2, T3, T4, T5, T6} <: AbstractParsingResult
        vars::T1
        coef::T2
        op::T3
        rhs::T4
        eqn::T5
        ij::T6
    end

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
    function RhoParsingResult(vars::AbstractVector{<:AbstractString},
                              coef::AbstractVector{<:Real}, op::AbstractString, rhs::Real,
                              eqn::AbstractString,
                              ij::AbstractVector{<:Union{<:Tuple{<:Integer, <:Integer},
                                                         <:Tuple{<:AbstractVector{<:Integer},
                                                                 <:AbstractVector{<:Integer}}}})
        @argcheck(length(vars) == length(coef),
                  DimensionMismatch("`vars` and `coef` must have the same length:\nlength(vars) => $(length(vars))\nlength(coef) => $(length(coef))"))
        return new{typeof(vars), typeof(coef), typeof(op), typeof(rhs), typeof(eqn),
                   typeof(ij)}(vars, coef, op, rhs, eqn, ij)
    end
end
Base.getindex(res::AbstractParsingResult, i) = i == 1 ? res : throw(BoundsError(res, i))

"""
    struct AssetSets{T1, T2} <: AbstractEstimator
        key::T1
        dict::T2
    end

Container for asset set and group information used in constraint generation.

`AssetSets` provides a unified interface for specifying the asset universe and any groupings or partitions of assets. It is used throughout constraint generation and estimator routines to expand group references, map group names to asset lists, and validate asset membership.

If a key in `dict` starts with the same value as `key`, it means that the corresponding group must have the same length as the asset universe, `dict[key]`. This is useful for defining partitions of the asset universe, for example when using [`asset_sets_matrix`](@ref) with [`NestedClustered`](@ref).

# Fields

  - `key`: The key in `dict` that identifies the primary list of assets.
  - `dict`: A dictionary mapping group names (or asset set names) to vectors of asset identifiers.

# Constructor

    AssetSets(; key::AbstractString = "nx", dict::AbstractDict{<:AbstractString, <:Any})

Keyword arguments correspond to the fields above.

## Validation

  - If a key in `dict` starts with the same value as `key`, `length(dict[nx]) == length(dict[key])`.
  - `!isempty(dict)`.
  - `haskey(dict, key)`.

# Examples

```jldoctest
julia> AssetSets(; key = "nx", dict = Dict("nx" => ["A", "B", "C"], "group1" => ["A", "B"]))
AssetSets
   key | String: "nx"
  dict | Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"], "group1" => ["A", "B"])
```

# Related

  - [`replace_group_by_assets`](@ref)
  - [`estimator_to_val`](@ref)
  - [`linear_constraints`](@ref)
"""
struct AssetSets{T1, T2} <: AbstractEstimator
    key::T1
    dict::T2
    function AssetSets(key::AbstractString, dict::AbstractDict{<:AbstractString, <:Any})
        @argcheck(!isempty(dict) && haskey(dict, key),
                  AssertionError("The following conditions must be met:\n`dict` must be non-empty => !isempty(dict) = $(!isempty(dict))\n`dict` must contain `key = $key``, haskey(dict, key) = $(haskey(dict, key))"))
        for k in keys(dict)
            if k == key
                continue
            elseif startswith(k, key)
                @argcheck(length(dict[k]) == length(dict[key]),
                          DimensionMismatch("$k starts with $key, so length(dict[$k]) => $(length(dict[k])), must be equal to length(dict[$key]) => $(length(dict[key]))"))
            end
        end
        return new{typeof(key), typeof(dict)}(key, dict)
    end
end
function AssetSets(; key::AbstractString = "nx",
                   dict::AbstractDict{<:AbstractString, <:Any})
    return AssetSets(key, dict)
end
function nothing_asset_sets_view(sets::AssetSets, i::AbstractVector)
    key = sets.key
    dict = typeof(sets.dict)()
    dict[key] = view(sets.dict[key], i)
    for (k, v) in sets.dict
        if k == key
            continue
        elseif startswith(k, key)
            v = view(v, i)
        end
        push!(dict, k => v)
    end
    return AssetSets(; key = key, dict = dict)
end
"""
    nothing_asset_sets_view(::Nothing, ::Any)

No-op fallback for indexing `nothing` asset sets.

# Returns

  - `nothing`.
"""
function nothing_asset_sets_view(::Nothing, ::Any)
    return nothing
end
"""
    group_to_val!(nx::AbstractVector, sdict::AbstractDict, key::Any, val::Real,
                  dict::Union{<:AbstractDict, <:Pair{<:AbstractString, <:Real},
                              <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                  arr::AbstractVector, strict::Bool)

Set values in a vector for all assets belonging to a specified group.

`group_to_val!` maps the assets in group `key` to their corresponding indices in the asset universe `nx`, and sets the corresponding entries in the vector `arr` to the value `val`. If the group is not found, the function either throws an error or issues a warning, depending on the `strict` flag.

# Arguments

  - `nx`: Vector of asset names.
  - `sdict`: Dictionary mapping group names to vectors of asset names.
  - `key`: Name of the group of assets to set values for.
  - `val`: The value to assign to the assets in the group.
  - `dict`: The original dictionary, vector of pairs, or pair being processed (used for logging messages).
  - `arr`: The array to be modified in-place.
  - `strict`: If `true`, throws an error if `key` is not found in `sdict`; if `false`, issues a warning.

# Details

  - If `key` is found in `sdict`, all assets in the group are mapped to their indices in `nx`, and the corresponding entries in `arr` are set to `val`.
  - If `key` is not found and `strict` is `true`, an `ArgumentError` is thrown; otherwise, a warning is issued.

# Returns

  - `nothing`: The operation is performed in-place on `arr`.

# Related

  - [`estimator_to_val`](@ref)
  - [`AssetSets`](@ref)
"""
function group_to_val!(nx::AbstractVector, sdict::AbstractDict, key::Any, val::Real,
                       dict::Union{<:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                   <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                       arr::AbstractVector, strict::Bool)
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
    estimator_to_val(dict::Union{<:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                 <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                     sets::AssetSets; val::Real = 0.0, strict::Bool = false)

Return value for assets or groups, based on a mapping and asset sets.

The function creates the vector and sets the values for assets or groups as specified by `dict`, using the asset universe and groupings in `sets`. If a key in `dict` is not found in the asset sets, the function either throws an error or issues a warning, depending on the `strict` flag.

# Arguments

  - `arr`: The array to be modified in-place.
  - `dict`: A dictionary, vector of pairs, or single pair mapping asset or group names to values.
  - `sets`: The [`AssetSets`](@ref) containing the asset universe and group definitions.
  - `val`: The default value to assign to assets not specified in `dict`.
  - `strict`: If `true`, throws an error if a key in `dict` is not found in the asset sets; if `false`, issues a warning.

# Details

  - Iterates over the (key, value) pairs in `dict`.

!!! warning

    If the same asset is found in subsequent iterations, its value will be overwritten in favour of the most recent one. To ensure determinism, use an [`OrderedDict`](https://juliacollections.github.io/OrderedCollections.jl/stable/#OrderedDicts) or a vector of pairs.

  - If a key in `dict` matches an asset in the universe, the corresponding entry in `arr` is set to the specified value.
  - If a key matches a group in `sets`, all assets in the group are set to the specified value using [`group_to_val!`](@ref).
  - If a key is not found and `strict` is `true`, an `ArgumentError` is thrown; otherwise, a warning is issued.
  - The operation is performed in-place on `arr`.

# Returns

  - `arr::Vector{<:Real}`: Value array.

# Related

  - [`group_to_val!`](@ref)
  - [`AssetSets`](@ref)
  - [`estimator_to_val`](@ref)
"""
function estimator_to_val(dict::Union{<:AbstractDict,
                                      <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
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
    estimator_to_val(val::Union{Nothing, <:Real, <:AbstractVector{<:Real}}, args...; kwargs...)

Fallback no-op for value mapping in asset/group estimators.

This method returns the input value `val` as-is, without modification or mapping. It serves as a fallback for cases where the input is already a numeric value, a vector of numeric values, or `nothing`, and no further processing is required.

# Arguments

  - `val`: A value of type `Nothing`, a single numeric value, or a vector of numeric values.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `val::Union{Nothing, <:Real, <:AbstractVector{<:Real}}`: The input `val`, unchanged.

# Related

  - [`estimator_to_val`](@ref)
  - [`group_to_val!`](@ref)
  - [`AssetSets`](@ref)
"""
function estimator_to_val(val::Union{Nothing, <:Real, <:AbstractVector{<:Real}}, args...;
                          kwargs...)
    return val
end
"""
    _eval_numeric_functions(expr)

Recursively evaluate numeric functions and constants in a Julia expression.

`_eval_numeric_functions` traverses a Julia expression tree and evaluates any sub-expressions that are purely numeric, including standard mathematical functions and constants (such as `Inf`). This is used to simplify constraint equations before further parsing and canonicalisation.

# Arguments

  - `expr`: The Julia expression to evaluate. Can be a `Number`, `Symbol`, or `Expr`.

# Details

  - `expr`:

      + `Real`: it is returned as-is.
      + `:Inf`: returns `Inf`.
      + `Expr`: representing a function call, and all arguments are numeric, the function is evaluated and replaced with its result.
      + Otherwise, the function recurses into sub-expressions, returning a new expression with numeric parts evaluated.

# Returns

  - The evaluated expression, with all numeric sub-expressions replaced by their computed values. Non-numeric or symbolic expressions are returned in their original or partially simplified form.

# Related

  - [`_collect_terms`](@ref)
  - [`_parse_equation`](@ref)
"""
function _eval_numeric_functions(expr)
    return if isa(expr, Expr)
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
    elseif isa(expr, Symbol) && expr == :Inf
        Inf
    else
        expr
    end
end
"""
    _collect_terms(expr::Union{Symbol, Expr, <:Number})

Expand and collect all terms from a Julia expression representing a linear constraint equation.

`_collect_terms` takes a Julia expression (such as the left-hand side of a constraint equation), recursively traverses its structure, and returns a vector of `(coefficient, variable)` pairs. It supports numeric constants, variables, and arithmetic operations (`+`, `-`, `*`, `/`), and is used to canonicalise linear constraint equations for further processing.

# Arguments

  - `expr`: The Julia expression to expand.

# Details

    - Calls [`_collect_terms!`](@ref) internally with an initial coefficient of `1.0` and an empty vector.
    - Numeric constants are collected as `(coefficient, nothing)`.
    - Variables are collected as `(coefficient, variable_name)`.
    - Arithmetic expressions are recursively expanded and collected.

# Returns

  - `terms::Vector{Tuple{Float64, Union{String, Nothing}}}`: A vector of `(coefficient, variable)` pairs, where `variable` is a string for variable terms or `nothing` for constant terms.

# Related

  - [`_collect_terms!`](@ref)
  - [`_parse_equation`](@ref)
"""
function _collect_terms(expr)
    terms = []
    _collect_terms!(expr, 1.0, terms)
    return terms
end
"""
    _collect_terms!(expr, coeff, terms)

Recursively collect and expand terms from a Julia expression for linear constraint parsing.

`_collect_terms!` traverses a Julia expression tree representing a linear equation, expanding and collecting all terms into a vector of `(coefficient, variable)` pairs. It handles numeric constants, variables, and arithmetic operations (`+`, `-`, `*`, `/`), supporting canonicalisation of linear constraint equations for further processing.

# Arguments

  - `expr`: The Julia expression to traverse.
  - `coeff`: The current numeric coefficient to apply.
  - `terms`: A vector to which `(coefficient, variable)` pairs are appended in-place. Each pair is of the form `(Float64, Union{String, Nothing})`, where `Nothing` indicates a constant term.

# Details

  - `expr`:

      + `Number`: appends `(coeff * oftype(coeff, expr), nothing)` to `terms`.

      + `Symbol`: appends `(coeff, string(expr))` to `terms`.
      + `Expr`:

          * For multiplication (`*`), distributes the coefficient to the numeric part.
          * For division (`/`), divides the coefficient by the numeric denominator.
          * For addition (`+`), recursively collects terms from all arguments.
          * For subtraction (`-`), recursively collects terms from all arguments except the last, which is negated.
          * For all other expressions, treats as a variable and appends as `(coeff, string(expr))`.

# Returns

  - `nothing`: The function modifies `terms` in-place.

# Related

  - [`_collect_terms`](@ref)
  - [`_parse_equation`](@ref)
"""
function _collect_terms!(expr, coeff, terms)
    if isa(expr, Number)
        push!(terms, (coeff * oftype(coeff, expr), nothing))
    elseif isa(expr, Symbol)
        push!(terms, (coeff, string(expr)))
    elseif isa(expr, Expr)
        if expr.head == :call && expr.args[1] == :*
            # Multiplication: find numeric and variable part
            a, b = expr.args[2], expr.args[3]
            if isa(a, Number)
                _collect_terms!(b, coeff * oftype(coeff, a), terms)
            elseif isa(b, Number)
                _collect_terms!(a, coeff * oftype(coeff, b), terms)
            else
                # e.g. x*y, treat as variable
                push!(terms, (coeff, string(expr)))
            end
        elseif expr.head == :call && expr.args[1] == :/
            a, b = expr.args[2], expr.args[3]
            if isa(b, Number)
                _collect_terms!(a, coeff / oftype(coeff, b), terms)
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
"""
    _format_term(coeff, var)

Format a single term in a linear constraint equation as a string.

`_format_term` takes a coefficient and a variable name and returns a string representation suitable for display in a canonicalised linear constraint equation. Handles special cases for coefficients of `1` and `-1` to avoid redundant notation.

# Arguments

  - `coeff`: Numeric coefficient for the variable.
  - `var`: Variable name as a string.

# Details

    - If `coeff == 1`, returns `"\$var"` (no explicit coefficient).
    - If `coeff == -1`, returns `"-\$(var)"` (no explicit coefficient).
    - Otherwise, returns `"\$(coeff)*\$(var)"`.

# Returns

  - `term_str::String`: The formatted term as a string.

# Related

  - [`_parse_equation`](@ref)
  - [`ParsingResult`](@ref)
"""
function _format_term(coeff, var)
    return if isone(coeff)
        "$var"
    elseif isone(-coeff)
        "-$var"
    else
        "$(coeff)*$var"
    end
end
"""
    _rethrow_parse_error(expr; side = :lhs)

Internal utility for error handling during equation parsing.

`_rethrow_parse_error` is used to detect and handle incomplete or invalid expressions encountered while parsing constraint equations. It is called on both sides of an equation during parsing to ensure that the expressions are valid and complete. If an incomplete expression is detected, a `Meta.ParseError` is thrown; otherwise, the function returns `nothing`.

# Arguments

  - `expr`: The parsed Julia expression to check. Can be an `Expr`, `Nothing`, or any other type.
  - `side`: Symbol indicating which side of the equation is being checked (`:lhs` or `:rhs`). Used for error messages.

# Details

  - If `expr` is `Nothing`, a warning is issued indicating that the side is empty and zero is assumed.
  - If `expr` is an incomplete expression (`expr.head == :incomplete`), a `Meta.ParseError` is thrown with a descriptive message.
  - For all other cases, the function returns `nothing` and does not modify the input.

# Validation

  - Throws a `Meta.ParseError` if the expression is incomplete.

# Returns

  - `nothing`: if the expression is valid or handled.

# Related

  - [`parse_equation`](@ref)
  - [`_parse_equation`](@ref)
"""
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
"""
    _parse_equation(lhs, opstr::AbstractString, rhs; datatype::DataType = Float64)

Parse and canonicalise a linear constraint equation from Julia expressions.

`_parse_equation` takes the left-hand side (`lhs`) and right-hand side (`rhs`) of a constraint equation, both as Julia expressions, and a comparison operator string (`opstr`). It evaluates numeric functions, moves all terms to the left-hand side, collects coefficients and variables, and returns a [`ParsingResult`](@ref) with the canonicalised equation.

# Arguments

  - `lhs`: Left-hand side of the equation as a Julia expression.
  - `opstr`: Comparison operator as a string.
  - `rhs`: Right-hand side of the equation as a Julia expression.
  - `datatype`: Numeric type for coefficients and right-hand side.

# Details

  - Recursively evaluates numeric functions and constants (e.g., `Inf`) on both sides.
  - Moves all terms to the left-hand side (`lhs - rhs == 0`).
  - Collects and sums like terms, separating variables and constants.
  - Moves the constant term to the right-hand side, variables to the left.
  - Formats the simplified equation as a string.
  - Returns a [`ParsingResult`](@ref) containing variable names, coefficients, operator, right-hand side value, and formatted equation.

# Returns

  - `res::ParsingResult`: Structured result with canonicalised variables, coefficients, operator, right-hand side, and formatted equation.

# Related

  - [`ParsingResult`](@ref)
  - [`parse_equation`](@ref)
"""
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
    parse_equation(eqn::Union{<:AbstractString, Expr,
                              <:AbstractVector{<:Union{<:AbstractString, Expr}}};
                   ops1::Tuple = ("==", "<=", ">="), ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                   datatype::DataType = Float64, kwargs...)

Parse a linear constraint equation from a string into a structured [`ParsingResult`](@ref).

# Arguments

  - `eqn`: The equation string to parse.

      + `eqn::AbstractVector`: Each element needs to meet the criteria below.

      + `eqn::AbstractString`: Must contain exactly one comparison operator from `ops1`.

          * `ops1`: Tuple of valid comparison operators as strings.
      + `eqn::Expr`: Must contain exactly one comparison operator from `ops1`.

          * `ops2`: Tuple of valid comparison operator expressions.

  - `datatype`: The numeric type to use for coefficients and right-hand side.
  - `kwargs...`: Additional keyword arguments, ignored.

# Validation

  - The equation must contain exactly one valid comparison operator from `ops1`.
  - Both sides of the equation must be valid Julia expressions.

# Details

  - If `eqn::AbstractVector`, the function is applied element-wise.

  - The function first checks for invalid operator patterns (e.g., `"++"`).
  - It searches for the first occurrence of a valid comparison operator from `ops1` in the equation string. Errors if there are more than one or none.
  - The equation is split into left- and right-hand sides using the detected operator.
  - If `eqn::AbstractString`:

      + Both sides are parsed into Julia expressions using `Meta.parse`.
  - If `eqn::Expr`:

      + Expression is ready as is.
  - Numeric functions and constants (e.g., `Inf`) are recursively evaluated.
  - All terms are moved to the left-hand side and collected, separating coefficients and variables.
  - The constant term is moved to the right-hand side, and the equation is formatted for display.
  - The result is returned as a [`ParsingResult`](@ref) containing the collected information.

# Returns

  - If `eqn::Union{<:AbstractString, Expr}`:

      + `res::ParsingResult`: Structured parsing result.

  - If `eqn::AbstractVector`:

      + `res::Vector{ParsingResult}`: Vector of structured parsing results.

# Examples

```jldoctest
julia> parse_equation("w_A + 2w_B <= 1")
ParsingResult
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
    replace_group_by_assets(res::Union{<:ParsingResult, <:AbstractVector{<:ParsingResult}},
                            sets::AssetSets; bl_flag::Bool = false, prior_flag::Bool = false,
                            rho_flag::Bool = false)

If `res` is a vector of [`ParsingResult`](@ref) objects, this function will be applied to each element of the vector.

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

  - `res::ParsingResult`: A new [`ParsingResult`](@ref) with all group and special variable references expanded to asset names.

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
"""
    get_linear_constraints(lcs::Union{<:ParsingResult, <:AbstractVector{<:ParsingResult}},
                           sets::AssetSets; datatype::DataType = Float64, strict::Bool = false)

Convert parsed linear constraint equations into a `LinearConstraint` object.

`get_linear_constraints` takes one or more [`ParsingResult`](@ref) objects (as produced by [`parse_equation`](@ref)), expands variable names using the provided [`AssetSets`](@ref), and assembles the corresponding constraint matrices and right-hand side vectors. The result is a [`LinearConstraint`](@ref) object containing both equality and inequality constraints, suitable for use in portfolio optimisation routines.

# Arguments

  - `lcs`: A single [`ParsingResult`](@ref) or a vector of such objects, representing parsed constraint equations.
  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.
  - `datatype`: Numeric type for coefficients and right-hand side.
  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning.

# Details

  - For each constraint, variable names are matched to the asset universe in `sets`.
  - Coefficient vectors are assembled for each constraint, with entries corresponding to the order of assets in `sets`.
  - Constraints are separated into equality (`==`) and inequality (`<=`, `>=`) types.
  - The function validates that all constraints reference valid assets or groups, using `@argcheck` for defensive programming.
  - Returns `nothing` if no valid constraints are found after processing.

# Returns

  - `lcs::LinearConstraint`: An object containing the assembled equality and inequality constraints, or `nothing` if no constraints are present.

# Related

  - [`ParsingResult`](@ref)
  - [`LinearConstraint`](@ref)
  - [`parse_equation`](@ref)
  - [`replace_group_by_assets`](@ref)
"""
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
        @argcheck(any(!iszero, At),
                  DomainError("At least one entry in At must be non-zero:\nany(!iszero, At) => $(any(!iszero, At))"))
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
"""
    struct LinearConstraintEstimator{T1} <: AbstractConstraintEstimator
        val::T1
    end

Container for one or more linear constraint equations to be parsed and converted into constraint matrices.

# Fields

  - `val`: A single equation as an `AbstractString` or `Expr`, or a vector of such equations.

# Constructor

    LinearConstraintEstimator(;
                              val::Union{<:AbstractString, Expr,
                                         <:AbstractVector{<:Union{<:AbstractString, Expr}}})

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(val)`.

# Examples

```jldoctest
julia> lce = LinearConstraintEstimator(; val = ["w_A + w_B == 1", "w_A >= 0.1"]);

julia> sets = AssetSets(; key = "nx", dict = Dict("nx" => ["w_A", "w_B"]));

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
    function LinearConstraintEstimator(val::Union{<:AbstractString, Expr,
                                                  <:AbstractVector{<:Union{<:AbstractString,
                                                                           Expr}}})
        if isa(val, Union{<:AbstractString, <:AbstractVector})
            @argcheck(!isempty(val))
        end
        return new{typeof(val)}(val)
    end
end
function LinearConstraintEstimator(;
                                   val::Union{<:AbstractString, Expr,
                                              <:AbstractVector{<:Union{<:AbstractString,
                                                                       Expr}}})
    return LinearConstraintEstimator(val)
end
"""
    linear_constraints(lcs::Union{Nothing, LinearConstraint}, args...; kwargs...)

No-op fallback for returning an existing `LinearConstraint` object or `nothing`.

This method is used to pass through an already constructed [`LinearConstraint`](@ref) object or `nothing` without modification. It enables composability and uniform interface handling in constraint generation workflows, allowing functions to accept either raw equations or pre-built constraint objects.

# Arguments

  - `lcs`: An existing [`LinearConstraint`](@ref) object or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `lcs`: The input, unchanged.

# Related

  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`linear_constraints`](@ref)
"""
function linear_constraints(lcs::Union{Nothing, LinearConstraint}, args...; kwargs...)
    return lcs
end
"""
    linear_constraints(eqn::Union{<:AbstractString, Expr,
                                  <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                       sets::AssetSets; ops1::Tuple = ("==", "<=", ">="),
                       ops2::Tuple = (:call, :(==), :(<=), :(>=)), datatype::DataType = Float64,
                       strict::Bool = false, bl_flag::Bool = false)

Parse and convert one or more linear constraint equations into a [`LinearConstraint`](@ref) object.

This function parses one or more constraint equations (as strings, expressions, or vectors thereof), replaces group or asset references using the provided [`AssetSets`](@ref), and constructs the corresponding constraint matrices. The result is a [`LinearConstraint`](@ref) object containing both equality and inequality constraints, suitable for use in portfolio optimisation routines.

# Arguments

  - `eqn`: A single constraint equation (as `AbstractString` or `Expr`), or a vector of such equations.
  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.
  - `ops1`: Tuple of valid comparison operators as strings.
  - `ops2`: Tuple of valid comparison operators as expression heads.
  - `datatype`: Numeric type for coefficients and right-hand side.
  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning.
  - `bl_flag`: If `true`, enables Black-Litterman-style group expansion.

# Details

  - Each equation is parsed using [`parse_equation`](@ref), supporting both string and expression input.
  - Asset and group references in the equations are expanded using [`replace_group_by_assets`](@ref) and the provided `sets`.
  - The function separates equality and inequality constraints, assembling the corresponding matrices and right-hand side vectors.
  - Input validation is performed using `@argcheck` to ensure non-empty and consistent constraints.
  - Returns `nothing` if no valid constraints are found after parsing and expansion.

# Returns

  - `lcs::LinearConstraint`: An object containing the assembled equality and inequality constraints, or `nothing` if no constraints are present.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx", dict = Dict("nx" => ["w_A", "w_B", "w_C"]));

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
  - [`linear_constraints`](@ref)
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
"""
    linear_constraints(lcs::Union{<:LinearConstraintEstimator,
                                  <:AbstractVector{<:LinearConstraintEstimator}},
                       sets::AssetSets; datatype::DataType = Float64, strict::Bool = false,
                       bl_flag::Bool = false)

If `lcs` is a vector of [`LinearConstraintEstimator`](@ref) objects, this function is broadcast over the vector.

This method is a wrapper calling:

    linear_constraints(lcs.val, sets; datatype = datatype, strict = strict, bl_flag = bl_flag)

It is used for type stability and to provide a uniform interface for processing constraint estimators, as well as simplifying the use of multiple estimators simultaneously.

# Related

  - [`linear_constraints`](@ref)
"""
function linear_constraints(lcs::LinearConstraintEstimator, sets::AssetSets;
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false)
    return linear_constraints(lcs.val, sets; datatype = datatype, strict = strict,
                              bl_flag = bl_flag)
end
function linear_constraints(lcs::AbstractVector{<:LinearConstraintEstimator},
                            sets::AssetSets; datatype::DataType = Float64,
                            strict::Bool = false, bl_flag::Bool = false)
    return [linear_constraints(lc, sets; datatype = datatype, strict = strict,
                               bl_flag = bl_flag) for lc in lcs]
end
"""
    struct RiskBudgetResult{T1} <: AbstractConstraintResult
        val::T1
    end

Container for the result of a risk budget constraint.

`RiskBudgetResult` stores the vector of risk budget allocations resulting from risk budget constraint generation or normalisation. This type is used to encapsulate the output of risk budgeting routines in a consistent, composable format for downstream processing and reporting.

# Fields

  - `val`: Vector of risk budget allocations (typically `AbstractVector{<:Real}`).

# Constructor

    RiskBudgetResult(; val::AbstractVector{<:Real})

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(val)`.
  - `all(x -> x >= zero(x), val)`.

# Examples

```jldoctest
julia> RiskBudgetResult(; val = [0.2, 0.3, 0.5])
RiskBudgetResult
  val | Vector{Float64}: [0.2, 0.3, 0.5]
```

# Related

  - [`RiskBudgetEstimator`](@ref)
  - [`risk_budget_constraints`](@ref)
  - [`AbstractConstraintResult`](@ref)
"""
struct RiskBudgetResult{T1} <: AbstractConstraintResult
    val::T1
    function RiskBudgetResult(val::AbstractVector{<:Real})
        @argcheck(!isempty(val) && all(x -> x >= zero(x), val),
                  AssertionError("`val` must be non-empty and all its entries must be non-negative:\n!isempty(val) => $(!isempty(val))\nall(x -> x >= zero(x), val) => $(all(x -> x >= zero(x), val))"))
        return new{typeof(val)}(val)
    end
end
function RiskBudgetResult(; val::AbstractVector{<:Real})
    return RiskBudgetResult(val)
end
function risk_budget_view(::Nothing, args...)
    return nothing
end
function risk_budget_view(rb::RiskBudgetResult, i::AbstractVector)
    val = nothing_scalar_array_view(rb.val, i)
    return RiskBudgetResult(; val = val)
end
"""
    struct RiskBudgetEstimator{T1} <: AbstractConstraintEstimator
        val::T1
    end

Container for a risk budget allocation mapping or vector.

`RiskBudgetEstimator` stores a mapping from asset or group names to risk budget values, or a vector of such pairs, for use in risk budgeting constraint generation. This type enables composable and validated workflows for specifying risk budgets in portfolio optimisation routines.

# Fields

  - `val`: A dictionary, pair, or vector of pairs mapping asset or group names to risk budget values.

# Constructor

    RiskBudgetEstimator(;
                        val::Union{<:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                   <:AbstractVector{<:Pair{<:AbstractString, <:Real}}})

Keyword arguments correspond to the fields above.

## Validation

  - `val` is validated with [`assert_nonneg_finite_val`](@ref).

# Examples

```jldoctest
julia> RiskBudgetEstimator(; val = Dict("A" => 0.2, "B" => 0.3, "C" => 0.5))
RiskBudgetEstimator
  val | Dict{String, Float64}: Dict("B" => 0.3, "A" => 0.2, "C" => 0.5)

julia> RiskBudgetEstimator(; val = ["A" => 0.2, "B" => 0.3, "C" => 0.5])
RiskBudgetEstimator
  val | Vector{Pair{String, Float64}}: ["A" => 0.2, "B" => 0.3, "C" => 0.5]
```

# Related

  - [`RiskBudgetResult`](@ref)
  - [`risk_budget_constraints`](@ref)
  - [`AssetSets`](@ref)
"""
struct RiskBudgetEstimator{T1} <: AbstractConstraintEstimator
    val::T1
    function RiskBudgetEstimator(val::Union{<:AbstractDict,
                                            <:Pair{<:AbstractString, <:Real},
                                            <:AbstractVector{<:Pair{<:AbstractString,
                                                                    <:Real}}})
        assert_nonneg_finite_val(val)
        return new{typeof(val)}(val)
    end
end
function RiskBudgetEstimator(;
                             val::Union{<:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                        <:AbstractVector{<:Union{<:Pair{<:AbstractString,
                                                                        <:Real}}}})
    return RiskBudgetEstimator(val)
end
function risk_budget_view(rb::RiskBudgetEstimator, ::Any)
    return rb
end
"""
    risk_budget_constraints(::Nothing, args...; N::Real, datatype::DataType = Float64,
                            kwargs...)

No-op fallback for risk budget constraint generation.

This method returns a uniform risk budget allocation when no explicit risk budget is provided. It creates a [`RiskBudgetResult`](@ref) with equal weights summing to one, using the specified number of assets `N` and numeric type `datatype`. This is useful as a default in workflows where a risk budget is optional or omitted.

# Arguments

  - `::Nothing`: Indicates that no risk budget is provided.
  - `args...`: Additional positional arguments (ignored).
  - `N::Real`: Number of assets (required).
  - `datatype::DataType`: Numeric type for the risk budget vector.
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `RiskBudgetResult`: A result object containing a uniform risk budget vector of length `N`, with each entry equal to `1/N`.

# Examples

```jldoctest
julia> risk_budget_constraints(nothing; N = 3)
RiskBudgetResult
  val | StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}: StepRangeLen(0.3333333333333333, 0.0, 3)
```

# Related

  - [`RiskBudgetResult`](@ref)
  - [`risk_budget_constraints`](@ref)
"""
function risk_budget_constraints(::Nothing, args...; N::Real, kwargs...)
    iN = inv(N)
    return RiskBudgetResult(; val = range(; start = iN, stop = iN, length = N))
end
"""
    risk_budget_constraints(rb::RiskBudgetResult, args...; kwargs...)

No-op fallback for risk budget constraint propagation.

This method returns the input [`RiskBudgetResult`](@ref) object unchanged. It is used to pass through an already constructed risk budget allocation result, enabling composability and uniform interface handling in risk budgeting workflows.

# Arguments

  - `rb`: An existing [`RiskBudgetResult`](@ref) object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `rb`: The input `RiskBudgetResult` object, unchanged.

# Examples

```jldoctest
julia> RiskBudgetResult(; val = [0.2, 0.3, 0.5])
RiskBudgetResult
  val | Vector{Float64}: [0.2, 0.3, 0.5]
```

# Related

  - [`RiskBudgetResult`](@ref)
  - [`risk_budget_constraints`](@ref)
"""
function risk_budget_constraints(rb::RiskBudgetResult, args...; kwargs...)
    return rb
end
"""
    risk_budget_constraints(rb::Union{<:AbstractDict{<:AbstractString, <:Real},
                                      <:Pair{<:AbstractString, <:Real},
                                      <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                            sets::AssetSets; N::Real = length(sets.dict[sets.key]),
                            strict::Bool = false)

Generate a risk budget allocation from asset/group mappings and asset sets.

This method constructs a [`RiskBudgetResult`](@ref) from a mapping of asset or group names to risk budget values, using the provided [`AssetSets`](@ref). The mapping can be a dictionary, a single pair, or a vector of pairs. Asset and group names are resolved using `sets`, and the resulting risk budget vector is normalised to sum to one.

# Arguments

  - `rb`: A dictionary, pair, or vector of pairs mapping asset or group names to risk budget values.
  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.
  - `N`: Number of assets in the universe.
  - `strict`: If `true`, throws an error if a key in `rb` is not found in `sets`; if `false`, issues a warning.

# Details

  - Asset and group names in `rb` are mapped to indices in the asset universe using `sets`.
  - If a key is a group, all assets in the group are assigned the specified value.
  - The resulting vector is normalised to sum to one.
  - If `strict` is `true`, missing keys cause an error; otherwise, a warning is issued.

# Returns

  - `RiskBudgetResult`: A result object containing the normalised risk budget vector.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx", dict = Dict("nx" => ["A", "B", "C"], "group1" => ["A", "B"]));

julia> risk_budget_constraints(Dict("A" => 0.2, "group1" => 0.8), sets)
RiskBudgetResult
  val | Vector{Float64}: [0.41379310344827586, 0.41379310344827586, 0.17241379310344826]
```

# Related

  - [`RiskBudgetResult`](@ref)
  - [`AssetSets`](@ref)
  - [`estimator_to_val`](@ref)
  - [`risk_budget_constraints`](@ref)
"""
function risk_budget_constraints(rb::Union{<:AbstractDict{<:AbstractString, <:Real},
                                           <:Pair{<:AbstractString, <:Real},
                                           <:AbstractVector{<:Pair{<:AbstractString,
                                                                   <:Real}}},
                                 sets::AssetSets; N::Real = length(sets.dict[sets.key]),
                                 strict::Bool = false)
    val = estimator_to_val(rb, sets, inv(N); strict = strict)
    return RiskBudgetResult(; val = val / sum(val))
end
"""
    risk_budget_constraints(rb::Union{<:RiskBudgetEstimator,
                                      <:AbstractVector{<:RiskBudgetEstimator}}, sets::AssetSets;
                            strict::Bool = false, kwargs...)

If `rb` is a vector of [`RiskBudgetEstimator`](@ref) objects, this function is broadcast over the vector.

This method is a wrapper calling:

    risk_budget_constraints(rb.val, sets; strict = strict)

It is used for type stability and to provide a uniform interface for processing constraint estimators, as well as simplifying the use of multiple estimators simulatneously.

# Related

  - [`risk_budget_constraints`](@ref)
"""
function risk_budget_constraints(rb::RiskBudgetEstimator, sets::AssetSets;
                                 strict::Bool = false, kwargs...)
    return risk_budget_constraints(rb.val, sets; strict = strict)
end
function risk_budget_constraints(rb::AbstractVector{<:RiskBudgetEstimator}, sets::AssetSets;
                                 strict::Bool = false, kwargs...)
    return [risk_budget_constraints(_rb, sets; strict = strict) for _rb in rb]
end
"""
    struct AssetSetsMatrixEstimator{T1} <: AbstractConstraintEstimator
        val::T1
    end

Estimator for constructing asset set membership matrices from asset groupings.

`AssetSetsMatrixEstimator` is a container type for specifying the key or group name used to generate a binary asset-group membership matrix from an [`AssetSets`](@ref) object. This is used in constraint generation and portfolio construction workflows that require mapping assets to groups or categories.

# Fields

  - `val`: The key or group name to extract from the asset sets.

# Constructor

    AssetSetsMatrixEstimator(; val::AbstractString)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(val)`.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx",
                        dict = Dict("nx" => ["A", "B", "C"],
                                    "sector" => ["Tech", "Tech", "Finance"]));

julia> est = AssetSetsMatrixEstimator(; val = "sector")
AssetSetsMatrixEstimator
  val | String: "sector"

julia> asset_sets_matrix(est, sets)
2×3 transpose(::BitMatrix) with eltype Bool:
 1  1  0
 0  0  1
```

# Related

  - [`AssetSets`](@ref)
  - [`asset_sets_matrix`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
"""
struct AssetSetsMatrixEstimator{T1} <: AbstractConstraintEstimator
    val::T1
    function AssetSetsMatrixEstimator(val::AbstractString)
        @argcheck(!isempty(val))
        return new{typeof(val)}(val)
    end
end
function AssetSetsMatrixEstimator(; val::AbstractString)
    return AssetSetsMatrixEstimator(val)
end
"""
    asset_sets_matrix(smtx::Union{Symbol, <:AbstractString}, sets::AssetSets)

Construct a binary asset-group membership matrix from asset set groupings.

`asset_sets_matrix` generates a binary (0/1) matrix indicating asset membership in groups or categories, based on the key or group name `smtx` in the provided [`AssetSets`](@ref). Each row corresponds to a unique group value, and each column to an asset in the universe. This is used in constraint generation and portfolio construction workflows that require mapping assets to groups or categories.

# Arguments

  - `smtx`: The key or group name to extract from the asset sets.
  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.

# Returns

  - `A::BitMatrix`: A binary matrix of size (number of groups) × (number of assets), where `A[i, j] == 1` if asset `j` belongs to group `i`.

# Details

  - The function checks that `smtx` exists in `sets.dict` and that its length matches the asset universe.
  - Each unique value in `sets.dict[smtx]` defines a group.
  - The output matrix is transposed so that rows correspond to groups and columns to assets.

# Validation

  - `haskey(sets.dict, smtx)`.
  - Throws an `AssertionError` if the length of `sets.dict[smtx]` does not match the asset universe.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx",
                        dict = Dict("nx" => ["A", "B", "C"],
                                    "sector" => ["Tech", "Tech", "Finance"]));

julia> asset_sets_matrix("sector", sets)
2×3 transpose(::BitMatrix) with eltype Bool:
 1  1  0
 0  0  1
```

# Related

  - [`AssetSets`](@ref)
  - [`AssetSetsMatrixEstimator`](@ref)
  - [`asset_sets_matrix_view`](@ref)
"""
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
"""
    asset_sets_matrix(smtx::Union{Nothing, <:AbstractMatrix}, args...)

No-op fallback for asset set membership matrix construction.

This method returns the input matrix `smtx` unchanged. It is used as a fallback when the asset set membership matrix is already provided as an `AbstractMatrix` or is `nothing`, enabling composability and uniform interface handling in constraint generation workflows.

# Arguments

  - `smtx`: An existing asset set membership matrix (`AbstractMatrix`) or `nothing`.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `smtx`: The input matrix or `nothing`, unchanged.

# Related

  - [`AssetSets`](@ref)
  - [`AssetSetsMatrixEstimator`](@ref)
  - [`asset_sets_matrix`](@ref)
"""
function asset_sets_matrix(smtx::Union{Nothing, <:AbstractMatrix}, args...)
    return smtx
end
"""
    asset_sets_matrix(smtx::AssetSetsMatrixEstimator, sets::AssetSets)

This method is a wrapper calling:

    asset_sets_matrix(smtx.val, sets)

It is used for type stability and to provide a uniform interface for processing constraint estimators, as well as simplifying the use of multiple estimators simulatneously.

# Related

  - [`asset_sets_matrix`](@ref)
"""
function asset_sets_matrix(smtx::AssetSetsMatrixEstimator, sets::AssetSets)
    return asset_sets_matrix(smtx.val, sets)
end
"""
    asset_sets_matrix(smtx::AbstractVector{<:Union{<:AbstractMatrix,
                                                   <:AssetSetsMatrixEstimator}},
                      sets::AssetSets)

Broadcasts [`asset_sets_matrix`](@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simulatneously.
"""
function asset_sets_matrix(smtx::AbstractVector{<:Union{<:AbstractMatrix,
                                                        <:AssetSetsMatrixEstimator}},
                           sets::AssetSets)
    return [asset_sets_matrix(_smtx, sets) for _smtx in smtx]
end
"""
"""
function asset_sets_matrix_view(smtx::AbstractMatrix, i::AbstractVector; kwargs...)
    return view(smtx, :, i)
end
function asset_sets_matrix_view(smtx::Union{Nothing, AssetSetsMatrixEstimator}, ::Any;
                                kwargs...)
    return smtx
end
function asset_sets_matrix_view(smtx::AbstractVector{<:Union{<:AbstractMatrix,
                                                             <:AssetSetsMatrixEstimator}},
                                i::AbstractVector; kwargs...)
    return [asset_sets_matrix_view(_smtx, i; kwargs...) for _smtx in smtx]
end

export AssetSets, PartialLinearConstraint, LinearConstraint, LinearConstraintEstimator,
       AssetSetsMatrixEstimator, RiskBudgetResult, RiskBudgetEstimator, ParsingResult,
       RhoParsingResult, parse_equation, replace_group_by_assets, estimator_to_val,
       linear_constraints, risk_budget_constraints, asset_sets_matrix
