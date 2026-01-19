"""
    struct PartialLinearConstraint{T1, T2} <: AbstractConstraintResult
        A::T1
        B::T2
    end

Container for a set of linear constraints (either equality or inequality) in the form `A * x = B` or `A * x ≤ B`.

`PartialLinearConstraint` stores the coefficient matrix `A` and right-hand side vector `B` for a group of linear constraints. This type is used internally by [`LinearConstraint`](@ref) to represent either the equality or inequality constraints in a portfolio optimisation problem.

# Fields

  - `A`: Coefficient matrix of the linear constraints.
  - `B`: Right-hand side vector of the linear constraints.

# Constructor

    PartialLinearConstraint(; A::MatNum, B::VecNum)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(A)`.
  - `!isempty(B)`.

# Examples

```jldoctest
julia> PartialLinearConstraint(; A = [1.0 2.0; 3.0 4.0], B = [5.0, 6.0])
PartialLinearConstraint
  A ┼ 2×2 Matrix{Float64}
  B ┴ Vector{Float64}: [5.0, 6.0]
```

# Related

  - [`LinearConstraint`](@ref)
  - [`LinearConstraintEstimator`](@ref)
"""
struct PartialLinearConstraint{T1, T2} <: AbstractConstraintResult
    A::T1
    B::T2
    function PartialLinearConstraint(A::MatNum, B::VecNum)
        @argcheck(!isempty(A), IsEmptyError)
        @argcheck(!isempty(B), IsEmptyError)
        return new{typeof(A), typeof(B)}(A, B)
    end
end
function PartialLinearConstraint(; A::MatNum, B::VecNum)
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

    LinearConstraint(; ineq::Option{<:PartialLinearConstraint} = nothing,
                     eq::Option{<:PartialLinearConstraint} = nothing)

Keyword arguments correspond to the fields above.

## Validation

  - `!(isnothing(ineq) && isnothing(eq))`, i.e. they cannot both be `nothing` at the same time.

# Examples

```jldoctest
julia> ineq = PartialLinearConstraint(; A = [1.0 2.0; 3.0 4.0], B = [5.0, 6.0]);

julia> eq = PartialLinearConstraint(; A = [7.0 8.0; 9.0 10.0], B = [11.0, 12.0]);

julia> LinearConstraint(; ineq = ineq, eq = eq)
LinearConstraint
  ineq ┼ PartialLinearConstraint
       │   A ┼ 2×2 Matrix{Float64}
       │   B ┴ Vector{Float64}: [5.0, 6.0]
    eq ┼ PartialLinearConstraint
       │   A ┼ 2×2 Matrix{Float64}
       │   B ┴ Vector{Float64}: [11.0, 12.0]
```

# Related

  - [`PartialLinearConstraint`](@ref)
  - [`LinearConstraintEstimator`](@ref)
"""
struct LinearConstraint{T1, T2} <: AbstractConstraintResult
    ineq::T1
    eq::T2
    function LinearConstraint(ineq::Option{<:PartialLinearConstraint},
                              eq::Option{<:PartialLinearConstraint})
        @argcheck(!(isnothing(ineq) && isnothing(eq)),
                  IsNothingError("ineq and eq cannot both be nothing. Got\nisnothing(ineq) => $(isnothing(ineq))\nisnothing(eq) => $(isnothing(eq))"))
        return new{typeof(ineq), typeof(eq)}(ineq, eq)
    end
end
function LinearConstraint(; ineq::Option{<:PartialLinearConstraint} = nothing,
                          eq::Option{<:PartialLinearConstraint} = nothing)
    return LinearConstraint(ineq, eq)
end
const VecLc = AbstractVector{<:LinearConstraint}
const Lc_VecLc = Union{<:LinearConstraint, <:VecLc}
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

All concrete types representing parsing results should subtype `AbstractParsingResult`.

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
    function ParsingResult(vars::VecStr, coef::VecNum, op::AbstractString, rhs::Number,
                           eqn::AbstractString)
        @argcheck(length(vars) == length(coef), DimensionMismatch)
        return new{typeof(vars), typeof(coef), typeof(op), typeof(rhs), typeof(eqn)}(vars,
                                                                                     coef,
                                                                                     op,
                                                                                     rhs,
                                                                                     eqn)
    end
end
const VecPR = AbstractVector{<:ParsingResult}
const PR_VecPR = Union{<:ParsingResult, <:VecPR}
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
    function RhoParsingResult(vars::VecStr, coef::VecNum, op::AbstractString, rhs::Number,
                              eqn::AbstractString,
                              ij::AbstractVector{<:Union{<:Tuple{<:Integer, <:Integer},
                                                         <:Tuple{<:VecInt, <:VecInt}}})
        @argcheck(length(vars) == length(coef), DimensionMismatch)
        return new{typeof(vars), typeof(coef), typeof(op), typeof(rhs), typeof(eqn),
                   typeof(ij)}(vars, coef, op, rhs, eqn, ij)
    end
end
"""
    struct AssetSets{T1, T2, T3} <: AbstractEstimator
        key::T1
        ukey::T2
        dict::T3
    end

Container for asset set and group information used in constraint generation.

`AssetSets` provides a unified interface for specifying the asset universe and any groupings or partitions of assets. It is used throughout constraint generation and estimator routines to expand group references, map group names to asset lists, and validate asset membership.

If a key in `dict` starts with the same value as `key`, it means that the corresponding group must have the same length as the asset universe, `dict[key]`. This is useful for defining partitions of the asset universe, for example when using [`asset_sets_matrix`]-(@ref) with [`NestedClustered`]-(@ref).

# Fields

  - `key`: The key in `dict` that identifies the primary list of assets. Groups prefixed by this `key` must have the same length as `dict[key]` as their lengths are preserved across views.
  - `ukey`: The key prefix used for asset sets with unique entries. If present, there must be an equivalent group prefixed by `key` with the same length as `dict[key]` as that group will be used to find the unique entries for the view.
  - `dict`: A dictionary mapping group names (or asset set names) to vectors of asset identifiers.

# Constructor

    AssetSets(; key::AbstractString = "nx", ukey::AbstractString = "ux",
              dict::AbstractDict{<:AbstractString, <:Any})

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(dict)`.
  - `haskey(dict, key)`.
  - `key !== ukey`.
  - `!startswith(key, ukey)`.
  - `!startswith(ukey, key)`.
  - If a key in `dict` starts with the same value as `key`, `length(dict[nx]) == length(dict[key])`.
  - If a key in `dict` starts with the same value as `ukey`, there must be a corresponding key in `dict` where the `ukey` prefix is replaced by the `key` prefix, and `length(dict[replace(k, ukey => key)]) == length(dict[key])`.

# Examples

```jldoctest
julia> AssetSets(; key = "nx", dict = Dict("nx" => ["A", "B", "C"], "group1" => ["A", "B"]))
AssetSets
   key ┼ String: "nx"
  ukey ┼ String: "ux"
  dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"], "group1" => ["A", "B"])
```

# Related

  - [`replace_group_by_assets`](@ref)
  - [`estimator_to_val`](@ref)
  - [`linear_constraints`](@ref)
"""
struct AssetSets{T1, T2, T3} <: AbstractEstimator
    key::T1
    ukey::T2
    dict::T3
    function AssetSets(key::AbstractString, ukey::AbstractString,
                       dict::AbstractDict{<:AbstractString, <:Any})
        @argcheck(!isempty(dict), IsEmptyError)
        @argcheck(haskey(dict, key), KeyError)
        @argcheck(key !== ukey, ValueError)
        @argcheck(!startswith(key, ukey))
        @argcheck(!startswith(ukey, key))
        for k in setdiff(keys(dict), (key,))
            if startswith(k, key)
                @argcheck(length(dict[k]) == length(dict[key]), DimensionMismatch)
            elseif startswith(k, ukey)
                tmp_key = replace(k, ukey => key)
                @argcheck(haskey(dict, tmp_key), KeyError)
                @argcheck(length(dict[tmp_key]) == length(dict[key]), DimensionMismatch)
            end
        end
        return new{typeof(key), typeof(ukey), typeof(dict)}(key, ukey, dict)
    end
end
function AssetSets(; key::AbstractString = "nx", ukey::AbstractString = "ux",
                   dict::AbstractDict{<:AbstractString, <:Any})
    return AssetSets(key, ukey, dict)
end
function nothing_asset_sets_view(sets::AssetSets, i)
    key = sets.key
    ukey = sets.ukey
    dict = typeof(sets.dict)()
    for (k, v) in sets.dict
        if startswith(k, key)
            v = view(v, i)
        elseif startswith(k, ukey)
            tmp_key = replace(k, ukey => key)
            v = unique(view(sets.dict[tmp_key], i))
        end
        push!(dict, k => v)
    end
    return AssetSets(; key = key, ukey = ukey, dict = dict)
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
    group_to_val!(nx::VecStr, sdict::AbstractDict, key::Any, val::Number,
                  dict::EstValType, arr::VecNum, strict::Bool)

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

  - `nothing`. The operation is performed in-place on `arr`.

# Related

  - [`estimator_to_val`](@ref)
  - [`AssetSets`](@ref)
"""
function group_to_val!(nx::VecStr, sdict::AbstractDict, key::Any, val::Number,
                       dict::EstValType, arr::VecNum, strict::Bool)
    assets = get(sdict, key, nothing)
    if isnothing(assets)
        msg = "$(key) is not in $(keys(sdict)).\n$(dict)"
        strict ? throw(ArgumentError(msg)) : @warn(msg)
    else
        unique!(assets)
        idx = [findfirst(x -> x == asset, nx) for asset in assets]
        N1 = length(idx)
        filter!(!isnothing, idx)
        N2 = length(idx)
        if N1 != N2
            msg = "Some assets in group `$(key)` are not in the asset universe.\nAssets in group `$key`: $(assets)\nAssets in universe: $(nx).\n$(dict)"
            strict ? throw(ArgumentError(msg)) : @warn(msg)
        end
        arr[idx] .= val
    end
    return nothing
end
"""
    estimator_to_val(dict::EstValType, sets::AssetSets, val::Option{<:Number} = nothing, key::Option{<:AbstractString} = nothing; strict::Bool = false)

Return value for assets or groups, based on a mapping and asset sets.

The function creates the vector and sets the values for assets or groups as specified by `dict`, using the asset universe and groupings in `sets`. If a key in `dict` is not found in the asset sets, the function either throws an error or issues a warning, depending on the `strict` flag.

# Arguments

  - `arr`: The array to be modified in-place.
  - `dict`: A dictionary, vector of pairs, or single pair mapping asset or group names to values.
  - `sets`: The [`AssetSets`](@ref) containing the asset universe and group definitions.
  - `val`: The default value to assign to assets not specified in `dict`.
  - `key`: (Optional) Key in the [`AssetSets`](@ref) to specify the asset universe for constraint generation. When provided, takes precedence over `key` field of [`AssetSets`](@ref).
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

  - `arr::VecNum`: Value array.

# Related

  - [`group_to_val!`](@ref)
  - [`AssetSets`](@ref)
  - [`estimator_to_val`](@ref)
"""
function estimator_to_val(dict::MultiEstValType, sets::AssetSets,
                          val::Option{<:Number} = nothing,
                          key::Option{<:AbstractString} = nothing;
                          datatype::DataType = Float64, strict::Bool = false)
    val = ifelse(isnothing(val), zero(datatype), val)
    nx = sets.dict[ifelse(isnothing(key), sets.key, key)]
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
function estimator_to_val(dict::PairStrNum, sets::AssetSets,
                          val::Option{<:Number} = nothing,
                          key::Option{<:AbstractString} = nothing;
                          datatype::DataType = Float64, strict::Bool = false)
    val = ifelse(isnothing(val), zero(datatype), val)
    nx = sets.dict[ifelse(isnothing(key), sets.key, key)]
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
    estimator_to_val(val::Option{<:Number}, args...; kwargs...)

Fallback no-op for value mapping in asset/group estimators.

This method returns the input value `val` as-is, without modification or mapping. It serves as a fallback for cases where the input is already a numeric value, a vector of numeric values, or `nothing`, and no further processing is required.

# Arguments

  - `val`: A value of type `Nothing` or a single numeric value.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `val::Option{<:Number}`: The input `val`, unchanged.

# Related

  - [`estimator_to_val`](@ref)
  - [`group_to_val!`](@ref)
  - [`AssetSets`](@ref)
"""
function estimator_to_val(val::Option{<:Number}, args...; kwargs...)
    return val
end
"""
    estimator_to_val(val::VecNum, sets::AssetSets, ::Any = nothing,
                     key::Option{<:AbstractString} = nothing; kwargs...)

Return a numeric vector for asset/group estimators, validating length against asset universe.

This method checks that the input vector `val` matches the length of the asset universe in `sets`, and returns it unchanged if valid. It is used as a fast path for workflows where the value vector is already constructed and requires only defensive validation.

# Arguments

  - `val`: Numeric vector to be mapped to assets/groups.
  - `sets`: [`AssetSets`](@ref) containing the asset universe and group definitions.
  - `::Any`: Fill value for API consistency (ignored).
  - `key`: (Optional) Key in the [`AssetSets`](@ref) to specify the asset universe for constraint generation. When provided, takes precedence over `key` field of [`AssetSets`](@ref).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `val::VecNum`: The input vector, unchanged.

# Validation

  - `length(val) == length(sets.dict[ifelse(isnothing(key), sets.key, key)]`.

# Related

  - [`estimator_to_val`](@ref)
  - [`AssetSets`](@ref)
  - [`group_to_val!`](@ref)
"""
function estimator_to_val(val::VecNum, sets::AssetSets, ::Any = nothing,
                          key::Option{<:AbstractString} = nothing; kwargs...)
    @argcheck(length(val) == length(sets.dict[ifelse(isnothing(key), sets.key, key)]),
              DimensionMismatch)
    return val
end
"""
    struct UniformValues <: AbstractEstimatorValueAlgorithm end

Custom weight bounds constraint for uniformly distributing asset weights, `1/N` for lower bounds and `1` for upper bounds, where `N` is the number of assets.

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> PortfolioOptimisers.estimator_to_val(UniformValues(), sets)
StepRangeLen(0.3333333333333333, 0.0, 3)
```

# Related

  - [`AbstractEstimatorValueAlgorithm`](@ref)
  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
"""
struct UniformValues <: AbstractEstimatorValueAlgorithm end
function estimator_to_val(::UniformValues, sets::AssetSets, ::Any = nothing,
                          key::Option{<:AbstractString} = nothing;
                          datatype::DataType = Float64, kwargs...)
    N = length(sets.dict[ifelse(isnothing(key), sets.key, key)])
    iN = datatype(inv(N))
    return range(; start = iN, stop = iN, length = N)
end
"""
    _eval_numeric_functions(expr)

Recursively evaluate numeric functions and constants in a Julia expression.

`_eval_numeric_functions` traverses a Julia expression tree and evaluates any sub-expressions that are purely numeric, including standard mathematical functions and constants (such as `Inf`). This is used to simplify constraint equations before further parsing and canonicalisation.

# Arguments

  - `expr`: The Julia expression to evaluate. Can be a `Number`, `Symbol`, or `Expr`.

# Details

  - `expr`:

      + `Number`: It is returned as-is.
      + `:Inf`: Returns `Inf`.
      + `Expr`: Representing a function call, and all arguments are numeric, the function is evaluated and replaced with its result.
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

  - `terms::Vector{Tuple{Float64, Option{<:String}}}`: A vector of `(coefficient, variable)` pairs, where `variable` is a string for variable terms or `nothing` for constant terms.

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
  - `terms`: A vector to which `(coefficient, variable)` pairs are appended in-place. Each pair is of the form `(Float64, Option{<:String})`, where `Nothing` indicates a constant term.

# Details

  - `expr`:

      + `Number`: Appends `(coeff * oftype(coeff, expr), nothing)` to `terms`.

      + `Symbol`: Appends `(coeff, string(expr))` to `terms`.
      + `Expr`:

          * For multiplication (`*`), distributes the coefficient to the numeric part.
          * For division (`/`), divides the coefficient by the numeric denominator.
          * For addition (`+`), recursively collects terms from all arguments.
          * For subtraction (`-`), recursively collects terms from all arguments except the last, which is negated.
          * For all other expressions, treats as a variable and appends as `(coeff, string(expr))`.

# Returns

  - `nothing`. The function modifies `terms` in-place.

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

  - `nothing`.

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
    @argcheck(expr.head != :incomplete,
              Meta.ParseError("$side is an incomplete expression.\n$expr"))
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
    parse_equation(eqn::EqnType;
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

  - If `eqn::Str_Expr`:

      + `res::ParsingResult`: Structured parsing result.

  - If `eqn::AbstractVector`:

      + `res::Vector{ParsingResult}`: Vector of structured parsing results.

# Examples

```jldoctest
julia> parse_equation("w_A + 2w_B <= 1")
ParsingResult
  vars ┼ Vector{String}: ["w_A", "w_B"]
  coef ┼ Vector{Float64}: [1.0, 2.0]
    op ┼ String: "<="
   rhs ┼ Float64: 1.0
   eqn ┴ SubString{String}: "w_A + 2.0*w_B <= 1.0"
```

# Related

  - [`ParsingResult`](@ref)
"""
function parse_equation(eqn::AbstractString; ops1::Tuple = ("==", "<=", ">="),
                        datatype::DataType = Float64, kwargs...)
    @argcheck(!occursin("++", eqn),
              Meta.ParseError("Invalid operator '++' detected in equation."))
    # 1. Identify the comparison operator
    op = findfirst(op -> occursin(op, eqn), ops1)
    @argcheck(!isnothing(op),
              Meta.ParseError("Equation must contain a valid comparison operator $(join(ops1,", ")) .\n$(eqn)"))
    opstr = ops1[op]
    parts = split(eqn, opstr)
    @argcheck(length(parts) == 2,
              Meta.ParseError("Equation must have exactly one comparison operator.\n$(eqn)"))
    lhs, rhs = strip.(parts)
    # 2. Parse both sides into Julia expressions
    lexpr = Meta.parse(lhs)
    _rethrow_parse_error(lexpr, :lhs)
    rexpr = Meta.parse(rhs)
    _rethrow_parse_error(rexpr, :rhs)
    return _parse_equation(lexpr, opstr, rexpr, datatype)
end
function _has_invalid_plus(expr)
    if !(isa(expr, Expr) && expr.head == :call)
        return false
    end
    # Check for nested :+ calls (e.g., :(+(+(a, b), c))) or more than two arguments
    if expr.args[1] == :++
        # If any argument is itself a :+ call, that's suspicious (from "++")
        return true
    end
    # Recurse into sub-expressions
    return any(_has_invalid_plus(arg) for arg in expr.args[2:end] if isa(arg, Expr))
end
function parse_equation(expr::Expr; ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                        datatype::DataType = Float64, kwargs...)
    # Recursively check for invalid "++" pattern in the expression tree
    @argcheck(!_has_invalid_plus(expr),
              Meta.ParseError("Invalid operator pattern '++' detected in equation expression:\n$expr"))
    # Ensure the expression is a call to a valid comparison operator
    @argcheck(expr.head == :call,
              Meta.ParseError("Expression must be a function call (comparison operator expected):\n$expr"))
    # Count how many valid operators are present
    op_count = count(op -> expr.args[1] == op, ops2[2:end])
    @argcheck(op_count == 1,
              Meta.ParseError("Expression must contain a valid comparison operator $(join(ops2[2:end], ", ")) .\n$expr"))
    opstr = string(expr.args[1])
    lhs, rhs = expr.args[2], expr.args[3]
    return _parse_equation(lhs, opstr, rhs, datatype)
end
function parse_equation(eqn::VecStr_Expr; ops1::Tuple = ("==", "<=", ">="),
                        ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                        datatype::DataType = Float64)
    return parse_equation.(eqn; ops1 = ops1, ops2 = ops2, datatype = datatype)
end
"""
    replace_group_by_assets(res::PR_VecPR,
                            sets::AssetSets; bl_flag::Bool = false, ep_flag::Bool = false,
                            rho_flag::Bool = false)

If `res` is a vector of [`ParsingResult`](@ref) objects, this function will be applied to each element of the vector.

Expand group or special variable references in a [`ParsingResult`](@ref) to their corresponding asset names.

This function takes a [`ParsingResult`](@ref) containing variable names (which may include group names, `prior(...)` expressions, or correlation views like `(A, B)`), and replaces these with the actual asset names from the provided [`AssetSets`](@ref). It supports Black-Litterman-style group expansion, entropy pooling prior views, and correlation view parsing for advanced constraint generation.

# Arguments

  - `res`: A [`ParsingResult`](@ref) object containing variables and coefficients to be expanded.
  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.
  - `bl_flag`: If `true`, enables Black-Litterman-style group expansion.
  - `ep_flag`: If `true`, enables expansion of `prior(...)` expressions for entropy pooling.
  - `rho_flag`: If `true`, enables expansion of correlation views `(A, B)` for entropy pooling.

# Validation

    - `bl_flag` can only be `true` if both `ep_flag` and `rho_flag` are `false`.
    - `rho_flag` can only be `true` if `ep_flag` is also `true`.

# Details

  - Group names in `res.vars` are replaced by the corresponding asset names from `sets.dict`.
  - If `bl_flag` is `true`, coefficients for group references are divided equally among the assets in the group.
  - If `ep_flag` is `true`, expands `prior(asset)` or `prior(group)` expressions for entropy pooling.
  - If `rho_flag` is `true`, expands correlation view expressions `(A, B)` or `prior(A, B)` for entropy pooling, mapping them to asset pairs.
  - If a variable or group is not found in `sets.dict`, it is skipped.

# Returns

  - `res::ParsingResult`: A new [`ParsingResult`](@ref) with all group and special variable references expanded to asset names.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx", dict = Dict("nx" => ["A", "B", "C"], "group1" => ["A", "B"]));

julia> res = parse_equation("group1 + 2C == 1")
ParsingResult
  vars ┼ Vector{String}: ["C", "group1"]
  coef ┼ Vector{Float64}: [2.0, 1.0]
    op ┼ String: "=="
   rhs ┼ Float64: 1.0
   eqn ┴ SubString{String}: "2.0*C + group1 == 1.0"

julia> replace_group_by_assets(res, sets)
ParsingResult
  vars ┼ Vector{String}: ["C", "A", "B"]
  coef ┼ Vector{Float64}: [2.0, 1.0, 1.0]
    op ┼ String: "=="
   rhs ┼ Float64: 1.0
   eqn ┴ String: "2.0*C + 1.0*A + 1.0*B == 1.0"
```

# Related

  - [`AssetSets`](@ref)
  - [`ParsingResult`](@ref)
  - [`parse_equation`](@ref)
"""
function replace_group_by_assets(res::ParsingResult, sets::AssetSets, bl_flag::Bool = false,
                                 ep_flag::Bool = false, rho_flag::Bool = false)
    @argcheck(!(bl_flag && (rho_flag || ep_flag)),
              ArgumentError("bl_flag can only be true if ep_flag and rho_flag are false. Got\nbl_flag => $(bl_flag)\nep_flag => $(ep_flag)\nrho_flag => $(rho_flag)."))
    @argcheck(!(rho_flag && !ep_flag),
              ArgumentError("rho_flag can only be true if ep_flag is also true. Got\nrho_flag => $rho_flag\nep_flag => $ep_flag"))
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
                @argcheck(ep_flag && rho_flag,
                          ArgumentError("The pattern '(a, b)' can only be used for rho_views (rho_flag is true) in entropy pooling (ep_flag is true). Got\nep_flag => $(ep_flag)\nrho_flag => $(rho_flag)."))
                @argcheck(!isnothing(n),
                          ArgumentError("Correlation views can only be of the form '(a, b)'. Got\nv => $v"))
                asset1 = n.captures[1]
                asset2 = n.captures[2]
                asset1 = get(sets.dict, asset1, nothing)
                asset2 = get(sets.dict, asset2, nothing)
                if isnothing(asset1) && isnothing(asset2)
                    continue
                end
                @argcheck(!isnothing(asset1), IsNothingError)
                @argcheck(!isnothing(asset2), IsNothingError)
                @argcheck(length(asset1) == length(asset2), DimensionMismatch)
                push!(variables_tmp, "([$(join(asset1, ", "))], [$(join(asset2, ", "))])")
                push!(coeffs_tmp, coeffs[i])
                push!(idx_rm, i)
            end
        else
            @argcheck(ep_flag,
                      ArgumentError("The pattern 'prior(a)' can only be used in entropy pooling (ep_flag is true). Got\nep_flag => $(ep_flag)."))
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
                @argcheck(rho_flag,
                          ArgumentError("The pattern 'prior(a, b)' can only be used for rho_views (rho_flag is true) in entropy pooling (ep_flag is true). Got\nep_flag => $(ep_flag)\nrho_flag => $(rho_flag)."))
                @argcheck(!isnothing(n),
                          ArgumentError("Correlation views prior can only be of the form 'prior(a, b)'. Got\nv => $v"))
                asset1 = n.captures[1]
                asset2 = n.captures[2]
                asset1 = get(sets.dict, asset1, nothing)
                asset2 = get(sets.dict, asset2, nothing)
                if isnothing(asset1) && isnothing(asset2)
                    continue
                end
                @argcheck(!isnothing(asset1), IsNothingError)
                @argcheck(!isnothing(asset2), IsNothingError)
                @argcheck(length(asset1) == length(asset2), DimensionMismatch)
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
function replace_group_by_assets(res::VecPR, sets::AssetSets, args...)
    return replace_group_by_assets.(res, sets, args...)
end
"""
    get_linear_constraints(lcs::PR_VecPR, sets::AssetSets,
                           key::Option{<:AbstractString} = nothing;
                           datatype::DataType = Float64, strict::Bool = false)

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
function get_linear_constraints(lcs::PR_VecPR, sets::AssetSets,
                                key::Option{<:AbstractString} = nothing;
                                datatype::DataType = Float64, strict::Bool = false)
    if isa(lcs, AbstractVector)
        @argcheck(!isempty(lcs), IsEmptyError)
    end
    A_ineq = Vector{datatype}(undef, 0)
    B_ineq = Vector{datatype}(undef, 0)
    A_eq = Vector{datatype}(undef, 0)
    B_eq = Vector{datatype}(undef, 0)
    nx = sets.dict[ifelse(isnothing(key), sets.key, key)]
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
    struct LinearConstraintEstimator{T1, T2} <: AbstractConstraintEstimator
        val::T1
        key::T2
    end

Container for one or more linear constraint equations to be parsed and converted into constraint matrices.

# Fields

  - `val`: A single equation as an `AbstractString` or `Expr`, or a vector of such equations.
  - `key`: (Optional) Key in the [`AssetSets`](@ref) to specify the asset universe for constraint generation. When provided, takes precedence over `key` field of [`AssetSets`](@ref).

# Constructor

    LinearConstraintEstimator(; val::EqnType, key::Option{<:AbstractString} = nothing)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(val)`.

# Examples

```jldoctest
julia> lce = LinearConstraintEstimator(; val = ["w_A + w_B == 1", "w_A >= 0.1"]);

julia> sets = AssetSets(; key = "nx", dict = Dict("nx" => ["w_A", "w_B"]));

julia> linear_constraints(lce, sets)
LinearConstraint
  ineq ┼ PartialLinearConstraint
       │   A ┼ 1×2 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       │   B ┴ Vector{Float64}: [-0.1]
    eq ┼ PartialLinearConstraint
       │   A ┼ 1×2 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       │   B ┴ Vector{Float64}: [1.0]
```

# Related

  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`parse_equation`](@ref)
  - [`linear_constraints`](@ref)
"""
struct LinearConstraintEstimator{T1, T2} <: AbstractConstraintEstimator
    val::T1
    key::T2
    function LinearConstraintEstimator(val::EqnType,
                                       key::Option{<:AbstractString} = nothing)
        if isa(val, Str_Vec)
            @argcheck(!isempty(val))
        end
        if !isnothing(key)
            @argcheck(!isempty(key))
        end
        return new{typeof(val), typeof(key)}(val, key)
    end
end
function LinearConstraintEstimator(; val::EqnType, key::Option{<:AbstractString} = nothing)
    return LinearConstraintEstimator(val, key)
end
const LcE_Lc = Union{<:LinearConstraintEstimator, <:LinearConstraint}
const VecLcE_Lc = AbstractVector{<:LcE_Lc}
const VecLcE = AbstractVector{<:LinearConstraintEstimator}
const LcE_Lc_VecLcE_Lc = Union{<:LcE_Lc, <:VecLcE_Lc}
const LcE_VecLcE = Union{<:LinearConstraintEstimator, <:VecLcE}
"""
    linear_constraints(lcs::Option{<:LinearConstraint}, args...; kwargs...)

No-op fallback for returning an existing `LinearConstraint` object or `nothing`.

This method is used to pass through an already constructed [`LinearConstraint`](@ref) object or `nothing` without modification. It enables composability and uniform interface handling in constraint generation workflows, allowing functions to accept either raw equations or pre-built constraint objects.

# Arguments

  - `lcs`: An existing [`LinearConstraint`](@ref) object or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `lcs::Option{<:LinearConstraint}`: The input, unchanged.

# Related

  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`linear_constraints`](@ref)
"""
function linear_constraints(lcs::Option{<:LinearConstraint}, args...; kwargs...)
    return lcs
end
"""
    linear_constraints(eqn::EqnType,
                       sets::AssetSets; ops1::Tuple = ("==", "<=", ">="),
                       key::Option{<:AbstractString} = nothing;
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
  ineq ┼ PartialLinearConstraint
       │   A ┼ 1×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       │   B ┴ Vector{Float64}: [-0.1]
    eq ┼ PartialLinearConstraint
       │   A ┼ 1×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       │   B ┴ Vector{Float64}: [1.0]
```

# Related

  - [`parse_equation`](@ref)
  - [`replace_group_by_assets`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`LinearConstraint`](@ref)
  - [`AssetSets`](@ref)
  - [`linear_constraints`](@ref)
"""
function linear_constraints(eqn::EqnType, sets::AssetSets,
                            key::Option{<:AbstractString} = nothing;
                            ops1::Tuple = ("==", "<=", ">="),
                            ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false)
    lcs = parse_equation(eqn; ops1 = ops1, ops2 = ops2, datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets, bl_flag)
    return get_linear_constraints(lcs, sets, key; datatype = datatype, strict = strict)
end
"""
    linear_constraints(lcs::LcE_VecLcE,
                       sets::AssetSets; datatype::DataType = Float64, strict::Bool = false,
                       bl_flag::Bool = false)

If `lcs` is a vector of [`LinearConstraintEstimator`](@ref) objects, this function is broadcast over the vector.

This method is a wrapper calling:

    linear_constraints(lcs.val, sets, lcs.key; datatype = datatype, strict = strict, bl_flag = bl_flag)

It is used for type stability and to provide a uniform interface for processing constraint estimators, as well as simplifying the use of multiple estimators simultaneously.

# Related

  - [`linear_constraints`](@ref)
"""
function linear_constraints(lcs::LinearConstraintEstimator, sets::AssetSets;
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false)
    return linear_constraints(lcs.val, sets, lcs.key; datatype = datatype, strict = strict,
                              bl_flag = bl_flag)
end
function linear_constraints(lcs::VecLcE, sets::AssetSets; datatype::DataType = Float64,
                            strict::Bool = false, bl_flag::Bool = false)
    return [linear_constraints(lc, sets; datatype = datatype, strict = strict,
                               bl_flag = bl_flag) for lc in lcs]
end

export AssetSets, PartialLinearConstraint, LinearConstraint, LinearConstraintEstimator,
       ParsingResult, RhoParsingResult, parse_equation, replace_group_by_assets,
       estimator_to_val, linear_constraints, UniformValues
