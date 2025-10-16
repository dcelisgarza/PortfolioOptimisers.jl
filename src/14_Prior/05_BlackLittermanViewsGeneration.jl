"""
    struct BlackLittermanViews{T1, T2} <: AbstractResult
        P::T1
        Q::T2
    end

Container for Black-Litterman investor views in canonical matrix form.

`BlackLittermanViews` stores the views matrix `P` and the expected returns vector `Q` for use in Black-Litterman prior construction and related portfolio optimisation routines. The matrix `P` encodes the linear relationships between assets for each view, while `Q` specifies the expected value for each view.

# Fields

  - `P`: Matrix of view coefficients, where each row represents a view and each column corresponds to an asset.
  - `Q`: Vector of expected returns or values for each view.

# Constructor

    BlackLittermanViews(; P::AbstractMatrix, Q::AbstractVector)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(P)` and `!isempty(Q)`.
  - `size(P, 1) == length(Q)`.

# Examples

```jldoctest
julia> BlackLittermanViews(; P = [1 2 3 4; 5 6 7 8], Q = [9; 10])
BlackLittermanViews
  P | 2×4 Matrix{Int64}
  Q | Vector{Int64}: [9, 10]
```

# Related

  - [`black_litterman_views`](@ref)
"""
struct BlackLittermanViews{T1, T2} <: AbstractResult
    P::T1
    Q::T2
    function BlackLittermanViews(P::AbstractMatrix, Q::AbstractVector)
        @argcheck(!isempty(P) && !isempty(Q))
        @argcheck(size(P, 1) == length(Q))
        return new{typeof(P), typeof(Q)}(P, Q)
    end
end
function BlackLittermanViews(; P::AbstractMatrix, Q::AbstractVector)
    return BlackLittermanViews(P, Q)
end
"""
    get_black_litterman_views(lcs::Union{<:ParsingResult, <:AbstractVector{<:ParsingResult}},
                              sets::AssetSets; datatype::DataType = Float64,
                              strict::Bool = false)

Convert parsed Black-Litterman view equations into a `BlackLittermanViews` object.

`get_black_litterman_views` takes one or more [`ParsingResult`](@ref) objects (as produced by [`parse_equation`](@ref)), expands variable names using the provided [`AssetSets`](@ref), and assembles the canonical views matrix `P` and expected returns vector `Q` for Black-Litterman prior construction. The result is a [`BlackLittermanViews`](@ref) object suitable for use in portfolio optimisation routines.

# Arguments

  - `lcs`: A single [`ParsingResult`](@ref) or a vector of such objects, representing parsed Black-Litterman view equations.
  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.
  - `datatype`: Numeric type for coefficients and expected returns.
  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning.

# Details

  - For each view, variable names are matched to the asset universe in `sets`.
  - Coefficient vectors are assembled for each view, with entries corresponding to the order of assets in `sets`.
  - The function validates that all views reference valid assets or groups, using `@argcheck` for defensive programming.
  - Returns `nothing` if no valid views are found after processing.

# Returns

  - `BlackLittermanViews`: An object containing the assembled views matrix `P` and expected returns vector `Q`, or `nothing` if no views are present.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx", dict = Dict("nx" => ["A", "B", "C"]));

julia> lcs = parse_equation(["A + B == 0.05", "C == 0.02"]);

julia> PortfolioOptimisers.get_black_litterman_views(lcs, sets)
BlackLittermanViews
  P | 2×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
  Q | Vector{Float64}: [0.05, 0.02]
```

# Related

  - [`BlackLittermanViews`](@ref)
  - [`parse_equation`](@ref)
  - [`AssetSets`](@ref)
"""
function get_black_litterman_views(lcs::Union{<:ParsingResult,
                                              <:AbstractVector{<:ParsingResult}},
                                   sets::AssetSets; datatype::DataType = Float64,
                                   strict::Bool = false)
    if isa(lcs, AbstractVector)
        @argcheck(!isempty(lcs))
    end
    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)
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
        append!(P, At)
        append!(Q, lc.rhs)
    end
    return if !isempty(P)
        P = transpose(reshape(P, length(nx), :))
        BlackLittermanViews(; P = P, Q = Q)
    else
        nothing
    end
end
"""
    black_litterman_views(views::Union{Nothing, <:BlackLittermanViews}, args...; kwargs...)
    black_litterman_views(views::Union{<:AbstractString, Expr,
                                       <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                          sets::AssetSets; datatype::DataType = Float64, strict::Bool = false)
    black_litterman_views(views::LinearConstraintEstimator, sets::AssetSets;
                          datatype::DataType = Float64, strict::Bool = false)

Unified interface for constructing or passing through Black-Litterman investor views.

`black_litterman_views` provides a composable API for handling Black-Litterman views in portfolio optimisation workflows. It supports passing through an existing [`BlackLittermanViews`](@ref) object, constructing views from equations or constraint estimators, and converting parsed view equations into canonical matrix form.

# Arguments

  - `views`:

      + If views is `nothing` or already a [`BlackLittermanViews`](@ref) object, it is returned unchanged.
      + If `views` is `Union{<:AbstractString, Expr, <:AbstractVector{<:Union{<:AbstractString, Expr}}}`, the view(s) are parsed, groups are replaced by their constituent assets using `sets`, calls [`get_black_litterman_views`](@ref) and constructs a [`BlackLittermanViews`](@ref) object is constructed.
      + If `views` is a [`LinearConstraintEstimator`](@ref), calls the method described above using the `val` field of the estimator.

  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.
  - `datatype`: Numeric type for coefficients and expected returns.
  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning.

# Returns

  - `blv::BlackLittermanViews`: An object containing the assembled views matrix `P` and expected returns vector `Q`, or `nothing` if no views are present.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx", dict = Dict("nx" => ["A", "B", "C"]));

julia> black_litterman_views(["A + B == 0.05", "C == 0.02"], sets)
BlackLittermanViews
  P | 2×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
  Q | Vector{Float64}: [0.05, 0.02]

julia> lce = LinearConstraintEstimator(; val = ["A == 0.03", "B + C == 0.04"]);

julia> black_litterman_views(lce, sets)
BlackLittermanViews
  P | 2×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
  Q | Vector{Float64}: [0.03, 0.04]
```

# Related

  - [`BlackLittermanViews`](@ref)
  - [`get_black_litterman_views`](@ref)
  - [`parse_equation`](@ref)
  - [`AssetSets`](@ref)
  - [`LinearConstraintEstimator`](@ref)
"""
function black_litterman_views(views::Union{Nothing, <:BlackLittermanViews}, args...;
                               kwargs...)
    return views
end
function black_litterman_views(eqn::Union{<:AbstractString, Expr,
                                          <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                               sets::AssetSets; datatype::DataType = Float64,
                               strict::Bool = false)
    lcs = parse_equation(eqn; ops1 = ("==",), ops2 = (:call, :(==)), datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets, true)
    return get_black_litterman_views(lcs, sets; datatype = datatype, strict = strict)
end
function black_litterman_views(lcs::LinearConstraintEstimator, sets::AssetSets;
                               datatype::DataType = Float64, strict::Bool = false)
    return black_litterman_views(lcs.val, sets; datatype = datatype, strict = strict)
end
"""
    assert_bl_views_conf(::Nothing, args...)
    assert_bl_views_conf(views_conf::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                         views::Union{<:AbstractString, Expr,
                                      <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                      <:LinearConstraintEstimator, <:BlackLittermanViews})

Validate Black-Litterman view confidence specification.

`assert_bl_views_conf` checks that the view confidence parameter(s) provided for Black-Litterman prior construction are valid. It supports scalar and vector confidence values, and works with views specified as equations, constraint estimators, or canonical views objects. The function enforces that confidence values are strictly between 0 and 1, and that the number of confidence values matches the number of views when a vector is provided.

# Arguments

  - `views_conf`: Scalar or vector of confidence values.
  - `views`: Black-Litterman views, which may be equations.

# Returns

  - `nothing`: Returns nothing if validation passes; throws an error otherwise.

# Validation

  - `views_conf`:

      + `::Nothing`, no-op.
      + `::Real`, `0 < views_conf < 1`.
      + `::AbstractVector{<:Real}`, `all(x -> 0 < x < 1, views_conf)`, and must have the same length as the number of views.

  - `views`:

      + `::Union{<:AbstractString, Expr}`, `length(views_conf) == 1`.
      + `::AbstractVector{<:Union{<:AbstractString, Expr}}`, `length(views_conf) == length(views)`.
      + `::LinearConstraintEstimator`, calls `assert_bl_views_conf(views_conf, views.val)`.
      + `::BlackLittermanViews`, `length(views_conf) == length(views.Q)`.

# Related

  - [`BlackLittermanViews`](@ref)
"""
function assert_bl_views_conf(::Nothing, args...)
    return nothing
end
function assert_bl_views_conf(views_conf::Real,
                              ::Union{<:AbstractString, Expr,
                                      <:AbstractVector{<:Union{<:AbstractString, Expr}}})
    @argcheck(zero(views_conf) < views_conf < one(views_conf))
    return nothing
end
function assert_bl_views_conf(views_conf::AbstractVector{<:Real},
                              val::Union{<:AbstractString, Expr,
                                         <:AbstractVector{<:Union{<:AbstractString, Expr}}})
    if isa(val, AbstractVector)
        @argcheck(length(val) == length(views_conf))
    else
        @argcheck(length(views_conf) == 1)
    end
    @argcheck(all(x -> zero(x) < x < one(x), views_conf))
    return nothing
end
function assert_bl_views_conf(views_conf::Union{<:Real, <:AbstractVector{<:Real}},
                              views::LinearConstraintEstimator)
    return assert_bl_views_conf(views_conf, views.val)
end
function assert_bl_views_conf(views_conf::Union{<:Real, <:AbstractVector{<:Real}},
                              views::BlackLittermanViews)
    return @argcheck(length(views_conf) == length(views.Q))
end

export black_litterman_views, BlackLittermanViews
