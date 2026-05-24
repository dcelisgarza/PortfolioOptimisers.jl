"""
$(DocStringExtensions.TYPEDEF)

Container for Black-Litterman investor views in canonical matrix form.

`BlackLittermanViews` stores the views matrix `P` and the expected returns vector `Q` for use in Black-Litterman prior construction and related portfolio optimisation routines. The matrix `P` encodes the linear relationships between assets for each view, while `Q` specifies the expected value for each view.

# Summary Statistics

```math
\\mathbf{P}\\,\\boldsymbol{\\mu} = \\boldsymbol{q}, \\qquad \\boldsymbol{\\Omega} = \\mathrm{diag}(\\tau\\,\\mathbf{P}\\boldsymbol{\\Sigma}\\mathbf{P}^{\\intercal})
```

where ``\\mathbf{P}`` is the ``K \\times N`` views matrix, ``\\boldsymbol{q}`` the ``K``-vector of view expected returns, ``\\boldsymbol{\\mu}`` the prior mean, and ``\\boldsymbol{\\Omega}`` the view uncertainty matrix.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BlackLittermanViews(;
        P::MatNum,
        Q::VecNum,
        excl::Option{<:VecInt} = nothing
    ) -> BlackLittermanViews

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(P)` and `!isempty(Q)`.
  - `size(P, 1) == length(Q)`.
  - If `excl` is provided, `!isempty(excl)` and `length(excl) <= length(Q)`.

# Examples

```jldoctest
julia> BlackLittermanViews(; P = [1 2 3 4; 5 6 7 8], Q = [9; 10])
BlackLittermanViews
     P ┼ 2×4 Matrix{Int64}
     Q ┼ Vector{Int64}: [9, 10]
  excl ┴ nothing
```

# Related

  - [`black_litterman_views`](@ref)
"""
@concrete struct BlackLittermanViews <: AbstractResult
    "$(field_dict[:P])"
    P
    "$(field_dict[:Q])"
    Q
    "$(field_dict[:excl])"
    excl
    function BlackLittermanViews(P::MatNum, Q::VecNum, excl::Option{<:VecInt})
        @argcheck(!isempty(P))
        @argcheck(!isempty(Q))
        @argcheck(size(P, 1) == length(Q))
        if !isnothing(excl)
            @argcheck(!isempty(excl))
            @argcheck(length(excl) <= length(Q))
        end
        return new{typeof(P), typeof(Q), typeof(excl)}(P, Q, excl)
    end
end
function BlackLittermanViews(; P::MatNum, Q::VecNum,
                             excl::Option{<:VecInt} = nothing)::BlackLittermanViews
    return BlackLittermanViews(P, Q, excl)
end
"""
    const Lc_BLV = Union{<:LinearConstraintEstimator, <:BlackLittermanViews}

Alias for a union of linear constraint estimator and Black-Litterman views types.

# Related

  - [`LinearConstraintEstimator`](@ref)
  - [`BlackLittermanViews`](@ref)
"""
const Lc_BLV = Union{<:LinearConstraintEstimator, <:BlackLittermanViews}
"""
    get_black_litterman_views(lcs::PR_VecPR,
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

  - `blv::BlackLittermanViews`: An object containing the assembled views matrix `P` and expected returns vector `Q`, or `nothing` if no views are present.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx", dict = Dict("nx" => ["A", "B", "C"]));

julia> lcs = parse_equation(["A + B == 0.05", "C == 0.02"]);

julia> PortfolioOptimisers.get_black_litterman_views(lcs, sets)
BlackLittermanViews
     P ┼ 2×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
     Q ┼ Vector{Float64}: [0.05, 0.02]
  excl ┴ nothing
```

# Related

  - [`BlackLittermanViews`](@ref)
  - [`parse_equation`](@ref)
  - [`AssetSets`](@ref)
"""
function get_black_litterman_views(lcs::PR_VecPR, sets::AssetSets;
                                   datatype::DataType = Float64, strict::Bool = false)
    if isa(lcs, AbstractVector)
        @argcheck(!isempty(lcs))
    end
    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)
    excl = Vector{Int}(undef, 0)
    nx = sets.dict[sets.key]
    At = Vector{datatype}(undef, length(nx))
    for (i, lc) in enumerate(lcs)
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
        if !any(!iszero, At)
            msg = "At least one entry in At must be non-zero:\nlc => $(lc)\nany(!iszero, At) => $(any(!iszero, At))"
            if strict
                throw(ArgumentError(msg))
            else
                @warn(msg)
                push!(excl, i)
                continue
            end
        end
        append!(P, At)
        append!(Q, lc.rhs)
    end
    return if !isempty(P)
        P = transpose(reshape(P, length(nx), :))
        BlackLittermanViews(; P = P, Q = Q, excl = isempty(excl) ? nothing : excl)
    else
        nothing
    end
end
"""
    black_litterman_views(views::Option{<:BlackLittermanViews}, args...; kwargs...)
    black_litterman_views(views::EqnType,
                          sets::AssetSets; datatype::DataType = Float64, strict::Bool = false)
    black_litterman_views(views::LinearConstraintEstimator, sets::AssetSets;
                          datatype::DataType = Float64, strict::Bool = false)

Unified interface for constructing or passing through Black-Litterman investor views.

`black_litterman_views` provides a composable API for handling Black-Litterman views in portfolio optimisation workflows. It supports passing through an existing [`BlackLittermanViews`](@ref) object, constructing views from equations or constraint estimators, and converting parsed view equations into canonical matrix form.

# Arguments

  - `views`:

      + `nothing` or [`BlackLittermanViews`](@ref): it is returned unchanged.
      + `EqnType`: The view(s) are parsed, groups are replaced by their constituent assets using `sets`, calls [`get_black_litterman_views`](@ref) and constructs a [`BlackLittermanViews`](@ref) object is constructed.
      + [`LinearConstraintEstimator`](@ref): calls the method described above using the `val` field of the estimator.

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
     P ┼ 2×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
     Q ┼ Vector{Float64}: [0.05, 0.02]
  excl ┴ nothing

julia> lce = LinearConstraintEstimator(; val = ["A == 0.03", "B + C == 0.04"]);

julia> black_litterman_views(lce, sets)
BlackLittermanViews
     P ┼ 2×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
     Q ┼ Vector{Float64}: [0.03, 0.04]
  excl ┴ nothing
```

# Related

  - [`BlackLittermanViews`](@ref)
  - [`get_black_litterman_views`](@ref)
  - [`parse_equation`](@ref)
  - [`AssetSets`](@ref)
  - [`LinearConstraintEstimator`](@ref)
"""
function black_litterman_views(views::Option{<:BlackLittermanViews}, args...; kwargs...)
    return views
end
function black_litterman_views(eqn::EqnType, sets::AssetSets; datatype::DataType = Float64,
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
    assert_bl_views_conf(views_conf::Option{<:Num_VecNum},
                         views::Union{<:EqnType, <:LinearConstraintEstimator, <:BlackLittermanViews})

Validate Black-Litterman view confidence specification.

`assert_bl_views_conf` checks that the view confidence parameter(s) provided for Black-Litterman prior construction are valid. It supports scalar and vector confidence values, and works with views specified as equations, constraint estimators, or canonical views objects. The function enforces that confidence values are strictly between 0 and 1, and that the number of confidence values matches the number of views when a vector is not `nothing`.

# Arguments

  - `views_conf`: Scalar or vector of confidence values.
  - `views`: Black-Litterman views, which may be equations.

# Returns

  - `nothing`.

# Validation

  - `views_conf`:

      + `::Nothing`, no-op.
      + `::Number`, `0 < views_conf < 1`.
      + `::VecNum`, `all(x -> 0 < x < 1, views_conf)`, and must have the same length as the number of views.

  - `views`:

      + `::Str_Expr`, `length(views_conf) == 1`.
      + `::VecStr_Expr`, `length(views_conf) == length(views)`.
      + `::LinearConstraintEstimator`, calls `assert_bl_views_conf(views_conf, views.val)`.
      + `::BlackLittermanViews`, `length(views_conf) == length(views.Q)`.

# Related

  - [`BlackLittermanViews`](@ref)
"""
function assert_bl_views_conf(::Nothing, args...)::Nothing
    return nothing
end
function assert_bl_views_conf(views_conf::Number, ::EqnType)::Nothing
    @argcheck(zero(views_conf) < views_conf < one(views_conf))
    return nothing
end
function assert_bl_views_conf(views_conf::VecNum, val::EqnType)::Nothing
    if isa(val, AbstractVector)
        @argcheck(length(val) == length(views_conf))
    else
        @argcheck(length(views_conf) == 1)
    end
    @argcheck(all(x -> zero(x) < x < one(x), views_conf))
    return nothing
end
function assert_bl_views_conf(views_conf::Num_VecNum,
                              views::LinearConstraintEstimator)::Nothing
    return assert_bl_views_conf(views_conf, views.val)
end
function assert_bl_views_conf(views_conf::Num_VecNum, views::BlackLittermanViews)::Nothing
    return @argcheck(length(views_conf) == length(views.Q))
end

export black_litterman_views, BlackLittermanViews
