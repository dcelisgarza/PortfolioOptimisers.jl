"""
$(DocStringExtensions.TYPEDEF)

Container for Black-Litterman investor views in canonical matrix form.

`BlackLittermanViews` stores the views matrix `P` and the expected returns vector `Q` for use in Black-Litterman prior construction and related portfolio optimisation routines. The matrix `P` encodes the linear relationships between assets for each view, while `Q` specifies the expected value for each view.

# Mathematical definition

This type stores the pair ``(\\mathbf{P}, \\boldsymbol{q})`` of the view creation model:

```math
\\begin{align}
\\mathbf{P}\\,\\boldsymbol{\\mu}_{e} &= \\boldsymbol{q} + \\boldsymbol{\\nu}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{P}``: ``K \\times N`` views matrix (each row encodes one view).
  - ``\\boldsymbol{\\mu}_{e}``: ``N \\times 1`` prior expected excess returns vector.
  - ``\\boldsymbol{q}``: ``K \\times 1`` vector of view expected returns.
  - ``\\boldsymbol{\\nu}``: ``K \\times 1`` estimation error of the views.

The view uncertainty matrix ``\\boldsymbol{\\Omega}`` is **not** stored here. It is derived from ``\\mathbf{P}`` and the covariance of the distribution the views update, by [`calc_omega`](@ref) and [`bl_preroll`](@ref).

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
  - [`calc_omega`](@ref)

# References

  - $(ref_dict[:black1992])
  - $(ref_dict[:cajas2025]) Section 5.1.3, Equations 5.5 and 5.7.
"""
@concrete struct BlackLittermanViews <: AbstractResult
    """
    $(field_dict[:P])
    """
    P
    """
    $(field_dict[:Q])
    """
    Q
    """
    $(field_dict[:excl])
    """
    excl
    function BlackLittermanViews(P::MatNum, Q::VecNum, excl::Option{<:VecInt})
        @argcheck(!isempty(P), IsEmptyError("P cannot be empty"))
        @argcheck(!isempty(Q), IsEmptyError("Q cannot be empty"))
        @argcheck(size(P, 1) == length(Q),
                  DimensionMismatch("size(P, 1) = $(size(P, 1)) must match length(Q) = $(length(Q))"))
        if !isnothing(excl)
            @argcheck(!isempty(excl), IsEmptyError("excl cannot be empty"))
            @argcheck(length(excl) <= length(Q),
                      DimensionMismatch("length(excl) = $(length(excl)) must be <= length(Q) = $(length(Q))"))
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
    get_black_litterman_views(lcs::PR_VecPR, sets::UniverseSets,
                              key::Option{<:AbstractString} = nothing;
                              datatype::DataType = Float64, strict::Bool = false)

Convert parsed Black-Litterman view equations into a `BlackLittermanViews` object.

`get_black_litterman_views` takes one or more [`ParsingResult`](@ref) objects (as produced by [`parse_equation`](@ref)), expands variable names using the provided [`UniverseSets`](@ref), and assembles the canonical views matrix `P` and expected returns vector `Q` for Black-Litterman prior construction. The result is a [`BlackLittermanViews`](@ref) object suitable for use in portfolio optimisation routines.

`key` selects **which** universe the view names resolve against, exactly as it does for [`get_linear_constraints`](@ref); `nothing` means `sets.xkey`. A view is never re-based — the estimator that owns it decides which distribution it lands on, and passes the matching key — so the assembled `P` is one row per view over `length(sets.dict[key])` columns, and the message an unresolved name produces names the axis via [`universe_axis`](@ref).

A view that resolves no name at all is **dropped, not refused**. Its index joins `excl`, the remaining rows keep their order, and [`remove_excl_views`](@ref) drops the matching entry of a per-view confidence vector. When every view is dropped there is no row left and the return is `nothing`; [`bl_preroll`](@ref) is the caller that turns that into a named error.

# Algorithm

 1. When `lcs` is a vector, check that it is not empty.
 2. Resolve the universe key `k`, which is `key` when it is given and `sets.xkey` otherwise. Read the universe `nx = sets.dict[k]`, and read its axis with [`universe_axis`](@ref), giving `axis` for the diagnostic messages.
 3. For each parsed view `lc`, in the order the caller wrote them, run steps 4 to 6 over the row accumulator `At`, which holds `length(nx)` coefficients and starts at zero.
 4. For each variable-coefficient pair `(v, c)` of `lc`, build the indicator `Ai = (nx .== v)`. When `Ai` selects no entry, report the unresolved name through [`strict_diagnostic`](@ref) and drop that term. Otherwise add `Ai * c` to `At`.
 5. When `At` is still all zeros, no name of this view resolved. Report the empty row through [`strict_diagnostic`](@ref), push the view's index `i` onto `excl`, and go on to the next view without writing a row.
 6. Append `At` to `P` and `lc.rhs` to `Q`.
 7. When `P` holds at least one row, reshape it to `length(nx)` rows and transpose it, so `P` is one view per row. Return a [`BlackLittermanViews`](@ref) over `P`, `Q` and `excl`, where an empty `excl` is passed as `nothing`. When `P` is empty, return `nothing`.

# Arguments

  - `lcs`: A single [`ParsingResult`](@ref) or a vector of such objects, representing parsed Black-Litterman view equations.
  - `sets`: A [`UniverseSets`](@ref) object specifying the universes and groupings.
  - $(arg_dict[:ekey])
  - `datatype`: Numeric type for coefficients and expected returns.
  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning.

# Validation

  - When `lcs` is a vector, `!isempty(lcs)`.
  - A name that matches no entry of `sets.dict[key]` raises through [`strict_diagnostic`](@ref) when `strict` is `true`, and warns otherwise. The message names the axis.
  - A view whose names all fail to resolve raises the same way, and is dropped when `strict` is `false`.
  - The assembled pair passes the [`BlackLittermanViews`](@ref) constructor's own checks.

# Returns

  - `blv::Option{<:BlackLittermanViews}`: The assembled views matrix `P`, one row per view over `length(sets.dict[key])` columns, the expected returns vector `Q`, and the indices `excl` of the views that resolved no name. `nothing` when no view resolved.

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\", dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> lcs = parse_equation([\"A + B == 0.05\", \"C == 0.02\"]);

julia> PortfolioOptimisers.get_black_litterman_views(lcs, sets)
BlackLittermanViews
     P ┼ 2×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
     Q ┼ Vector{Float64}: [0.05, 0.02]
  excl ┴ nothing
```

# Related

  - [`BlackLittermanViews`](@ref)
  - [`parse_equation`](@ref)
  - [`UniverseSets`](@ref)
  - [`strict_diagnostic`](@ref) Decides whether an unresolved name raises or warns.
  - [`universe_axis`](@ref)
  - [`remove_excl_views`](@ref) Drops the confidences of the views this function excluded.
  - [`bl_preroll`](@ref) The caller that refuses a `nothing` answer with a named error.
"""
function get_black_litterman_views(lcs::PR_VecPR, sets::UniverseSets,
                                   key::Option{<:AbstractString} = nothing;
                                   datatype::DataType = Float64, strict::Bool = false)
    if isa(lcs, AbstractVector)
        @argcheck(!isempty(lcs), IsEmptyError("lcs cannot be empty"))
    end
    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)
    excl = Vector{Int}(undef, 0)
    k = ifelse(isnothing(key), sets.xkey, key)
    nx = sets.dict[k]
    axis = universe_axis(sets, k)
    At = Vector{datatype}(undef, length(nx))
    for (i, lc) in enumerate(lcs)
        fill!(At, zero(eltype(At)))
        for (v, c) in zip(lc.vars, lc.coef)
            Ai = (nx .== v)
            if !any(isone, Ai)
                msg = unknown_variable_msg(v, nx, k; axis = axis)
                strict_diagnostic(msg, strict)
                continue
            end
            At += Ai * c
        end
        if !any(!iszero, At)
            msg = empty_row_msg(lc.eqn, nx, k; noun = "view", axis = axis)
            strict_diagnostic(msg, strict)
            push!(excl, i)
            continue
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
    black_litterman_views(views::EqnType, sets::UniverseSets,
                          key::Option{<:AbstractString} = nothing;
                          datatype::DataType = Float64, strict::Bool = false)
    black_litterman_views(views::LinearConstraintEstimator, sets::UniverseSets,
                          key::Option{<:AbstractString} = nothing;
                          datatype::DataType = Float64, strict::Bool = false)

Unified interface for constructing or passing through Black-Litterman investor views.

`black_litterman_views` provides a composable API for handling Black-Litterman views in portfolio optimisation workflows. It supports passing through an existing [`BlackLittermanViews`](@ref) object, constructing views from equations or constraint estimators, and converting parsed view equations into canonical matrix form.

The two routes agree. A [`LinearConstraintEstimator`](@ref) assembled here and the [`BlackLittermanViews`](@ref) result of that same assembly, passed back in, give the same `P` and the same `Q`, so a caller who precomputes the pair loses nothing but the name resolution.

# Algorithm

 1. When `views` is `nothing` or a [`BlackLittermanViews`](@ref), return it unchanged. The pair was assembled against whatever universe the caller held, so `sets`, `key`, `datatype` and `strict` are all ignored.
 2. When `views` is a [`LinearConstraintEstimator`](@ref), pick the key: `views.key` when the estimator carries one, and the `key` argument otherwise. Call step 3 on `views.val` with that key.
 3. When `views` is an `EqnType`, parse it with [`parse_equation`](@ref) under the `==` operator alone, giving the parsed views `lcs`. A Black-Litterman view is an equality, so no inequality operator is admitted.
 4. Expand every group name in `lcs` into its member assets with [`replace_group_by_assets`](@ref), under `sets`.
 5. Assemble the canonical pair from `lcs` with [`get_black_litterman_views`](@ref), under the key of step 2, and return what it gives.

# Arguments

  - `views`:

      + `nothing` or [`BlackLittermanViews`](@ref): it is returned unchanged, `key` and all — a precomputed `P` was assembled against whatever universe the caller had, and nothing here can re-check it.
      + `EqnType`: The view(s) are parsed, groups are replaced by their constituent members using `sets`, calls [`get_black_litterman_views`](@ref) and constructs a [`BlackLittermanViews`](@ref) object is constructed.
      + [`LinearConstraintEstimator`](@ref): calls the method described above using the `val` field of the estimator. Its own `key` **wins** over the one the estimator passes, which is the same precedence [`rebase_linear_constraints`](@ref) uses: the argument is the axis the caller is written against, the field is the user overriding it.

  - `sets`: A [`UniverseSets`](@ref) object specifying the universes and groupings.

  - $(arg_dict[:ekey])

  - `datatype`: Numeric type for coefficients and expected returns.

  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning.

# Returns

  - `blv::BlackLittermanViews`: An object containing the assembled views matrix `P` and expected returns vector `Q`, or `nothing` if no views are present.

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\", dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> black_litterman_views([\"A + B == 0.05\", \"C == 0.02\"], sets)
BlackLittermanViews
     P ┼ 2×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
     Q ┼ Vector{Float64}: [0.05, 0.02]
  excl ┴ nothing

julia> lce = LinearConstraintEstimator(; val = [\"A == 0.03\", \"B + C == 0.04\"]);

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
  - [`replace_group_by_assets`](@ref) Expands a group name into its member assets.
  - [`UniverseSets`](@ref)
  - [`LinearConstraintEstimator`](@ref)
  - [`Lc_BLV`](@ref) The union of the two shapes this function admits.
"""
function black_litterman_views(views::Option{<:BlackLittermanViews}, args...; kwargs...)
    return views
end
function black_litterman_views(eqn::EqnType, sets::UniverseSets,
                               key::Option{<:AbstractString} = nothing;
                               datatype::DataType = Float64, strict::Bool = false)
    lcs = parse_equation(eqn; ops1 = ("==",), ops2 = (:call, :(==)), datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets, true)
    return get_black_litterman_views(lcs, sets, key; datatype = datatype, strict = strict)
end
function black_litterman_views(lcs::LinearConstraintEstimator, sets::UniverseSets,
                               key::Option{<:AbstractString} = nothing;
                               datatype::DataType = Float64, strict::Bool = false)
    return black_litterman_views(lcs.val, sets, ifelse(isnothing(lcs.key), key, lcs.key);
                                 datatype = datatype, strict = strict)
end
"""
    assert_bl_views_conf(::Nothing, args...)
    assert_bl_views_conf(views_conf::Number, ::EqnType)
    assert_bl_views_conf(views_conf::VecNum, val::EqnType)
    assert_bl_views_conf(views_conf::Num_VecNum, views::LinearConstraintEstimator)
    assert_bl_views_conf(views_conf::Num_VecNum, views::BlackLittermanViews)

Validate Black-Litterman view confidence specification.

`assert_bl_views_conf` checks that the view confidence parameter(s) provided for Black-Litterman prior construction are valid. It supports scalar and vector confidence values, and works with views specified as equations, constraint estimators, or canonical views objects. The function enforces that confidence values are strictly between 0 and 1, and that the number of confidence values matches the number of views when a vector is given.

A **scalar** confidence is one confidence for every view, whatever the number of views. Only a **vector** states one confidence per view, so only a vector is counted against the views. Both shapes reach the same [`calc_omega`](@ref) answer: a scalar ``v`` and the constant vector of ``v`` give the same ``\\mathbf{\\Omega}``.

The unit-interval bound is load-bearing rather than cosmetic. [`calc_omega`](@ref) maps a confidence ``v`` to the scale ``1/v - 1``, which is negative for every ``v > 1`` and for every ``v < 0``. A view uncertainty matrix with a negative diagonal entry is not a covariance, and the estimator returns an answer built from it rather than raising.

# Arguments

  - `views_conf`: Scalar or vector of confidence values.
  - `views`: Black-Litterman views, which may be equations.
  - `val`: The equations of a [`LinearConstraintEstimator`](@ref), unwrapped.

# Validation

Each method selects one shape of `views_conf` and one shape of `views`, and refuses what that pair does not admit.

  - `(::Nothing, args...)`: no confidence was given, so nothing is checked.
  - `(::Number, ::EqnType)`: `0 < views_conf < 1`, through [`assert_unit_interval`](@ref). The count is not checked, because one scalar covers any number of equations.
  - `(::VecNum, ::EqnType)`: when `val` is a vector of equations, `length(val) == length(views_conf)`; when `val` is one equation, `length(views_conf) == 1`. Then `all(x -> 0 < x < 1, views_conf)`.
  - `(::Num_VecNum, ::LinearConstraintEstimator)`: selects nothing itself, and forwards `views.val` to the two methods above.
  - `(::Num_VecNum, ::BlackLittermanViews)`: when `views_conf` is a vector, `length(views_conf) == length(views.Q)`. Then `all(x -> 0 < x < 1, views_conf)`. This is the only site that sees the confidences of a precomputed pair, because such a pair resolves no name.

# Returns

  - `nothing`.

# Related

  - [`BlackLittermanViews`](@ref)
  - [`calc_omega`](@ref)
  - [`assert_unit_interval`](@ref)
"""
function assert_bl_views_conf(::Nothing, args...)::Nothing
    return nothing
end
function assert_bl_views_conf(views_conf::Number, ::EqnType)::Nothing
    assert_unit_interval(views_conf, :views_conf)
    return nothing
end
function assert_bl_views_conf(views_conf::VecNum, val::EqnType)::Nothing
    if isa(val, AbstractVector)
        @argcheck(length(val) == length(views_conf),
                  DimensionMismatch("length(val) = $(length(val)) must match length(views_conf) = $(length(views_conf))"))
    else
        @argcheck(length(views_conf) == 1,
                  DimensionMismatch("views_conf must have length 1 for a single view, got $(length(views_conf))"))
    end
    @argcheck(all(x -> zero(x) < x < one(x), views_conf),
              DomainError("all views_conf values must be in (0, 1), got $views_conf"))
    return nothing
end
function assert_bl_views_conf(views_conf::Num_VecNum,
                              views::LinearConstraintEstimator)::Nothing
    return assert_bl_views_conf(views_conf, views.val)
end
function assert_bl_views_conf(views_conf::Num_VecNum, views::BlackLittermanViews)::Nothing
    # A scalar broadcasts over every view, which is what `calc_omega`'s scalar branch is for,
    # and the equation-shaped routes above accept it over any number of views. Only a vector
    # states one confidence per view, so only a vector is length-checked. Checking a scalar
    # here refused `views_conf = 0.4` over two precomputed views while accepting `[0.4, 0.4]`,
    # which reaches `calc_omega` as the same number.
    if isa(views_conf, AbstractVector)
        @argcheck(length(views_conf) == length(views.Q),
                  DimensionMismatch("length(views_conf) ($(length(views_conf))) must match length(views.Q) ($(length(views.Q)))"))
    end
    # Precomputed views resolve no names, so this is the only site that sees their confidences.
    # Without the bound a confidence outside `(0, 1)` reaches `calc_omega`, whose `1/v - 1` scale
    # is then negative, and the estimator answers from a view uncertainty matrix that is not a
    # covariance. The equation-shaped view routes above already refuse the same input.
    @argcheck(all(x -> zero(x) < x < one(x), views_conf),
              DomainError("all views_conf values must be in (0, 1), got $views_conf"))
    return nothing
end

export black_litterman_views, BlackLittermanViews
