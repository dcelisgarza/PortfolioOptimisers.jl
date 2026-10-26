"""
$(DocStringExtensions.TYPEDEF)

Black-Litterman views in matrix form: the views matrix `P` and the view returns `Q`.

Each row of `P` holds the coefficients of one view over the names of one axis, assets or factors. The same entry of `Q` holds the expected return of that view. [`black_litterman_views`](@ref) assembles the pair from view equations. A caller can also construct the pair, and a Black-Litterman prior estimator then uses it as it is.

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

[black1991](@citet) write the estimation error of the views as ``\\boldsymbol{\\varepsilon}``. The name "view creation model" and the symbol ``\\boldsymbol{\\nu}`` come from [cajas2025](@citet).

This type does **not** hold the view uncertainty matrix ``\\boldsymbol{\\Omega}``. [`calc_omega`](@ref) and [`bl_preroll`](@ref) make it from ``\\mathbf{P}`` and from the covariance of the distribution that the views update.

`excl` holds the indices of the views that the caller wrote and that `P` and `Q` do not hold. The indices count over all the views that the caller wrote, so `excl` can be longer than `Q`. [`remove_excl_views`](@ref) reads `excl` to remove the confidences of those views.

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

  - `!isempty(P)` and `!isempty(Q)`. Raises an [`IsEmptyError`](@ref).
  - `size(P, 1) == length(Q)`. Raises a `DimensionMismatch`.
  - If `excl` is not `nothing`:
      + `!isempty(excl)`. Raises an [`IsEmptyError`](@ref).
      + `allunique(excl)`. Raises an `ArgumentError`.
      + Each entry of `excl` is in `1:(length(Q) + length(excl))`. Raises a `DomainError`.

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

  - $(ref_dict[:black1991]) Appendix, item 7. $(ref_dict[:no_original_source])
  - $(ref_dict[:cajas2025]) Section 5.1.3, Equation 5.5.
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
            @argcheck(allunique(excl),
                      ArgumentError("excl must not repeat an index, got $excl"))
            # `excl` indexes the views the caller wrote, the kept rows and the dropped ones, so
            # it can be longer than `Q`. `remove_excl_views` reads it over that same count.
            n = length(Q) + length(excl)
            @argcheck(all(i -> one(i) <= i <= n, excl),
                      DomainError(excl,
                                  "`excl` indexes the views that the caller wrote, `length(Q) + length(excl) = $n`, so every entry must lie in `1:$n`."))
        end
        return new{typeof(P), typeof(Q), typeof(excl)}(P, Q, excl)
    end
end
function BlackLittermanViews(; P::MatNum, Q::VecNum,
                             excl::Option{<:VecInt} = nothing)::BlackLittermanViews
    return BlackLittermanViews(P, Q, excl)
end
"""
    const Lc_BLV = Union{LinearConstraintEstimator, BlackLittermanViews}

Alias for the union of the two shapes of Black-Litterman views: a [`LinearConstraintEstimator`](@ref) of view equations, and a precomputed [`BlackLittermanViews`](@ref).

# Related

  - [`LinearConstraintEstimator`](@ref)
  - [`BlackLittermanViews`](@ref)
"""
const Lc_BLV = Union{LinearConstraintEstimator, BlackLittermanViews}
"""
    get_black_litterman_views(lcs::PR_VecPR, sets::UniverseSets,
                              key::Option{<:AbstractString} = nothing;
                              datatype::DataType = Float64, strict::Bool = false,
                              ledger::Option{<:AbstractVector} = nothing)

Assemble parsed Black-Litterman view equations into a [`BlackLittermanViews`](@ref).

The function reads one or more [`ParsingResult`](@ref) objects from [`parse_equation`](@ref). It resolves each name against one universe of a [`UniverseSets`](@ref), and it writes one row of `P` and one entry of `Q` for each view that it keeps.

`key` selects the universe that the names resolve against, as it does for [`get_linear_constraints`](@ref). If `key` is `nothing`, the universe is `sets.dict[sets.xkey]`. The function does not re-base a view. The estimator that owns the views selects the distribution that they update, and it passes the matching key. `P` has one row for each kept view and `length(sets.dict[key])` columns. The message for a name that does not resolve names the axis, which [`universe_axis`](@ref) reads.

The function **drops** a view that it cannot assemble, and it does not refuse it. The index of the view goes into `excl`, and the other rows keep their order. The indices count over all the views that the caller wrote, so `excl` can be longer than `Q`. [`remove_excl_views`](@ref) removes the matching entries of a confidence vector. When the function drops every view, no row is left and it returns `nothing`. The caller, [`bl_preroll`](@ref), decides what that result means.

**The function drops a full row.** A view is one statement about several names, with one right-hand side. Thus a name that does not resolve removes the full row, and not only its own term. Without `c`, the view `a + c == 0.05` becomes `a == 0.05`, which is a different and stronger statement than the caller wrote.

The cause of the failure decides only whether the function reports the drop. [`counterpart_axis_names`](@ref) reads the **counterpart axis**. In practice it is the Non-Investable Axis, which [`investable_views`](@ref) makes for the caller that derives the Investable Mask. A name on the counterpart axis was correct over the universe that the caller received, and the data removed it. The function drops its row with no report, under both settings of `strict`. The caller that derives the Investable Mask reports the departure once, through [`announce_non_investable`](@ref). A name on neither axis is a typo, and the function reports it through [`strict_diagnostic`](@ref).

# Algorithm

 1. When `lcs` is a vector, check that it is not empty.
 2. Find the universe key `k`, which is `key` when it is not `nothing` and `sets.xkey` otherwise. Read the universe `nx = sets.dict[k]`. Read its axis `axis` for the messages with [`universe_axis`](@ref), and read the counterpart axis with [`counterpart_axis_names`](@ref).
 3. For each parsed view `lc`, in the order that the caller wrote them, set each of the `length(nx)` coefficients of the row accumulator `At` to zero, mark the view as kept, and do steps 4 to 7.
 4. For each pair `(v, c)` of a variable and its coefficient in `lc`, build the indicator `Ai = (nx .== v)`. When `Ai` selects no entry, mark the row as dropped. If `v` is not on the counterpart axis, report it through [`strict_diagnostic`](@ref). If it is, record the row in `ledger` with [`record_non_investable_drop!`](@ref), once for each row. Otherwise add `Ai * c` to `At`. The function reads each name of the row, so it reports both typos of a row that holds two.
 5. When the row is marked as dropped, push the index `i` of the view onto `excl`, and go to the next view.
 6. When each entry of `At` is zero, each name of the view resolved and the coefficients cancel. Report the row through [`strict_diagnostic`](@ref), push `i` onto `excl`, and go to the next view. Do not write a row.
 7. Append `At` to `P` and `lc.rhs` to `Q`.
 8. When `P` holds at least one row, reshape it to `length(nx)` rows and transpose it, so each row of `P` is one view. Return a [`BlackLittermanViews`](@ref) of `P`, `Q` and `excl`, with `nothing` in place of an empty `excl`. When `P` is empty, return `nothing`.

# Arguments

  - `lcs`: One [`ParsingResult`](@ref) or a vector of them, the parsed view equations.
  - `sets`: The [`UniverseSets`](@ref) that the names resolve against.
  - $(arg_dict[:ekey])
  - `datatype`: The numeric type of the coefficients and of the view returns.
  - `strict`: If `true`, a name that resolves against neither axis raises an `ArgumentError`, and a row whose coefficients cancel raises an [`IsZeroError`](@ref). If `false`, both warn.
  - $(arg_dict[:ledger]) The function records in it each row that it drops for a name on the counterpart axis, through [`record_non_investable_drop!`](@ref).

# Validation

  - When `lcs` is a vector, `!isempty(lcs)`.
  - A name that matches no entry of `sets.dict[key]` and no name of the counterpart axis raises an error through [`strict_diagnostic`](@ref) when `strict` is `true`, and warns otherwise. The message names the axis. The function drops the row in both cases.
  - A name on the counterpart axis drops its row with no report, under both settings of `strict`.
  - A view whose names all resolve and whose coefficients cancel raises an [`IsZeroError`](@ref) when `strict` is `true`, and warns otherwise. The function drops the row in both cases.
  - The assembled pair passes the checks of the [`BlackLittermanViews`](@ref) constructor.

# Returns

  - `blv::Option{<:BlackLittermanViews}`: The assembled views: `P`, with one row for each kept view and `length(sets.dict[key])` columns, the view returns `Q`, and the indices `excl` of the dropped views, or `nothing` in `excl` when the function drops no view. The result is `nothing` when the function drops every view.

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
  - [`counterpart_axis_names`](@ref)
  - [`record_non_investable_drop!`](@ref)
  - [`investable_views`](@ref)
  - [`remove_excl_views`](@ref) Drops the confidences of the views this function excluded.
  - [`bl_preroll`](@ref) The caller that decides what a `nothing` answer means.
"""
function get_black_litterman_views(lcs::PR_VecPR, sets::UniverseSets,
                                   key::Option{<:AbstractString} = nothing;
                                   datatype::DataType = Float64, strict::Bool = false,
                                   ledger::Option{<:AbstractVector} = nothing)
    if isa(lcs, AbstractVector)
        @argcheck(!isempty(lcs), IsEmptyError("lcs cannot be empty"))
    end
    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)
    excl = Vector{Int}(undef, 0)
    k = ifelse(isnothing(key), sets.xkey, key)
    nx = sets.dict[k]
    axis = universe_axis(sets, k)
    other = counterpart_axis_names(sets, k)
    At = Vector{datatype}(undef, length(nx))
    for (i, lc) in enumerate(lcs)
        fill!(At, zero(eltype(At)))
        dropped = false
        for (v, c) in zip(lc.vars, lc.coef)
            Ai = (nx .== v)
            if !any(isone, Ai)
                # A view is a **row**: `a + c == 0.05` fitted without `c` asserts
                # `a == 0.05`, which is a different and stronger claim than the caller
                # wrote — and one the Black-Litterman update would then blend the prior
                # against. So the row goes whole, whatever the name's failure was. A name on
                # the counterpart axis goes in silence, because it was correct over the
                # universe the caller was handed and the data moved it; a name on neither
                # axis is still a typo and is still reported. See ADR 0125.
                if v ∉ other
                    msg = unknown_variable_msg(v, nx, k; axis = axis,
                                               consequence = "row dropped")
                    strict_diagnostic(msg, strict; id = @log_id(v, k))
                elseif !dropped
                    # Once per row, not once per departed name in it: the row is the unit
                    # that went, and `ni` already names every asset that left.
                    record_non_investable_drop!(ledger, "the view row `$(lc.eqn)`")
                end
                dropped = true
                continue
            end
            At += Ai * c
        end
        # `excl` carries the index either way, so `remove_excl_views` drops the confidence
        # of a row that went to a departure exactly as it drops one that resolved nothing.
        if dropped
            push!(excl, i)
            continue
        end
        if !any(!iszero, At)
            # Every name of the row resolved — one that did not took the row with it above
            # — so this row resolved and still summed to zero, by its own cancelling
            # coefficients or by holding no name at all. It is not a typo.
            msg = empty_row_msg(lc.eqn, nx, k; noun = "view", axis = axis)
            strict_diagnostic(msg, strict; id = @log_id(lc.eqn, k), exception = IsZeroError)
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
                          datatype::DataType = Float64, strict::Bool = false,
                          ledger::Option{<:AbstractVector} = nothing)
    black_litterman_views(views::LinearConstraintEstimator, sets::UniverseSets,
                          key::Option{<:AbstractString} = nothing;
                          datatype::DataType = Float64, strict::Bool = false,
                          ledger::Option{<:AbstractVector} = nothing)

Assemble Black-Litterman views into a [`BlackLittermanViews`](@ref), or return precomputed views unchanged.

The two routes give the same pair. When a caller passes back the [`BlackLittermanViews`](@ref) that this function assembled from a [`LinearConstraintEstimator`](@ref), it gets the same `P` and the same `Q`. A caller that precomputes the pair loses only the resolution of names.

# Algorithm

 1. When `views` is `nothing` or a [`BlackLittermanViews`](@ref), return it unchanged. The caller assembled the pair over its own universe, and this function cannot check it. The method ignores `sets`, `key`, `datatype`, `strict` and `ledger`.
 2. When `views` is a [`LinearConstraintEstimator`](@ref), select the key: `views.key` when it is not `nothing`, and the `key` argument otherwise. Do step 3 on `views.val` with that key.
 3. When `views` is an `EqnType`, parse it with [`parse_equation`](@ref) under the `==` operator alone. The result is the parsed views `lcs`. A Black-Litterman view is an equality, so the function accepts no inequality operator.
 4. Replace each group name in `lcs` by its member assets with [`replace_group_by_assets`](@ref), under `sets`. That function removes the departed members of a group *before* it divides the coefficient of the group, so the coefficient is divided among the members that remain.
 5. Assemble the pair from `lcs` with [`get_black_litterman_views`](@ref), under the key of step 2, and return its result.

# Arguments

  - `views`:

      + `nothing` or [`BlackLittermanViews`](@ref): The method returns it unchanged. The caller assembled a precomputed `P` over its own universe, and this function cannot check it.
      + `EqnType`: The method parses the equations, replaces each group by its member assets with `sets`, and assembles the pair with [`get_black_litterman_views`](@ref).
      + [`LinearConstraintEstimator`](@ref): The method calls the `EqnType` method on the `val` field of the estimator. When the `key` field of the estimator is not `nothing`, it takes precedence over the `key` argument. [`rebase_linear_constraints`](@ref) uses the same precedence: the argument is the axis that the caller uses, and the field is the override of the user.

  - `sets`: The [`UniverseSets`](@ref) that the names and the groups resolve against.

  - $(arg_dict[:ekey])

  - `datatype`: The numeric type of the coefficients and of the view returns.

  - `strict`: Passed to [`get_black_litterman_views`](@ref), which states what it raises.

  - $(arg_dict[:ledger]) The method passes it to [`replace_group_by_assets`](@ref) and to [`get_black_litterman_views`](@ref), so the ledger records a group that loses members and a row that the assembly drops.

# Returns

  - `blv::Option{<:BlackLittermanViews}`: The assembled views, the input when it is precomputed, or `nothing` when no view is left.

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
                               datatype::DataType = Float64, strict::Bool = false,
                               ledger::Option{<:AbstractVector} = nothing)
    lcs = parse_equation(eqn; ops1 = ("==",), ops2 = (:call, :(==)), datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets, true; ledger = ledger)
    return get_black_litterman_views(lcs, sets, key; datatype = datatype, strict = strict,
                                     ledger = ledger)
end
function black_litterman_views(lcs::LinearConstraintEstimator, sets::UniverseSets,
                               key::Option{<:AbstractString} = nothing;
                               datatype::DataType = Float64, strict::Bool = false,
                               ledger::Option{<:AbstractVector} = nothing)
    return black_litterman_views(lcs.val, sets, ifelse(isnothing(lcs.key), key, lcs.key);
                                 datatype = datatype, strict = strict, ledger = ledger)
end
"""
    assert_bl_views_conf(::Nothing, args...)
    assert_bl_views_conf(views_conf::Number, ::EqnType)
    assert_bl_views_conf(views_conf::VecNum, val::EqnType)
    assert_bl_views_conf(views_conf::Num_VecNum, views::LinearConstraintEstimator)
    assert_bl_views_conf(views_conf::Num_VecNum, views::BlackLittermanViews)

Check the view confidences of a Black-Litterman prior estimator against its views.

Each confidence must lie strictly in `(0, 1)`. A **scalar** is one confidence for all the views, for any number of views. A **vector** holds one confidence for each view that the caller wrote, so the function counts only a vector against the views. A scalar ``v`` and the constant vector of ``v`` give the same ``\\mathbf{\\Omega}`` in [`calc_omega`](@ref).

A precomputed [`BlackLittermanViews`](@ref) with `excl` holds `length(Q) + length(excl)` views that the caller wrote. [`remove_excl_views`](@ref) then removes the confidences of the views in `excl`, so one confidence remains for each row of `P`.

The bound is necessary. [`calc_omega`](@ref) changes a confidence ``v`` into the scale ``1/v - 1``, which is negative for each ``v > 1`` and each ``v < 0``. A view uncertainty matrix with a negative diagonal entry is not a covariance. Without the bound, the estimator returns a result made from such a matrix, and it raises no error. [`calc_omega`](@ref) states why the bound also refuses the two endpoints.

# Arguments

  - `views_conf`: One confidence, or a vector of confidences.
  - `views`: The views, a [`LinearConstraintEstimator`](@ref) or a [`BlackLittermanViews`](@ref).
  - `val`: The equations of a [`LinearConstraintEstimator`](@ref).

# Validation

Each method selects one shape of `views_conf` and one shape of `views`, and refuses the input that the pair does not accept.

  - `(::Nothing, args...)`: There is no confidence, so the method checks nothing.
  - `(::Number, ::EqnType)`: `0 < views_conf < 1`, through [`assert_unit_interval`](@ref). The method does not check the count, because one scalar is for any number of equations.
  - `(::VecNum, ::EqnType)`: When `val` is a vector of equations, `length(val) == length(views_conf)`. When `val` is one equation, `length(views_conf) == 1`. Then `all(x -> 0 < x < 1, views_conf)`.
  - `(::Num_VecNum, ::LinearConstraintEstimator)`: The method checks nothing itself, and sends `views.val` to the two methods above.
  - `(::Num_VecNum, ::BlackLittermanViews)`: When `views_conf` is a vector, `length(views_conf) == length(views.Q) + length(views.excl)`, with no `excl` term when `views.excl` is `nothing`. Then `all(x -> 0 < x < 1, views_conf)`. A precomputed pair resolves no name, so this is the only site that sees its confidences.

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
              DomainError(views_conf,
                          "`views_conf` holds $(count(x -> !(zero(x) < x < one(x)), views_conf)) entries outside `(0, 1)`. A view confidence scales the view uncertainty by `1 / v - 1`, so every entry must lie in `(0, 1)`."))
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
        # A vector holds one confidence for each view the caller wrote, as on the
        # equation-shaped routes, and `remove_excl_views` then drops the entries of `excl`.
        # Counted against `Q` alone, a vector over the kept rows lost an entry there and
        # `calc_omega` broadcast the one that was left over every row.
        n = length(views.Q) + (isnothing(views.excl) ? 0 : length(views.excl))
        @argcheck(length(views_conf) == n,
                  DimensionMismatch("length(views_conf) ($(length(views_conf))) must match the number of views that the caller wrote, length(views.Q) + length(views.excl) ($n)"))
    end
    # Precomputed views resolve no names, so this is the only site that sees their confidences.
    # Without the bound a confidence outside `(0, 1)` reaches `calc_omega`, whose `1/v - 1` scale
    # is then negative, and the estimator answers from a view uncertainty matrix that is not a
    # covariance. The equation-shaped view routes above already refuse the same input.
    @argcheck(all(x -> zero(x) < x < one(x), views_conf),
              DomainError(views_conf,
                          "`views_conf` holds $(count(x -> !(zero(x) < x < one(x)), views_conf)) entries outside `(0, 1)`. A view confidence scales the view uncertainty by `1 / v - 1`, so every entry must lie in `(0, 1)`."))
    return nothing
end

export black_litterman_views, BlackLittermanViews
