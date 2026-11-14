"""
    assert_universe_axis_order(sets::UniverseSets, rd::ReturnsResult) -> Nothing
    assert_universe_axis_order(sets::UniverseSets, ax::NamedTuple) -> Nothing
    assert_universe_axis_order(est::Union{AbstractEstimator, AbstractAlgorithm}, rd::ReturnsResult) -> Nothing

Assert that every [`UniverseSets`](@ref) on the asset axis of `rd` names the columns of `rd`, name for name and in order.

A name in `sets.dict[sets.xkey]` resolves to its position in that list, and the position indexes a column of the returns matrix. Nothing else ties the two lists. If they differ in order, a view, a bound or a group binds the wrong asset, and the fit still runs. If `sets` holds a name that `rd` does not have, the fit runs too. This function turns that silent wrong answer into an error.

The estimator method walks the tree of `est` and checks each universe that it finds. The walk follows the fields that [`asset_axis_fields`](@ref) names, because a universe in such a field names the columns of `rd`. A prior that fits a child on the factor returns names that child in [`factor_axis_fields`](@ref), and the walk checks the universes of the child against `rd.nf`. The walk stops at these values:

  - A nested optimisation head, an [`OptimisationEstimator`](@ref). A meta-optimiser gives each inner head returns data of its own, and the head checks its tree against that data at its own entry.
  - A [`TimeDependent`](@ref) schedule. Its entries resolve at each fold, against the universe of that fold.
  - A result, or any value that is not an estimator or an algorithm. A result is data, and it holds no configuration to resolve.

# Algorithm

The method that Julia selects is the algorithm.

 1. `sets` is a [`UniverseSets`](@ref): check the asset axis, `sets.dict[sets.xkey]` against `rd.nx`, and the factor axis, `sets.dict[sets.tfkey]` against `rd.nf`. An axis is skipped when `rd` holds no names for it, or when `sets` declares no key for it. A list that is `===` to the names of `rd` passes with no comparison.
 2. `est` is an estimator or an algorithm: walk the fields of `est` with [`universe_axis_children!`](@ref). An `IdSet` holds each universe that the walk checked, so a universe that many estimators share is checked once.

# Arguments

  - `sets`: The universe to check.
  - `est`: The estimator whose tree to check, such as a prior, an optimiser or an algorithm. The walk does not stop at `est` itself, even when it is an optimisation head.
  - $(arg_dict[:rd])
  - `ax`: The names to check against, from [`returns_universe_axes`](@ref). The walk passes them, so that a child on the factor axis is checked against the factor names.

# Validation

  - Each declared list has the length of the names of `rd`. A `DimensionMismatch` is thrown otherwise.
  - Each declared list equals the names of `rd`, in order. An `ArgumentError` is thrown otherwise.

Both messages come from [`universe_axis_msg`](@ref), and name the first position where the lists differ.

# Returns

  - `nothing`.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\", \"C\", \"B\"], X = [0.1 0.2 0.3; 0.4 0.5 0.6]);

julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> PortfolioOptimisers.assert_universe_axis_order(EqualWeighted(; sets = sets), rd)
ERROR: ArgumentError: the asset universe declared under key `nx` does not describe the returns data: both have 3 assets but the order differs, first at position 2: `B` vs `C`. Position is the only link between a name and a column, so this attaches constraints, bounds and groups to the wrong asset rather than failing. Set `sets.dict[\"nx\"]` to `rd.nx`, or slice the sets to match the data.
[...]
```

# Related

  - [`UniverseSets`](@ref)
  - [`asset_axis_fields`](@ref)
  - [`factor_axis_fields`](@ref)
  - [`misaligned_axis_msg`](@ref)
  - [`investable_reduction`](@ref)
  - [`prior`](@ref)
"""
function assert_universe_axis_order(sets::UniverseSets, rd::ReturnsResult)
    return assert_universe_axis_order(sets, returns_universe_axes(rd))
end
function assert_universe_axis_order(est::Union{AbstractEstimator, AbstractAlgorithm},
                                    rd::ReturnsResult)
    universe_axis_children!(IdSet{UniverseSets}(), est, returns_universe_axes(rd))
    return nothing
end
function assert_universe_axis_order(sets::UniverseSets, ax::NamedTuple)
    for (axis, key, names, sym) in
        ((ax.axis, sets.xkey, ax.nx, ax.sym), ("factor", sets.tfkey, ax.nf, "nf"))
        if isnothing(names) || !haskey(sets.dict, key)
            continue
        end
        declared = sets.dict[key]
        if declared === names
            continue
        end
        @argcheck(length(declared) == length(names),
                  DimensionMismatch(universe_axis_msg(sets, declared, names, axis, key, sym,
                                                      ax.nf)))
        @argcheck(declared == names,
                  ArgumentError(universe_axis_msg(sets, declared, names, axis, key, sym,
                                                  ax.nf)))
    end
    return nothing
end
"""
    universe_axis_msg(sets::UniverseSets, declared, names, axis, key, sym, nf) -> String

Build the error text of [`assert_universe_axis_order`](@ref).

The text is the one of [`misaligned_axis_msg`](@ref). One more sentence follows it when the list under the asset key is the list of the factor names `nf`. That is the shape of a universe that names the factors under the asset key, which a consumer of the factors once read. The sentence names the factor key, where the factors go.

# Arguments

  - `sets`: The universe that the check refused.
  - `declared`, `names`, `axis`, `key`, `sym`: The arguments of [`misaligned_axis_msg`](@ref).
  - `nf`: The names of the factor axis of the check, or `nothing`.

# Returns

  - `msg::String`: The error text.

# Related

  - [`assert_universe_axis_order`](@ref)
  - [`misaligned_axis_msg`](@ref)
"""
function universe_axis_msg(sets::UniverseSets, declared, names, axis, key, sym, nf)
    msg = misaligned_axis_msg(declared, names, axis, key, sym)
    if key != sets.xkey || isnothing(nf) || !isequal(declared, nf)
        return msg
    end
    return msg *
           " The list under `$(key)` is `rd.nf`, the names of the factors. Declare the factors under the factor key `$(sets.tfkey)`, and the assets under `$(key)`."
end
"""
    returns_universe_axes(rd::ReturnsResult) -> NamedTuple
    returns_universe_axes(ax::NamedTuple, ::Val{:factor}) -> NamedTuple

Return the names that a [`UniverseSets`](@ref) is checked against, as [`assert_universe_axis_order`](@ref) reads them.

The tuple has four entries. `nx` holds the names of the columns that the asset key of a universe names, and `nf` holds the names of the factor axis. `axis` and `sym` name the axis and the field of `rd` that `nx` comes from, for the error message.

# Algorithm

The method that Julia selects is the algorithm.

 1. `rd`: the asset key names `rd.nx` and the factor key names `rd.nf`.
 2. `ax` and `Val(:factor)`: the axes of a child that a prior fits on the factor returns. The child reads the factor returns as its asset axis, so its asset key names `ax.nf`. The factor returns have no factor axis of their own, so `nf` is `nothing`.

# Arguments

  - $(arg_dict[:rd])
  - `ax`: The axes of the parent.

# Returns

  - `ax::NamedTuple`: The tuple `(; nx, nf, axis, sym)`.

# Related

  - [`assert_universe_axis_order`](@ref)
  - [`factor_axis_fields`](@ref)
"""
function returns_universe_axes(rd::ReturnsResult)
    return (; nx = rd.nx, nf = rd.nf, axis = "asset", sym = "nx")
end
function returns_universe_axes(ax::NamedTuple, ::Val{:factor})
    return (; nx = ax.nf, nf = nothing, axis = "factor", sym = "nf")
end
"""
    factor_axis_fields(x) -> Tuple{Vararg{Symbol}}

Return the names of the fields of `x` that hold a child that `x` fits on the factor returns, or a universe of the factors.

The child reads the columns of the factor returns, so a [`UniverseSets`](@ref) in that child must name the factors of the returns data, in order. [`asset_axis_fields`](@ref) leaves such a child out, because the child is not on the asset axis. [`universe_axis_children!`](@ref) walks these fields against the factor names instead.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`FactorPrior`](@ref) or a [`FactorBlackLittermanPrior`](@ref): return `(:pe,)`, its factor prior.
 2. An [`AugmentedBlackLittermanPrior`](@ref): return `(:f_pe,)`, its factor prior.
 3. A [`FactorRiskContribution`](@ref): return `(:sets,)`. The head solves for the factor weights, so the asset key of its universe names the factors of the returns data. Its method of [`asset_axis_fields`](@ref) leaves `sets` out for the same reason.
 4. Any other value: return `()`.

A [`CrossSectionalFactorPrior`](@ref) fits its factor prior on the factor returns that it estimates, and the returns data holds no names for those factors. So it is not on this list.

# Arguments

  - `x`: The estimator to read.

# Returns

  - `fields::Tuple{Vararg{Symbol}}`: The names of the fields on the factor axis.

# Related

  - [`asset_axis_fields`](@ref)
  - [`assert_universe_axis_order`](@ref)
"""
function factor_axis_fields(::Any)
    return ()
end
function factor_axis_fields(::Union{FactorPrior, FactorBlackLittermanPrior})
    return (:pe,)
end
function factor_axis_fields(::AugmentedBlackLittermanPrior)
    return (:f_pe,)
end
"""
    universe_axis_children!(seen::IdSet{UniverseSets}, x, ax::NamedTuple) -> Nothing

Walk the fields of an estimator, and check each [`UniverseSets`](@ref) in them on the axes `ax`, for [`assert_universe_axis_order`](@ref).

[`universe_axis_visit!`](@ref) handles the value of each field.

# Algorithm

 1. Visit the value of each field of [`asset_axis_fields`](@ref) on `ax`.
 2. When `ax.nf` holds names, visit the value of each field of [`factor_axis_fields`](@ref) on the factor axes of [`returns_universe_axes`](@ref), with a new `IdSet`. A universe on the factor axis names other columns than a universe on the asset axis, so the two do not share a set.

# Arguments

  - `seen`: The universes that the walk checked.
  - `x`: The estimator whose fields to walk.
  - `ax`: The axes, from [`returns_universe_axes`](@ref).

# Validation

  - The rules of [`assert_universe_axis_order`](@ref), for each universe that the walk finds.

# Returns

  - `nothing`.

# Related

  - [`assert_universe_axis_order`](@ref)
  - [`universe_axis_visit!`](@ref)
  - [`asset_axis_fields`](@ref)
  - [`factor_axis_fields`](@ref)
"""
function universe_axis_children!(seen::IdSet{UniverseSets}, x, ax::NamedTuple)
    for f in asset_axis_fields(x)
        universe_axis_visit!(seen, getfield(x, f), ax)
    end
    ffs = factor_axis_fields(x)
    if isempty(ffs) || isnothing(ax.nf)
        return nothing
    end
    fseen = IdSet{UniverseSets}()
    fax = returns_universe_axes(ax, Val(:factor))
    for f in ffs
        universe_axis_visit!(fseen, getfield(x, f), fax)
    end
    return nothing
end
"""
    universe_axis_visit!(seen::IdSet{UniverseSets}, x, ax::NamedTuple) -> Nothing

Handle one value of a field in the walk of [`universe_axis_children!`](@ref).

# Algorithm

The method that Julia selects on the type of `x` is the algorithm.

 1. A [`UniverseSets`](@ref) that `seen` holds: return. Any other universe: add it to `seen`, and check it with [`assert_universe_axis_order`](@ref).
 2. An [`OptimisationEstimator`](@ref) or a [`TimeDependent`](@ref): return, for the reasons that [`assert_universe_axis_order`](@ref) states.
 3. An estimator, an algorithm or a covariance estimator: walk its fields with `universe_axis_children!`.
 4. A vector of estimators or algorithms: visit each element.
 5. Any other value: return.

# Arguments

  - `seen`: The universes that the walk checked.
  - `x`: The value of a field.
  - `ax`: The axes, from [`returns_universe_axes`](@ref).

# Validation

  - The rules of [`assert_universe_axis_order`](@ref), for each universe that the walk finds.

# Returns

  - `nothing`.

# Related

  - [`universe_axis_children!`](@ref)
  - [`assert_universe_axis_order`](@ref)
"""
function universe_axis_visit!(seen::IdSet{UniverseSets}, sets::UniverseSets, ax::NamedTuple)
    if sets in seen
        return nothing
    end
    push!(seen, sets)
    return assert_universe_axis_order(sets, ax)
end
function universe_axis_visit!(::IdSet{UniverseSets},
                              ::Union{OptimisationEstimator, TimeDependent}, ::NamedTuple)
    return nothing
end
function universe_axis_visit!(seen::IdSet{UniverseSets},
                              x::Union{AbstractEstimator, AbstractAlgorithm,
                                       StatsBase.CovarianceEstimator}, ax::NamedTuple)
    return universe_axis_children!(seen, x, ax)
end
function universe_axis_visit!(seen::IdSet{UniverseSets},
                              x::AbstractVector{<:Union{Nothing, AbstractEstimator,
                                                        AbstractAlgorithm}}, ax::NamedTuple)
    for xi in x
        universe_axis_visit!(seen, xi, ax)
    end
    return nothing
end
function universe_axis_visit!(::IdSet{UniverseSets}, ::Any, ::NamedTuple)
    return nothing
end
