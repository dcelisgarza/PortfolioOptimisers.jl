"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `val` is non-empty.

No-op for `Pair` and `Number` inputs; emptiness does not apply to scalars.

# Arguments

  - `val`: Container to check; one of `AbstractDict`, `VecPair`, or `ArrNum`.
  - `sym`: Symbolic name used in the error message.

# Validation

  - `!isempty(val)`, which raises an [`IsEmptyError`](@ref) naming `sym`.

# Returns

  - `nothing`.

# Related

  - [`assert_finite`](@ref)
  - [`assert_nonneg`](@ref)
  - [`assert_gt0`](@ref)
  - [`assert_nonempty_nonneg_finite_val`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
  - [`assert_nonempty_finite_val`](@ref)
"""
function assert_nonempty(val::Union{<:AbstractDict, <:VecPair, <:ArrNum},
                         sym::Sym_Str = :val)::Nothing
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($sym) must hold. Got\n!isempty($sym) => $(!isempty(val))"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

No-op overload of [`assert_nonempty`](@ref) for scalar inputs.

Emptiness does not apply to `Pair` or `Number` values.

# Arguments

  - `::Union{<:Pair, <:Number}`: Scalar value, not read.
  - `::Sym_Str = :val`: Symbolic name, not read.

# Returns

  - `nothing`.

# Related

  - [`assert_nonempty`](@ref)
"""
function assert_nonempty(::Union{<:Pair, <:Number}, ::Sym_Str = :val)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `val` contains at least one finite element.

# Algorithm

The method Julia selects on the type of `val` is the algorithm. Each method checks one predicate and raises a `DomainError` that names `sym` and the predicate it failed.

 1. An `AbstractDict` checks `any(isfinite, values(val))`, so the keys are not read.
 2. A `VecPair` checks `any(isfinite, getindex.(val, 2))`, so the second element of each pair is the value.
 3. An `ArrNum` checks `any(isfinite, val)`.
 4. A `Pair` checks `isfinite(val[2])`.
 5. A `Number` checks `isfinite(val)`.

# Arguments

  - `val`: Value to check.
  - `sym`: Symbolic name used in the error message.

# Validation

  - At least one element of `val` is finite, under the predicate that its type selects. A breach raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`assert_nonempty`](@ref)
  - [`assert_nonneg`](@ref)
  - [`assert_gt0`](@ref)
  - [`assert_nonempty_nonneg_finite_val`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
  - [`assert_nonempty_finite_val`](@ref)
"""
function assert_finite(val::AbstractDict, sym::Sym_Str = :val)::Nothing
    @argcheck(any(isfinite, values(val)),
              DomainError("any(isfinite, values($sym)) must hold. Got\nany(isfinite, values($sym)) => $(any(isfinite, values(val)))"))
    return nothing
end
function assert_finite(val::VecPair, sym::Sym_Str = :val)::Nothing
    @argcheck(any(isfinite, getindex.(val, 2)),
              DomainError("any(isfinite, getindex.($sym, 2)) must hold. Got\nany(isfinite, getindex.($sym, 2)) => $(any(isfinite, getindex.(val, 2)))"))
    return nothing
end
function assert_finite(val::ArrNum, sym::Sym_Str = :val)::Nothing
    @argcheck(any(isfinite, val),
              DomainError("any(isfinite, $sym) must hold. Got\nany(isfinite, $sym) => $(any(isfinite, val))"))
    return nothing
end
function assert_finite(val::Pair, sym::Sym_Str = :val)::Nothing
    @argcheck(isfinite(val[2]),
              DomainError("isfinite($sym[2]) must hold. Got\nisfinite($sym[2]) => $(isfinite(val[2]))"))
    return nothing
end
function assert_finite(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(isfinite(val),
              DomainError("isfinite($sym) must hold. Got\nisfinite($sym) => $(isfinite(val))"))
    return nothing
end
"""
    assert_all_finite(val::ArrNum, sym::Sym_Str = :val)

Assert that *every* element of `val` is finite, failing closed with an [`IsNonFiniteError`](@ref) otherwise.

Unlike [`assert_finite`](@ref), which only requires *one* finite element, this demands the whole array be finite. It guards the comparison-based covariance estimators ([`GerberCovariance`](@ref), [`SmythBrobyCovariance`](@ref)): their `X .>= sd` / `X .<= -sd` comparisons silently evaluate a `NaN` entry as `false`, masking it as "no co-movement" and yielding a finite, plausible, *wrong* covariance rather than an error. Clean returns first with an asset selector (e.g. [`CompleteAssetSelector`](@ref)) or [`MissingDataFilter`](@ref) — non-finite entries in a returns matrix are a supported input to *those*, but not to a comparison-based estimator. The message reports the count of offending entries and the first offending index only — never the data values.

# Arguments

  - `val`: Array to check.
  - `sym`: Symbolic name used in the error message.

# Validation

  - `all(isfinite, val)`, which raises an [`IsNonFiniteError`](@ref). The message carries the count of offending entries and the first offending index, and never a data value.

# Returns

  - `nothing`.

# Related

  - [`assert_finite`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
function assert_all_finite(val::ArrNum, sym::Sym_Str = :val)::Nothing
    @argcheck(all(isfinite, val),
              IsNonFiniteError("all(isfinite, $sym) must hold. Got $(count(!isfinite, val)) non-finite entries; first at $(findfirst(!isfinite, val))."))
    return nothing
end
"""
    assert_resource_cap(val::Integer, cap::Integer, sym::Sym_Str, knob::Sym_Str)

Assert that an untrusted sizing integer `val` does not exceed the active [`RESOURCE_LIMITS`](@ref) ceiling `cap`, failing closed with a `DomainError` otherwise.

`sym` names the offending field in the message (e.g. `:n_sim`) and `knob` names the [`ResourceLimits`](@ref) field to raise (e.g. `:max_n_sim`), so the error tells the caller both what was rejected and how to allow it deliberately.

# Arguments

  - `val`: The requested size.
  - `cap`: The active ceiling.
  - `sym`: Symbolic name of the offending field.
  - `knob`: Symbolic name of the [`ResourceLimits`](@ref) field that raises the ceiling.

# Validation

  - `val <= cap`, which raises a `DomainError` naming `val`, `sym`, `cap` and `knob`.

# Returns

  - `nothing`.

# Related

  - [`RESOURCE_LIMITS`](@ref)
  - [`set_resource_limits!`](@ref)
  - [`with_resource_limits`](@ref)
  - [`assert_frontier_sweep_cap`](@ref)
"""
function assert_resource_cap(val::Integer, cap::Integer, sym::Sym_Str,
                             knob::Sym_Str)::Nothing
    @argcheck(val <= cap,
              DomainError(val,
                          "$sym = $val exceeds RESOURCE_LIMITS[].$knob = $cap. Raise it with set_resource_limits!(; $knob) — or with_resource_limits for a single scope — for genuinely large machine-authored runs."))
    return nothing
end
"""
    resolve_rng(rng::Random.AbstractRNG, seed::Option{<:Integer})

Resolve which random number generator to draw from given an optional `seed`.

A supplied `seed` yields a fresh, private generator — a `copy` of `rng` reseeded with `seed`
via `Random.seed!` — so a seeded estimator is reproducible **without** reseeding, and thereby
silently derandomising, the task-global RNG the caller may also own (the default `rng` is
`Random.default_rng()`, a shared object). When `seed` is `nothing`, `rng` is returned unchanged
and used as-is.

Copying `rng` before seeding (rather than constructing a fixed generator type such as
`Random.Xoshiro(seed)`) preserves both the caller's generator *object* — it is never mutated —
and its *type*, so a caller-supplied portable generator (e.g. `StableRNGs.StableRNG`) keeps
producing the same stream `Random.seed!(rng, seed)` did in place. The observable draws are thus
identical to the old in-place seeding; only the side effect on the caller's stream disappears.

# Arguments

  - `rng`: Fallback random number generator, used verbatim when `seed` is `nothing`.
  - `seed`: Optional seed. If set, a private `Random.seed!(copy(rng), seed)` is returned instead of touching `rng`.

# Algorithm

 1. Return `rng` unchanged when `seed` is `nothing`, so the caller's generator is used as it stands.
 2. Otherwise copy `rng`, reseed the copy with `seed`, and return the copy. The caller's own generator is never mutated.

# Returns

  - `Random.AbstractRNG`: the generator to draw from.

# Related

  - [`assert_resource_cap`](@ref)
"""
function resolve_rng(rng::Random.AbstractRNG, seed::Option{<:Integer})
    return isnothing(seed) ? rng : Random.seed!(copy(rng), seed)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that all elements of `val` are non-negative (`>= 0`).

# Algorithm

The method Julia selects on the type of `val` is the algorithm. Each method checks one predicate and raises a `DomainError` that names `sym` and the predicate it failed.

 1. An `AbstractDict` checks `all(x -> 0 <= x, values(val))`, so the keys are not read.
 2. A `VecPair` checks `all(x -> 0 <= x[2], val)`, so the second element of each pair is the value.
 3. An `ArrNum` checks `all(x -> 0 <= x, val)`.
 4. A `Pair` checks `0 <= val[2]`.
 5. A `Number` checks `0 <= val`.

# Arguments

  - `val`: Value to check.
  - `sym`: Symbolic name used in the error message.

# Validation

  - Every element of `val` is non-negative, under the predicate that its type selects. A breach raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`assert_nonempty`](@ref)
  - [`assert_finite`](@ref)
  - [`assert_gt0`](@ref)
  - [`assert_nonempty_nonneg_finite_val`](@ref)
"""
function assert_nonneg(val::AbstractDict, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x) <= x, values(val)),
              DomainError("all(x -> 0 <= x, values($sym)) must hold. Got\nall(x -> 0 <= x, values($sym)) => $(all(x -> zero(x) <= x, values(val)))"))
    return nothing
end
function assert_nonneg(val::VecPair, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x[2]) <= x[2], val),
              DomainError("all(x -> 0 <= x[2], $sym) must hold. Got\nall(x -> 0 <= x[2], $sym) => $(all(x -> zero(x[2]) <= x[2], val))"))
    return nothing
end
function assert_nonneg(val::ArrNum, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x) <= x, val),
              DomainError("all(x -> 0 <= x, $sym) must hold. Got\nall(x -> 0 <= x, $sym) => $(all(x -> zero(x) <= x, val))"))
    return nothing
end
function assert_nonneg(val::Pair, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val[2]) <= val[2],
              DomainError("0 <= $sym[2] must hold. Got\n$sym[2] => $(val[2])"))
    return nothing
end
function assert_nonneg(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val) <= val, DomainError("0 <= $sym must hold. Got\n$sym => $(val)"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that all elements of `val` are strictly positive (`> 0`).

# Algorithm

The method Julia selects on the type of `val` is the algorithm. Each method checks one predicate and raises a `DomainError` that names `sym` and the predicate it failed.

 1. An `AbstractDict` checks `all(x -> 0 < x, values(val))`, so the keys are not read.
 2. A `VecPair` checks `all(x -> 0 < x[2], val)`, so the second element of each pair is the value.
 3. An `ArrNum` checks `all(x -> 0 < x, val)`.
 4. A `Pair` checks `0 < val[2]`.
 5. A `Number` checks `0 < val`.

# Arguments

  - `val`: Value to check.
  - `sym`: Symbolic name used in the error message.

# Validation

  - Every element of `val` is strictly positive, under the predicate that its type selects. A breach raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`assert_nonempty`](@ref)
  - [`assert_finite`](@ref)
  - [`assert_nonneg`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
"""
function assert_gt0(val::AbstractDict, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x) < x, values(val)),
              DomainError("all(x -> 0 < x, values($sym)) must hold. Got\nall(x -> 0 < x, values($sym)) => $(all(x -> zero(x) < x, values(val)))"))
    return nothing
end
function assert_gt0(val::VecPair, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x[2]) < x[2], val),
              DomainError("all(x -> 0 < x[2], $sym) must hold. Got\nall(x -> 0 < x[2], $sym) => $(all(x -> zero(x[2]) < x[2], val))"))
    return nothing
end
function assert_gt0(val::ArrNum, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x) < x, val),
              DomainError("all(x -> 0 < x, $sym) must hold. Got\nall(x -> 0 < x, $sym) => $(all(x -> zero(x) < x, val))"))
    return nothing
end
function assert_gt0(val::Pair, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val[2]) < val[2],
              DomainError("0 < $sym[2] must hold. Got\n$sym[2] => $(val[2])"))
    return nothing
end
function assert_gt0(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val) < val, DomainError("0 < $sym must hold. Got\n$sym => $(val)"))
    return nothing
end
"""
    assert_unit_interval(val::Number, sym::Union{Symbol,<:AbstractString} = :val)
    assert_unit_interval(args...)

Assert that `val` lies strictly inside the open unit interval (`0 < val < 1`).

A value of any other type selects the `args...` method, which checks nothing. That is what lets a caller validate a slot whose bound admits more than a number without a branch of its own, on the terms [`assert_nonempty_gt0_finite_val`](@ref) already sets. A **Calibration Role** ([`AbstractCalibrationEstimator`](@ref)) states no number at construction, so the range is checked when the rebuild runs, against the number its rule returned.

# Arguments

  - `val`: Value to check.
  - `sym`: Symbolic name used in the error message.

# Validation

  - `::Number`: `0 < val < 1`, which raises a `DomainError` naming `sym` and `val`.
  - Any other type: no rule, so the call always passes.

# Returns

  - `nothing`.

# Related

  - [`assert_closed_unit_interval`](@ref)
  - [`assert_nonneg`](@ref)
  - [`assert_gt0`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
"""
function assert_unit_interval(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val) < val < one(val),
              DomainError("0 < $sym < 1 must hold. Got\n$sym => $(val)"))
    return nothing
end
function assert_unit_interval(args...)::Nothing
    return nothing
end
"""
    assert_closed_unit_interval(val::Number, sym::Union{Symbol,<:AbstractString} = :val)
    assert_closed_unit_interval(args...)

Assert that `val` lies inside the closed unit interval (`0 <= val <= 1`).

This is the closed sibling of [`assert_unit_interval`](@ref), and the two differ only in whether the ends belong to the interval. A weight that a template may switch off entirely, or hand its full value to, reaches both ends, so it needs this rule and not the open one.

A value of any other type selects the `args...` method, which checks nothing, on the terms [`assert_unit_interval`](@ref) already sets.

# Arguments

  - `val`: Value to check.
  - `sym`: Symbolic name used in the error message.

# Validation

  - `::Number`: `0 <= val <= 1`, which raises a `DomainError` naming `sym` and `val`.
  - Any other type: no rule, so the call always passes.

# Returns

  - `nothing`.

# Related

  - [`assert_unit_interval`](@ref)
  - [`assert_nonneg`](@ref)
  - [`assert_nonempty_nonneg_finite_val`](@ref)
"""
function assert_closed_unit_interval(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val) <= val <= one(val),
              DomainError("0 <= $sym <= 1 must hold. Got\n$sym => $(val)"))
    return nothing
end
function assert_closed_unit_interval(args...)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that a matrix-source selector names one of the two carriers.

Source selectors pick which of the two carriers a matrix is read from: `:prior` reads the prior result, `:data` reads the raw returns result. `x_src` selects the returns matrix `X`.

# Arguments

  - `src`: Selector to check.
  - `sym`: Symbolic name used in the error message.

# Validation

  - `src in (:prior, :data)`, which raises an `ArgumentError` naming `sym` and `src`.

# Returns

  - `nothing`.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`HierarchicalOptimiser`](@ref)
  - [`NestedClustered`](@ref)
"""
function assert_source_selector(src::Symbol, sym::Sym_Str = :x_src)::Nothing
    @argcheck(src in (:prior, :data),
              ArgumentError("$sym must be :prior or :data, got $(repr(src))"))
    return nothing
end
"""
    assert_nonempty_nonneg_finite_val(
        val::Union{<:AbstractDict, <:VecPair, <:ArrNum, Pair, Number},
        val_sym::Union{Symbol,<:AbstractString} = :val
    )
    assert_nonempty_nonneg_finite_val(args...)

Validate that the input value is non-empty, non-negative and finite.

# Algorithm

 1. Call [`assert_nonempty`](@ref), then [`assert_finite`](@ref), then [`assert_nonneg`](@ref), each on `val` and `val_sym`. The order is the order of the raises, so the first rule a value breaks is the one it is told about.
 2. A value of any other type selects the `args...` method, which checks nothing. That is what lets a caller validate an optional field without a branch of its own.

# Arguments

  - `val`: Input value to validate.
  - `val_sym`: Symbolic name used in the error messages.

# Validation

Each rule is the one that `val`'s own type selects in the three functions of step 1.

  - `::AbstractDict`: `!isempty(val)`, `any(isfinite, values(val))`, `all(x -> x >= 0, values(val))`.
  - `::VecPair`: `!isempty(val)`, `any(isfinite, getindex.(val, 2))`, `all(x -> x[2] >= 0, val)`.
  - `::ArrNum`: `!isempty(val)`, `any(isfinite, val)`, `all(x -> x >= 0, val)`.
  - `::Pair`: `isfinite(val[2])` and `val[2] >= 0`.
  - `::Number`: `isfinite(val)` and `val >= 0`.
  - Any other type: no rule, so the call always passes.

# Returns

  - `nothing`.

# Related

  - [`assert_nonempty_finite_val`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
  - [`assert_nonempty`](@ref)
  - [`assert_finite`](@ref)
  - [`assert_nonneg`](@ref)
"""
function assert_nonempty_nonneg_finite_val(val::Union{<:AbstractDict, <:VecPair, <:ArrNum,
                                                      <:Pair, <:Number},
                                           val_sym::Sym_Str = :val)::Nothing
    assert_nonempty(val, val_sym)
    assert_finite(val, val_sym)
    assert_nonneg(val, val_sym)
    return nothing
end
function assert_nonempty_nonneg_finite_val(args...)::Nothing
    return nothing
end
"""
    assert_nonempty_gt0_finite_val(
        val::Union{<:AbstractDict, <:VecPair, <:ArrNum, Pair, Number},
        val_sym::Union{Symbol,<:AbstractString} = :val
    )
    assert_nonempty_gt0_finite_val(args...)

Validate that the input value is non-empty, greater than zero, and finite.

# Algorithm

 1. Call [`assert_nonempty`](@ref), then [`assert_finite`](@ref), then [`assert_gt0`](@ref), each on `val` and `val_sym`. The order is the order of the raises, so the first rule a value breaks is the one it is told about.
 2. A value of any other type selects the `args...` method, which checks nothing. That is what lets a caller validate an optional field without a branch of its own.

# Arguments

  - `val`: Input value to validate.
  - `val_sym`: Symbolic name used in the error messages.

# Validation

Each rule is the one that `val`'s own type selects in the three functions of step 1.

  - `::AbstractDict`: `!isempty(val)`, `any(isfinite, values(val))`, `all(x -> x > 0, values(val))`.
  - `::VecPair`: `!isempty(val)`, `any(isfinite, getindex.(val, 2))`, `all(x -> x[2] > 0, val)`.
  - `::ArrNum`: `!isempty(val)`, `any(isfinite, val)`, `all(x -> x > 0, val)`.
  - `::Pair`: `isfinite(val[2])` and `val[2] > 0`.
  - `::Number`: `isfinite(val)` and `val > 0`.
  - Any other type: no rule, so the call always passes.

# Returns

  - `nothing`.

# Related

  - [`assert_nonempty_nonneg_finite_val`](@ref)
  - [`assert_nonempty_finite_val`](@ref)
  - [`assert_nonempty`](@ref)
  - [`assert_finite`](@ref)
  - [`assert_gt0`](@ref)
"""
function assert_nonempty_gt0_finite_val(val::Union{<:AbstractDict, <:VecPair, <:ArrNum,
                                                   <:Pair, <:Number},
                                        val_sym::Sym_Str = :val)::Nothing
    assert_nonempty(val, val_sym)
    assert_finite(val, val_sym)
    assert_gt0(val, val_sym)
    return nothing
end
function assert_nonempty_gt0_finite_val(args...)::Nothing
    return nothing
end
"""
    assert_nonempty_finite_val(
        val::Union{<:AbstractDict, <:VecPair, <:ArrNum, Pair, Number},
        val_sym::Union{Symbol,<:AbstractString} = :val
    )
    assert_nonempty_finite_val(args...)

Validate that the input value is non-empty and finite.

# Algorithm

 1. Call [`assert_nonempty`](@ref), then [`assert_finite`](@ref), each on `val` and `val_sym`. The order is the order of the raises, so the first rule a value breaks is the one it is told about.
 2. A value of any other type selects the `args...` method, which checks nothing. That is what lets a caller validate an optional field without a branch of its own.

# Arguments

  - `val`: Input value to validate.
  - `val_sym`: Symbolic name used in the error messages.

# Validation

Each rule is the one that `val`'s own type selects in the two functions of step 1.

  - `::AbstractDict`: `!isempty(val)`, `any(isfinite, values(val))`.
  - `::VecPair`: `!isempty(val)`, `any(isfinite, getindex.(val, 2))`.
  - `::ArrNum`: `!isempty(val)`, `any(isfinite, val)`.
  - `::Pair`: `isfinite(val[2])`.
  - `::Number`: `isfinite(val)`.
  - Any other type: no rule, so the call always passes.

# Returns

  - `nothing`.

# Related

  - [`assert_nonempty_nonneg_finite_val`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
  - [`assert_nonempty`](@ref)
  - [`assert_finite`](@ref)
"""
function assert_nonempty_finite_val(val::Union{<:AbstractDict, <:VecPair, <:ArrNum, <:Pair,
                                               <:Number}, val_sym::Sym_Str = :val)::Nothing
    assert_nonempty(val, val_sym)
    assert_finite(val, val_sym)
    return nothing
end
function assert_nonempty_finite_val(args...)::Nothing
    return nothing
end
"""
    assert_matrix_issquare(X::MatNum, X_sym::Symbol = :X)

Assert that the input matrix is square.

# Arguments

  - `X`: Input matrix to validate.
  - `X_sym`: Symbolic name used in error messages.

# Validation

  - `size(X, 1) == size(X, 2)`, which raises a `DimensionMismatch` naming `X_sym` and both sizes.

# Returns

  - `nothing`.

# Related

  - [`MatNum`](@ref)
  - [`assert_dims`](@ref)
"""
function assert_matrix_issquare(X::MatNum, X_sym::Symbol = :X)::Nothing
    @argcheck(size(X, 1) == size(X, 2),
              DimensionMismatch("size($X_sym, 1) == size($X_sym, 2) must hold. Got\nsize($X_sym, 1) => $(size(X, 1))\nsize($X_sym, 2) => $(size(X, 2))."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `dims` selects a valid matrix dimension (`dims in (1, 2)`).

# Arguments

  - `dims`: Dimension selector to check.
  - `sym`: Symbolic name used in the error message.

# Validation

  - `dims in (1, 2)`, which raises a `DomainError` naming `sym` and `dims`.

# Returns

  - `nothing`.

# Related

  - [`assert_matrix_issquare`](@ref)
  - [`dims_oriented`](@ref)
"""
function assert_dims(dims::Integer, sym::Sym_Str = :dims)::Nothing
    @argcheck(dims in (1, 2),
              DomainError(dims, "$sym must be 1 or 2. Got\n$sym => $(dims)"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate `dims` and return the matrices with the observations along the rows.

The guard and the orientation are one call, so a caller cannot orient a matrix without validating `dims`. This is the single decision point: a leaf that spelled the guard and the `transpose` by hand could omit the guard and answer a `dims` of `3` with the raw input.

# Algorithm

 1. Validate `dims` with [`assert_dims`](@ref).
 2. Return each matrix untouched when `dims` is `1`, because the observations already lie along the rows.
 3. Return the `transpose` of each matrix when `dims` is `2`. A `nothing` passes through unchanged.

# Arguments

  - `dims`: Dimension along which the observations lie.
  - `A`, `B`, `Cs...`: Matrices to orient. A `nothing` passes through unchanged, so an optional matrix needs no branch of its own.

# Validation

  - `dims in (1, 2)`, by [`assert_dims`](@ref).

# Returns

  - `A`: The oriented matrix, when one matrix is given.
  - `(A, B, Cs...)`: A tuple of the oriented matrices, when more than one is given.

# Related

  - [`assert_dims`](@ref)
  - [`MatNum`](@ref)
  - [`Option`](@ref)
"""
function dims_oriented(dims::Integer, A::Option{<:AbstractMatrix})
    assert_dims(dims)
    return isnothing(A) || isone(dims) ? A : transpose(A)
end
function dims_oriented(dims::Integer, A::Option{<:AbstractMatrix},
                       B::Option{<:AbstractMatrix}, Cs::Option{<:AbstractMatrix}...)
    assert_dims(dims)
    return map(x -> dims_oriented(dims, x), (A, B, Cs...))
end
