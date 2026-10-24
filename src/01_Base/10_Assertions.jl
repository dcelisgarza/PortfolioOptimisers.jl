"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `val` is non-empty.

A `Pair` or a `Number` input does nothing, because a scalar cannot be empty.

# Arguments

  - `val`: The container to check. It is an `AbstractDict`, a `VecPair` or an `ArrNum`.
  - $(arg_dict[:sym_msg])

# Validation

  - `!isempty(val)`, which raises an [`IsEmptyError`](@ref) that names `sym`.

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

Method of [`assert_nonempty`](@ref) for scalar inputs, which does nothing.

A `Pair` or a `Number` value cannot be empty.

# Arguments

  - `::Union{<:Pair, <:Number}`: The scalar value. The method does not read it.
  - `::Sym_Str = :val`: The symbolic name. The method does not read it.

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

Assert that every element of `val` is finite.

# Algorithm

The method that Julia selects on the type of `val` is the algorithm. Each method checks one predicate, and raises a `DomainError` that names `sym` and the predicate that failed.

 1. An `AbstractDict` checks `all(isfinite, values(val))`, so the method does not read the keys.
 2. A `VecPair` checks `all(isfinite, getindex.(val, 2))`, so the second element of each pair is the value.
 3. An `ArrNum` checks `all(isfinite, val)`.
 4. A `Pair` checks `isfinite(val[2])`.
 5. A `Number` checks `isfinite(val)`.

# Arguments

  - `val`: The value to check.
  - $(arg_dict[:sym_msg])

# Validation

  - Every element of `val` is finite, under the predicate that its type selects. A `NaN` and an infinity both breach it, and a breach raises a `DomainError`.

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
    @argcheck(all(isfinite, values(val)),
              DomainError(val,
                          "all(isfinite, values($sym)) must hold. Got\ncount(!isfinite, values($sym)) => $(count(!isfinite, values(val)))"))
    return nothing
end
function assert_finite(val::VecPair, sym::Sym_Str = :val)::Nothing
    @argcheck(all(isfinite, getindex.(val, 2)),
              DomainError(val,
                          "all(isfinite, getindex.($sym, 2)) must hold. Got\ncount(!isfinite, getindex.($sym, 2)) => $(count(!isfinite, getindex.(val, 2)))"))
    return nothing
end
function assert_finite(val::ArrNum, sym::Sym_Str = :val)::Nothing
    @argcheck(all(isfinite, val),
              DomainError(val,
                          "all(isfinite, $sym) must hold. Got\ncount(!isfinite, $sym) => $(count(!isfinite, val))"))
    return nothing
end
function assert_finite(val::Pair, sym::Sym_Str = :val)::Nothing
    @argcheck(isfinite(val[2]),
              DomainError(val[2], "isfinite($sym[2]) must hold. Got\n$sym[2] => $(val[2])"))
    return nothing
end
function assert_finite(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(isfinite(val),
              DomainError(val, "isfinite($sym) must hold. Got\n$sym => $val"))
    return nothing
end
"""
    assert_all_finite(val::ArrNum, sym::Sym_Str = :val)

Assert that every element of `val` is finite, and raise an [`IsNonFiniteError`](@ref) when one is not.

[`assert_finite`](@ref) checks the same predicate and raises a `DomainError`, as [`assert_nonneg`](@ref) and [`assert_gt0`](@ref) do. This function raises an [`IsNonFiniteError`](@ref), and it reports the position of the breach. With the position, a caller can find the bad entry, and the message does not quote the data.

This function guards the covariance estimators that compare returns with a threshold, [`GerberCovariance`](@ref) and [`SmythBrobyCovariance`](@ref). Their comparisons `X .>= sd` and `X .<= -sd` give `false` for a `NaN` entry, with no message. The estimator then reads the `NaN` as "no co-movement", and gives a finite covariance that looks correct and is wrong. Clean the returns first with an asset selector, such as [`CompleteAssetSelector`](@ref), or with [`MissingDataFilter`](@ref). Those accept non-finite entries in a returns matrix, and an estimator that compares returns does not. The message reports the count of bad entries and the first bad index, and never a data value.

# Arguments

  - `val`: The array to check.
  - $(arg_dict[:sym_msg])

# Validation

  - `all(isfinite, val)`, which raises an [`IsNonFiniteError`](@ref). The message gives the count of bad entries and the first bad index, and never a data value.

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

Assert that an untrusted size `val` is not above the active cap `cap` of [`RESOURCE_LIMITS`](@ref), and raise a `DomainError` when it is.

`sym` is the name of the field that the message reports, for example `:n_sim`. `knob` is the field of [`ResourceLimits`](@ref) to raise, for example `:max_n_sim`. The error tells the caller what the function refused, and how to allow it.

# Arguments

  - `val`: The requested size.
  - `cap`: The active cap.
  - `sym`: The symbolic name of the field that holds `val`.
  - `knob`: The symbolic name of the [`ResourceLimits`](@ref) field that raises the cap.

# Validation

  - `val <= cap`, which raises a `DomainError` that names `val`, `sym`, `cap` and `knob`.

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
    assert_ep_grid_size(K::Integer)

Assert that `K`, the number of grid points of a grid formulation of a tail view, is odd, positive and not above the active cap `max_ep_grid` of [`RESOURCE_LIMITS`](@ref).

Each grid point is one binary variable of the mixed-integer program that an upper-bound or equality view builds, and one dense row over the posterior probabilities. `K` is an untrusted size of the kind that [`assert_resource_cap`](@ref) checks. An odd `K` makes the centre of the grid a point of the grid. The two grid formulations state one rule, so both call this function.

# Arguments

  - `K`: The requested number of grid points.

# Validation

  - $(val_dict[:ep_gridK])

# Returns

  - `nothing`.

# Related

  - [`assert_resource_cap`](@ref)
  - [`RESOURCE_LIMITS`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
"""
function assert_ep_grid_size(K::Integer)::Nothing
    @argcheck(K >= one(K) && isodd(K), DomainError(K, "K must be odd and >= 1"))
    assert_resource_cap(K, RESOURCE_LIMITS[].max_ep_grid, :K, :max_ep_grid)
    return nothing
end
"""
    resolve_rng(rng::Random.AbstractRNG, seed::Option{<:Integer})

Resolve the random number generator to draw from, for an optional `seed`.

A `seed` gives a fresh, private generator. It is a `copy` of `rng` that `Random.seed!` reseeds
with `seed`. A seeded estimator is thus reproducible, and it does not reseed the task-global
generator that the caller can also use. The default `rng` is `Random.default_rng()`, which is a
shared object, and a reseed of it removes the randomness from the other draws of the caller with
no message. When `seed` is `nothing`, the function returns `rng` unchanged.

The function copies `rng` before it seeds it, and does not construct a fixed generator type such
as `Random.Xoshiro(seed)`. The copy keeps the generator object of the caller unchanged, and it
keeps the type of that generator. A portable generator from the caller, such as
`StableRNGs.StableRNG`, thus gives the same stream as `Random.seed!(rng, seed)` gave in place. The
draws are the same as the draws of the old in-place seed. The only difference is that the stream
of the caller does not change.

# Arguments

  - `rng`: The fallback random number generator. The function returns it unchanged when `seed` is `nothing`.
  - `seed`: The optional seed. When it is set, the function returns a private `Random.seed!(copy(rng), seed)`, and does not change `rng`.

# Algorithm

 1. Return `rng` unchanged when `seed` is `nothing`, so that the caller's generator is the generator to draw from.
 2. Otherwise copy `rng`, reseed the copy with `seed`, and return the copy. The function never mutates the caller's own generator.

# Returns

  - `Random.AbstractRNG`: The generator to draw from.

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

The method that Julia selects on the type of `val` is the algorithm. Each method checks one predicate, and raises a `DomainError` that names `sym` and the predicate that failed.

 1. An `AbstractDict` checks `all(x -> 0 <= x, values(val))`, so the method does not read the keys.
 2. A `VecPair` checks `all(x -> 0 <= x[2], val)`, so the second element of each pair is the value.
 3. An `ArrNum` checks `all(x -> 0 <= x, val)`.
 4. A `Pair` checks `0 <= val[2]`.
 5. A `Number` checks `0 <= val`.

# Arguments

  - `val`: The value to check.
  - $(arg_dict[:sym_msg])

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
              DomainError(val,
                          "all(x -> 0 <= x, values($sym)) must hold. Got\ncount(x -> !(0 <= x), values($sym)) => $(count(x -> !(zero(x) <= x), values(val)))"))
    return nothing
end
function assert_nonneg(val::VecPair, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x[2]) <= x[2], val),
              DomainError(val,
                          "all(x -> 0 <= x[2], $sym) must hold. Got\ncount(x -> !(0 <= x[2]), $sym) => $(count(x -> !(zero(x[2]) <= x[2]), val))"))
    return nothing
end
function assert_nonneg(val::ArrNum, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x) <= x, val),
              DomainError(val,
                          "all(x -> 0 <= x, $sym) must hold. Got\ncount(x -> !(0 <= x), $sym) => $(count(x -> !(zero(x) <= x), val))"))
    return nothing
end
function assert_nonneg(val::Pair, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val[2]) <= val[2],
              DomainError(val[2], "0 <= $sym[2] must hold. Got\n$sym[2] => $(val[2])"))
    return nothing
end
function assert_nonneg(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val) <= val, DomainError(val, "0 <= $sym must hold. Got\n$sym => $val"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that all elements of `val` are strictly positive (`> 0`).

# Algorithm

The method that Julia selects on the type of `val` is the algorithm. Each method checks one predicate, and raises a `DomainError` that names `sym` and the predicate that failed.

 1. An `AbstractDict` checks `all(x -> 0 < x, values(val))`, so the method does not read the keys.
 2. A `VecPair` checks `all(x -> 0 < x[2], val)`, so the second element of each pair is the value.
 3. An `ArrNum` checks `all(x -> 0 < x, val)`.
 4. A `Pair` checks `0 < val[2]`.
 5. A `Number` checks `0 < val`.

# Arguments

  - `val`: The value to check.
  - $(arg_dict[:sym_msg])

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
              DomainError(val,
                          "all(x -> 0 < x, values($sym)) must hold. Got\ncount(x -> !(0 < x), values($sym)) => $(count(x -> !(zero(x) < x), values(val)))"))
    return nothing
end
function assert_gt0(val::VecPair, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x[2]) < x[2], val),
              DomainError(val,
                          "all(x -> 0 < x[2], $sym) must hold. Got\ncount(x -> !(0 < x[2]), $sym) => $(count(x -> !(zero(x[2]) < x[2]), val))"))
    return nothing
end
function assert_gt0(val::ArrNum, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x) < x, val),
              DomainError(val,
                          "all(x -> 0 < x, $sym) must hold. Got\ncount(x -> !(0 < x), $sym) => $(count(x -> !(zero(x) < x), val))"))
    return nothing
end
function assert_gt0(val::Pair, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val[2]) < val[2],
              DomainError(val[2], "0 < $sym[2] must hold. Got\n$sym[2] => $(val[2])"))
    return nothing
end
function assert_gt0(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val) < val, DomainError(val, "0 < $sym must hold. Got\n$sym => $val"))
    return nothing
end
"""
    assert_unit_interval(val::Number, sym::Union{Symbol,<:AbstractString} = :val)
    assert_unit_interval(args...)

Assert that `val` lies strictly inside the open unit interval (`0 < val < 1`).

A value of a different type selects the `args...` method, which checks nothing. With that method, a caller can check a slot whose bound accepts more than a number and write no branch, as with [`assert_nonempty_gt0_finite_val`](@ref). A Calibration Rule, an [`AbstractCalibrationAlgorithm`](@ref), states no number at construction. The library checks the range later, when the rebuild runs, on the number that the rule returns.

# Arguments

  - `val`: The value to check.
  - $(arg_dict[:sym_msg])

# Validation

  - `::Number`: `0 < val < 1`, which raises a `DomainError` that names `sym` and `val`.
  - Each other type: No rule, so the call always passes.

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
              DomainError(val, "0 < $sym < 1 must hold. Got\n$sym => $val"))
    return nothing
end
function assert_unit_interval(args...)::Nothing
    return nothing
end
"""
    assert_closed_unit_interval(val::Number, sym::Union{Symbol,<:AbstractString} = :val)
    assert_closed_unit_interval(args...)

Assert that `val` lies inside the closed unit interval (`0 <= val <= 1`).

It is the closed form of [`assert_unit_interval`](@ref). The two functions differ only in whether the ends belong to the interval. A template can turn a weight fully off, or give it its full value, so the weight reaches both ends and needs this rule and not the open one.

A value of a different type selects the `args...` method, which checks nothing, as in [`assert_unit_interval`](@ref).

# Arguments

  - `val`: The value to check.
  - $(arg_dict[:sym_msg])

# Validation

  - `::Number`: `0 <= val <= 1`, which raises a `DomainError` that names `sym` and `val`.
  - Each other type: No rule, so the call always passes.

# Returns

  - `nothing`.

# Related

  - [`assert_unit_interval`](@ref)
  - [`assert_nonneg`](@ref)
  - [`assert_nonempty_nonneg_finite_val`](@ref)
"""
function assert_closed_unit_interval(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val) <= val <= one(val),
              DomainError(val, "0 <= $sym <= 1 must hold. Got\n$sym => $val"))
    return nothing
end
function assert_closed_unit_interval(args...)::Nothing
    return nothing
end
"""
    assert_nonempty_nonneg_finite_val(
        val::Union{<:AbstractDict, <:VecPair, <:ArrNum, Pair, Number},
        val_sym::Union{Symbol,<:AbstractString} = :val
    )
    assert_nonempty_nonneg_finite_val(args...)

Check that the input value is non-empty, non-negative and finite.

# Algorithm

 1. Call [`assert_nonempty`](@ref), then [`assert_finite`](@ref), then [`assert_nonneg`](@ref), each on `val` and `val_sym`. The calls run in this order, so the error names the first rule that the value breaks.
 2. A value of any other type selects the `args...` method, which checks nothing. With that method, a caller can check an optional field and write no branch.

# Arguments

  - `val`: The input value to check.
  - `val_sym`: The symbolic name that the error messages show.

# Validation

Each rule is the rule that the type of `val` selects in the three functions of step 1.

  - `::AbstractDict`: `!isempty(val)`, `all(isfinite, values(val))`, `all(x -> x >= 0, values(val))`.
  - `::VecPair`: `!isempty(val)`, `all(isfinite, getindex.(val, 2))`, `all(x -> x[2] >= 0, val)`.
  - `::ArrNum`: `!isempty(val)`, `all(isfinite, val)`, `all(x -> x >= 0, val)`.
  - `::Pair`: `isfinite(val[2])` and `val[2] >= 0`.
  - `::Number`: `isfinite(val)` and `val >= 0`.
  - Each other type: No rule, so the call always passes.

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

Check that the input value is non-empty, greater than zero, and finite.

# Algorithm

 1. Call [`assert_nonempty`](@ref), then [`assert_finite`](@ref), then [`assert_gt0`](@ref), each on `val` and `val_sym`. The calls run in this order, so the error names the first rule that the value breaks.
 2. A value of any other type selects the `args...` method, which checks nothing. With that method, a caller can check an optional field and write no branch.

# Arguments

  - `val`: The input value to check.
  - `val_sym`: The symbolic name that the error messages show.

# Validation

Each rule is the rule that the type of `val` selects in the three functions of step 1.

  - `::AbstractDict`: `!isempty(val)`, `all(isfinite, values(val))`, `all(x -> x > 0, values(val))`.
  - `::VecPair`: `!isempty(val)`, `all(isfinite, getindex.(val, 2))`, `all(x -> x[2] > 0, val)`.
  - `::ArrNum`: `!isempty(val)`, `all(isfinite, val)`, `all(x -> x > 0, val)`.
  - `::Pair`: `isfinite(val[2])` and `val[2] > 0`.
  - `::Number`: `isfinite(val)` and `val > 0`.
  - Each other type: No rule, so the call always passes.

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

Check that the input value is non-empty and finite.

# Algorithm

 1. Call [`assert_nonempty`](@ref), then [`assert_finite`](@ref), each on `val` and `val_sym`. The calls run in this order, so the error names the first rule that the value breaks.
 2. A value of any other type selects the `args...` method, which checks nothing. With that method, a caller can check an optional field and write no branch.

# Arguments

  - `val`: The input value to check.
  - `val_sym`: The symbolic name that the error messages show.

# Validation

Each rule is the rule that the type of `val` selects in the two functions of step 1.

  - `::AbstractDict`: `!isempty(val)`, `all(isfinite, values(val))`.
  - `::VecPair`: `!isempty(val)`, `all(isfinite, getindex.(val, 2))`.
  - `::ArrNum`: `!isempty(val)`, `all(isfinite, val)`.
  - `::Pair`: `isfinite(val[2])`.
  - `::Number`: `isfinite(val)`.
  - Each other type: No rule, so the call always passes.

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

  - `X`: The input matrix to check.
  - `X_sym`: The symbolic name that the error messages show.

# Validation

  - `size(X, 1) == size(X, 2)`, which raises a `DimensionMismatch` that names `X_sym` and both sizes.

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

  - $(arg_dict[:dims_chk])
  - $(arg_dict[:sym_msg])

# Validation

  - `dims in (1, 2)`, which raises a `DomainError` that names `sym` and `dims`.

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

Check `dims`, and return the matrices with the observations along the rows.

The check and the orientation are one call, so a caller cannot orient a matrix and skip the check of `dims`. All callers use this one function. A caller that writes the check and the `transpose` itself can omit the check, and then return the raw input for a `dims` of `3`.

# Algorithm

 1. Check `dims` with [`assert_dims`](@ref).
 2. Return each matrix unchanged when `dims` is `1`, because the observations are already along the rows.
 3. Return the `transpose` of each matrix when `dims` is `2`. A `nothing` passes through unchanged.

# Arguments

  - `dims`: The dimension that holds the observations.
  - `A`, `B`, `Cs...`: The matrices to orient. A `nothing` passes through unchanged, so an optional matrix needs no branch of its own.

# Validation

  - `dims in (1, 2)`. [`assert_dims`](@ref) checks it.

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
public resolve_rng
