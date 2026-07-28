"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for feature matrix estimators.

An `AbstractFeatureMatrixEstimator` is a **producer**: it computes the derived feature matrix `Z` that [`FeaturePrior`](@ref) attaches to a prior result. Producers exist so that a feature matrix on an [`AbstractPriorResult`](@ref) means exactly one thing — *someone declared these are features* — rather than being a second live spelling of a matrix that is also reachable elsewhere on the same result.

All producers implement [`feature_matrix`](@ref).

# Related

  - [`feature_matrix`](@ref)
  - [`RegressionFeatures`](@ref)
  - [`FeaturePrior`](@ref)
  - [`FeatureDistance`](@ref)
"""
abstract type AbstractFeatureMatrixEstimator <: AbstractEstimator end
"""
    feature_matrix(ze::MatNum_Arr3Num, pr::AbstractPriorResult, X, F, sets; kwargs...)
    feature_matrix(ze::RegressionFeatures, pr::AbstractPriorResult, X, F, sets; kwargs...)

Compute the derived feature matrix and its squareness flag.

Returns the pair `(Z, z_sq)` that [`FeaturePrior`](@ref) writes onto the prior result. `Z` is **canonically assets-major** — `assets × features` when static, `observations × assets × features` when time-varying — and is validated as such by [`LowOrderPrior`](@ref).

`z_sq` is **stated here, by the producer**, because a producer runs inside `prior(pe, X, F; …)` with raw matrices and no feature names: a prior result carries none, so [`features_are_assets`](@ref) is structurally unavailable and squareness cannot be derived. A producer that has both name vectors — one whose features *are* the assets, given `sets` — must still run the `nx == nz` comparison rather than assume, because a `z_sq` that lies is silent, not loud: [`feature_matrix_view`](@ref) slices the feature axis that is not there.

`pr` is the **already-computed** wrapped prior result, not an estimator. That ordering is required: [`RegressionFeatures`](@ref) reads `pr.rr`, which does not exist until the wrapped prior has run.

# Arguments

  - `ze`: Feature matrix producer, or a literal feature matrix.
  - `pr`: Prior result returned by the wrapped estimator.
  - `X`: Asset returns matrix `observations × assets`.
  - `F`: Factor returns matrix, or `nothing`.
  - `sets`: Asset sets supplying asset names, or `nothing`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators.

# Returns

  - `(Z, z_sq)::Tuple{<:MatNum_Arr3Num, Bool}`.

# Related

  - [`AbstractFeatureMatrixEstimator`](@ref)
  - [`FeaturePrior`](@ref)
  - [`RegressionFeatures`](@ref)
  - [`features_are_assets`](@ref)
"""
function feature_matrix end
# A literal matrix declares nothing about its own axes, so it is never treated as square. To
# reuse an `assets × assets` adjacency matrix as features — where an asset view must slice
# the feature axis too — use a producer that states `z_sq` rather than passing the matrix in
# bare.
function feature_matrix(ze::MatNum_Arr3Num, ::AbstractPriorResult, args...; kwargs...)
    return ze, false
end
"""
$(DocStringExtensions.TYPEDEF)

Feature matrix producer that reads the regression loadings off the wrapped prior result.

`RegressionFeatures` treats `pr.rr.L` — the coordinate system a factor model places each asset in — as an `assets × features` feature matrix. It is the cheapest real feature source in the library: a factor prior already computes it, and already refits it per fold, so the features track the fold with no extra plumbing.

# The matrix is `L`, not `M`

`L` is `assets × reduced_dimensions`, set by [`DimensionReductionRegression`](@ref): the low-dimensional coordinate system the asset actually lives in. `M` is the *reconstructed* full-factor loadings. `pr.rr.L` always resolves — [`Regression`](@ref) swaps in `M` when `L` is unset — so this producer needs no branch and works behind every regression estimator.

Both are assets-major and both are row-sliced by `port_opt_view(re, i)`, so the carried layout holds with no transpose. Sets `z_sq = false`: reduced dimensions are not assets.

# Validation

  - The wrapped prior must carry a regression (see [`assert_prior_regression`](@ref)). `BlackLittermanPrior` forwards neither `rr` nor `f_mu`/`f_sigma`, so `FeaturePrior(; pe = BlackLittermanPrior(…), ze = RegressionFeatures())` throws; nest the other way round instead.

# Examples

```jldoctest
julia> RegressionFeatures()
RegressionFeatures()
```

# Related

  - [`AbstractFeatureMatrixEstimator`](@ref)
  - [`feature_matrix`](@ref)
  - [`FeaturePrior`](@ref)
  - [`FactorPrior`](@ref)
  - [`Regression`](@ref)
  - [`assert_prior_regression`](@ref)
"""
struct RegressionFeatures <: AbstractFeatureMatrixEstimator end
function feature_matrix(::RegressionFeatures, pr::AbstractPriorResult, args...; kwargs...)
    assert_prior_regression(pr, :pe)
    return pr.rr.L, false
end
"""
    feature_estimator_view(ze::AbstractFeatureMatrixEstimator, i, args...)
    feature_estimator_view(ze::MatNum_Arr3Num, i, args...)

Subselect a [`FeaturePrior`](@ref)'s `ze` slot by assets `i`.

A producer is *configuration*: it recomputes from the viewed prior on the next call, so it passes through unchanged. A literal feature matrix is *data* and must be sliced on its asset axis, exactly as the carried matrix is — otherwise its columns would keep pointing at the full universe while the rows point at a cluster.

The literal path slices with `sq = false`, matching the `z_sq = false` that [`feature_matrix`](@ref) reports for a bare matrix.

# Related

  - [`FeaturePrior`](@ref)
  - [`feature_matrix_view`](@ref)
  - [`port_opt_view`](@ref)
"""
function feature_estimator_view(ze::AbstractFeatureMatrixEstimator, args...)
    return ze
end
function feature_estimator_view(ze::MatNum_Arr3Num, i, args...)
    return feature_matrix_view(ze, false, :, i)
end
"""
$(DocStringExtensions.TYPEDEF)

Prior estimator that attaches a feature matrix to the prior it wraps.

`FeaturePrior` delegates every moment to the wrapped estimator `pe` and adds nothing but `Z` and `z_sq`, so it is a **provably pure addition**: the moments it returns are the wrapped estimator's, unchanged. That is what makes every existing prior feature-capable without any of them knowing about features.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FeaturePrior(;
        pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
        ze::Union{<:MatNum_Arr3Num, <:AbstractFeatureMatrixEstimator},
        sets::Option{<:AssetSets} = nothing
    ) -> FeaturePrior

Keywords correspond to the struct's fields.

# Details

  - **Moments first, features second.** The wrapped prior is computed before `ze` runs, because a producer may need the result — [`RegressionFeatures`](@ref) reads `pr.rr`.
  - **Nesting order does not matter.** Every wrapping prior estimator forwards `Z` and `z_sq`, so `BlackLittermanPrior(; pe = FeaturePrior(…))` and `FeaturePrior(; pe = BlackLittermanPrior(…))` both arrive at [`distance`](@ref) with the same feature matrix. The exception is the estimators whose wrapped prior is fit on *factors* — [`FactorPrior`](@ref), [`FactorBlackLittermanPrior`](@ref) — which drop it deliberately; wrap those from the outside.
  - **The outermost declaration wins.** Nesting one `FeaturePrior` inside another overwrites the inner feature matrix rather than merging.
  - **A literal `ze` must be static to survive a fold.** An `assets × features` matrix is observation-independent and is correct under any cross-validation fold. A time-varying literal is not: folds slice observations *before* the prior is fit and never touch the estimator, so its observation axis would no longer match — which [`LowOrderPrior`](@ref) rejects at construction. Use a producer to derive a time-varying `Z` per fold.

# Validation

  - If `ze` is a literal matrix, it is non-empty.
  - If `sets` is not `nothing`, `length(sets.dict[sets.key]) == size(X, 2)`.

# Examples

```jldoctest
julia> pe = FeaturePrior(; pe = EmpiricalPrior(), ze = RegressionFeatures());

julia> pe.ze
RegressionFeatures()

julia> isnothing(pe.sets)
true

julia> pe.pe
EmpiricalPrior
       ce ┼ PortfolioOptimisersCovariance
          │   ce ┼ Covariance
          │      │    me ┼ SimpleExpectedReturns
          │      │       │   w ┴ nothing
          │      │    ce ┼ GeneralCovariance
          │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
          │      │       │    w ┴ nothing
          │      │   alg ┴ FullMoment()
          │   mp ┼ MatrixProcessing
          │      │     pdm ┼ Posdef
          │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
          │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
          │      │      dn ┼ nothing
          │      │      dt ┼ nothing
          │      │     alg ┼ nothing
          │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
       me ┼ SimpleExpectedReturns
          │   w ┴ nothing
  horizon ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractFeatureMatrixEstimator`](@ref)
  - [`RegressionFeatures`](@ref)
  - [`feature_matrix`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`FeatureDistance`](@ref)
  - [`prior`](@ref)
"""
@propagatable @concrete struct FeaturePrior <: AbstractLowOrderPriorEstimator_AF
    """
    $(field_dict[:pe])
    """
    @fprop pe
    """
    $(field_dict[:ze])
    """
    @fprop ze
    """
    $(field_dict[:sets])
    """
    sets
    function FeaturePrior(pe::AbstractLowOrderPriorEstimator_A_F_AF,
                          ze::Union{<:MatNum_Arr3Num, <:AbstractFeatureMatrixEstimator},
                          sets::Option{<:AssetSets})
        if isa(ze, MatNum_Arr3Num)
            assert_nonempty(ze, :ze)
        end
        return new{typeof(pe), typeof(ze), typeof(sets)}(pe, ze, sets)
    end
end
function FeaturePrior(; pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                      ze::Union{<:MatNum_Arr3Num, <:AbstractFeatureMatrixEstimator},
                      sets::Option{<:AssetSets} = nothing)::FeaturePrior
    return FeaturePrior(pe, ze, sets)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties FeaturePrior begin
    forward(pe, me, ce)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`FeaturePrior`](@ref) restricted to assets at index `i`.

Hand-written rather than generated by [`@vprop`](@ref): the `ze` slot holds either a producer or a literal feature matrix, and the two need different treatment — see [`feature_estimator_view`](@ref).

# Related

  - [`FeaturePrior`](@ref)
  - [`feature_estimator_view`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(pe::FeaturePrior, i, args...)::FeaturePrior
    return FeaturePrior(; pe = port_opt_view(pe.pe, i, args...),
                        ze = feature_estimator_view(pe.ze, i, args...),
                        sets = port_opt_view(pe.sets, i, args...))
end
"""
    prior(pe::FeaturePrior, X::MatNum, F::Option{<:MatNum} = nothing;
          dims::Int = 1, kwargs...)

Compute the wrapped prior's moments and attach a feature matrix to them.

Every moment is the wrapped estimator's, untouched. The only additions are `Z` and `z_sq`, produced by `pe.ze` from the *already-computed* prior result — the ordering [`RegressionFeatures`](@ref) needs, since it reads `pr.rr`.

# Arguments

  - `pe`: Feature prior estimator.
  - `X`: Asset returns matrix `observations × assets`.
  - `F`: Optional factor matrix (default: `nothing`).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the wrapped estimator and the producer.

# Returns

  - `pr::LowOrderPrior`: The wrapped result, with `Z` and `z_sq` set.

# Validation

  - `dims in (1, 2)`.
  - If `pe.sets` is not `nothing`, `length(pe.sets.dict[pe.sets.key]) == size(X, 2)`.
  - `Z` is validated against `X` by [`LowOrderPrior`](@ref).

# Related

  - [`FeaturePrior`](@ref)
  - [`feature_matrix`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
function prior(pe::FeaturePrior, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1,
               kwargs...)
    assert_dims(dims)
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    if !isnothing(pe.sets)
        @argcheck(length(pe.sets.dict[pe.sets.key]) == size(X, 2),
                  DimensionMismatch("length(pe.sets.dict[pe.sets.key]) ($(length(pe.sets.dict[pe.sets.key]))) must match size(X, 2) ($(size(X, 2)))"))
    end
    pr = prior(pe.pe, X, F; kwargs...)
    Z, z_sq = feature_matrix(pe.ze, pr, X, F, pe.sets; kwargs...)
    return LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma, chol = pr.chol, w = pr.w,
                         ens = pr.ens, kld = pr.kld, ow = pr.ow, rr = pr.rr, f_mu = pr.f_mu,
                         f_sigma = pr.f_sigma, f_w = pr.f_w, Z = Z, z_sq = z_sq)
end

export AbstractFeatureMatrixEstimator, RegressionFeatures, FeaturePrior, feature_matrix
