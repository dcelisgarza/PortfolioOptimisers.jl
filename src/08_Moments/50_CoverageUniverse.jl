"""
    coverage_mask(X::MatNum, pnl::Nothing; dims::Int = 1) -> Option{BitVector}
    coverage_mask(X::MatNum, pnl::AssetPanel{<:Any, Nothing, Nothing};
                  dims::Int = 1) -> Option{BitVector}
    coverage_mask(X::MatNum, pnl::AssetPanel; dims::Int = 1) -> Option{BitVector}

Derive the Coverage Universe of one fit: `true` at every asset a plain moment estimator can be fitted on.

An asset is in the Coverage Universe when its return is finite and the panel's active mask is `true` at **every** row of the window. The estimation mask is not read: it names the assets that enter a cross-sectional estimate, and a moment is not one. A static panel carries no mask, so the rule there is finiteness alone, as it is with no panel at all.

The all-covered case returns `nothing` rather than a mask of every `true`. That sentinel skips both the slice and the expansion, exactly as the `nothing` of [`investable_mask`](@ref) skips the optimiser's two halves.

The rule has one cost, and no docstring may hide it. **One** non-finite return, or **one** inactive row, inside the window puts the asset outside the Coverage Universe for that fit. A caller with a holiday imputes it in [`prices_to_returns`](@ref), or reaches for a mask-aware estimator, which takes the whole window and emits its own frame.

The two scans are **independent**, and only the asset axes must agree. A prior that reweights observations works on the axis its nested prior answered, and a nested prior may drop rows, so `X` and the panel can carry different observation counts and no row of one pairs with a row of the other. Pairing them by position would read the wrong date, and pairing them by the tail would assume a warm-up that no contract states. Reading each over its own rows assumes nothing. It is conservative where the two axes differ: an asset that the panel reports inactive at a row the sample no longer holds is outside the Coverage Universe of that fit.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Scan each column of `X`. An asset stays in while its return is finite at every row.
 3. Scan each column of the active mask, where a panel carries one. An asset stays in while the mask is `true` at every row.
 4. Throw an `IsEmptyError` when no asset is covered. A moment over no asset has no answer to give.
 5. Return `nothing` when every asset is covered, and the mask otherwise.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])
  - The asset axis of `pnl.amsk` must be the asset axis of the oriented `X`.
  - At least one asset must be covered.

# Returns

  - `cmsk::Option{BitVector}`: `true` at every covered asset, or `nothing` when every asset is covered.

# Related

  - [`coverage_reduction`](@ref)
  - [`expand_moment`](@ref)
  - [`investable_mask`](@ref)
  - [`AssetPanel`](@ref)
  - [`IsEmptyError`](@ref)
"""
function coverage_mask(X::MatNum, ::Union{Nothing, <:AssetPanel{<:Any, Nothing, Nothing}};
                       dims::Int = 1)::Option{BitVector}
    Xo = dims_oriented(dims, X)
    cmsk = trues(size(Xo, 2))
    for i in axes(Xo, 2)
        for t in axes(Xo, 1)
            if !isfinite(Xo[t, i])
                cmsk[i] = false
                break
            end
        end
    end
    return coverage_sentinel(cmsk)
end
function coverage_mask(X::MatNum, pnl::AssetPanel; dims::Int = 1)::Option{BitVector}
    Xo = dims_oriented(dims, X)
    amsk = pnl.amsk
    @argcheck(size(Xo, 2) == size(amsk, 2),
              DimensionMismatch("the returns matrix and the active mask of the Asset Panel describe the same assets, so their asset axes must be the same length, got $(size(Xo, 2)) columns of X against $(size(amsk, 2)) of pnl.amsk"))
    cmsk = trues(size(Xo, 2))
    for i in axes(Xo, 2)
        for t in axes(Xo, 1)
            if !isfinite(Xo[t, i])
                cmsk[i] = false
                break
            end
        end
    end
    for i in axes(amsk, 2)
        for t in axes(amsk, 1)
            if !amsk[t, i]
                cmsk[i] = false
                break
            end
        end
    end
    return coverage_sentinel(cmsk)
end
"""
    coverage_sentinel(cmsk::BitVector) -> Option{BitVector}

Refuse an empty Coverage Universe, and collapse a complete one onto the `nothing` sentinel.

This is the tail that the two [`coverage_mask`](@ref) methods share, so the refusal and the sentinel are written once and cannot drift apart.

# Algorithm

 1. Throw an `IsEmptyError` when `cmsk` holds no `true`.
 2. Return `nothing` when `cmsk` holds no `false`.
 3. Return `cmsk` otherwise.

# Arguments

  - `cmsk`: The raw coverage mask, one entry per asset.

# Validation

  - At least one asset must be covered.

# Returns

  - `cmsk::Option{BitVector}`: `cmsk` itself, or `nothing` when every asset is covered.

# Related

  - [`coverage_mask`](@ref)
  - [`IsEmptyError`](@ref)
"""
function coverage_sentinel(cmsk::BitVector)::Option{BitVector}
    @argcheck(any(cmsk),
              IsEmptyError("no asset is in the Coverage Universe of this window: every asset carries a non-finite return, or an inactive row of the Asset Panel, at some observation. Check that the window holds at least one asset that is listed and quoted throughout it."))
    return all(cmsk) ? nothing : cmsk
end
"""
    coverage_reduction(X::MatNum, pnl::Option{<:AssetPanel};
                       dims::Int = 1) -> Tuple{Option{BitVector}, MatNum}

Reduce a returns matrix to its Coverage Universe, and return the mask beside the clean block.

The reduction keeps the caller's orientation, so the plain moment verb that receives the block sees exactly the matrix it saw before the Asset Panel existed, and its own `dims` still describes it. The `nothing` sentinel returns `X` itself, so a complete window allocates nothing.

# Algorithm

 1. Derive the Coverage Universe with [`coverage_mask`](@ref).
 2. Return `(nothing, X)` on the sentinel.
 3. Slice the asset axis of `X` otherwise: its columns when `dims = 1`, and its rows when `dims = 2`.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])

# Returns

  - `(cmsk, Xc)::Tuple{Option{BitVector}, MatNum}`: The Coverage Universe, and `X` reduced to it.

# Related

  - [`coverage_mask`](@ref)
  - [`expand_moment`](@ref)
"""
function coverage_reduction(X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1)
    cmsk = coverage_mask(X, pnl; dims = dims)
    return if isnothing(cmsk)
        cmsk, X
    elseif isone(dims)
        cmsk, X[:, cmsk]
    else
        cmsk, X[cmsk, :]
    end
end
"""
    coverage_reduced_pair(A::MatNum, B::MatNum, cmsk::Nothing) -> Tuple{MatNum, MatNum}
    coverage_reduced_pair(A::MatNum, B::MatNum, cmsk::BitVector) -> Tuple{MatNum, MatNum}

Slice two `observations × assets` matrices onto the same Coverage Universe.

An estimator that reads a second per-asset panel beside its returns, as [`ImpliedVolatility`](@ref) reads an implied volatility surface, must slice both, so that the two describe the same universe. The `nothing` sentinel returns both untouched.

# Arguments

  - `A`, `B`: Two matrices, both `observations × assets`.
  - `cmsk`: The Coverage Universe, or `nothing`.

# Returns

  - `(A, B)::Tuple{MatNum, MatNum}`: The two matrices on the Coverage Universe.

# Related

  - [`coverage_mask`](@ref)
  - [`coverage_reduction`](@ref)
"""
function coverage_reduced_pair(A::MatNum, B::MatNum, ::Nothing)
    return A, B
end
function coverage_reduced_pair(A::MatNum, B::MatNum, cmsk::BitVector)
    return A[:, cmsk], B[:, cmsk]
end
"""
    coverage_nan_frame(A::AbstractArray, sz::Dims) -> AbstractArray

Allocate the `NaN` frame that a moment expands into.

The frame takes its element type from the block through `similar`, and not from `eltype(A)`. [`MatNum`](@ref) admits a `JuMP` scalar as well as a number, so `eltype` of one splits into two branches at inference and the numeric half is the only one a moment ever takes. Reading the type off the block instead keeps the frame in the block's own element type and leaves no unreachable branch behind.

# Algorithm

 1. Allocate an array of `sz`, of the same kind and element type as `A`.
 2. Fill it with `NaN`, and return it.

# Arguments

  - `A`: The block that will be written into the frame.
  - `sz`: Size of the frame.

# Returns

  - `frame::AbstractArray`: An array of size `sz`, filled with `NaN`.

# Related

  - [`expand_moment`](@ref)
"""
function coverage_nan_frame(A::AbstractArray, sz::Dims)
    frame = similar(A, sz)
    fill!(frame, NaN)
    return frame
end
"""
    coverage_pair_index(cmsk::BitVector) -> Vector{Int}

Map the columns of a reduced co-moment tensor onto the columns of the full-universe one.

A coskewness tensor is `assets × assets²` and a cokurtosis matrix is `assets² × assets²`. Both index an asset pair `(a, b)` at `(a - 1) * N + b`, because both are built from `kron(o, Y) ⊙ kron(Y, o)`, whose column `(a - 1) * N + b` is the elementwise product of column `b` and column `a`. The reduced tensor uses the same rule over the reduced width, and its pairs run with `a` outermost, so the index that this returns is in the order that the reduced columns already have.

# Algorithm

 1. Take the positions of the covered assets.
 2. Return `(a - 1) * N + b` over every ordered pair of them, with `a` outermost.

# Arguments

  - `cmsk`: The Coverage Universe.

# Returns

  - `idx::Vector{Int}`: The full-universe pair columns of the reduced ones, in the reduced order.

# Related

  - [`expand_moment`](@ref)
  - [`coskewness`](@ref)
  - [`cokurtosis`](@ref)
"""
function coverage_pair_index(cmsk::BitVector)
    idx = findall(cmsk)
    N = length(cmsk)
    return [(a - 1) * N + b for a in idx for b in idx]
end
"""
    expand_columns(A::MatNum, cmsk::Nothing) -> A
    expand_columns(A::MatNum, cmsk::BitVector) -> MatNum

Write a block whose **columns** are the asset axis back into a `NaN` frame of the full width.

This is one of the three primitives of the expansion, beside [`expand_rows`](@ref) and [`expand_vector`](@ref). [`expand_moment`](@ref) is the verb-facing name over them, and a prior that carries a block of its own reaches for them directly: a reconstructed returns matrix and a Cholesky factor are `observations × assets` and `factors × assets`, so both expand along their columns.

# Arguments

  - `A`: The block that the reduced fit produced, `anything × assets`.
  - `cmsk`: The Coverage Universe, or `nothing`.

# Returns

  - The block on the full asset universe, carrying `NaN` outside the Coverage Universe.

# Related

  - [`expand_moment`](@ref)
  - [`expand_regression`](@ref)
  - [`coverage_nan_frame`](@ref)
"""
function expand_columns(A::MatNum, ::Nothing)
    return A
end
function expand_columns(A::MatNum, cmsk::BitVector)
    frame = coverage_nan_frame(A, (size(A, 1), length(cmsk)))
    frame[:, cmsk] = A
    return frame
end
"""
    expand_rows(A::MatNum, cmsk::Nothing) -> A
    expand_rows(A::MatNum, cmsk::BitVector) -> MatNum

Write a block whose **rows** are the asset axis back into a `NaN` frame of the full width.

This is [`expand_columns`](@ref) along the other axis. A loadings matrix is `assets × factors`, so it expands along its rows.

# Arguments

  - `A`: The block that the reduced fit produced, `assets × anything`.
  - `cmsk`: The Coverage Universe, or `nothing`.

# Returns

  - The block on the full asset universe, carrying `NaN` outside the Coverage Universe.

# Related

  - [`expand_columns`](@ref)
  - [`expand_moment`](@ref)
  - [`expand_regression`](@ref)
"""
function expand_rows(A::MatNum, ::Nothing)
    return A
end
function expand_rows(A::MatNum, cmsk::BitVector)
    frame = coverage_nan_frame(A, (length(cmsk), size(A, 2)))
    frame[cmsk, :] = A
    return frame
end
"""
    expand_vector(v::VecNum, cmsk::Nothing) -> v
    expand_vector(v::VecNum, cmsk::BitVector) -> VecNum

Write a per-asset vector back into a `NaN` frame of the full width.

This is [`expand_columns`](@ref) over one axis alone. A mean vector, an intercept vector and a per-asset idiosyncratic variance all take it.

# Arguments

  - `v`: The per-asset vector that the reduced fit produced.
  - `cmsk`: The Coverage Universe, or `nothing`.

# Returns

  - The vector on the full asset universe, carrying `NaN` outside the Coverage Universe.

# Related

  - [`expand_columns`](@ref)
  - [`expand_moment`](@ref)
  - [`expand_idiosyncratic_covariance`](@ref)
"""
function expand_vector(v::VecNum, ::Nothing)
    return v
end
function expand_vector(v::VecNum, cmsk::BitVector)
    frame = coverage_nan_frame(v, (length(cmsk),))
    frame[cmsk] = v
    return frame
end
"""
    expand_regression(re, cmsk::Nothing) -> re
    expand_regression(re::Regression, cmsk::BitVector) -> Regression

Write a regression result fitted on the Coverage Universe back onto the full asset universe.

A Prior Result has no mask field, so **every** block it carries lives on the full asset universe, and the regression result it carries is one of those blocks. Its loadings, its intercepts and its idiosyncratic covariance are all per-asset, so all three expand along their asset axis.

`L` is read with `getfield`, as [`port_opt_view`](@ref) reads it: the `swap(L, M)` property rule makes `re.L` return `re.M` where `L` is unset, so an expanded result would materialise `L` as a copy of `M` and lose the unset-ness that the rule exists to express.

# Arguments

  - `re`: The regression result the reduced fit produced.
  - `cmsk`: The Coverage Universe, or `nothing`.

# Returns

  - `re::Regression`: The regression result on the full asset universe.

# Related

  - [`Regression`](@ref)
  - [`port_opt_view`](@ref)
  - [`expand_rows`](@ref)
  - [`expand_moment`](@ref)
"""
function expand_regression(re, ::Nothing)
    return re
end
function expand_regression(re::Regression, cmsk::BitVector)
    L = getfield(re, :L)
    b = getfield(re, :b)
    return Regression(; M = expand_rows(re.M, cmsk),
                      L = isnothing(L) ? nothing : expand_rows(L, cmsk),
                      b = isnothing(b) ? nothing : expand_vector(b, cmsk),
                      esigma = expand_idiosyncratic_covariance(re.esigma, cmsk))
end
"""
    expand_idiosyncratic_covariance(esigma::Nothing, cmsk) -> nothing
    expand_idiosyncratic_covariance(esigma::VecNum, cmsk::BitVector) -> VecNum
    expand_idiosyncratic_covariance(esigma::MatNum, cmsk::BitVector) -> MatNum

Expand the idiosyncratic covariance a regression result carries, in either of the two shapes it takes.

The block is a per-asset variance vector where the residuals are taken as uncorrelated, and an `assets × assets` matrix where they are not. This is the expansion counterpart of [`idiosyncratic_covariance_view`](@ref), which slices the same two shapes.

# Arguments

  - `esigma`: The idiosyncratic covariance, or `nothing`.
  - `cmsk`: The Coverage Universe.

# Returns

  - `esigma`: The idiosyncratic covariance on the full asset universe.

# Related

  - [`expand_regression`](@ref)
  - [`idiosyncratic_covariance_view`](@ref)
"""
function expand_idiosyncratic_covariance(::Nothing, ::BitVector)
    return nothing
end
function expand_idiosyncratic_covariance(esigma::VecNum, cmsk::BitVector)
    return expand_vector(esigma, cmsk)
end
function expand_idiosyncratic_covariance(esigma::MatNum, cmsk::BitVector)
    return expand_moment(esigma, cmsk)
end
"""
    expand_moment(m, cmsk::Nothing) -> m
    expand_moment(m, cmsk::Nothing, dims::Int) -> m
    expand_moment(m, cmsk::Nothing, ::Val{:kt}) -> m
    expand_moment(sigma::MatNum, cmsk::BitVector) -> MatNum
    expand_moment(mu::VecNum, cmsk::BitVector, dims::Int) -> VecNum
    expand_moment(mu::MatNum, cmsk::BitVector, dims::Int) -> MatNum
    expand_moment(skV::Tuple{<:MatNum, <:MatNum}, cmsk::BitVector) -> Tuple{MatNum, MatNum}
    expand_moment(kt::MatNum, cmsk::BitVector, ::Val{:kt}) -> MatNum

Write a moment estimated on the Coverage Universe back into a `NaN` frame of the full width.

A Prior Result lives on the **full** asset universe, and an asset that the prior could not estimate carries `NaN`, so every block a reduced fit produced is expanded before it is carried. This is the second half of the seam whose first half is [`coverage_reduction`](@ref), and the `nothing` sentinel returns the argument untouched.

The four shapes need four methods, because their types do not separate them: a covariance matrix and a cokurtosis matrix are both square, and a mean and a coskewness tensor are both rectangular. The arity and the `Val` marker separate them instead.

# Algorithm

The method that Julia selects is the algorithm.

 1. A `nothing` mask returns the moment untouched, at every arity.
 2. A covariance-like matrix, `assets × assets`, is written at `(cmsk, cmsk)`.
 3. A marginal, `1 × assets` under `dims = 1` and `assets × 1` under `dims = 2`, is written along its asset axis.
 4. A coskewness pair expands its tensor at the rows `cmsk` and the pair columns of [`coverage_pair_index`](@ref), and expands its negative spectral skewness matrix as a covariance-like matrix.
 5. A cokurtosis matrix is written at the pair index on both axes.

# Arguments

  - `m`, `sigma`, `mu`, `skV`, `kt`: The moment that the reduced fit produced.
  - `cmsk`: The Coverage Universe, or `nothing`.
  - `dims`: The dimension that the marginal was computed along.
  - `::Val{:kt}`: The marker that names a cokurtosis matrix, which a covariance matrix cannot be told from by type.

# Returns

  - The moment on the full asset universe, carrying `NaN` outside the Coverage Universe.

# Related

  - [`coverage_mask`](@ref)
  - [`coverage_reduction`](@ref)
  - [`coverage_pair_index`](@ref)
  - [`investable_mask`](@ref)
"""
function expand_moment(m, ::Nothing)
    return m
end
function expand_moment(m, ::Nothing, ::Int)
    return m
end
function expand_moment(m, ::Nothing, ::Val{:kt})
    return m
end
function expand_moment(sigma::MatNum, cmsk::BitVector)
    N = length(cmsk)
    frame = coverage_nan_frame(sigma, (N, N))
    frame[cmsk, cmsk] = sigma
    return frame
end
function expand_moment(mu::VecNum, cmsk::BitVector, ::Int)
    return expand_vector(mu, cmsk)
end
function expand_moment(mu::MatNum, cmsk::BitVector, dims::Int)
    N = length(cmsk)
    return if isone(dims)
        frame = coverage_nan_frame(mu, (1, N))
        frame[1, cmsk] = mu
        frame
    else
        frame = coverage_nan_frame(mu, (N, 1))
        frame[cmsk, 1] = mu
        frame
    end
end
function expand_moment(skV::Tuple{<:MatNum, <:MatNum}, cmsk::BitVector)
    sk, V = skV
    N = length(cmsk)
    frame = coverage_nan_frame(sk, (N, N^2))
    frame[cmsk, coverage_pair_index(cmsk)] = sk
    return frame, expand_moment(V, cmsk)
end
function expand_moment(kt::MatNum, cmsk::BitVector, ::Val{:kt})
    N = length(cmsk)
    idx = coverage_pair_index(cmsk)
    frame = coverage_nan_frame(kt, (N^2, N^2))
    frame[idx, idx] = kt
    return frame
end
"""
    assert_finite_sample(X::ArrNum) -> nothing

Refuse a non-finite sample in a plain moment estimator, and name the two ways out.

A plain moment estimator has no correct answer for an asset whose window carries a gap, so it refuses the whole sample rather than returning a number that reads as an estimate. The message names the two paths a caller has: fit through a prior, which reduces to the Coverage Universe and expands, or reach for a mask-aware estimator, which takes the whole window and emits its own frame.

This is [`assert_all_finite`](@ref) with the message that this refusal owes its caller. The generic message names the count and the position alone, which tells a caller what is wrong and not what to do about it.

# Arguments

  - $(arg_dict[:X])

# Validation

  - Every entry of `X` must be finite.

# Returns

  - `nothing`.

# Related

  - [`assert_all_finite`](@ref)
  - [`coverage_reduction`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
function assert_finite_sample(X::ArrNum)::Nothing
    @argcheck(all(isfinite, X),
              IsNonFiniteError("a plain moment estimator has no correct answer for a gapped sample, so it refuses one: all(isfinite, X) must hold, and got $(count(!isfinite, X)) non-finite entries, the first at $(findfirst(!isfinite, X)). Fit through a prior, which reduces to the Coverage Universe and expands, or use a mask-aware estimator, which reads the active mask of an Asset Panel."))
    return nothing
end
"""
    panel_moment_masks(pnl::Nothing) -> Tuple{Nothing, Nothing}
    panel_moment_masks(pnl::AssetPanel) -> Tuple

Read the two universe masks a mask-aware moment estimator takes, out of an optional Asset Panel.

A mask-aware estimator overrides the panel method of its verb and takes the whole window, so it needs the masks themselves rather than the Coverage Universe that [`coverage_mask`](@ref) derives. No panel, and a static panel, both carry no mask, so both give the pair of `nothing` that is the estimator's own unmasked path.

# Arguments

  - $(arg_dict[:pnl_moment])

# Returns

  - `(amsk, emsk)::Tuple`: The active mask and the estimation mask, or two `nothing`.

# Related

  - [`AssetPanel`](@ref)
  - [`coverage_mask`](@ref)
"""
function panel_moment_masks(::Nothing)
    return nothing, nothing
end
function panel_moment_masks(pnl::AssetPanel)
    return pnl.amsk, pnl.emsk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fit a moment on the Coverage Universe of an Asset Panel, and expand it to the full asset universe.

This is the covariance root, and the six verbs beside it — `cor`, `var`, `std`, `mean`, [`coskewness`](@ref) and [`cokurtosis`](@ref) — each carry the same root and refer to this one for the rule.

The Asset Panel travels as the **third positional argument** of every moment verb, as it travels as the third positional argument of [`prior`](@ref). This is the root method of each verb, and it is the reduce-and-expand of ADR 0117: it reduces `X` to its Coverage Universe, hands the clean block to the plain estimator, and writes the answer back into a `NaN` frame of the full width.

A plain estimator needs no method of its own and no declaration. A **mask-aware** estimator overrides this method and reads `pnl.amsk` itself, because it alone knows its warm-up, its freezes and its resets, and it emits its own frame over the whole window.

The cost of the rule is [`coverage_mask`](@ref)'s: one non-finite return, or one inactive row, inside the window puts the asset outside the Coverage Universe for that fit, and its entries of the answer are `NaN`.

# Algorithm

 1. Reduce `X` to its Coverage Universe with [`coverage_reduction`](@ref).
 2. Call the plain estimator on the clean block, with the caller's `dims` and keywords.
 3. Expand the answer with [`expand_moment`](@ref).

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:me])
  - $(arg_dict[:ske])
  - $(arg_dict[:kte])
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Validation

  - $(val_dict[:dims])
  - At least one asset must be in the Coverage Universe.

# Returns

  - The moment on the full asset universe, carrying `NaN` outside the Coverage Universe.

# Related

  - [`coverage_mask`](@ref)
  - [`coverage_reduction`](@ref)
  - [`expand_moment`](@ref)
  - [`AssetPanel`](@ref)
"""
function Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    cmsk, Xc = coverage_reduction(X, pnl; dims = dims)
    return expand_moment(Statistics.cov(ce, Xc; dims = dims, kwargs...), cmsk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The correlation root of the Asset Panel seam. It is the reduce-and-expand of [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref), which states the rule and its cost.

# Related

  - [`coverage_reduction`](@ref)
  - [`expand_moment`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::AbstractCovarianceEstimator, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    cmsk, Xc = coverage_reduction(X, pnl; dims = dims)
    return expand_moment(Statistics.cor(ce, Xc; dims = dims, kwargs...), cmsk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The variance root of the Asset Panel seam. It is the reduce-and-expand of [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref), which states the rule and its cost. The answer is a marginal, so it expands along its asset axis alone.

# Related

  - [`coverage_reduction`](@ref)
  - [`expand_moment`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.var(ce::AbstractCovarianceEstimator, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    cmsk, Xc = coverage_reduction(X, pnl; dims = dims)
    return expand_moment(Statistics.var(ce, Xc; dims = dims, kwargs...), cmsk, dims)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The standard deviation root of the Asset Panel seam. It is the reduce-and-expand of [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref), which states the rule and its cost. The answer is a marginal, so it expands along its asset axis alone.

# Related

  - [`coverage_reduction`](@ref)
  - [`expand_moment`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.std(ce::AbstractCovarianceEstimator, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    cmsk, Xc = coverage_reduction(X, pnl; dims = dims)
    return expand_moment(Statistics.std(ce, Xc; dims = dims, kwargs...), cmsk, dims)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The expected returns root of the Asset Panel seam. It is the reduce-and-expand of [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref), which states the rule and its cost. The answer is a marginal, so it expands along its asset axis alone.

# Related

  - [`coverage_reduction`](@ref)
  - [`expand_moment`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.mean(me::AbstractExpectedReturnsEstimator, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    cmsk, Xc = coverage_reduction(X, pnl; dims = dims)
    return expand_moment(Statistics.mean(me, Xc; dims = dims, kwargs...), cmsk, dims)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The coskewness root of the Asset Panel seam. It is the reduce-and-expand of [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref), which states the rule and its cost. The answer is a pair, so the tensor expands at the pair index and the negative spectral skewness matrix expands as a covariance-like matrix.

# Related

  - [`coverage_reduction`](@ref)
  - [`expand_moment`](@ref)
  - [`coverage_pair_index`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function coskewness(ske::CoskewnessEstimator, X::MatNum, pnl::Option{<:AssetPanel};
                    dims::Int = 1, kwargs...)
    cmsk, Xc = coverage_reduction(X, pnl; dims = dims)
    return expand_moment(coskewness(ske, Xc; dims = dims, kwargs...), cmsk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The cokurtosis root of the Asset Panel seam. It is the reduce-and-expand of [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref), which states the rule and its cost. The answer is `assets² × assets²`, so it expands at the pair index on both axes.

# Related

  - [`coverage_reduction`](@ref)
  - [`expand_moment`](@ref)
  - [`coverage_pair_index`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function cokurtosis(kte::CokurtosisEstimator, X::MatNum, pnl::Option{<:AssetPanel};
                    dims::Int = 1, kwargs...)
    cmsk, Xc = coverage_reduction(X, pnl; dims = dims)
    return expand_moment(cokurtosis(kte, Xc; dims = dims, kwargs...), cmsk, Val(:kt))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The point-in-time series root of the Asset Panel seam.

[`variance_series`](@ref) refits its estimator once per observation, so every window meets the refusal of [`assert_finite_sample`](@ref) on its own. A window is therefore reduced to **its own** Coverage Universe rather than to the sample's: an asset that lists inside the window is outside the Coverage Universe of every window that reaches back past its listing, and inside the Coverage Universe of none. That is one reduction per row, and it is what a point-in-time series of a gapped panel means.

A **mask-aware** estimator overrides this method and takes the whole window, as it overrides [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref).

# Algorithm

 1. Orient `X` so that the observations lie on the rows.
 2. Allocate a `NaN` frame of the shape of `X` with [`coverage_nan_frame`](@ref).
 3. For each observation `t`, take the assets whose window is finite throughout and, when the Asset Panel carries an active mask, active throughout.
 4. Fit the estimator on that block, and write the answer into the covered entries of row `t`. An asset outside the window's Coverage Universe keeps its `NaN`, and a window that covers no asset leaves the whole row at `NaN`.
 5. Return the series, transposed when `dims == 2`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Validation

  - $(val_dict[:dims])

# Returns

  - `val::Matrix{<:Number}`: Variance series on the full asset universe, shaped as `(T, N)` if `dims == 1` or `(N, T)` if `dims == 2`, carrying `NaN` outside each window's Coverage Universe.

# Related

  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`coverage_nan_frame`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
  - [`AssetPanel`](@ref)
"""
function variance_series(ce::AbstractCovarianceEstimator, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    amsk, _ = panel_moment_masks(pnl)
    val = coverage_nan_frame(X, size(X))
    for t in axes(X, 1)
        Xt = view(X, 1:t, :)
        cmsk = vec(all(isfinite, Xt; dims = 1))
        if !isnothing(amsk)
            cmsk .&= vec(all(view(amsk, 1:t, :); dims = 1))
        end
        if !any(cmsk)
            continue
        end
        val[t, cmsk] = vec(Statistics.var(ce, Xt[:, cmsk]; dims = 1, kwargs...))
    end
    return isone(dims) ? val : permutedims(val)
end
