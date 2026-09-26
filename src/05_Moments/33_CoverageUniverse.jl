"""
    coverage_mask(X::MatNum, pnl::Nothing; dims::Int = 1) -> Option{BitVector}
    coverage_mask(X::MatNum, pnl::AssetPanel{<:Any, Nothing, Nothing};
                  dims::Int = 1) -> Option{BitVector}
    coverage_mask(X::MatNum, pnl::AssetPanel; dims::Int = 1) -> Option{BitVector}

Derive the Coverage Universe of one fit, the mask that is `true` at each asset a plain moment estimator can fit.

An asset is in the Coverage Universe when its return is finite, and the active mask of the panel is `true`, at every row of the window. The method does not read the estimation mask. That mask names the assets of a cross-sectional estimate, and a moment is not a cross-sectional estimate. A static panel carries no mask, so there the rule is finiteness alone, as it is with no panel.

When every asset is covered, the method returns `nothing`, not a mask of every `true`. The caller then skips the slice and the expansion, as the `nothing` of [`investable_mask`](@ref) lets the optimiser skip its reduction and its expansion.

The rule has a cost. One non-finite return, or one inactive row, in the window puts the asset outside the Coverage Universe of that fit. A caller with a holiday in the window can fill the price gap with [`PriceGapFill`](@ref), or use a mask-aware estimator, which takes the whole window and writes its own frame.

The method scans `X` and the active mask separately, and only their asset axes must agree. A prior that reweights observations fits on the observations that its nested prior returned, and a nested prior can drop rows. So `X` and the panel can have different observation counts, and no row of one pairs with a row of the other. A pairing by position reads the wrong date, and a pairing by the last rows assumes a warm-up that no contract states. The method reads each input over its own rows, so it assumes neither. Where the two row counts differ, the result is conservative: an asset that the panel marks inactive at a row that the sample no longer holds is outside the Coverage Universe of that fit.

# Mathematical definition

```math
\\begin{align}
\\mathcal{C} &= \\left\\{ i \\in \\{1, \\ldots, N\\} : x_{t,\\,i} \\in \\mathbb{R} \\text{ for } t = 1, \\ldots, T \\text{ and } \\tilde{a}_{s,\\,i} = 1 \\text{ for } s = 1, \\ldots, T_{a} \\right\\}\\,.
\\end{align}
```

Where:

  - $(math_dict[:C_cvg_univ])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:a_tilde_ti_act])
  - $(math_dict[:T])
  - ``T_{a}``: Number of rows of the active mask, ``0`` when the panel carries no mask.
  - $(math_dict[:N])

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Scan each column of `X`. An asset stays in while its return is finite at every row.
 3. Scan each column of the active mask, where a panel carries one. An asset stays in while the mask is `true` at every row.
 4. Throw an `IsEmptyError` when no asset is covered, because a moment over no asset has no value.
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
    coverage_mask(X::MatNum, iv::MatNum, pnl::Option{<:AssetPanel};
                  dims::Int = 1) -> Option{BitVector}

Derive the Coverage Universe of one [`ImpliedVolatility`](@ref) fit, which reads an implied volatility surface beside its returns.

The estimator fits on the assets whose returns pass [`coverage_mask`](@ref), and whose implied volatilities are finite at every row of the window. An implied volatility series holds `NaN` where it has no quote, as a return series does. So one rule names an asset that the estimator cannot price and an asset that it cannot estimate, and the reduce-and-expand writes a `NaN` row and column for both.

The second input narrows the asset universe. It adds no second mask and no second Coverage Universe. The result names no entry of the implied volatility axis, and each consumer of the expanded moment reads a `NaN` of this method as it reads the `NaN` of an absent return.

The narrowing is in the reduce-and-expand of the estimator, not in the carrier that assembled `iv`. A caller of `Statistics.cov(ce, X; iv = …)` with its own surface passes through no carrier, and a nested estimator that receives the forwarded `iv` passes through none either. Both come after this reduction, so both see a block whose implied volatilities are complete.

# Mathematical definition

```math
\\begin{align}
\\mathcal{C}_{\\sigma} &= \\mathcal{C} \\cap \\left\\{ i : \\sigma_{t,\\,i} \\in \\mathbb{R} \\text{ for } t = 1, \\ldots, T_{\\sigma} \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{C}_{\\sigma}``: Coverage Universe of the implied volatility fit.
  - $(math_dict[:C_cvg_univ])
  - ``\\sigma_{t,\\,i}``: Implied volatility of asset ``i`` at observation ``t``.
  - ``T_{\\sigma}``: Number of rows of the implied volatility surface.

# Algorithm

 1. Orient `X` and `iv` to `observations × assets` with [`dims_oriented`](@ref).
 2. Check that `iv` has as many columns as `X`.
 3. Derive the Coverage Universe of `X` with [`coverage_mask`](@ref).
 4. Scan each column of `iv`. An asset stays in while its implied volatility is finite at every row.
 5. Intersect the two, and hand the result to [`coverage_sentinel`](@ref) with a refusal that names the implied volatilities.

# Arguments

  - $(arg_dict[:X])
  - `iv`: Implied volatility surface `observations × assets`.
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])
  - The asset axis of the oriented `iv` must be the asset axis of the oriented `X`.
  - The asset axis of `pnl.amsk` must be the asset axis of the oriented `X`.
  - At least one asset must be covered.

# Returns

  - `cmsk::Option{BitVector}`: `true` at every covered asset, or `nothing` when every asset is covered.

# Related

  - [`ImpliedVolatility`](@ref)
  - [`coverage_mask`](@ref)
  - [`coverage_sentinel`](@ref)
  - [`expand_moment`](@ref)
"""
function coverage_mask(X::MatNum, iv::MatNum, pnl::Option{<:AssetPanel};
                       dims::Int = 1)::Option{BitVector}
    Xo, ivo = dims_oriented(dims, X, iv)
    @argcheck(size(ivo, 2) == size(Xo, 2),
              DimensionMismatch("the returns matrix and the implied volatility surface describe the same assets, so their asset axes must be the same length, got $(size(Xo, 2)) columns of X against $(size(ivo, 2)) of iv"))
    cmsk = coverage_mask(Xo, pnl; dims = 1)
    imsk = isnothing(cmsk) ? trues(size(ivo, 2)) : copy(cmsk)
    for i in axes(ivo, 2)
        if !imsk[i]
            continue
        end
        for t in axes(ivo, 1)
            if !isfinite(ivo[t, i])
                imsk[i] = false
                break
            end
        end
    end
    return coverage_sentinel(imsk,
                             "no asset is in the Coverage Universe of this window: every asset carries a non-finite return, an inactive row of the Asset Panel, or a non-finite implied volatility, at some observation. Check that the window holds at least one asset that is listed and quoted throughout it, and whose implied volatility series is complete over it.")
end
"""
    coverage_sentinel(cmsk::BitVector) -> Option{BitVector}
    coverage_sentinel(cmsk::BitVector, msg::AbstractString) -> Option{BitVector}

Refuse an empty Coverage Universe, and collapse a complete one onto the `nothing` sentinel.

Every [`coverage_mask`](@ref) method ends with this call, so the file holds one copy of the refusal and of the sentinel. A method that reads one more input passes its own `msg`, because this function cannot know why an asset left the Coverage Universe. The implied volatility method of [`coverage_mask`](@ref) reads a surface too, and a message that names only the returns and the Asset Panel sends its caller to the wrong input.

# Algorithm

 1. Throw an `IsEmptyError` carrying `msg` when `cmsk` holds no `true`.
 2. Return `nothing` when `cmsk` holds no `false`.
 3. Return `cmsk` otherwise.

# Arguments

  - `cmsk`: The raw coverage mask, one entry per asset.
  - `msg`: The refusal message, which names every input that the caller read to derive the mask.

# Validation

  - At least one asset must be covered.

# Returns

  - `cmsk::Option{BitVector}`: `cmsk` itself, or `nothing` when every asset is covered.

# Related

  - [`coverage_mask`](@ref)
  - [`IsEmptyError`](@ref)
"""
function coverage_sentinel(cmsk::BitVector)::Option{BitVector}
    return coverage_sentinel(cmsk,
                             "no asset is in the Coverage Universe of this window: every asset carries a non-finite return, or an inactive row of the Asset Panel, at some observation. Check that the window holds at least one asset that is listed and quoted throughout it.")
end
function coverage_sentinel(cmsk::BitVector, msg::AbstractString)::Option{BitVector}
    @argcheck(any(cmsk), IsEmptyError(msg))
    return all(cmsk) ? nothing : cmsk
end
"""
    coverage_reduction(X::MatNum, pnl::Option{<:AssetPanel};
                       dims::Int = 1) -> Tuple{Option{BitVector}, MatNum}

Reduce a returns matrix to its Coverage Universe, and return the mask beside the clean block.

The reduction keeps the orientation of the caller. The plain moment verb that receives the block sees a matrix of the same shape as a sample with no Asset Panel, and its own `dims` still describes it. The `nothing` sentinel returns `X` itself, so a complete window allocates nothing.

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
    coverage_reduction(rd::AbstractReturnsResult) -> Tuple{Option{BitVector}, AbstractReturnsResult}
    coverage_reduction(X::MatNum, rd::AbstractReturnsResult) -> Tuple{Option{BitVector}, AbstractReturnsResult}
    coverage_reduction(cmsk::Nothing,
                       rd::AbstractReturnsResult) -> Tuple{Nothing, AbstractReturnsResult}
    coverage_reduction(cmsk::BitVector,
                       rd::AbstractReturnsResult) -> Tuple{BitVector, AbstractReturnsResult}

Reduce a returns carrier to the Coverage Universe of its own window.

[`coverage_reduction(X::MatNum, pnl::Option{<:AssetPanel})`](@ref) gives a moment estimator a clean block. This method gives a consumer of the whole carrier a clean carrier. [`port_opt_view`](@ref) slices `nx`, `X`, the benchmark, the implied volatility surface and the [`AssetPanel`](@ref) at the same asset index. The caller is pre-selection. [`fit_preprocessing`](@ref) over an [`AbstractAssetSelector`](@ref) reduces the carrier here, so each selector ranks the live assets alone.

The four methods choose each branch by dispatch, not by a condition, as [`coverage_reduction(opt::AbstractOptimisationEstimator, rd::ReturnsResult)`](@ref) does. The first method passes the returns matrix of the carrier as the first argument of the second. The second method derives the mask. The `nothing` method is the path where every asset is covered, and it returns the carrier unchanged. The `BitVector` method takes the view.

The split between the first two methods is the refusal of the family. A carrier whose `X` is not an `observations × assets` matrix has no asset axis to reduce. [`PredictionReturnsResult`](@ref) holds one portfolio return series, for example. Such a carrier matches no method, and the call throws a `MethodError` that names the carrier. The refusal comes before the method reads the panel. So the error names the returns matrix of a carrier that has neither the fields `nx` and `X` nor the fields `nx`, `X` and `pnl`, not a missing field.

A window with no covered asset throws an `IsEmptyError` in [`coverage_mask`](@ref), which derives the mask, so every caller gets the same refusal.

# Algorithm

 1. Pass `rd.X` as the first argument of the second method, which refuses a carrier with no asset axis.
 2. Derive the Coverage Universe of `X` and `rd.pnl` with [`coverage_mask`](@ref), with `dims = 1`, because a carrier holds its observations along the rows.
 3. Return the mask and the carrier unchanged when the mask is `nothing`.
 4. Otherwise return the mask beside a [`port_opt_view`](@ref) of the carrier at `findall(cmsk)`.

# Arguments

  - $(arg_dict[:rd])
  - $(arg_dict[:X])
  - `cmsk`: The Coverage Universe, or `nothing`.

# Validation

  - The carrier must hold an `observations × assets` returns matrix.
  - At least one asset must be in the Coverage Universe.

# Returns

  - `(cmsk, rdc)::Tuple{Option{BitVector}, AbstractReturnsResult}`: The Coverage Universe, and the carrier reduced to it.

# Related

  - [`coverage_mask`](@ref)
  - [`coverage_reduction(X::MatNum, pnl::Option{<:AssetPanel})`](@ref)
  - [`fit_preprocessing`](@ref)
  - [`port_opt_view`](@ref)
"""
function coverage_reduction(rd::AbstractReturnsResult)
    return coverage_reduction(rd.X, rd)
end
function coverage_reduction(X::MatNum, rd::AbstractReturnsResult)
    return coverage_reduction(coverage_mask(X, rd.pnl; dims = 1), rd)
end
function coverage_reduction(::Nothing, rd::AbstractReturnsResult)
    return nothing, rd
end
function coverage_reduction(cmsk::BitVector, rd::AbstractReturnsResult)
    return cmsk, port_opt_view(rd, findall(cmsk))
end
"""
    coverage_reduced_pair(A::MatNum, B::MatNum, cmsk::Nothing) -> Tuple{MatNum, MatNum}
    coverage_reduced_pair(A::MatNum, B::MatNum, cmsk::BitVector) -> Tuple{MatNum, MatNum}

Slice two `observations × assets` matrices onto the same Coverage Universe.

An estimator that reads a second per-asset panel beside its returns, as [`ImpliedVolatility`](@ref) reads an implied volatility surface, must slice both, so that the two describe the same assets. The `nothing` sentinel returns both unchanged.

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
    coverage_nan_frame(frame::AbstractArray) -> AbstractArray

Allocate the `NaN` frame that a moment expands into.

The one-argument method fills a frame that the caller allocated, after the same refusal. [`coverage_series_frame`](@ref) calls it, because the frame of a variance series takes the element type of a variance, not the element type of the sample.

The frame takes its element type from the block through `similar`, not from `eltype(A)`. [`MatNum`](@ref) admits a `JuMP` scalar as well as a number, so at inference `eltype` of a `MatNum` splits into two branches, and a moment only takes the numeric one. `similar` keeps the frame in the element type of the block and leaves no branch that no call reaches.

The element type of the block must hold `NaN`. An `Integer` and a `Rational` have no `NaN`, and a moment of a `Rational` sample stays `Rational`, so the function refuses both before it writes the frame. The function does not widen the type, because a sentinel is not an arithmetic operation, and the data sets the type. A caller with a `Rational` sample and a gap converts the sample to a floating-point type, or fills the gap so that every asset is in the Coverage Universe.

# Algorithm

 1. Allocate `frame`, an array of `sz`, of the same kind and element type as `A`. The one-argument method takes `frame` from the caller.
 2. Refuse the frame when its element type is an `Integer` or a `Rational`.
 3. Fill it with `NaN`, and return it.

# Arguments

  - `A`: The block that the caller writes into the frame.
  - `sz`: Size of the frame.
  - `frame`: A frame that the caller allocated.

# Validation

  - $(val_dict[:nan_frame])

# Returns

  - `frame::AbstractArray`: An array of size `sz`, filled with `NaN`.

# Related

  - [`expand_moment`](@ref)
"""
function coverage_nan_frame(A::AbstractArray, sz::Dims)
    return coverage_nan_frame(similar(A, sz))
end
function coverage_nan_frame(frame::AbstractArray)
    @argcheck(!(eltype(frame) <: Union{Integer, Rational}),
              ArgumentError("a moment on the full asset universe carries NaN outside the Coverage Universe, and its element type $(eltype(frame)) cannot hold NaN. Convert the sample to a floating-point type, or fill its gaps so that every asset is in the Coverage Universe."))
    fill!(frame, NaN)
    return frame
end
"""
    coverage_pair_index(cmsk::BitVector) -> Vector{Int}

Map the columns of a reduced co-moment tensor onto the columns of the full-universe one.

A coskewness tensor is `assets × assets²` and a cokurtosis matrix is `assets² × assets²`. Both index an asset pair `(a, b)` at `(a - 1) * N + b`, because the estimators build both from `kron(o, Y) ⊙ kron(Y, o)`, whose column `(a - 1) * N + b` is the elementwise product of column `b` and column `a`. The reduced tensor uses the same rule over the reduced width, with `a` outermost, so the returned index is in the order of the reduced columns.

# Mathematical definition

```math
\\begin{align}
\\mathcal{P} &= \\left( (a - 1) N + b \\right)_{a \\in \\mathcal{C},\\, b \\in \\mathcal{C}}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{P}``: Pair index, a sequence of ``\\lvert \\mathcal{C} \\rvert^{2}`` full-universe columns, ordered with ``a`` outermost and each of ``a`` and ``b`` ascending.
  - $(math_dict[:C_cvg_univ])
  - $(math_dict[:N])

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

Write a block whose columns are the asset axis back into a `NaN` frame of the full width.

The expansion has three basic functions, this one, [`expand_rows`](@ref) and [`expand_vector`](@ref). The moment verbs call them through [`expand_moment`](@ref), and a prior that carries a block of its own calls them directly. A reconstructed returns matrix is `observations × assets` and a Cholesky factor is `factors × assets`, so both expand along their columns.

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
    reduce_columns(A::MatNum, cmsk::Nothing) -> A
    reduce_columns(A::MatNum, cmsk::BitVector) -> MatNum

Take the columns of an `observations × assets` block that a mask keeps.

This is the inverse of [`expand_columns`](@ref), and the file keeps the two side by side so that they stay consistent. [`coverage_reduction`](@ref) takes the same slice for a mask that it derives itself. This function is for a caller that already holds a mask. An [`AugmentedBlackLittermanPrior`](@ref) reduces its returns by the Investable Mask of its asset prior. That mask can differ from the mask of the returns alone, because a column can be quoted over the whole window and still have no estimate.

The result is a copy, not a view. The block goes on to a regression, and the compiler cannot infer through a `SubArray` of an array whose element type is not concrete.

`nothing` is the path where every asset is kept, and it returns the block unchanged.

# Arguments

  - `A`: The block to reduce, `observations × assets`.
  - `cmsk`: The mask to keep, or `nothing`.

# Returns

  - The block over the kept columns.

# Related

  - [`expand_columns`](@ref)
  - [`coverage_reduction`](@ref)
  - [`investable_mask`](@ref)
"""
function reduce_columns(A::MatNum, ::Nothing)
    return A
end
function reduce_columns(A::MatNum, cmsk::BitVector)
    return A[:, cmsk]
end
"""
    expand_rows(A::MatNum, cmsk::Nothing) -> A
    expand_rows(A::MatNum, cmsk::BitVector) -> MatNum

Write a block whose rows are the asset axis back into a `NaN` frame of the full width.

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
    expand_count(x::Nothing, cmsk::BitVector) -> nothing
    expand_count(x::VecNum, cmsk::BitVector) -> VecNum

Write a per-asset count of a loadings block, `edof` or `ediv`, back onto the full asset universe.

A count differs from a moment at an absent asset. No observation of an asset outside the Coverage Universe entered the fit, so its count is zero. Zero is a true count, not a gap. Every number type has a zero, so an integer count expands in its own type, and a `NaN` frame needs a type that holds `NaN`.

# Arguments

  - `x`: The count on the Coverage Universe, or `nothing`.
  - `cmsk`: The Coverage Universe.

# Returns

  - `x::Option{<:VecNum}`: The count on the full asset universe, zero outside the Coverage Universe, or `nothing`.

# Related

  - [`expand_regression`](@ref)
  - [`expand_vector`](@ref)
  - [`Regression`](@ref)
"""
function expand_count(::Nothing, ::BitVector)::Nothing
    return nothing
end
function expand_count(x::VecNum, cmsk::BitVector)
    frame = similar(x, length(cmsk))
    fill!(frame, zero(eltype(x)))
    frame[cmsk] = x
    return frame
end
"""
    expand_regression(re, cmsk::Nothing) -> re
    expand_regression(re::Regression, cmsk::BitVector) -> Regression

Write a regression result fitted on the Coverage Universe back onto the full asset universe.

A Prior Result has no mask field, so every block that it carries is on the full asset universe, and its regression result is one of those blocks. The loadings, the intercepts, the idiosyncratic covariance and the counts `edof` and `ediv` of that covariance are all per asset, so each expands along its asset axis. An asset outside the Coverage Universe reads zero in each count, through [`expand_count`](@ref).

The method reads `L` with `getfield`, as [`port_opt_view`](@ref) does. The property rule `swap(L, M)` makes `re.L` return `re.M` when `L` is `nothing`. A read through the property writes a copy of `M` into the `L` of the expanded result, and the result no longer shows that `L` is unset.

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
  - [`expand_count`](@ref)
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
                      esigma = expand_idiosyncratic_covariance(re.esigma, cmsk),
                      edof = expand_count(re.edof, cmsk),
                      ediv = expand_count(re.ediv, cmsk))
end
"""
    expand_idiosyncratic_covariance(esigma::Nothing, cmsk) -> nothing
    expand_idiosyncratic_covariance(esigma::VecNum, cmsk::BitVector) -> VecNum
    expand_idiosyncratic_covariance(esigma::MatNum, cmsk::BitVector) -> MatNum

Expand the idiosyncratic covariance a regression result carries, in either of the two shapes it takes.

The block is a per-asset variance vector when the regression takes the residuals as uncorrelated, and an `assets × assets` matrix when it does not. [`idiosyncratic_covariance_view`](@ref) slices the same two shapes, and this function expands them.

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

A Prior Result is on the full asset universe, and an asset that the prior cannot estimate carries `NaN`. So the prior expands every block of a reduced fit before it stores the block. [`coverage_reduction`](@ref) is the first half of the seam, and this function is the second half. The `nothing` sentinel returns the argument unchanged.

The four shapes need four methods, and their types do not tell them apart. A covariance matrix and a cokurtosis matrix are both square, and a mean and a coskewness tensor are both rectangular. The number of arguments and the `Val` marker tell them apart.

# Mathematical definition

For ``i, a, b, c, d \\in \\mathcal{C}``,

```math
\\begin{align}
\\tilde{m}_{i} &= m_{k(i)}\\,, \\\\
\\tilde{\\mathbf{B}}_{i,\\,j} &= \\mathbf{B}_{k(i),\\,k(j)}\\,, \\\\
\\tilde{\\mathbf{S}}_{i,\\,(a - 1) N + b} &= \\mathbf{S}_{k(i),\\,(k(a) - 1) n + k(b)}\\,, \\\\
\\tilde{\\mathbf{K}}_{(a - 1) N + b,\\,(c - 1) N + d} &= \\mathbf{K}_{(k(a) - 1) n + k(b),\\,(k(c) - 1) n + k(d)}\\,,
\\end{align}
```

and every other entry of the four frames is `NaN`.

Where:

  - $(math_dict[:C_cvg_univ])
  - ``n = \\lvert \\mathcal{C} \\rvert``: Number of assets in the Coverage Universe.
  - ``k(i)``: Position of asset ``i`` in ``\\mathcal{C}`` in ascending order, so that ``k(i) \\in \\{1, \\ldots, n\\}``.
  - ``\\boldsymbol{m}``, ``\\tilde{\\boldsymbol{m}}``: A marginal, of ``n`` entries on the Coverage Universe and of ``N`` entries on the full universe.
  - ``\\mathbf{B}``, ``\\tilde{\\mathbf{B}}``: A covariance-like matrix, ``n \\times n`` and ``N \\times N``.
  - ``\\mathbf{S}``, ``\\tilde{\\mathbf{S}}``: A coskewness tensor, ``n \\times n^{2}`` and ``N \\times N^{2}``.
  - ``\\mathbf{K}``, ``\\tilde{\\mathbf{K}}``: A cokurtosis matrix, ``n^{2} \\times n^{2}`` and ``N^{2} \\times N^{2}``.
  - $(math_dict[:N])

# Algorithm

Dispatch selects one step of this list for each call.

 1. Return the moment unchanged when the mask is `nothing`, at every arity.
 2. Write a covariance-like matrix, `assets × assets`, into `frame`, a `NaN` frame of `N × N`, at `(cmsk, cmsk)`.
 3. Write a marginal into `frame` along its asset axis. A matrix marginal is `1 × assets` under `dims = 1` and `assets × 1` under `dims = 2`, and a vector marginal goes to [`expand_vector`](@ref).
 4. Write the coskewness tensor of a pair into `frame`, a `NaN` frame of `N × N²`, at the rows `cmsk` and the columns of [`coverage_pair_index`](@ref). Expand the negative spectral skewness matrix of the pair as in step 2.
 5. Write a cokurtosis matrix into `frame`, a `NaN` frame of `N² × N²`, at the pair index on both axes.

# Arguments

  - `m`, `sigma`, `mu`, `skV`, `kt`: The moment that the reduced fit produced.
  - `cmsk`: The Coverage Universe, or `nothing`.
  - `dims`: The dimension that the marginal was computed along.
  - `::Val{:kt}`: The marker that names a cokurtosis matrix, which a covariance matrix cannot be told from by type.

# Validation

  - $(val_dict[:nan_frame])

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

A plain moment estimator has no correct value for an asset whose window has a gap. So it refuses the whole sample, and does not return a number that looks like an estimate. The message names the two paths that a caller has. The caller can fit through a prior, which reduces the sample to the Coverage Universe and expands the result. Or the caller can use a mask-aware estimator, which takes the whole window and writes its own frame.

This is [`assert_all_finite`](@ref) with a message for this refusal. The message of [`assert_all_finite`](@ref) names the count and the position of the non-finite entries alone. It tells a caller what is wrong, and not what to do.

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

Read the two universe masks that a mask-aware moment estimator takes out of an optional Asset Panel.

A mask-aware estimator overrides the panel method of its verb and takes the whole window. So it needs the masks, not the Coverage Universe that [`coverage_mask`](@ref) derives. No panel and a static panel both carry no mask, so both give two `nothing`, which select the unmasked path of the estimator.

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
    coverage_series_frame(X::MatNum, pnl::Option{<:AssetPanel}) -> Tuple

Allocate the `NaN` frame of a point-in-time variance series, and read the active mask that the series pairs with `X` row by row.

Row `t` of the series is a fit on rows `1` to `t` of `X` and of the active mask, so the series pairs the two by position, and the mask must have the size of `X`. [`coverage_mask`](@ref) pairs no rows, and it lets the two observation axes differ.

A variance divides, so the frame takes [`float_if_integer`](@ref) of the element type of `X`. An integer sample gives a `Float64` frame, and a `Float32` sample gives a `Float32` frame. A `Rational` sample keeps its type, which has no `NaN`, so [`coverage_nan_frame`](@ref) refuses it.

# Algorithm

 1. Read `amsk`, the active mask of `pnl`, with [`panel_moment_masks`](@ref).
 2. Check that `amsk` has the size of `X` when the panel carries one.
 3. Allocate an array of the size of `X` with the element type [`float_if_integer`](@ref) of `eltype(X)`, and give it to [`coverage_nan_frame`](@ref), which refuses a type with no `NaN` and returns `val`, the array filled with `NaN`.

# Arguments

  - `X`: Returns matrix, already oriented to `observations × assets`.
  - $(arg_dict[:pnl_moment])

# Validation

  - The active mask of `pnl` must have the size of `X`.
  - $(val_dict[:nan_frame])

# Returns

  - `(val, amsk)::Tuple`: The `NaN` frame of the series, and the active mask or `nothing`.

# Related

  - [`coverage_variance_series`](@ref)
  - [`panel_moment_masks`](@ref)
  - [`float_if_integer`](@ref)
"""
function coverage_series_frame(X::MatNum, pnl::Option{<:AssetPanel})
    amsk, _ = panel_moment_masks(pnl)
    @argcheck(isnothing(amsk) || size(amsk) == size(X),
              DimensionMismatch("a variance series pairs row t of the active mask of the Asset Panel with row t of the returns matrix, so the two must be the same size, got $(size(X)) for X, oriented to observations × assets, against $(isnothing(amsk) ? nothing : size(amsk)) for pnl.amsk"))
    val = coverage_nan_frame(similar(X, float_if_integer(eltype(X)), size(X)))
    return val, amsk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fit a moment on the Coverage Universe of an Asset Panel, and expand it to the full asset universe.

This is the root method of `cov`. The six other verbs, `cor`, `var`, `std`, `mean`, [`coskewness`](@ref) and [`cokurtosis`](@ref), have root methods of the same shape, and their docstrings refer to this one for the rule.

The Asset Panel is the third positional argument of every moment verb, as it is of [`prior`](@ref). The root method of each verb is the reduce-and-expand. It reduces `X` to its Coverage Universe, gives the clean block to the plain estimator, and writes the result into a `NaN` frame of the full width.

A plain estimator needs no method of its own and no declaration. A mask-aware estimator overrides this method and reads `pnl.amsk` itself, because its warm-up, its freezes and its resets depend on the mask. It writes its own frame over the whole window.

The rule has the cost that [`coverage_mask`](@ref) states. One non-finite return, or one inactive row, in the window puts the asset outside the Coverage Universe of that fit, and its entries of the result are `NaN`.

# Algorithm

 1. Reduce `X` to its Coverage Universe with [`coverage_reduction`](@ref).
 2. Call the plain estimator on the clean block, with the caller's `dims` and keywords.
 3. Expand the result with [`expand_moment`](@ref).

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
  - $(val_dict[:nan_frame])

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

The variance root of the Asset Panel seam. It is the reduce-and-expand of [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref), which states the rule and its cost. The result is a marginal, so it expands along its asset axis alone.

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

The standard deviation root of the Asset Panel seam. It is the reduce-and-expand of [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref), which states the rule and its cost. The result is a marginal, so it expands along its asset axis alone.

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

The expected returns root of the Asset Panel seam. It is the reduce-and-expand of [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref), which states the rule and its cost. The result is a marginal, so it expands along its asset axis alone.

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

The coskewness root of the Asset Panel seam. It is the reduce-and-expand of [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref), which states the rule and its cost. The result is a pair, so the tensor expands at the pair index and the negative spectral skewness matrix expands as a covariance-like matrix.

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

The cokurtosis root of the Asset Panel seam. It is the reduce-and-expand of [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref), which states the rule and its cost. The result is `assets² × assets²`, so it expands at the pair index on both axes.

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

Fit a point-in-time variance series on the Coverage Universe of each expanding window, and write it into a `NaN` frame of the full width.

[`variance_series`](@ref) refits its estimator once per observation, so each window meets the refusal of [`assert_finite_sample`](@ref) on its own. Each window is reduced to its own Coverage Universe, not to the Coverage Universe of the whole sample. Every window starts at the first row, so an asset that lists after the first row is outside the Coverage Universe of every window, and its column is `NaN` throughout.

The method calls [`coverage_variance_series`](@ref) with no policy, and its `Nothing` method runs the steps. A mask-aware estimator overrides this method and takes the whole window, as it overrides [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref).

# Mathematical definition

```math
\\begin{align}
v_{t,\\,i} &= \\begin{cases} \\left[ \\hat{\\boldsymbol{\\sigma}}^{2}\\left( \\mathbf{X}_{1:t,\\,\\mathcal{C}_{t}} \\right) \\right]_{i} & i \\in \\mathcal{C}_{t} \\\\ \\mathrm{NaN} & i \\notin \\mathcal{C}_{t} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``v_{t,\\,i}``: Entry of the series for asset ``i`` at observation ``t``.
  - ``\\mathcal{C}_{t}``: Coverage Universe of the window of rows ``1`` to ``t``, the assets whose returns are finite, and whose active mask entries are ``1``, at each of those rows.
  - ``\\mathbf{X}_{1:t,\\,\\mathcal{C}_{t}}``: Rows ``1`` to ``t`` of the returns matrix, over the columns of ``\\mathcal{C}_{t}``.
  - ``\\hat{\\boldsymbol{\\sigma}}^{2}(\\cdot)``: Variance vector that the estimator fits on a block, indexed by asset.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Validation

  - $(val_dict[:dims])
  - $(val_dict[:nan_frame]) The series writes its frame before its first row, so it needs this type for every sample, with or without a gap.
  - The active mask of `pnl` must have the size of the oriented `X`.

# Returns

  - `val::Matrix{<:Number}`: Variance series on the full asset universe, shaped as `(T, N)` if `dims == 1` or `(N, T)` if `dims == 2`, carrying `NaN` outside each window's Coverage Universe.

# Related

  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`coverage_variance_series`](@ref)
  - [`coverage_series_frame`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
  - [`AssetPanel`](@ref)
"""
function variance_series(ce::AbstractCovarianceEstimator, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    return coverage_variance_series(ce, nothing, X, pnl; dims = dims, kwargs...)
end
"""
    coverage_panel_moment(f, est, cvg, X, pnl, expand; dims::Int = 1, kwargs...)

Route the Asset Panel method of a moment verb to the Coverage Universe seam or to the available-case seam.

An estimator that carries a [`CoveragePolicy`](@ref) is a mask-aware estimator, in the sense of the root method of `cov` in this file. It takes the whole window and reads `pnl.amsk` itself. So the panel method must give it the mask, and must not reduce the window to its Coverage Universe. The caller passes the `cvg` field as the third argument, so dispatch chooses the seam. The `Nothing` method is the reduce-and-expand of the root methods.

`expand` is the framing that the verb needs, because the four moment shapes frame differently. A covariance frames as a matrix, a marginal along its asset axis alone, and a co-moment at the pair index. The verb passes the framing, so the file holds one copy of this routing for every verb.

# Arguments

  - `f`: The plain, panel-free method of the verb.
  - `est`: The estimator.
  - `cvg`: The policy the estimator carries, which selects the seam.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - `expand`: The framing, called as `expand(val, cmsk)` on the Coverage Universe seam and unused on the other.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - `val`: The moment on the full asset universe.

# Related

  - [`CoveragePolicy`](@ref)
  - [`coverage_reduction`](@ref)
  - [`expand_moment`](@ref)
  - [`panel_moment_masks`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function coverage_panel_moment end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`Nothing` method of [`coverage_panel_moment`](@ref), the Coverage Universe seam. It reduces the window with [`coverage_reduction`](@ref), fits the plain estimator on the clean block, and frames the result with `expand`.

# Related

  - [`coverage_panel_moment`](@ref)
  - [`coverage_reduction`](@ref)
"""
function coverage_panel_moment(f::F, est, ::Nothing, X::MatNum, pnl::Option{<:AssetPanel},
                               expand::E; dims::Int = 1, kwargs...) where {F, E}
    cmsk, Xc = coverage_reduction(X, pnl; dims = dims)
    return expand(f(est, Xc; dims = dims, kwargs...), cmsk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`CoveragePolicy`](@ref) method of [`coverage_panel_moment`](@ref), the available-case seam. It gives the estimator the whole window and the active mask of the panel, turned to the orientation of `X`, and the estimator writes its own frame.

# Related

  - [`coverage_panel_moment`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`panel_moment_masks`](@ref)
"""
function coverage_panel_moment(f::F, est, ::CoveragePolicy, X::MatNum,
                               pnl::Option{<:AssetPanel}, ::E; dims::Int = 1,
                               kwargs...) where {F, E}
    amsk, _ = dims_oriented(dims, panel_moment_masks(pnl)...)
    return f(est, X; dims = dims, active_mask = amsk, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Asset Panel method of `Statistics.mean` for a [`SimpleExpectedReturns`](@ref), which routes on its `cvg` field with [`coverage_panel_moment`](@ref).

# Related

  - [`coverage_panel_moment`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function Statistics.mean(me::SimpleExpectedReturns, X::MatNum, pnl::Option{<:AssetPanel};
                         dims::Int = 1, kwargs...)
    return coverage_panel_moment(Statistics.mean, me, me.cvg, X, pnl,
                                 (m, cmsk) -> expand_moment(m, cmsk, dims); dims = dims,
                                 kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Asset Panel method of `Statistics.var` for a [`SimpleVariance`](@ref), which routes on its `cvg` field with [`coverage_panel_moment`](@ref).

# Related

  - [`coverage_panel_moment`](@ref)
  - [`SimpleVariance`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function Statistics.var(ve::SimpleVariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    return coverage_panel_moment(Statistics.var, ve, ve.cvg, X, pnl,
                                 (m, cmsk) -> expand_moment(m, cmsk, dims); dims = dims,
                                 kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Asset Panel method of `Statistics.std` for a [`SimpleVariance`](@ref), which routes on its `cvg` field with [`coverage_panel_moment`](@ref).

# Related

  - [`coverage_panel_moment`](@ref)
  - [`SimpleVariance`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function Statistics.std(ve::SimpleVariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    return coverage_panel_moment(Statistics.std, ve, ve.cvg, X, pnl,
                                 (m, cmsk) -> expand_moment(m, cmsk, dims); dims = dims,
                                 kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Asset Panel method of `Statistics.cov` for a [`Covariance`](@ref), which routes on its `cvg` field with [`coverage_panel_moment`](@ref).

# Related

  - [`coverage_panel_moment`](@ref)
  - [`Covariance`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function Statistics.cov(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1,
                        kwargs...)
    return coverage_panel_moment(Statistics.cov, ce, ce.cvg, X, pnl, expand_moment;
                                 dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Asset Panel method of `Statistics.cor` for a [`Covariance`](@ref), which routes on its `cvg` field with [`coverage_panel_moment`](@ref).

# Related

  - [`coverage_panel_moment`](@ref)
  - [`Covariance`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function Statistics.cor(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1,
                        kwargs...)
    return coverage_panel_moment(Statistics.cor, ce, ce.cvg, X, pnl, expand_moment;
                                 dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Asset Panel method of `Statistics.var` for a [`Covariance`](@ref), which routes on its `cvg` field with [`coverage_panel_moment`](@ref). The result is a marginal, so it frames along its asset axis alone.

# Related

  - [`coverage_panel_moment`](@ref)
  - [`Covariance`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function Statistics.var(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1,
                        kwargs...)
    return coverage_panel_moment(Statistics.var, ce, ce.cvg, X, pnl,
                                 (m, cmsk) -> expand_moment(m, cmsk, dims); dims = dims,
                                 kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Asset Panel method of `Statistics.std` for a [`Covariance`](@ref), which routes on its `cvg` field with [`coverage_panel_moment`](@ref). The result is a marginal, so it frames along its asset axis alone.

# Related

  - [`coverage_panel_moment`](@ref)
  - [`Covariance`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function Statistics.std(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1,
                        kwargs...)
    return coverage_panel_moment(Statistics.std, ce, ce.cvg, X, pnl,
                                 (m, cmsk) -> expand_moment(m, cmsk, dims); dims = dims,
                                 kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Asset Panel method of [`coskewness`](@ref) for a [`Coskewness`](@ref), which routes on its `cvg` field with [`coverage_panel_moment`](@ref). The result is a pair, so the Coverage Universe seam frames the tensor at the pair index and the negative spectral skewness matrix as a covariance-like matrix.

# Related

  - [`coverage_panel_moment`](@ref)
  - [`Coskewness`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function coskewness(ske::Coskewness, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1,
                    kwargs...)
    return coverage_panel_moment(coskewness, ske, ske.cvg, X, pnl, expand_moment;
                                 dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Asset Panel method of [`cokurtosis`](@ref) for a [`Cokurtosis`](@ref), which routes on its `cvg` field with [`coverage_panel_moment`](@ref). The result is `assets² × assets²`, so the Coverage Universe seam frames it at the pair index on both axes.

# Related

  - [`coverage_panel_moment`](@ref)
  - [`Cokurtosis`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function cokurtosis(kte::Cokurtosis, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1,
                    kwargs...)
    return coverage_panel_moment(cokurtosis, kte, kte.cvg, X, pnl,
                                 (m, cmsk) -> expand_moment(m, cmsk, Val(:kt)); dims = dims,
                                 kwargs...)
end
"""
    coverage_variance_series(ce, cvg, X, pnl; dims::Int = 1, kwargs...) -> MatNum

Route a point-in-time variance series to the Coverage Universe seam or to the available-case seam.

This is the series counterpart of [`coverage_panel_moment`](@ref). It is a separate function because the series refits the estimator at each observation and writes the frame one row at a time, not once at the end.

# Arguments

  - $(arg_dict[:ce])
  - `cvg`: The policy the estimator carries, which selects the seam.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Validation

  - $(val_dict[:dims])
  - The active mask of `pnl` must have the size of the oriented `X`.

# Returns

  - `val::Matrix{<:Number}`: Variance series on the full asset universe.

# Related

  - [`coverage_panel_moment`](@ref)
  - [`variance_series`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function coverage_variance_series end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`Nothing` method of [`coverage_variance_series`](@ref), the Coverage Universe seam.

It reduces each window to its own Coverage Universe, and fits the estimator on that block. [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref) states the rule and its mathematics.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Allocate `val`, the `NaN` frame, and read `amsk`, the active mask, with [`coverage_series_frame`](@ref).
 3. For each observation `t`, take `Xt`, rows `1` to `t` of `X`.
 4. Set `cmsk` to the assets whose returns are finite at every row of `Xt` and, when `amsk` is not `nothing`, whose entries of `amsk` are `true` at every one of those rows.
 5. Skip row `t` when `cmsk` holds no asset, so the whole row keeps its `NaN`.
 6. Otherwise fit the variance of the estimator on the columns `cmsk` of `Xt`, and write it into the entries `cmsk` of row `t` of `val`.
 7. Return `val`, transposed when `dims == 2`.

# Related

  - [`coverage_variance_series`](@ref)
  - [`coverage_series_frame`](@ref)
  - [`variance_series`](@ref)
"""
function coverage_variance_series(ce::AbstractCovarianceEstimator, ::Nothing, X::MatNum,
                                  pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    val, amsk = coverage_series_frame(X, pnl)
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`CoveragePolicy`](@ref) method of [`coverage_variance_series`](@ref), the available-case seam.

It fits each window over the whole universe, and does not reduce it to its own Coverage Universe. An asset that lists after the first row therefore carries a number once the estimator admits it, where the Coverage Universe seam leaves its column at `NaN`.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Allocate `val`, the `NaN` frame, and read `amsk`, the active mask, with [`coverage_series_frame`](@ref). The active mask of an Asset Panel is `observations × assets` already.
 3. For each observation `t`, fit the variance of the estimator on rows `1` to `t` of `X`, with rows `1` to `t` of `amsk` as its active mask, and write it into row `t` of `val`.
 4. Return `val`, transposed when `dims == 2`.

# Related

  - [`coverage_variance_series`](@ref)
  - [`coverage_series_frame`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function coverage_variance_series(ce::AbstractCovarianceEstimator, ::CoveragePolicy,
                                  X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1,
                                  kwargs...)
    X = dims_oriented(dims, X)
    val, amsk = coverage_series_frame(X, pnl)
    for t in axes(X, 1)
        val[t, :] = vec(Statistics.var(ce, view(X, 1:t, :); dims = 1,
                                       active_mask = if isnothing(amsk)
                                           nothing
                                       else
                                           view(amsk, 1:t, :)
                                       end, kwargs...))
    end
    return isone(dims) ? val : permutedims(val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Asset Panel method of [`variance_series`](@ref) for a [`Covariance`](@ref), which routes on its `cvg` field with [`coverage_variance_series`](@ref).

# Related

  - [`coverage_variance_series`](@ref)
  - [`Covariance`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function variance_series(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel};
                         dims::Int = 1, kwargs...)
    return coverage_variance_series(ce, ce.cvg, X, pnl; dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Asset Panel method of [`variance_series`](@ref) for a [`SimpleVariance`](@ref), which routes on its `cvg` field with [`coverage_variance_series`](@ref).

# Related

  - [`coverage_variance_series`](@ref)
  - [`SimpleVariance`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function variance_series(ve::SimpleVariance, X::MatNum, pnl::Option{<:AssetPanel};
                         dims::Int = 1, kwargs...)
    return coverage_variance_series(ve, ve.cvg, X, pnl; dims = dims, kwargs...)
end
"""
    coverage_floor(cvg::Nothing)
    coverage_floor(cvg::CoveragePolicy)
    coverage_floor(me::AbstractExpectedReturnsEstimator)
    coverage_floor(ce::StatsBase.CovarianceEstimator)
    coverage_floor(me::SimpleExpectedReturns)
    coverage_floor(ve::SimpleVariance)
    coverage_floor(ce::Covariance)
    coverage_floor(ce::PortfolioOptimisersCovariance)
    coverage_floor(a::Nothing, b::Nothing)
    coverage_floor(a::Nothing, b::Real)
    coverage_floor(a::Real, b::Nothing)
    coverage_floor(a::Real, b::Real)

Read the coverage floor at which an estimator admits an asset, and take the tighter floor of two arms.

A [`CoveragePolicy`](@ref) is a field of one estimator, so a fit has no single floor until a caller reads the floor of each arm. The one-argument methods do that read. An estimator that carries a policy returns its own `cvg`, a wrapper forwards to the estimator it holds, and every other estimator returns `nothing`. `nothing` means that the arm states no floor, not a floor of zero. The two-argument methods take the maximum of two arms, because an asset must pass both arms. The Investable Mask needs `mu` and the diagonal of `sigma` finite, and [`coverage_admission`](@ref) reads the same per-asset count for both. So a floor on either arm bounds every investable column, whatever the other arm does.

[`resolve_fill_limit`](@ref) reads the floor, and turns it into the share of a [`scenario_fill`](@ref) that fills without a warning. An estimator with no method of its own returns `nothing` through the method of its root type. So a policy inside a wrapper that this function does not forward through gives a warning for every fill, not for none, which is the conservative result.

The two-argument methods choose by dispatch, not by a branch, so a call site that holds two `Option` values finds a method for each member of its union split.

# Arguments

  - `cvg`: A [`CoveragePolicy`](@ref) or `nothing`.
  - `me`, `ve`, `ce`: The moment estimator whose floor the method reads.
  - `a`, `b`: The floors of two arms, each a `Real` or `nothing`.

# Returns

  - `floor::Option{<:Real}`: The coverage floor, or `nothing` when none is stated.

# Related

  - [`CoveragePolicy`](@ref)
  - [`admits`](@ref)
  - [`coverage_admission`](@ref)
  - [`resolve_fill_limit`](@ref)
  - [`scenario_fill`](@ref)
"""
function coverage_floor(::Nothing)
    return nothing
end
function coverage_floor(cvg::CoveragePolicy)
    return cvg.min_coverage
end
function coverage_floor(::AbstractExpectedReturnsEstimator)
    return nothing
end
function coverage_floor(::StatsBase.CovarianceEstimator)
    return nothing
end
function coverage_floor(me::SimpleExpectedReturns)
    return coverage_floor(me.cvg)
end
function coverage_floor(ve::SimpleVariance)
    return coverage_floor(ve.cvg)
end
function coverage_floor(ce::Covariance)
    return coverage_floor(ce.cvg)
end
function coverage_floor(ce::PortfolioOptimisersCovariance)
    return coverage_floor(ce.ce)
end
function coverage_floor(::Nothing, ::Nothing)
    return nothing
end
function coverage_floor(::Nothing, b::Real)
    return b
end
function coverage_floor(a::Real, ::Nothing)
    return a
end
function coverage_floor(a::Real, b::Real)
    return max(a, b)
end
