"""
$(DocStringExtensions.TYPEDEF)

Holds the coordinates and the order of the last fit of a [`SpectralSeriation`](@ref), so that the next fit keeps its orientation.

The online arm of a walk-forward writes it into the `cache` field of the member after each fold, see [`carry_fold_state`](@ref). The next fit projects these coordinates onto its own eigenspace, and breaks a tie by this order. So the order does not reverse from one fold to the next. The memory changes the turnover little, because the recursive bisection of a reversed order gives almost the same weights. The memory is not a sufficient statistic of the observations, so [`merge_states`](@ref) refuses two states.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SpectralSeriationState(;
        coordinates::VecNum,
        order::AbstractVector{<:Integer}
    ) -> SpectralSeriationState

Keywords correspond to the struct's fields.

## Validation

  - Every entry of `order` is an index of `coordinates`, and no entry repeats. The constructor throws an `ArgumentError` otherwise.
  - The entries of `coordinates` at `order` are finite, and the others are `NaN`. The constructor throws an `ArgumentError` otherwise.

# Related

  - [`SpectralSeriation`](@ref)
  - [`SpectralSeriationResult`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct SpectralSeriationState <: AbstractPartialFitState
    """
    The coordinate of each asset of the universe, `NaN` for an asset that the last fit did not order, so that an asset can leave and come back.
    """
    coordinates
    """
    The assets that the last fit ordered, in their order, as indices of the universe.
    """
    order
    function SpectralSeriationState(coordinates::VecNum, order::AbstractVector{<:Integer})
        @argcheck(allunique(order) && all(in(eachindex(coordinates)), order),
                  ArgumentError("`SpectralSeriationState.order` is $order, which does not name distinct assets of the $(length(coordinates)) coordinates."))
        kept = falses(length(coordinates))
        kept[order] .= true
        @argcheck(all(isfinite, view(coordinates, kept)) &&
                  all(isnan, view(coordinates, .!kept)),
                  ArgumentError("`SpectralSeriationState.coordinates` must be finite at the assets of `order` and `NaN` at the others."))
        return new{typeof(coordinates), typeof(order)}(coordinates, order)
    end
end
function SpectralSeriationState(; coordinates::VecNum,
                                order::AbstractVector{<:Integer})::SpectralSeriationState
    return SpectralSeriationState(coordinates, order)
end
function Base.copy(x::SpectralSeriationState)
    return SpectralSeriationState(copy(x.coordinates), copy(x.order))
end
function merge_states(::SpectralSeriationState, ::SpectralSeriationState)
    return throw(ArgumentError("a `SpectralSeriationState` is the orientation of the last fit, not a sufficient statistic of the observations, so two states do not merge. Fold the rows in order instead."))
end
"""
    port_opt_view(x::SpectralSeriationState, i, args...) -> SpectralSeriationState

Keep the assets `i` of a spectral seriation state.

The coordinates are those of the assets `i`, and the order keeps the assets of `i` in their order, each numbered by its position in `i`.

# Arguments

  - `x`: The state.
  - `i`: The indices of the assets that the view keeps.
  - `args...`: Not read.

# Returns

  - `x::SpectralSeriationState`: The state over the assets `i`.

# Related

  - [`SpectralSeriationState`](@ref)
  - [`seriation_view_order`](@ref)
"""
function port_opt_view(x::SpectralSeriationState, i, args...)
    return SpectralSeriationState(x.coordinates[i],
                                  seriation_view_order(x.order, i, length(x.coordinates)))
end
"""
$(DocStringExtensions.TYPEDEF)

Orders the assets by their coordinates in the leading eigenspace of a Laplacian of their distances, with no tree.

This is the spectral seriation of [atkins1998](@cite). Two assets with a small distance get close coordinates, and the order sorts the coordinates. A seriation from a tree can move a whole branch when a small change of the distances changes a merge. The spectral order reads the whole distance matrix, so a small change of the distances moves the coordinates a little. With its memory in `cache`, the order keeps its orientation and its ties from fold to fold. Neither property bounds the turnover of the weights: the order changes where two coordinates cross.

# Mathematical definition

```math
\\begin{align}
B_{ij} &= \\frac{\\max \\mathbf{S} - S_{ij}}{\\max \\mathbf{S} - \\min \\mathbf{S}}\\,,\\\\
\\mathbf{L} &= \\operatorname{diag}(\\mathbf{B} \\mathbf{1}) - \\mathbf{B}\\,,\\\\
\\mathbf{L} \\mathbf{U} &= \\lambda_1 \\mathbf{U}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{S}``: The similarity matrix of the distances, from `sim`, through [`distance_to_similarity`](@ref).
  - ``\\mathbf{B}``: The dissimilarity, scaled to ``[0, 1]``, with a zero diagonal.
  - ``\\mathbf{L}``: The Laplacian of ``\\mathbf{B}``.
  - ``\\lambda_1``: The largest eigenvalue of ``\\mathbf{L}``.
  - ``\\mathbf{U}``: An orthonormal basis of the eigenspace of ``\\lambda_1``, the eigenvalues within ``64 \\epsilon N \\lambda_1`` of it, with ``\\epsilon`` the machine epsilon and ``N`` the number of assets.

The constant vector is in the null space of ``\\mathbf{L}``, so ``\\mathbf{U}`` is orthogonal to it. The eigenspace of ``\\lambda_1`` is the Fiedler eigenspace of the Laplacian of the affinity ``\\mathbf{1}\\mathbf{1}^\\intercal - \\mathbf{B}``. With the default `sim`, [`MaximumDistanceSimilarity`](@ref), ``\\mathbf{B} = (\\mathbf{D} / \\max \\mathbf{D})^{\\circ 2}``. When every distance is equal, ``\\mathbf{B} = \\mathbf{0}``, and ``\\mathbf{U}`` is the Helmert basis of the subspace orthogonal to the constant vector. [`spectral_seriation`](@ref) states how the coordinate is chosen in ``\\mathbf{U}`` and how ties are broken.

!!! warning

    A multiplicity above one, or a graph of ``\\mathbf{1}\\mathbf{1}^\\intercal - \\mathbf{B}`` that is not connected, gives an eigenspace of more than one dimension. The order then depends on the anchor or on the memory of the last fit, and a small change of the distances can split the eigenvalue and change the order. The result reports the multiplicity and the spectral gap, and the fit gives no warning.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SpectralSeriation(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
        sim::AbstractSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
        x_src::Option{<:AbstractReturnsSource} = nothing,
        cache::Option{<:SpectralSeriationState} = nothing
    ) -> SpectralSeriation

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:ce_cov])
  - `x_src in ALL_RETURNS_SOURCES`. [`assert_returns_source`](@ref) throws an `ArgumentError` otherwise.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ce`: Recursively updated through [`factory`](@ref).
  - `de`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> SpectralSeriation()
SpectralSeriation
     ce ┼ PortfolioOptimisersCovariance
        │   ce ┼ Covariance
        │      │    me ┼ SimpleExpectedReturns
        │      │       │   w ┴ nothing
        │      │    ce ┼ GeneralCovariance
        │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
        │      │       │    w ┴ nothing
        │      │   alg ┼ FullMoment()
        │      │     w ┴ nothing
        │   mp ┼ MatrixProcessing
        │      │     pdm ┼ Posdef
        │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │      │      dn ┼ nothing
        │      │      dt ┼ nothing
        │      │     alg ┼ nothing
        │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
     de ┼ Distance
        │   power ┼ nothing
        │     alg ┴ CanonicalDistance()
    sim ┼ MaximumDistanceSimilarity()
  x_src ┼ nothing
  cache ┴ nothing
```

# Related

  - [`AbstractSeriationEstimator`](@ref)
  - [`SpectralSeriationResult`](@ref)
  - [`SpectralSeriationState`](@ref)
  - [`HierarchicalSeriation`](@ref)
  - [`spectral_seriation`](@ref)
  - [`seriate`](@ref)

# References

  - $(ref_dict[:atkins1998])
"""
@propagatable @concrete struct SpectralSeriation <: AbstractSeriationEstimator
    """
    $(field_dict[:ce])
    """
    @fprop ce
    """
    $(field_dict[:de])
    """
    @fprop de
    """
    The similarity matrix algorithm that maps the distances to the similarities whose Laplacian gives the coordinates.
    """
    sim
    """
    $(field_dict[:x_src_graph])
    """
    x_src
    """
    The memory of the last fit, or `nothing` for a fresh fit.
    """
    @vprop cache
    function SpectralSeriation(ce::StatsBase.CovarianceEstimator,
                               de::AbstractDistanceEstimator,
                               sim::AbstractSimilarityMatrixAlgorithm,
                               x_src::Option{<:AbstractReturnsSource},
                               cache::Option{<:SpectralSeriationState})
        assert_covariance_estimator(ce, :ce)
        assert_returns_source(x_src, ALL_RETURNS_SOURCES, :SpectralSeriation)
        return new{typeof(ce), typeof(de), typeof(sim), typeof(x_src), typeof(cache)}(ce,
                                                                                      de,
                                                                                      sim,
                                                                                      x_src,
                                                                                      cache)
    end
end
function SpectralSeriation(;
                           ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                           de::AbstractDistanceEstimator = Distance(;
                                                                    alg = CanonicalDistance()),
                           sim::AbstractSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
                           x_src::Option{<:AbstractReturnsSource} = nothing,
                           cache::Option{<:SpectralSeriationState} = nothing)::SpectralSeriation
    return SpectralSeriation(ce, de, sim, x_src, cache)
end
function returns_source(ser::SpectralSeriation)
    return ser.x_src
end
function reader_frames(ser::SpectralSeriation, path::AbstractString, ::Bool)
    return source_frames(ser.x_src, path, true)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the order of a [`SpectralSeriation`](@ref) and the diagnostics of its eigenspace.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SpectralSeriationResult(;
        order::AbstractVector{<:Integer},
        coordinates::VecNum,
        multiplicity::Integer,
        gap::Number
    ) -> SpectralSeriationResult

Keywords correspond to the struct's fields.

## Validation

  - `isperm(order)` and `length(order) == length(coordinates)`. The constructor throws an `ArgumentError` or a `DimensionMismatch` otherwise.
  - `multiplicity >= 0`. The constructor throws a `DomainError` otherwise.

# Related

  - [`AbstractSeriationResult`](@ref)
  - [`SpectralSeriation`](@ref)
  - [`SpectralSeriationState`](@ref)
"""
@concrete struct SpectralSeriationResult <: AbstractSeriationResult
    """
    $(field_dict[:ser_order])
    """
    order
    """
    The coordinate of each asset in the eigenspace. The vector has unit norm when the fit orders two assets or more, and it is zero for one asset.
    """
    coordinates
    """
    The dimension of the eigenspace of the largest eigenvalue, zero for fewer than two assets.
    """
    multiplicity
    """
    The relative separation ``(\\lambda_1 - \\lambda_2) / \\lambda_1`` of the two largest eigenvalues, zero when they are tied, and `NaN` for fewer than three assets or for equal distances.
    """
    gap
    function SpectralSeriationResult(order::AbstractVector{<:Integer}, coordinates::VecNum,
                                     multiplicity::Integer, gap::Number)
        @argcheck(isperm(order),
                  ArgumentError("`SpectralSeriationResult.order` is $order, which is not a permutation of `1:$(length(order))`."))
        @argcheck(length(order) == length(coordinates), DimensionMismatch)
        @argcheck(multiplicity >= 0,
                  DomainError(multiplicity,
                              "`SpectralSeriationResult.multiplicity` is the dimension of an eigenspace, so it is at least `0`."))
        return new{typeof(order), typeof(coordinates), typeof(multiplicity), typeof(gap)}(order,
                                                                                          coordinates,
                                                                                          multiplicity,
                                                                                          gap)
    end
end
function SpectralSeriationResult(; order::AbstractVector{<:Integer}, coordinates::VecNum,
                                 multiplicity::Integer,
                                 gap::Number)::SpectralSeriationResult
    return SpectralSeriationResult(order, coordinates, multiplicity, gap)
end
"""
    port_opt_view(ser::SpectralSeriationResult, i, args...) -> SpectralSeriationResult

Keep the assets `i` of a spectral order, with their coordinates. The diagnostics of the eigenspace describe the fit over every asset, and the view keeps them.

# Arguments

  - `ser`: The result.
  - `i`: The indices of the assets that the view keeps.
  - `args...`: Not read.

# Returns

  - `ser::SpectralSeriationResult`: The order of the assets `i`.

# Related

  - [`SpectralSeriationResult`](@ref)
  - [`seriation_view_order`](@ref)
"""
function port_opt_view(ser::SpectralSeriationResult, i, args...)
    return rebuild_struct(ser,
                          (; order = seriation_view_order(ser.order, i, length(ser.order)),
                           coordinates = ser.coordinates[i]))
end
"""
    helmert_basis(::Type{T}, n::Integer) -> Matrix{T}

The orthonormal Helmert basis of the subspace of ``\\mathbb{R}^n`` that is orthogonal to the constant vector, one vector per column.

Column ``k`` holds ``1`` in its first ``k`` entries and ``-k`` in entry ``k + 1``, divided by ``\\sqrt{k (k + 1)}``. This is the form of `scipy.linalg.helmert(n, full = False)`, transposed.

# Arguments

  - `T`: The element type.
  - `n`: The dimension, at least `2`.

# Returns

  - `U::Matrix{T}`: The basis, `n × (n - 1)`.

# Related

  - [`spectral_seriation`](@ref)
"""
function helmert_basis(::Type{T}, n::Integer) where {T}
    U = zeros(T, n, n - 1)
    for k in 1:(n - 1)
        s = sqrt(T(k * (k + 1)))
        U[1:k, k] .= inv(s)
        U[k + 1, k] = -k / s
    end
    return U
end
"""
    spectral_coordinate(U::MatNum, cache::Option{<:SpectralSeriationState}, tol::Number) -> Vector

Choose the coordinate of each asset in the eigenspace `U`.

# Algorithm

 1. With a memory, centre the coordinates of the memory on the assets that it holds, scale them to unit norm, and project them onto `U`. Return the projection, scaled to unit norm, when its norm is at least ``\\sqrt{\\epsilon}``.
 2. Without a memory, or when the projection is too small, and when `U` has one column: return that column, with its sign set so that its first entry of the largest magnitude, within `tol`, is not negative.
 3. Otherwise, take the anchor, the first asset with the largest diagonal of ``\\mathbf{U}\\mathbf{U}^\\intercal`` within `tol`. Return ``\\mathbf{U}\\mathbf{U}^\\intercal \\boldsymbol{e}_{a}``, scaled to unit norm. Neither rule depends on the basis of the eigenspace.

# Arguments

  - `U`: An orthonormal basis of the eigenspace, `assets × multiplicity`.
  - `cache`: The memory of the last fit, over the same assets, or `nothing`.
  - `tol`: The tolerance of a tie.

# Returns

  - `c::Vector`: The coordinate of each asset.

# Related

  - [`spectral_seriation`](@ref)
  - [`SpectralSeriationState`](@ref)
"""
function spectral_coordinate(U::MatNum, cache::Option{<:SpectralSeriationState},
                             tol::Number)
    T = eltype(U)
    c = zeros(T, size(U, 1))
    if !isnothing(cache)
        prev = cache.coordinates
        ovl = isfinite.(prev)
        if any(ovl)
            c[ovl] .= view(prev, ovl) .- Statistics.mean(view(prev, ovl))
            rn = LinearAlgebra.norm(c)
            if rn > zero(rn)
                c = U * (transpose(U) * (c / rn))
            end
        end
    end
    pn = LinearAlgebra.norm(c)
    if pn >= sqrt(eps(T))
        return c / pn
    elseif size(U, 2) == 1
        c = U[:, 1]
        mag = abs.(c)
        a = findfirst(x -> maximum(mag) - x <= tol, mag)
        return c[a] >= zero(T) ? c : -c
    end
    pd = vec(sum(abs2, U; dims = 2))
    a = findfirst(x -> maximum(pd) - x <= tol, pd)
    c = U * U[a, :]
    return c / LinearAlgebra.norm(c)
end
"""
    order_coordinates(c::VecNum, prev::Option{<:AbstractVector{<:Integer}}, tol::Number) -> Vector{Int}

Sort the coordinates, and break a tie by the order of the last fit, then by the position of the asset.

# Algorithm

 1. Rank each asset by its position in `prev`. An asset that `prev` does not hold takes a rank after every asset of `prev`, in the order of its position. Without `prev`, the rank is the position.
 2. Sort the assets by their coordinates, in a stable sort.
 3. Form each group of the assets whose coordinates are within `tol` of the first coordinate of the group, and sort each group by the rank of step 1, in a stable sort.

# Arguments

  - `c`: The coordinate of each asset.
  - `prev`: The order of the last fit over the same assets, or `nothing`.
  - `tol`: The tolerance of a tie.

# Returns

  - `order::Vector{Int}`: The assets in their order.

# Related

  - [`spectral_seriation`](@ref)
"""
function order_coordinates(c::VecNum, prev::Option{<:AbstractVector{<:Integer}},
                           tol::Number)
    n = length(c)
    rank = collect(1:n)
    if !isnothing(prev)
        rank .+= length(prev)
        rank[prev] .= 1:length(prev)
    end
    order = sortperm(c)
    s = 1
    while s <= n
        e = s + 1
        while e <= n && c[order[e]] - c[order[s]] <= tol
            e += 1
        end
        g = order[s:(e - 1)]
        order[s:(e - 1)] .= g[sortperm(view(rank, g))]
        s = e
    end
    return order
end
"""
    spectral_seriation(S::MatNum, cache::Option{<:SpectralSeriationState} = nothing) -> SpectralSeriationResult

Order the assets by the leading eigenspace of the Laplacian of the dissimilarity of the similarity matrix `S`.

[`SpectralSeriation`](@ref) states the matrices. The rules of the coordinate, the sign and the ties are those of the spectral seriation of the oracle, which the parity tests of this library read.

# Algorithm

 1. With no asset, return an empty order. With one asset, return it, with the coordinate zero. The multiplicity is zero and the gap is `NaN` in both cases.
 2. Set the tolerance `tol` to ``64 \\epsilon N``.
 3. When ``\\max \\mathbf{S} > \\min \\mathbf{S}``, scale the dissimilarity to `B`, with a zero diagonal, and form the Laplacian `L`. Take the eigenvectors `U` of the eigenvalues within `tol * |λ₁|` of the largest, `λ₁`. With three assets or more, the gap is zero when the second largest eigenvalue is in the eigenspace, and ``(\\lambda_1 - \\lambda_2) / \\lambda_1`` otherwise.
 4. Otherwise, take the Helmert basis of [`helmert_basis`](@ref) as `U`, and keep the gap `NaN`.
 5. Choose the coordinates `c` in `U` with [`spectral_coordinate`](@ref), from the memory `cache`.
 6. Sort `c` into the order with [`order_coordinates`](@ref), with the order of `cache` for the ties.

# Arguments

  - `S`: The similarity matrix of the assets, symmetric.
  - `cache`: The memory of the last fit over the same assets, or `nothing` for a fresh fit.

# Returns

  - `res::SpectralSeriationResult`: The order, the coordinates, the multiplicity and the gap.

# Related

  - [`SpectralSeriation`](@ref)
  - [`spectral_coordinate`](@ref)
  - [`order_coordinates`](@ref)
  - [`helmert_basis`](@ref)

# References

  - $(ref_dict[:atkins1998])
"""
function spectral_seriation(S::MatNum, cache::Option{<:SpectralSeriationState} = nothing)
    T = eltype(S)
    n = size(S, 1)
    if n <= 1
        return SpectralSeriationResult(collect(1:n), zeros(T, n), 0, T(NaN))
    end
    tol = 64 * eps(T) * n
    smin, smax = extrema(S)
    gap = T(NaN)
    U = if smax > smin
        B = (smax .- S) ./ (smax - smin)
        B[LinearAlgebra.diagind(B)] .= zero(T)
        L = LinearAlgebra.Diagonal(vec(sum(B; dims = 2))) - B
        vals, vecs = LinearAlgebra.eigen(LinearAlgebra.Symmetric(L))
        sel = vals[end] .- vals .<= tol * abs(vals[end])
        if n > 2
            gap = sel[end - 1] ? zero(T) : (vals[end] - vals[end - 1]) / vals[end]
        end
        vecs[:, sel]
    else
        helmert_basis(T, n)
    end
    c = spectral_coordinate(U, cache, tol)
    order = order_coordinates(c, isnothing(cache) ? nothing : cache.order, tol)
    return SpectralSeriationResult(order, c, size(U, 2), gap)
end
"""
    seriate(ser::SpectralSeriation, X::MatNum; dims::Int = 1, kwargs...)

Order the assets with a [`SpectralSeriation`](@ref).

# Algorithm

 1. Estimate the similarity matrix `S` and the distance matrix `D` from `X` with [`cor_and_dist`](@ref), under `ser.de` and `ser.ce`.
 2. Refuse each asset that the reader could not estimate with [`assert_distance_reader_rows`](@ref). Then refuse any other `NaN` in `D`.
 3. Check with [`assert_similarity_domain`](@ref) that `D` lies in the domain that `ser.sim` needs, and map `D` to the similarity with [`distance_to_similarity`](@ref).
 4. Order the assets with [`spectral_seriation`](@ref), from the memory `ser.cache`.

# Arguments

  - `ser`: The spectral seriation.
  - `X`: The data matrix, with one row per observation and one column per asset when `dims = 1`.
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments for [`cor_and_dist`](@ref).

# Validation

  - No diagonal entry of `D` is `NaN`. Raises a [`DistanceReaderRefusalError`](@ref) that names the columns of the assets that the reader could not estimate. A fit catches it and drops those assets.
  - `D` holds no other `NaN`. Raises a `DomainError` through [`assert_no_nan_distance`](@ref).

# Returns

  - `res::SpectralSeriationResult`: The order of the assets.

# Related

  - [`SpectralSeriation`](@ref)
  - [`spectral_seriation`](@ref)
"""
function seriate(ser::SpectralSeriation, X::MatNum; dims::Int = 1, kwargs...)
    S, D = cor_and_dist(ser.de, ser.ce, X; dims = dims, kwargs...)
    assert_distance_reader_rows(D)
    assert_no_nan_distance(D, "a spectral seriation")
    assert_similarity_domain(ser.sim, ser.de, D)
    return spectral_seriation(distance_to_similarity(ser.sim; S = S, D = D), ser.cache)
end
"""
    partial_fit!(ser::SpectralSeriation, X::MatNum; dims::Int = 1, kwargs...) -> SpectralSeriation

Order the assets of `X` from the memory of the last fit, and keep the order as the memory of the next fit.

This is the step of the memory. [`seriate`](@ref) reads the memory and keeps none. A fit with `cache = nothing` starts fresh.

# Algorithm

 1. Order the assets of `X` with [`seriate`](@ref), giving `res`.
 2. Return `ser` with `cache` set to the [`SpectralSeriationState`](@ref) of the coordinates and the order of `res`.

# Arguments

  - `ser`: The spectral seriation.
  - `X`: The data matrix, with one row per observation and one column per asset when `dims = 1`. Its assets are the assets of the memory.
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments for [`seriate`](@ref).

# Returns

  - `ser::SpectralSeriation`: The seriation, with its new memory.

# Related

  - [`SpectralSeriation`](@ref)
  - [`SpectralSeriationState`](@ref)
  - [`seriate`](@ref)
"""
function partial_fit!(ser::SpectralSeriation, X::MatNum; dims::Int = 1, kwargs...)
    res = seriate(ser, X; dims = dims, kwargs...)
    return rebuild_struct(ser, (; cache = seriation_state(res, nothing)))
end
"""
    seriation_state(res::SpectralSeriationResult, imsk::Option{<:BitVector}) -> SpectralSeriationState

The memory of a spectral fit over the assets of `imsk`, as a state over the universe of `imsk`.

The universe has `length(imsk)` assets, or the assets of the fit when `imsk` is `nothing`.

A fit orders the assets of the Investable Mask alone. The state holds a coordinate for each asset of the universe, `NaN` for an asset outside the mask, so that an asset can leave and come back. Its order names the assets by their index in the universe.

# Arguments

  - `res`: The result of the fit, over the assets of `imsk`.
  - `imsk`: The Investable Mask of the fit, or `nothing` when the fit ordered every asset.

# Returns

  - `state::SpectralSeriationState`: The memory over the universe.

# Related

  - [`SpectralSeriationState`](@ref)
  - [`carry_fold_state`](@ref)
"""
function seriation_state(res::SpectralSeriationResult, imsk::Option{<:BitVector})
    N = isnothing(imsk) ? length(res.order) : length(imsk)
    idx = isnothing(imsk) ? collect(1:N) : findall(imsk)
    c = fill(convert(eltype(res.coordinates), NaN), N)
    c[idx] .= res.coordinates
    return SpectralSeriationState(c, idx[res.order])
end

export SpectralSeriation, SpectralSeriationResult, SpectralSeriationState
