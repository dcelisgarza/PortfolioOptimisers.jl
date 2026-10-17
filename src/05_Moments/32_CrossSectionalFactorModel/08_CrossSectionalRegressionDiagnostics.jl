"""
    cs_estimation_mask_weights(B::Arr3Num, w::Nothing)
    cs_estimation_mask_weights(B::Arr3Num, w::MatNum)

Return the eligibility mask of a cross-sectional regression history, and the weights that go with it.

An asset enters the fit of an observation when every one of its exposures is finite, and when the weight it carries is positive. An infinite weight is positive, so it enters, and every answer of its observation is then `NaN`. The fit of [`cross_sectional_regression`](@ref) refuses such a weight, so a block never carries one. The verb returns both answers because every diagnostic of this file needs both. The mask counts the assets, and the weights scale the sums. A method for `nothing` handles an absent weight matrix, so no caller writes an `isnothing` test.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.

# Validation

  - `!isempty(B)`.
  - `size(w) == (size(B, 1), size(B, 2))`, when `w` is present.

# Returns

  - `mask::Matrix{Bool}`: `observations × assets`. Entry `(t, i)` is `true` when asset `i` enters the fit of observation `t`.
  - `u::Matrix{<:Real}`: `observations × assets`. The weight of an eligible pair, and zero outside the mask.

# Related

  - [`cs_gram`](@ref)
  - [`exposure_vif`](@ref)
  - [`cs_regression_t_stats`](@ref)
"""
function cs_estimation_mask_weights(B::Arr3Num, ::Nothing)
    @argcheck(!isempty(B), IsEmptyError("B cannot be empty"))
    T, N, K = size(B)
    Tf = real(eltype(B))
    mask = fill(false, T, N)
    u = zeros(Tf, T, N)
    for i in 1:N, t in 1:T
        ok = true
        for k in 1:K
            if !isfinite(B[t, i, k])
                ok = false
                break
            end
        end
        mask[t, i] = ok
        u[t, i] = ok ? one(Tf) : zero(Tf)
    end
    return mask, u
end
function cs_estimation_mask_weights(B::Arr3Num, w::MatNum)
    @argcheck(!isempty(B), IsEmptyError("B cannot be empty"))
    T, N, K = size(B)
    @argcheck(size(w, 1) == T && size(w, 2) == N,
              DimensionMismatch("w ($(size(w, 1))×$(size(w, 2))) must match B ($T×$N on its first two axes)"))
    Tf = promote_type(real(eltype(B)), real(eltype(w)))
    mask = fill(false, T, N)
    u = zeros(Tf, T, N)
    for i in 1:N, t in 1:T
        ok = w[t, i] > zero(eltype(w))
        if ok
            for k in 1:K
                if !isfinite(B[t, i, k])
                    ok = false
                    break
                end
            end
        end
        mask[t, i] = ok
        u[t, i] = ok ? Tf(w[t, i]) : zero(Tf)
    end
    return mask, u
end
"""
    cs_gram(B::Arr3Num, w::Option{<:MatNum} = nothing) -> Array{<:Real, 3}

Return the weighted Gram history of a cross-sectional regression, one slice per observation.

The slice of observation ``t`` is the normal matrix of the weighted design. The variance inflation factors, the condition number and the standard errors of the factor returns all read this one history. The kernel applies the mask first, so an asset whose exposures are not all finite contributes nothing, and neither does an asset whose weight is zero.

# Mathematical definition

```math
\\mathbf{G}_{t} = \\mathbf{B}_{t}^{\\intercal} \\mathbf{W}_{t} \\mathbf{B}_{t}
```

Where:

  - ``\\mathbf{B}_{t}``: Exposure matrix of observation ``t``, ``N \\times K``, whose row of an ineligible asset is zero.
  - ``\\mathbf{W}_{t}``: Diagonal weight matrix of observation ``t``, whose entry of an ineligible asset is zero.
  - $(math_dict[:N])
  - $(math_dict[:K])

# Algorithm

 1. Take the mask and the weights with [`cs_estimation_mask_weights`](@ref).
 2. For each observation and each pair of factors ``(k, l)`` with ``k \\le l``, sum ``u_{t,i} \\, B_{t,i,k} \\, B_{t,i,l}`` over the assets inside the mask.
 3. Copy each sum to the entry ``(l, k)``, so the slice is exactly symmetric. The kernel takes no square root, so an integer or `Rational` history gives an exact Gram history.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights. A weight of zero excludes the pair.

# Validation

  - `!isempty(B)`.
  - `size(w) == (size(B, 1), size(B, 2))`, when `w` is present.

# Returns

  - `G::Array{<:Real, 3}`: Gram history `observations × factors × factors`.

# Examples

```jldoctest
julia> B = reshape([1.0, 1.0, 0.0, 1.0], 1, 2, 2);

julia> cs_gram(B)
1×2×2 Array{Float64, 3}:
[:, :, 1] =
 2.0  1.0

[:, :, 2] =
 1.0  1.0
```

# Related

  - [`cs_estimation_mask_weights`](@ref)
  - [`exposure_vif`](@ref)
  - [`exposure_condition_number`](@ref)
  - [`cs_regression_t_stats`](@ref)
"""
function cs_gram(B::Arr3Num, w::Option{<:MatNum} = nothing)
    _, u = cs_estimation_mask_weights(B, w)
    return cs_gram_from_weights(B, u)
end
"""
    cs_gram_from_weights(B::Arr3Num, u::MatNum)

Return the weighted Gram history from a mask that a caller has already resolved.

[`cs_gram`](@ref) resolves the mask itself and calls this method. A diagnostic that has already built its own weights, such as one that also excludes a pair whose residual is not finite, calls this method instead. It then resolves the mask once rather than twice.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `u`: Resolved weight history `observations × assets`, zero outside the mask.

# Returns

  - `G::Array{<:Real, 3}`: Gram history `observations × factors × factors`.

# Related

  - [`cs_gram`](@ref)
  - [`cs_estimation_mask_weights`](@ref)
"""
function cs_gram_from_weights(B::Arr3Num, u::MatNum)
    T, N, K = size(B)
    Tf = promote_type(real(eltype(B)), real(eltype(u)))
    G = zeros(Tf, T, K, K)
    for l in 1:K, k in 1:l, i in 1:N, t in 1:T
        # A pair outside the mask can carry an exposure that is not finite.
        if !iszero(u[t, i])
            G[t, k, l] += Tf(u[t, i]) * Tf(B[t, i, k]) * Tf(B[t, i, l])
        end
    end
    for l in 1:K, k in (l + 1):K, t in 1:T
        G[t, k, l] = G[t, l, k]
    end
    return G
end
"""
    cs_gram_inverse_diagonal(G::Arr3Num) -> Matrix{<:Real}

Return the diagonal of the inverse of every slice of a Gram history.

The verb inverts each slice through its singular value decomposition. One formulation covers both cases a design presents. A full-rank slice gets its inverse. A collinear slice gets the limit of the inverse of ``\\mathbf{G}_{t} + \\lambda \\mathbf{I}`` as ``\\lambda`` falls to zero. That limit is the diagonal of the pseudo-inverse for a coefficient the design identifies, and `Inf` for one it does not identify, because the variance of such a coefficient has no bound.

A design identifies coefficient ``k`` when the unit vector ``\\boldsymbol{e}_{k}`` lies in the row space of the slice. The directions whose singular value falls under the tolerance span the null space, so the verb reads the share of ``\\boldsymbol{e}_{k}`` that they hold. The share is round-off for an identified coefficient and of order one for another. The diagonal of the pseudo-inverse alone is not the answer for an unidentified coefficient: it is the round-off of the other directions, and a ratio of such round-off looks like a real answer.

The diagonal is the only part any diagnostic of this file reads. The variance inflation factor multiplies it by the diagonal of the slice itself, and the standard error of a factor return scales it by the residual variance.

A Gram matrix formed in floating point squares the condition number of the weighted design, so its rank test drops a direction from a design condition number of about ``10^{8}``, where [`cross_sectional_rank`](@ref) keeps every column up to about ``10^{14} / K``. So a diagnostic that holds the design reads it through [`cs_inverse_diagonal!`](@ref) instead, at the rank of the fit. This verb is for a caller that holds only the Gram history.

# Mathematical definition

```math
\\begin{align}
(\\mathbf{G}_{t}^{-1})_{kk} &= \\begin{cases} \\displaystyle\\sum_{i \\,:\\, \\sigma_{i} > \\tau} \\frac{u_{ki} \\, v_{ki}}{\\sigma_{i}} & \\text{if } m_{k} \\leq \\sqrt{\\varepsilon}\\,,\\\\ \\infty & \\text{otherwise}\\,,\\end{cases}\\\\
m_{k} &= \\sum_{i \\,:\\, \\sigma_{i} \\leq \\tau} v_{ki}^{2}\\,.
\\end{align}
```

Where:

  - ``\\sigma_{i}``, ``\\boldsymbol{u}_{i}``, ``\\boldsymbol{v}_{i}``: ``i``-th singular value and the two singular vectors of the slice.
  - ``\\tau = K \\, \\varepsilon \\, \\max_{i} \\sigma_{i}``: Tolerance below which the verb drops a direction. It is the default tolerance of `LinearAlgebra.pinv`.
  - ``m_{k}``: Share of ``\\boldsymbol{e}_{k}`` that the null space of the slice holds.
  - ``\\varepsilon``: Machine epsilon of the element type.
  - $(math_dict[:K])

# Arguments

  - `G`: Gram history `observations × factors × factors`.

# Returns

  - `D::Matrix{<:Real}`: `observations × factors`. Row `t` is the diagonal of the inverse of slice `t`, `Inf` for a coefficient that slice does not identify.

# Related

  - [`cs_gram`](@ref)
  - [`cs_gram_slice!`](@ref)
  - [`cs_inverse_diagonal!`](@ref)
  - [`exposure_vif`](@ref)
  - [`cs_regression_t_stats`](@ref)
"""
function cs_gram_inverse_diagonal(G::Arr3Num)
    T = size(G, 1)
    K = size(G, 2)
    # A decomposition leaves an integer or a `Rational` type, as a square root does.
    Tf = typeof(sqrt(one(float_if_integer(real(eltype(G))))))
    D = Matrix{Tf}(undef, T, K)
    Gt = Matrix{Tf}(undef, K, K)
    for t in 1:T
        cs_gram_slice!(Gt, G, t)
        cs_gram_inverse_diagonal!(D, Gt, t)
    end
    return D
end
"""
    cs_gram_slice!(Gt::AbstractMatrix, G::Arr3Num, t::Integer)

Copy one slice of a Gram history into a working matrix.

The verbs that read a Gram history read a slice: [`cs_gram_inverse_diagonal`](@ref), [`cs_regression_t_stats`](@ref) when its caller supplies `G`, and [`exposure_condition_number`](@ref). Each needs it as a matrix of one concrete element type, and each reuses one buffer across the observations rather than allocating one per slice.

# Arguments

  - `Gt`: Working matrix `factors × factors`, written in place.
  - `G`: Gram history `observations × factors × factors`.
  - `t`: Observation to copy.

# Returns

  - `nothing`.

# Related

  - [`cs_gram`](@ref)
  - [`cs_gram_inverse_diagonal`](@ref)
  - [`exposure_condition_number`](@ref)
"""
function cs_gram_slice!(Gt::AbstractMatrix, G::Arr3Num, t::Integer)::Nothing
    Tf = eltype(Gt)
    for l in axes(Gt, 2), k in axes(Gt, 1)
        Gt[k, l] = Tf(G[t, k, l])
    end
    return nothing
end
"""
    cs_inverse_diagonal!(D::AbstractMatrix, Dt::MatNum, t::Integer,
                         Pt::Option{<:AbstractMatrix} = nothing)

Write the diagonal of the inverse Gram matrix of one weighted design into the answer, at the rank of the fit.

The Gram matrix of the design ``\\mathbf{D}_{t} = \\mathbf{U} \\mathbf{\\Sigma} \\mathbf{V}^{\\intercal}`` is ``\\mathbf{V} \\mathbf{\\Sigma}^{2} \\mathbf{V}^{\\intercal}``, so its inverse is the sum over the singular directions of ``\\boldsymbol{v}_{i} \\boldsymbol{v}_{i}^{\\intercal} / \\sigma_{i}^{2}``. The verb keeps the first `r` directions, where `r` is the rank that [`cross_sectional_rank`](@ref) reads, the rank test of the fit. So the answer is the inverse of a full-rank design, and the pseudo-inverse of a collinear one on each coefficient the design identifies. A coefficient whose unit vector the dropped directions hold by more than the square root of the machine epsilon is not identified, and its entry is `Inf`. [`cs_gram_inverse_diagonal`](@ref) states the rule.

The verb reads the design rather than its Gram matrix, because a Gram matrix formed in floating point squares the condition number. On a design whose condition number lies between about ``10^{8}`` and ``10^{14} / K``, the fit solves at full rank, and a test on the Gram matrix would drop a direction that the fit kept.

At an observation with an Unseen Member, the design is the changed design, and the factor returns are ``\\mathbf{P}_{t}`` times its coefficients. The verb then writes the diagonal of ``\\mathbf{P}_{t} \\mathbf{G}_{t}^{+} \\mathbf{P}_{t}^{\\intercal}``. It changes the singular vectors by ``\\mathbf{P}_{t}``, and it measures the share of row ``k`` of ``\\mathbf{P}_{t}`` that the dropped directions hold against the squared norm of the row. A zero row, an Unseen Member, has a variance of zero, because the rule states its return. A column that [`DependentColumnDrop`](@ref) dropped takes the same route, with a zero column in `Dt` and a zero column in `Pt`.

The count of the kept directions is the numeric rank of the design. The standard error of a factor return subtracts it from the asset count, so the verb returns it rather than a second decomposition finding it again.

# Arguments

  - `D`: Answer `observations × factors`, written in place.
  - `Dt`: Weighted design of the observation, `eligible assets × factors`, from [`cs_design_slice`](@ref).
  - `t`: Observation to write.
  - `Pt`: The change of the observation from [`cs_regression_data`](@ref), or `nothing` when it has none.

# Returns

  - `r::Int`: The numeric rank of the design.

# Related

  - [`cs_gram_inverse_diagonal`](@ref)
  - [`cs_gram_inverse_diagonal!`](@ref)
  - [`cs_design_slice`](@ref)
  - [`cross_sectional_rank`](@ref)
  - [`unseen_member_design`](@ref)
"""
function cs_inverse_diagonal!(D::AbstractMatrix, Dt::MatNum, t::Integer,
                              Pt::Option{<:AbstractMatrix} = nothing)::Int
    Tf = eltype(D)
    K = size(Dt, 2)
    r = cross_sectional_rank(Dt)
    # A design of rank zero keeps no direction, so every direction of the identity is null.
    F = iszero(r) ? nothing : LinearAlgebra.svd(Dt; full = true)
    V = isnothing(F) ? Matrix{Tf}(LinearAlgebra.I, K, K) : F.V
    lam = isnothing(F) ? zeros(Tf, K) : inv.(Tf.(F.S) .^ 2)
    return cs_write_inverse_diagonal!(D, t, isnothing(Pt) ? V : Pt * V, lam, r)
end
"""
    cs_gram_inverse_diagonal!(D::AbstractMatrix, Gt::AbstractMatrix, t::Integer,
                              Pt::Option{<:AbstractMatrix} = nothing)

Write the diagonal of the inverse of one Gram slice into the answer.

The verb is [`cs_inverse_diagonal!`](@ref) for a caller that holds the Gram slice alone. It drops a direction whose singular value falls under the tolerance ``\\tau`` of [`cs_gram_inverse_diagonal`](@ref), which is a test at the scale of the square of the condition number of the design.

# Arguments

  - `D`: Answer `observations × factors`, written in place.
  - `Gt`: One Gram slice `factors × factors`.
  - `t`: Observation to write.
  - `Pt`: The change of the observation, or `nothing` when it has none.

# Returns

  - `r::Int`: The numeric rank of the slice.

# Related

  - [`cs_gram_inverse_diagonal`](@ref)
  - [`cs_inverse_diagonal!`](@ref)
  - [`cs_gram_slice!`](@ref)
"""
function cs_gram_inverse_diagonal!(D::AbstractMatrix, Gt::AbstractMatrix, t::Integer,
                                   Pt::Option{<:AbstractMatrix} = nothing)::Int
    Tf = eltype(D)
    F = LinearAlgebra.svd(Gt)
    tol = minimum(size(Gt)) * eps(Tf) * maximum(F.S)
    # The singular values fall, so the directions over the tolerance lead.
    r = count(>(tol), F.S)
    return cs_write_inverse_diagonal!(D, t, isnothing(Pt) ? F.V : Pt * F.V, inv.(Tf.(F.S)),
                                      r)
end
"""
    cs_write_inverse_diagonal!(D::AbstractMatrix, t::Integer, V::AbstractMatrix,
                               lam::AbstractVector, r::Integer)

Write the diagonal of an inverse, given its eigenvectors and the reciprocals of its eigenvalues, into the answer.

The first `r` directions are kept, and the others span the null space. A coefficient whose row of `V` the null space holds by more than the square root of the machine epsilon of its squared norm is not identified, and its entry is `Inf`. A zero row has a variance of zero.

# Arguments

  - `D`: Answer `observations × factors`, written in place.
  - `t`: Observation to write.
  - `V`: Eigenvectors of the Gram matrix, one column per direction, already changed by the change of the observation.
  - `lam`: Reciprocal of the eigenvalue of each kept direction.
  - `r`: Number of kept directions.

# Returns

  - `r::Int`: The number of kept directions.

# Related

  - [`cs_inverse_diagonal!`](@ref)
  - [`cs_gram_inverse_diagonal!`](@ref)
"""
function cs_write_inverse_diagonal!(D::AbstractMatrix, t::Integer, V::AbstractMatrix,
                                    lam::AbstractVector, r::Integer)::Int
    Tf = eltype(D)
    # The share of a coordinate that the null directions hold is round-off when the design
    # identifies the coefficient, and of order one when it does not.
    idtol = sqrt(eps(Tf))
    for k in axes(D, 2)
        d = zero(Tf)
        m = zero(Tf)
        s = zero(Tf)
        for i in axes(V, 2)
            v2 = Tf(V[k, i])^2
            s += v2
            if i <= r
                d += v2 * Tf(lam[i])
            else
                m += v2
            end
        end
        # `s` is the squared norm of the row of the change, one without a change.
        D[t, k] = m > idtol * s ? Tf(Inf) : d
    end
    return r
end
"""
    cs_design_slice(B::Arr3Num, u::MatNum, t::Integer, ::Type{T})

Return the weighted design of one observation, over the pairs of positive weight.

The rows are the eligible assets of the observation, each scaled by the square root of its weight, which is the design the fit solved. The verb leaves out a pair outside the mask rather than giving it a zero row, so the tolerance of [`cross_sectional_rank`](@ref) reads the count of eligible assets, as the rank test of the fit does.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `u`: Resolved weight history `observations × assets`, zero outside the mask.
  - `t`: Observation to read.
  - `T`: Element type of the answer.

# Returns

  - `Dt::Matrix{T}`: `eligible assets × factors`.

# Related

  - [`cs_inverse_diagonal!`](@ref)
  - [`cs_score_regressors`](@ref)
  - [`exposure_vif`](@ref)
  - [`exposure_condition_number`](@ref)
"""
function cs_design_slice(B::Arr3Num, u::MatNum, t::Integer, ::Type{T}) where {T}
    idx = findall(i -> u[t, i] > zero(u[t, i]), axes(B, 2))
    Dt = Matrix{T}(undef, length(idx), size(B, 3))
    for k in axes(Dt, 2), j in eachindex(idx)
        Dt[j, k] = sqrt(T(u[t, idx[j]])) * T(B[t, idx[j], k])
    end
    return Dt
end
"""
    cs_regression_data(csfm::CrossSectionalFactorModel)

Return the lag-aligned regression history a cross-sectional diagnostic reads off a factor model block.

Every block method of this file starts here. The verb trims the exposures at the tail and the return-like histories at the head, so the exposures of observation ``t - \\ell`` line up with the returns of observation ``t``. The answer is the design the regression ran on. `csr.f` holds the factors the fit estimated, on the reduced axis when the block carries a family re-basis, so the verb maps the exposures onto the reduced axis, and it drops the observed factors, which are the trailing columns of that axis and which the regression did not estimate. Every answer that carries a factor axis is then on the axis of `csr.f`.

# Algorithm

 1. Refuse a block that carries no exposure history, or no cross-sectional fit.
 2. Trim the exposure history at the tail by `csfm.lag`, and the factor returns, the residuals and the regression weights at the head by the same count.
 3. Map the exposures onto the design of the regression with [`cs_regression_design`](@ref): onto the reduced axis when `csfm.fcb` is set, changed under the Unseen Member rule `csfm.unseen`, and without the observed factors that `csfm.fx` states. The factor returns `csr.f` are already on that axis.

# Arguments

  - `csfm`: A cross-sectional factor model block.

# Validation

  - `csfm.Ms` is not `nothing`, else an `IsNothingError` naming `Ms` is raised.
  - `csfm.csr` is not `nothing`, else an `IsNothingError` naming `csr` is raised.

# Returns

  - `data::NamedTuple`: `(; B, f, eps, w, P, dc)`, the lag-aligned exposures, factor returns, residuals and regression weights, on the design of the regression, the `t => P_t` pairs of [`unseen_member_design`](@ref) on the rows of the answer, and the lag-aligned dropped-column mask of the fit. `w` is `nothing` when the block carries no regression weight history, and `dc` is `nothing` when the fit states no mask. [`cs_dropped_design`](@ref) composes the mask with the pairs.

# Validation

  - The design and `csr.f` agree on the factor axis. Raises a `DimensionMismatch` that names both counts.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`reduce_exposures`](@ref)
  - [`cs_regression_design`](@ref)
  - [`cs_regression_t_stats`](@ref)
"""
function cs_regression_data(csfm::CrossSectionalFactorModel)
    return cs_regression_data(csfm, csfm.Ms, csfm.csr)
end
function cs_regression_data(::CrossSectionalFactorModel, ::Nothing,
                            ::Option{<:CrossSectionalRegression})
    return throw(IsNothingError("Ms cannot be nothing: a cross-sectional regression diagnostic reads the exposure history of the block"))
end
function cs_regression_data(::CrossSectionalFactorModel, ::Arr3Num, ::Nothing)
    return throw(IsNothingError("csr cannot be nothing: a cross-sectional regression diagnostic reads the factor returns and the residuals of the block"))
end
function cs_regression_data(csfm::CrossSectionalFactorModel, Ms::Arr3Num,
                            csr::CrossSectionalRegression)
    lag = cs_regression_lag(csfm.lag)
    T = size(Ms, 1)
    @argcheck(T > lag,
              DimensionMismatch("Ms ($T observations) must carry more observations than lag ($lag)"))
    B = Ms[1:(T - lag), :, :]
    rows = (lag + 1):size(csr.f, 1)
    f = csr.f[rows, :]
    eps = csr.eps[rows, :]
    w = cs_lagged_rows(csfm.rw, rows)
    rd = cs_regression_design(csfm.fcb, csfm.unseen, B, w,
                              isnothing(csfm.fx) ? 0 : size(csfm.fx, 2))
    @argcheck(size(rd.B, 3) == size(f, 2),
              DimensionMismatch("the regression of the block ran on $(size(rd.B, 3)) factors, the reduced axis less the observed ones, and csr.f carries $(size(f, 2)). A block states csr.f on the axis the fit estimated, which is the reduced axis under a family re-basis"))
    return (; B = rd.B, f = f, eps = eps, w = w, P = rd.P,
            dc = nothing_scalar_array_getindex_odd_order(csr.dc, rows, :))
end
"""
    cs_regression_lag(lag::Nothing)
    cs_regression_lag(lag::Integer)

Return the exposure lag of a factor model block as a count.

A block whose `lag` is `nothing` has no lag, so the verb answers `0` for it and the stated count otherwise. Two methods make the choice, so no caller writes an `isnothing` test.

# Arguments

  - `lag`: The `lag` field of a [`CrossSectionalFactorModel`](@ref), or `nothing`.

# Returns

  - `lag::Int`: The lag as a count.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`cs_regression_data`](@ref)
"""
function cs_regression_lag(::Nothing)::Int
    return 0
end
function cs_regression_lag(lag::Integer)::Int
    return Int(lag)
end
"""
    cs_lagged_rows(A::Nothing, rows)
    cs_lagged_rows(A::MatNum, rows)

Trim a per-asset history at the head, keeping the rows a lag alignment leaves.

# Arguments

  - `A`: A per-asset history `observations × assets`, or `nothing`.
  - `rows`: Indices of the observations to keep.

# Returns

  - `A::Option{<:MatNum}`: The trimmed history, or `nothing` when the block carries none.

# Related

  - [`cs_regression_data`](@ref)
"""
function cs_lagged_rows(::Nothing, args...)::Nothing
    return nothing
end
function cs_lagged_rows(A::MatNum, rows)
    return A[rows, :]
end
"""
    cs_regression_design(fcb::Nothing, rule::AbstractUnseenMemberRule, B::Arr3Num,
                         w::Option{<:MatNum}, no::Integer)
    cs_regression_design(fcb::FactorFamilyBasis, rule::AbstractUnseenMemberRule, B::Arr3Num,
                         w::Option{<:MatNum}, no::Integer)

Map a lag-aligned exposure history onto the design the cross-sectional regression ran on.

A block that carries a family re-basis has a rank-deficient design on the raw axis, because every constrained family sums to zero, so the regression ran on the reduced axis, and the diagnostics answer there too. The basis has the rows of the exposure history, and [`cs_regression_data`](@ref) trimmed the exposures at the tail, so the verb slices the basis to the rows of `B` before the basis maps the exposures. A block that carries no re-basis is already on its own axis. The observed factors are the last `no` columns of either axis, and the regression did not estimate them, so the verb drops them.

At an observation with an Unseen Member, the fit regressed on the reduced exposures times a change ``\\mathbf{P}_{t}``, so the verb changes the design with [`unseen_member_design`](@ref) under the rule of the block. The sample of each observation is its pairs of positive regression weight, as in the fit. The observed factors are not in a family, so the verb keeps the block of ``\\mathbf{P}_{t}`` on the estimated factors.

# Arguments

  - `fcb`: The `fcb` field of a [`CrossSectionalFactorModel`](@ref), or `nothing`.
  - `rule`: The `unseen` field of the block.
  - `B`: Lag-aligned exposure history `observations × assets × factors`, on the raw axis.
  - `w`: Lag-aligned regression weight history `observations × assets`, or `nothing` for equal weights.
  - `no`: The number of observed factors.

# Returns

  - `B::Arr3Num`: The exposure history of the factors the regression estimated, changed at each observation with an Unseen Member.
  - `P`: The `t => P_t` pairs of [`unseen_member_design`](@ref) on the estimated factors, empty or `()` when the rule changes no observation.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_exposures`](@ref)
  - [`unseen_member_design`](@ref)
  - [`cs_regression_data`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function cs_regression_design(::Nothing, ::AbstractUnseenMemberRule, B::Arr3Num,
                              ::Option{<:MatNum}, no::Integer)
    return (; B = B[:, :, 1:(size(B, 3) - no)], P = ())
end
function cs_regression_design(fcb::FactorFamilyBasis, rule::AbstractUnseenMemberRule,
                              B::Arr3Num, w::Option{<:MatNum}, no::Integer)
    fb = factor_basis_slice(fcb, axes(B, 1))
    u = cs_estimation_weights_only(w, size(B, 1), size(B, 2),
                                   float_if_integer(real(eltype(B))))
    ud = unseen_member_design(rule, fb, B, reduce_exposures(fb, B), u)
    ke = 1:(size(ud.Z, 3) - no)
    return (; B = ud.Z[:, :, ke], P = map(p -> first(p) => last(p)[ke, ke], ud.P))
end
"""
    cs_diagnostic_mask_weights(B::Arr3Num, eps::MatNum, w::Option{<:MatNum})

Return the eligibility mask of a cross-sectional regression diagnostic, and the weights that go with it.

A diagnostic of the fit reads the residual of every pair it counts, so a pair whose residual is not finite leaves the mask on top of the rule [`cs_estimation_mask_weights`](@ref) states. Every block method of this file resolves this mask before it calls the method that reads the arrays. So an answer from a block reproduces the fit rather than the exposure history alone.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.

# Validation

  - `size(eps) == (size(B, 1), size(B, 2))`.

# Returns

  - `mask::Matrix{Bool}`: `observations × assets`.
  - `u::Matrix{<:Real}`: `observations × assets`, zero outside the mask.

# Related

  - [`cs_estimation_mask_weights`](@ref)
  - [`cs_regression_t_stats`](@ref)
"""
function cs_diagnostic_mask_weights(B::Arr3Num, eps::MatNum, w::Option{<:MatNum})
    @argcheck(size(eps, 1) == size(B, 1) && size(eps, 2) == size(B, 2),
              DimensionMismatch("eps ($(size(eps, 1))×$(size(eps, 2))) must match B ($(size(B, 1))×$(size(B, 2)) on its first two axes)"))
    mask, u = cs_estimation_mask_weights(B, w)
    for i in axes(mask, 2), t in axes(mask, 1)
        if mask[t, i] && !isfinite(eps[t, i])
            mask[t, i] = false
            u[t, i] = zero(eltype(u))
        end
    end
    return mask, u
end
"""
    cs_regression_t_stats(
        B::Arr3Num,
        f::MatNum,
        eps::MatNum,
        w::Option{<:MatNum} = nothing;
        G::Option{<:Arr3Num} = nothing,
        P = ()
    ) -> Matrix{<:Real}
    cs_regression_t_stats(csfm::CrossSectionalFactorModel) -> FactorDiagnosticResult

Return the t-statistic of every factor return, one row per observation.

The statistic says how many standard errors a factor return of one observation sits from zero, so a rule of thumb reads an absolute value above two as significant at about the five per cent level. The fit stores no standard error, so this verb recomputes it from the Gram matrix and the residuals, and the fit result gains no field.

# Mathematical definition

```math
\\begin{align}
t_{t,k} &= \\frac{f_{t,k}}{\\mathrm{SE}_{t,k}}\\,,\\\\
\\mathrm{SE}_{t,k} &= \\sqrt{\\hat{\\sigma}^{2}_{t} \\, (\\mathbf{G}_{t}^{-1})_{kk}}\\,,\\\\
\\hat{\\sigma}^{2}_{t} &= \\frac{\\mathrm{RSS}_{t}}{n_{t} - r_{t}}\\,,\\\\
\\mathrm{RSS}_{t} &= \\sum_{i} u_{t,i} \\, \\varepsilon_{t,i}^{2}\\,.
\\end{align}
```

Where:

  - ``f_{t,k}``: Factor return of factor ``k`` at observation ``t``, which is the coefficient of the cross-sectional fit.
  - ``\\mathbf{G}_{t}``: Gram matrix of observation ``t``, which [`cs_gram`](@ref) defines.
  - ``(\\mathbf{G}_{t}^{-1})_{kk}``: Diagonal of the inverse of ``\\mathbf{G}_{t}``, which [`cs_gram_inverse_diagonal`](@ref) defines. On a collinear slice it is the diagonal of the pseudo-inverse for a coefficient the design identifies, and `Inf` for another. At an observation with an Unseen Member it is the diagonal of ``\\mathbf{P}_{t} \\mathbf{G}_{t}^{+} \\mathbf{P}_{t}^{\\intercal}``, which [`cs_inverse_diagonal!`](@ref) states, because the factor returns are ``\\mathbf{P}_{t}`` times the coefficients of the changed design.
  - ``r_{t}``: Numeric rank of the weighted design of observation ``t``, which [`cross_sectional_rank`](@ref) reads as the fit reads it. It is ``K`` on a full-rank design. The residuals of a collinear fit span ``n_{t} - r_{t}`` dimensions, so the unbiased variance divides by that count.
  - ``u_{t,i}``: Resolved regression weight of asset ``i`` at observation ``t``, zero outside the mask.
  - ``\\varepsilon_{t,i}``: Residual of asset ``i`` at observation ``t``.
  - ``n_{t}``: Number of eligible assets at observation ``t``.
  - $(math_dict[:K])

# Algorithm

 1. Resolve the mask and the weights with [`cs_diagnostic_mask_weights`](@ref), which also excludes a pair whose residual is not finite.
 2. Take the diagonal of the inverse Gram matrix and the rank of every observation with [`cs_slice_inverse_diagonal!`](@ref), mapped by the change of the observation in `P` when it has one. Without `G` it reads the weighted design of the observation, so the rank and the inverse are those of the fit. With `G` it reads the slice of `G`.
 3. Take the residual sum of squares of every observation, and divide it by the degrees of freedom.
 4. Scale the diagonal by that variance, and take the square root, which is the standard error.
 5. Divide the factor returns by the standard errors. Answer `NaN` where the degrees of freedom are not positive, where a factor return of that observation is not finite, and where the standard error is zero or `Inf`. A coefficient that a collinear design does not identify has no t-statistic, because the fit states no value of it. An Unseen Member has a standard error of zero, so it has no t-statistic either, because the rule states its return and the data do not estimate it.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `f`: Factor return matrix `observations × factors`, already lagged.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.
  - `G`: Gram history `observations × factors × factors` the caller already holds, or `nothing` to read the design. A caller that supplies one states that it built the history from the same mask. The verb does not check this. A Gram history formed in floating point squares the condition number of the design, so on a design whose condition number exceeds about ``10^{8}`` it drops a direction that the fit kept, and the t-statistics of the near-dependent factors are `NaN`. Leave `G` unset to read the design.
  - `P`: The `t => P_t` pairs of [`unseen_member_design`](@ref) on the rows of `B`, where `B` is the changed design and `f` holds the factor returns. The block method passes the pairs of [`cs_regression_data`](@ref). It is `()` for a design that no rule changed.
  - `csfm`: A cross-sectional factor model block. The answer is on the reduced factor axis when the block carries a family re-basis. When the fit states a dropped-column mask, the block method reads the design after [`cs_dropped_design`](@ref), so a factor that [`DependentColumnDrop`](@ref) dropped has no t-statistic, and the kept factors of its dependent group have the t-statistics of the kept columns.

# Validation

  - `!isempty(B)`, and every history agrees with `B` on the axes it shares.
  - On the block method, `csfm.Ms` and `csfm.csr` are not `nothing`.

# Returns

  - `t::Matrix{<:Real}`: `observations × factors`, on the method over histories.
  - `r::FactorDiagnosticResult`: On the block method, the same matrix in `X`, with the names and the family labels of the design axis. The block method returned the bare matrix in earlier releases, so a caller that indexed it now reads `r.X`.

# Examples

```jldoctest
julia> B = reshape([1.0, 1.0, 1.0, -1.0, 1.0, 0.0], 1, 3, 2);

julia> round.(cs_regression_t_stats(B, [2.0 1.0], [0.1 -0.1 0.05]); digits = 4)
1×2 Matrix{Float64}:
 23.094  9.4281
```

# Related

  - [`cs_gram`](@ref)
  - [`cs_gram_inverse_diagonal`](@ref)
  - [`cs_regression_t_stat_exceedance_rate`](@ref)
  - [`exposure_vif`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function cs_regression_t_stats(B::Arr3Num, f::MatNum, eps::MatNum,
                               w::Option{<:MatNum} = nothing;
                               G::Option{<:Arr3Num} = nothing, P = ())
    K = size(B, 3)
    @argcheck(size(f, 1) == size(B, 1) && size(f, 2) == K,
              DimensionMismatch("f ($(size(f, 1))×$(size(f, 2))) must match B ($(size(B, 1)) observations, $K factors)"))
    mask, u = cs_diagnostic_mask_weights(B, eps, w)
    cs_assert_gram(G, B)
    T = size(B, 1)
    # A decomposition leaves an integer or a `Rational` type, as a square root does.
    Td = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(B)), real(eltype(u)),
                                                       cs_gram_eltype(G))))))
    D = Matrix{Td}(undef, T, K)
    Tf = promote_type(Td, real(eltype(f)), real(eltype(eps)))
    # The answer starts absent, so an observation the loop skips needs no branch of its own.
    t = fill(convert(Tf, NaN), T, K)
    for tt in 1:T
        dof = count(view(mask, tt, :)) -
              cs_slice_inverse_diagonal!(D, G, B, u, tt, unseen_member_change_at(P, tt))
        if dof > zero(dof) && cs_row_is_finite(f, tt)
            s2 = cs_weighted_rss(eps, u, mask, tt, Tf) / Tf(dof)
            cs_t_stat_row!(t, f, D, s2, tt)
        end
    end
    return t
end
function cs_regression_t_stats(csfm::CrossSectionalFactorModel)
    data = cs_regression_data(csfm)
    dd = cs_dropped_design(data.B, data.P, data.dc)
    return cs_design_result(csfm,
                            cs_regression_t_stats(dd.B, data.f, data.eps, data.w; P = dd.P))
end
"""
    cs_row_is_finite(A::MatNum, t::Integer)

Return whether every entry of one row of a history is finite.

A cross-sectional fit whose factor returns are not all finite has no t-statistic at that observation, and this is the test that says so.

# Arguments

  - `A`: A history `observations × columns`.
  - `t`: Observation to read.

# Returns

  - `val::Bool`: `true` when every entry of the row is finite.

# Related

  - [`cs_regression_t_stats`](@ref)
"""
function cs_row_is_finite(A::MatNum, t::Integer)::Bool
    for k in axes(A, 2)
        if !isfinite(A[t, k])
            return false
        end
    end
    return true
end
"""
    cs_weighted_rss(eps::MatNum, u::MatNum, mask::AbstractMatrix{Bool}, t::Integer, ::Type{T})

Return the weighted residual sum of squares of one observation.

The sum reads the resolved weights, not the normalised ones. The standard error of a factor return divides this sum by its degrees of freedom. [`cs_regression_score_parts`](@ref) normalises instead, because a score compares observations of different sizes.

# Arguments

  - `eps`: Residual matrix `observations × assets`.
  - `u`: Resolved weight history `observations × assets`, zero outside the mask.
  - `mask`: Eligibility mask `observations × assets`.
  - `t`: Observation to read.
  - `T`: Element type of the answer.

# Returns

  - `rss::Real`: The weighted residual sum of squares of observation `t`.

# Related

  - [`cs_regression_t_stats`](@ref)
  - [`cs_regression_score_parts`](@ref)
"""
function cs_weighted_rss(eps::MatNum, u::MatNum, mask::AbstractMatrix{Bool}, t::Integer,
                         ::Type{T}) where {T}
    rss = zero(T)
    for i in axes(mask, 2)
        if mask[t, i]
            rss += T(u[t, i]) * T(eps[t, i])^2
        end
    end
    return rss
end
"""
    cs_t_stat_row!(t::MatNum, f::MatNum, D::MatNum, s2, tt::Integer)

Write the t-statistics of one observation into the answer.

A factor whose standard error is zero, or `Inf` because the design does not identify it, keeps the absent answer the caller filled the row with, so this writes only the entries that have one.

# Arguments

  - `t`: T-statistic matrix `observations × factors`, written in place.
  - `f`: Factor return matrix `observations × factors`.
  - `D`: Diagonal of the inverse Gram, `observations × factors`.
  - `s2`: Residual variance of observation `tt`.
  - `tt`: Observation to write.

# Returns

  - `nothing`.

# Related

  - [`cs_regression_t_stats`](@ref)
  - [`cs_gram_inverse_diagonal`](@ref)
"""
function cs_t_stat_row!(t::MatNum, f::MatNum, D::MatNum, s2, tt::Integer)::Nothing
    Tf = eltype(t)
    for k in axes(t, 2)
        se = sqrt(max(s2 * Tf(D[tt, k]), zero(Tf)))
        # An unbounded standard error belongs to a coefficient the design does not
        # identify, and the ratio `f / Inf = 0` would state a value it does not have.
        if zero(Tf) < se < Tf(Inf)
            t[tt, k] = Tf(f[tt, k]) / se
        end
    end
    return nothing
end
"""
    cs_assert_gram(G::Nothing, B::Arr3Num)
    cs_assert_gram(G::Arr3Num, B::Arr3Num)

Check that a Gram history a caller supplied matches the design.

# Arguments

  - `G`: Gram history the caller holds, or `nothing`.
  - `B`: Exposure history `observations × assets × factors`, already lagged.

# Validation

  - `size(G) == (size(B, 1), size(B, 3), size(B, 3))`, when `G` is present. Raises a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`cs_regression_t_stats`](@ref)
  - [`cs_slice_inverse_diagonal!`](@ref)
"""
function cs_assert_gram(::Nothing, ::Arr3Num)::Nothing
    return nothing
end
function cs_assert_gram(G::Arr3Num, B::Arr3Num)::Nothing
    @argcheck(size(G, 1) == size(B, 1) &&
              size(G, 2) == size(B, 3) &&
              size(G, 3) == size(B, 3),
              DimensionMismatch("G ($(size(G, 1))×$(size(G, 2))×$(size(G, 3))) must match B ($(size(B, 1)) observations, $(size(B, 3)) factors)"))
    return nothing
end
"""
    cs_gram_eltype(G::Nothing)
    cs_gram_eltype(G::Arr3Num)

Return the element type that a Gram history a caller supplied adds to the arithmetic of a diagnostic.

`Bool` is the type that adds nothing, because it promotes to every other number type. So a diagnostic without a Gram history takes the type of its design and its weights alone.

# Arguments

  - `G`: Gram history the caller holds, or `nothing`.

# Returns

  - `T::Type`: `Bool` without a history, and the real element type of `G` otherwise.

# Related

  - [`cs_regression_t_stats`](@ref)
"""
function cs_gram_eltype(::Nothing)
    return Bool
end
function cs_gram_eltype(G::Arr3Num)
    return real(eltype(G))
end
"""
    cs_slice_inverse_diagonal!(D::AbstractMatrix, G::Nothing, B::Arr3Num, u::MatNum,
                               t::Integer, Pt::Option{<:AbstractMatrix})
    cs_slice_inverse_diagonal!(D::AbstractMatrix, G::Arr3Num, B::Arr3Num, u::MatNum,
                               t::Integer, Pt::Option{<:AbstractMatrix})

Write the diagonal of the inverse Gram matrix of one observation into the answer, from the design or from the Gram history the caller supplied.

Without a Gram history the verb reads the weighted design through [`cs_inverse_diagonal!`](@ref), at the rank of the fit. With one, it reads the slice through [`cs_gram_inverse_diagonal!`](@ref), whose rank test reads the square of the condition number of the design.

# Arguments

  - `D`: Answer `observations × factors`, written in place.
  - `G`: Gram history the caller holds, or `nothing`.
  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `u`: Resolved weight history `observations × assets`, zero outside the mask.
  - `t`: Observation to write.
  - `Pt`: The change of the observation, or `nothing`.

# Returns

  - `r::Int`: The numeric rank of the observation.

# Related

  - [`cs_regression_t_stats`](@ref)
  - [`cs_inverse_diagonal!`](@ref)
  - [`cs_gram_inverse_diagonal!`](@ref)
"""
function cs_slice_inverse_diagonal!(D::AbstractMatrix, ::Nothing, B::Arr3Num, u::MatNum,
                                    t::Integer, Pt::Option{<:AbstractMatrix})::Int
    return cs_inverse_diagonal!(D, cs_design_slice(B, u, t, eltype(D)), t, Pt)
end
function cs_slice_inverse_diagonal!(D::AbstractMatrix, G::Arr3Num, ::Arr3Num, ::MatNum,
                                    t::Integer, Pt::Option{<:AbstractMatrix})::Int
    Gt = Matrix{eltype(D)}(undef, size(G, 2), size(G, 3))
    cs_gram_slice!(Gt, G, t)
    return cs_gram_inverse_diagonal!(D, Gt, t, Pt)
end
"""
    cs_regression_t_stat_exceedance_rate(t::MatNum; threshold::Number = 2) -> Vector{<:Real}
    cs_regression_t_stat_exceedance_rate(
        t::FactorDiagnosticResult;
        threshold::Number = 2
    ) -> FactorDiagnosticResult
    cs_regression_t_stat_exceedance_rate(
        csfm::CrossSectionalFactorModel;
        threshold::Number = 2
    ) -> FactorDiagnosticResult

Return the fraction of observations at which a factor's t-statistic exceeds a threshold.

A factor whose true cross-sectional coefficient is zero, and whose t-statistics are about Gaussian, exceeds a threshold of two at about five per cent of the observations. A rate above that reference level says that the factor is repeatedly significant rather than significant once. An observation whose t-statistic is `NaN` counts in neither the numerator nor the denominator.

# Mathematical definition

```math
\\mathrm{rate}_{k} = \\frac{\\#\\{t : |t_{t,k}| > \\tau\\}}{\\#\\{t : t_{t,k} \\text{ is finite}\\}}
```

Where:

  - ``t_{t,k}``: T-statistic of factor ``k`` at observation ``t``, which [`cs_regression_t_stats`](@ref) returns.
  - ``\\tau``: Absolute threshold.

# Arguments

  - `t`: T-statistic matrix `observations × factors`, or the [`FactorDiagnosticResult`](@ref) of [`cs_regression_t_stats`](@ref), whose labels the answer keeps.
  - `csfm`: A cross-sectional factor model block. The answer is on the reduced factor axis when the block carries a family re-basis.
  - `threshold`: Absolute t-statistic above which an observation counts as significant.

# Validation

  - `!isempty(t)`.

# Returns

  - `rate::Vector{<:Real}`: One entry per factor, on the method over a matrix. A factor with no finite t-statistic answers zero.
  - `r::FactorDiagnosticResult`: On the other two methods, the same vector in `X`, with the names and the family labels of the factor axis of `t`. The block method returned the bare vector in earlier releases, so a caller that indexed it now reads `r.X`.

# Examples

```jldoctest
julia> cs_regression_t_stat_exceedance_rate([3.0 1.0; 1.0 1.0; NaN 1.0])
2-element Vector{Float64}:
 0.5
 0.0
```

# Related

  - [`cs_regression_t_stats`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function cs_regression_t_stat_exceedance_rate(t::MatNum; threshold::Number = 2)
    @argcheck(!isempty(t), IsEmptyError("t cannot be empty"))
    K = size(t, 2)
    Tf = float_if_integer(real(eltype(t)))
    rate = zeros(Tf, K)
    for k in 1:K
        n = 0
        s = 0
        for tt in axes(t, 1)
            if isfinite(t[tt, k])
                n += 1
                if abs(t[tt, k]) > threshold
                    s += 1
                end
            end
        end
        rate[k] = iszero(n) ? zero(Tf) : Tf(s) / Tf(n)
    end
    return rate
end
function cs_regression_t_stat_exceedance_rate(t::FactorDiagnosticResult;
                                              threshold::Number = 2)
    return FactorDiagnosticResult(cs_regression_t_stat_exceedance_rate(t.X;
                                                                       threshold = threshold),
                                  t.nf, t.fam, (1,))
end
function cs_regression_t_stat_exceedance_rate(csfm::CrossSectionalFactorModel;
                                              threshold::Number = 2)
    return cs_regression_t_stat_exceedance_rate(cs_regression_t_stats(csfm);
                                                threshold = threshold)
end
"""
    cs_regression_score_parts(B::Arr3Num, f::MatNum, eps::MatNum, w::Option{<:MatNum})

Return the pieces every cross-sectional regression score is built from.

The four scores read the same three quantities: the eligible asset count, the weight-normalised residual sum of squares, and the coefficient of determination. This verb computes them once per score, and no Result holds them, because the scores are independent statistics with no identity between them.

The mask of a score is the finiteness of the asset return, which the exposures, the factor returns and the residuals reconstruct. That differs from the mask of a Gram diagnostic, which reads the exposures and the residuals separately.

# Algorithm

 1. Reconstruct the asset returns as the systematic part plus the residual.
 2. Mask a pair whose reconstructed return is not finite, and a pair whose weight is not positive.
 3. Normalise the weights of every observation to sum to one.
 4. Take the weighted residual sum of squares, the weighted mean return and the weighted total sum of squares. Clamp the mean to the range of the returns inside the mask, so a constant cross-section has a total sum of squares of exactly zero.
 5. Answer the coefficient of determination as one less the ratio of the two sums, and `NaN` where the total sum of squares is zero.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `f`: Factor return matrix `observations × factors`, already lagged.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.

# Validation

  - `!isempty(B)`, and every history agrees with `B` on the axes it shares.

# Returns

  - `n::Vector{Int}`: Eligible asset count of every observation.
  - `rss::Vector{<:Real}`: Weight-normalised residual sum of squares of every observation.
  - `r2::Vector{<:Real}`: Coefficient of determination of every observation.
  - `Q::Matrix{<:Real}`: `observations × assets`. The normalised weights, zero outside the mask, which [`cs_score_regressors`](@ref) reads for the rank of each design.

# Related

  - [`cs_score_regressors`](@ref)
  - [`cs_regression_r2`](@ref)
  - [`cs_regression_adjusted_r2`](@ref)
  - [`cs_regression_aic`](@ref)
  - [`cs_regression_bic`](@ref)
"""
function cs_regression_score_parts(B::Arr3Num, f::MatNum, eps::MatNum, w::Option{<:MatNum})
    T, N, K = size(B)
    @argcheck(!isempty(B), IsEmptyError("B cannot be empty"))
    @argcheck(size(f, 1) == T && size(f, 2) == K,
              DimensionMismatch("f ($(size(f, 1))×$(size(f, 2))) must match B ($T observations, $K factors)"))
    @argcheck(size(eps, 1) == T && size(eps, 2) == N,
              DimensionMismatch("eps ($(size(eps, 1))×$(size(eps, 2))) must match B ($T×$N on its first two axes)"))
    Tf = float_if_integer(promote_type(real(eltype(B)), real(eltype(f)), real(eltype(eps))))
    u0 = cs_estimation_weights_only(w, T, N, Tf)
    n = zeros(Int, T)
    rss = Vector{Tf}(undef, T)
    r2 = Vector{Tf}(undef, T)
    r = Vector{Tf}(undef, N)
    Q = Matrix{Tf}(undef, T, N)
    for t in 1:T
        q = view(Q, t, :)
        n[t] = cs_score_observation!(r, q, B, f, eps, u0, t)
        rss[t], tss = cs_weighted_score_sums(r, q, eps, t)
        r2[t] = iszero(tss) ? Tf(NaN) : one(Tf) - rss[t] / tss
    end
    return n, rss, r2, Q
end
"""
    cs_score_regressors(k::Integer, B::Arr3Num, Q::MatNum)
    cs_score_regressors(k::Nothing, B::Arr3Num, Q::MatNum)

Return the effective number of regressors of every cross-sectional fit, which a score charges for.

A fit spends one degree of freedom on each direction of its design, which is the rank of the design rather than its column count. The two differ on a collinear observation, such as one where a level of a constrained family has no member. So `nothing` takes the numeric rank of the weighted design of each observation, on the mask of the score, through [`cross_sectional_rank`](@ref), which is the rank test of the fit. A test on the Gram matrix would square the condition number, and count fewer directions than the fit spent on a design whose condition number exceeds about ``10^{8}``. An integer is a count the caller states, and every observation takes it.

# Arguments

  - `k`: Number of regressors, or `nothing` for the rank of each observation.
  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `Q`: Normalised weight history `observations × assets` of the score, zero outside its mask.

# Returns

  - `k::Vector{<:Integer}`: One count per observation.

# Related

  - [`cs_regression_score_parts`](@ref)
  - [`cs_regression_adjusted_r2`](@ref)
  - [`cs_regression_aic`](@ref)
  - [`cs_regression_bic`](@ref)
"""
function cs_score_regressors(k::Integer, B::Arr3Num, ::MatNum)
    return fill(k, size(B, 1))
end
function cs_score_regressors(::Nothing, B::Arr3Num, Q::MatNum)
    # A decomposition leaves an integer or a `Rational` type, as a square root does.
    Tf = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(B)), real(eltype(Q)))))))
    return [cross_sectional_rank(cs_design_slice(B, Q, t, Tf)) for t in axes(B, 1)]
end
"""
    cs_score_observation!(
        r::AbstractVector,
        q::AbstractVector,
        B::Arr3Num,
        f::MatNum,
        eps::MatNum,
        u0::MatNum,
        t::Integer
    )

Reconstruct the asset returns of one observation, and normalise its weights to sum to one.

The mask of a cross-sectional regression score is the finiteness of the reconstructed asset return, which differs from the mask of a Gram diagnostic. This verb applies that mask. An ineligible pair gets a zero return and a zero weight, so every sum the score takes over the cross-section ignores it.

# Arguments

  - `r`: Reconstructed asset returns of the observation, written in place.
  - `q`: Normalised weights of the observation, written in place.
  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `f`: Factor return matrix `observations × factors`, already lagged.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `u0`: Regression weights `observations × assets`, zero where the weight is not positive.
  - `t`: Observation to read.

# Returns

  - `n::Int`: Number of eligible assets at observation `t`.

# Related

  - [`cs_regression_score_parts`](@ref)
  - [`cs_weighted_score_sums`](@ref)
"""
function cs_score_observation!(r::AbstractVector, q::AbstractVector, B::Arr3Num, f::MatNum,
                               eps::MatNum, u0::MatNum, t::Integer)::Int
    Tf = eltype(r)
    n = 0
    wsum = zero(Tf)
    for i in eachindex(r)
        ri = cs_systematic_return(B, f, t, i, Tf) + Tf(eps[t, i])
        if isfinite(ri) && u0[t, i] > zero(Tf)
            r[i] = ri
            q[i] = Tf(u0[t, i])
            n += 1
            wsum += Tf(u0[t, i])
        else
            r[i] = zero(Tf)
            q[i] = zero(Tf)
        end
    end
    scale = wsum > zero(Tf) ? inv(wsum) : zero(Tf)
    for i in eachindex(q)
        q[i] *= scale
    end
    return n
end
"""
    cs_systematic_return(B::Arr3Num, f::MatNum, t::Integer, i::Integer, ::Type{T})

Return the part of one asset's return the factors of one observation span.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `f`: Factor return matrix `observations × factors`, already lagged.
  - `t`: Observation to read.
  - `i`: Asset to read.
  - `T`: Element type of the answer.

# Returns

  - `sys::Real`: The exposures of asset `i` at observation `t`, weighted by the factor returns of that observation.

# Related

  - [`cs_score_observation!`](@ref)
  - [`cs_regression_score_parts`](@ref)
"""
function cs_systematic_return(B::Arr3Num, f::MatNum, t::Integer, i::Integer,
                              ::Type{T}) where {T}
    sys = zero(T)
    for k in axes(B, 3)
        sys += T(B[t, i, k]) * T(f[t, k])
    end
    return sys
end
"""
    cs_weighted_score_sums(r::AbstractVector, q::AbstractVector, eps::MatNum, t::Integer)

Return the weighted residual and total sums of squares of one observation.

The two sums are the numerator and the denominator of the coefficient of determination, and both read the weights [`cs_score_observation!`](@ref) normalised.

# Arguments

  - `r`: Reconstructed asset returns of the observation, zero outside the mask.
  - `q`: Normalised weights of the observation, zero outside the mask.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `t`: Observation to read.

# Returns

  - `rss::Real`: Weighted residual sum of squares.
  - `tss::Real`: Weighted total sum of squares about the weighted mean return.

# Related

  - [`cs_regression_score_parts`](@ref)
  - [`cs_score_observation!`](@ref)
"""
function cs_weighted_score_sums(r::AbstractVector, q::AbstractVector, eps::MatNum,
                                t::Integer)
    Tf = eltype(r)
    rss = zero(Tf)
    mean = zero(Tf)
    lo = Tf(Inf)
    hi = Tf(-Inf)
    for i in eachindex(r)
        if q[i] > zero(Tf)
            rss += q[i] * Tf(eps[t, i])^2
            mean += q[i] * r[i]
            lo = min(lo, r[i])
            hi = max(hi, r[i])
        end
    end
    # The clamp makes the mean of a constant cross-section exactly its common value, so its
    # total sum of squares is exactly zero rather than a rounding residue.
    mean = clamp(mean, lo, hi)
    tss = zero(Tf)
    for i in eachindex(r)
        if q[i] > zero(Tf)
            tss += q[i] * (r[i] - mean)^2
        end
    end
    return rss, tss
end
"""
    cs_estimation_weights_only(w::Nothing, T::Integer, N::Integer, ::Type{Tf})
    cs_estimation_weights_only(w::MatNum, T::Integer, N::Integer, ::Type{Tf})

Return the regression weights of a score, without reading the exposures.

A cross-sectional regression score masks on the reconstructed asset return rather than on the exposures, so it resolves its weights here rather than through [`cs_estimation_mask_weights`](@ref). An absent weight matrix gives every pair a unit weight.

# Arguments

  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.
  - `T`: Number of observations.
  - `N`: Number of assets.
  - `Tf`: Element type of the answer.

# Validation

  - `size(w) == (T, N)`, when `w` is present.

# Returns

  - `u::Matrix{<:Real}`: `observations × assets`. A non-positive weight reads back as zero.

# Related

  - [`cs_regression_score_parts`](@ref)
  - [`cs_estimation_mask_weights`](@ref)
"""
function cs_estimation_weights_only(::Nothing, T::Integer, N::Integer,
                                    ::Type{Tf}) where {Tf}
    return ones(Tf, T, N)
end
function cs_estimation_weights_only(w::MatNum, T::Integer, N::Integer,
                                    ::Type{Tf}) where {Tf}
    @argcheck(size(w, 1) == T && size(w, 2) == N,
              DimensionMismatch("w ($(size(w, 1))×$(size(w, 2))) must match the history ($T×$N)"))
    u = zeros(Tf, T, N)
    for i in 1:N, t in 1:T
        u[t, i] = w[t, i] > zero(eltype(w)) ? Tf(w[t, i]) : zero(Tf)
    end
    return u
end
"""
    cs_regression_r2(
        B::Arr3Num,
        f::MatNum,
        eps::MatNum,
        w::Option{<:MatNum} = nothing
    ) -> Vector{<:Real}
    cs_regression_r2(csfm::CrossSectionalFactorModel) -> Vector{<:Real}

Return the weighted cross-sectional coefficient of determination, one entry per observation.

The score says what share of the weighted cross-sectional variance of the returns the factors of that observation explain. The verb normalises the weights to sum to one, so an observation with many assets and an observation with few are on one scale.

# Mathematical definition

```math
R^{2}_{t} = 1 - \\frac{\\sum_{i} q_{t,i} \\, \\varepsilon_{t,i}^{2}}
                     {\\sum_{i} q_{t,i} \\, (r_{t,i} - \\bar{r}_{t})^{2}}
```

Where:

  - ``q_{t,i}``: Regression weight of asset ``i`` at observation ``t``, normalised over the eligible assets to sum to one.
  - ``\\varepsilon_{t,i}``: Residual of asset ``i`` at observation ``t``.
  - ``r_{t,i}``: Return of asset ``i`` at observation ``t``, reconstructed as the systematic part plus the residual.
  - ``\\bar{r}_{t}``: Weighted mean return of observation ``t``.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `f`: Factor return matrix `observations × factors`, already lagged.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(B)`, and every history agrees with `B` on the axes it shares.
  - On the block method, `csfm.Ms` and `csfm.csr` are not `nothing`.

# Returns

  - `r2::Vector{<:Real}`: One entry per observation, `NaN` where the weighted total sum of squares is zero, as it is for a cross-section of equal returns.

# Examples

```jldoctest
julia> B = reshape([1.0, 1.0, -1.0, 1.0], 1, 2, 2);

julia> cs_regression_r2(B, [1.0 1.0], [0.0 0.0])
1-element Vector{Float64}:
 1.0
```

# Related

  - [`cs_regression_adjusted_r2`](@ref)
  - [`cs_regression_aic`](@ref)
  - [`cs_regression_bic`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function cs_regression_r2(B::Arr3Num, f::MatNum, eps::MatNum, w::Option{<:MatNum} = nothing)
    _, _, r2, _ = cs_regression_score_parts(B, f, eps, w)
    return r2
end
function cs_regression_r2(csfm::CrossSectionalFactorModel)
    data = cs_regression_data(csfm)
    return cs_regression_r2(data.B, data.f, data.eps, data.w)
end
"""
    cs_regression_adjusted_r2(
        B::Arr3Num,
        f::MatNum,
        eps::MatNum,
        w::Option{<:MatNum} = nothing;
        k::Option{<:Integer} = nothing
    ) -> Vector{<:Real}
    cs_regression_adjusted_r2(
        csfm::CrossSectionalFactorModel;
        k::Option{<:Integer} = nothing
    ) -> Vector{<:Real}

Return the cross-sectional coefficient of determination adjusted for the regressor count, one entry per observation.

The adjustment charges the score for every regressor, so a factor that explains nothing lowers it rather than leaving it flat. The adjusted score therefore never exceeds [`cs_regression_r2`](@ref), and it is the score to read when two designs of different sizes are compared.

# Mathematical definition

```math
\\bar{R}^{2}_{t} = 1 - (1 - R^{2}_{t}) \\, \\frac{n_{t} - 1}{n_{t} - k_{t} - 1}
```

Where:

  - ``R^{2}_{t}``: Coefficient of determination of observation ``t``, which [`cs_regression_r2`](@ref) defines.
  - ``n_{t}``: Number of eligible assets at observation ``t``.
  - ``k_{t}``: Effective number of regressors of observation ``t``.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `f`: Factor return matrix `observations × factors`, already lagged.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.
  - `k`: Effective number of regressors, or `nothing` for the rank of the design of each observation, which [`cs_score_regressors`](@ref) takes. The rank is what the fit spent: the factor count of the reduced axis on a full-rank observation, and less on a collinear one.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(B)`, and every history agrees with `B` on the axes it shares.
  - On the block method, `csfm.Ms` and `csfm.csr` are not `nothing`.

# Returns

  - `adj::Vector{<:Real}`: One entry per observation, `NaN` where the eligible asset count does not exceed `k_t + 1`.

# Examples

```jldoctest
julia> B = reshape([1.0, 1.0, 1.0, -1.0, 1.0, 0.0], 1, 3, 2);

julia> cs_regression_adjusted_r2(B, [1.0 0.0], [0.0 0.0 0.0])
1-element Vector{Float64}:
 NaN
```

# Related

  - [`cs_regression_r2`](@ref)
  - [`cs_regression_aic`](@ref)
  - [`cs_regression_bic`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function cs_regression_adjusted_r2(B::Arr3Num, f::MatNum, eps::MatNum,
                                   w::Option{<:MatNum} = nothing;
                                   k::Option{<:Integer} = nothing)
    n, _, r2, Q = cs_regression_score_parts(B, f, eps, w)
    kt = cs_score_regressors(k, B, Q)
    Tf = eltype(r2)
    adj = Vector{Tf}(undef, length(r2))
    for t in eachindex(adj)
        adj[t] = if n[t] > kt[t] + 1
            one(Tf) - (one(Tf) - r2[t]) * Tf(n[t] - 1) / Tf(n[t] - kt[t] - 1)
        else
            Tf(NaN)
        end
    end
    return adj
end
function cs_regression_adjusted_r2(csfm::CrossSectionalFactorModel;
                                   k::Option{<:Integer} = nothing)
    data = cs_regression_data(csfm)
    return cs_regression_adjusted_r2(data.B, data.f, data.eps, data.w; k = k)
end
"""
    cs_regression_aic(
        B::Arr3Num,
        f::MatNum,
        eps::MatNum,
        w::Option{<:MatNum} = nothing;
        k::Option{<:Integer} = nothing
    ) -> Vector{<:Real}
    cs_regression_aic(
        csfm::CrossSectionalFactorModel;
        k::Option{<:Integer} = nothing
    ) -> Vector{<:Real}

Return the Akaike information criterion of every cross-sectional fit, one entry per observation.

The criterion trades the fit of an observation against the size of its design, and a lower value is the better trade. It shares its residual term with [`cs_regression_bic`](@ref) and differs only in the penalty, which is flat in the asset count here and grows with it there.

# Mathematical definition

```math
\\mathrm{AIC}_{t} = n_{t} \\ln (\\mathrm{RSS}_{t}) + 2 k_{t}
```

Where:

  - ``\\mathrm{RSS}_{t}``: Weight-normalised residual sum of squares of observation ``t``.
  - ``n_{t}``: Number of eligible assets at observation ``t``.
  - ``k_{t}``: Effective number of regressors of observation ``t``.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `f`: Factor return matrix `observations × factors`, already lagged.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.
  - `k`: Effective number of regressors, or `nothing` for the rank of the design of each observation, which [`cs_score_regressors`](@ref) takes. The rank is what the fit spent: the factor count of the reduced axis on a full-rank observation, and less on a collinear one.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(B)`, and every history agrees with `B` on the axes it shares.
  - On the block method, `csfm.Ms` and `csfm.csr` are not `nothing`.

# Returns

  - `aic::Vector{<:Real}`: One entry per observation, `NaN` where the eligible asset count does not exceed `k_t`, and `-Inf` where the residual sum of squares is zero.

# Examples

```jldoctest
julia> B = reshape([1.0, 1.0, 1.0, -1.0, 1.0, 0.0], 1, 3, 2);

julia> cs_regression_aic(B, [1.0 0.0], [0.1 0.1 0.1]; k = 1)
1-element Vector{Float64}:
 -11.815510557964274
```

# Related

  - [`cs_regression_bic`](@ref)
  - [`cs_regression_r2`](@ref)
  - [`cs_regression_adjusted_r2`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function cs_regression_aic(B::Arr3Num, f::MatNum, eps::MatNum,
                           w::Option{<:MatNum} = nothing; k::Option{<:Integer} = nothing)
    n, rss, _, Q = cs_regression_score_parts(B, f, eps, w)
    kt = cs_score_regressors(k, B, Q)
    # A logarithm leaves a `Rational` type.
    Tf = typeof(log(one(eltype(rss))))
    aic = Vector{Tf}(undef, length(rss))
    for t in eachindex(aic)
        aic[t] = n[t] > kt[t] ? Tf(n[t]) * log(rss[t]) + Tf(2 * kt[t]) : Tf(NaN)
    end
    return aic
end
function cs_regression_aic(csfm::CrossSectionalFactorModel; k::Option{<:Integer} = nothing)
    data = cs_regression_data(csfm)
    return cs_regression_aic(data.B, data.f, data.eps, data.w; k = k)
end
"""
    cs_regression_bic(
        B::Arr3Num,
        f::MatNum,
        eps::MatNum,
        w::Option{<:MatNum} = nothing;
        k::Option{<:Integer} = nothing
    ) -> Vector{<:Real}
    cs_regression_bic(
        csfm::CrossSectionalFactorModel;
        k::Option{<:Integer} = nothing
    ) -> Vector{<:Real}

Return the Bayesian information criterion of every cross-sectional fit, one entry per observation.

The criterion trades the fit of an observation against the size of its design, and a lower value is the better trade. Its penalty grows with the logarithm of the eligible asset count, so it charges a large cross-section more for a regressor than [`cs_regression_aic`](@ref) does.

# Mathematical definition

```math
\\mathrm{BIC}_{t} = n_{t} \\ln (\\mathrm{RSS}_{t}) + k_{t} \\ln (n_{t})
```

Where:

  - ``\\mathrm{RSS}_{t}``: Weight-normalised residual sum of squares of observation ``t``.
  - ``n_{t}``: Number of eligible assets at observation ``t``.
  - ``k_{t}``: Effective number of regressors of observation ``t``.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `f`: Factor return matrix `observations × factors`, already lagged.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.
  - `k`: Effective number of regressors, or `nothing` for the rank of the design of each observation, which [`cs_score_regressors`](@ref) takes. The rank is what the fit spent: the factor count of the reduced axis on a full-rank observation, and less on a collinear one.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(B)`, and every history agrees with `B` on the axes it shares.
  - On the block method, `csfm.Ms` and `csfm.csr` are not `nothing`.

# Returns

  - `bic::Vector{<:Real}`: One entry per observation, `NaN` where the eligible asset count does not exceed `k_t`, and `-Inf` where the residual sum of squares is zero.

# Examples

```jldoctest
julia> B = reshape([1.0, 1.0, 1.0, -1.0, 1.0, 0.0], 1, 3, 2);

julia> cs_regression_bic(B, [1.0 0.0], [0.1 0.1 0.1]; k = 1)
1-element Vector{Float64}:
 -12.716898269296165
```

# Related

  - [`cs_regression_aic`](@ref)
  - [`cs_regression_r2`](@ref)
  - [`cs_regression_adjusted_r2`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function cs_regression_bic(B::Arr3Num, f::MatNum, eps::MatNum,
                           w::Option{<:MatNum} = nothing; k::Option{<:Integer} = nothing)
    n, rss, _, Q = cs_regression_score_parts(B, f, eps, w)
    kt = cs_score_regressors(k, B, Q)
    Tf = typeof(log(one(eltype(rss))))
    bic = Vector{Tf}(undef, length(rss))
    for t in eachindex(bic)
        bic[t] = n[t] > kt[t] ? Tf(n[t]) * log(rss[t]) + Tf(kt[t]) * log(Tf(n[t])) : Tf(NaN)
    end
    return bic
end
function cs_regression_bic(csfm::CrossSectionalFactorModel; k::Option{<:Integer} = nothing)
    data = cs_regression_data(csfm)
    return cs_regression_bic(data.B, data.f, data.eps, data.w; k = k)
end

export cs_gram, cs_regression_t_stats, cs_regression_t_stat_exceedance_rate,
       cs_regression_r2, cs_regression_adjusted_r2, cs_regression_aic, cs_regression_bic
