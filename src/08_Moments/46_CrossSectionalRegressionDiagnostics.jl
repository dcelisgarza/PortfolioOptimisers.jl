"""
    cs_estimation_mask_weights(B::Arr3Num, w::Nothing)
    cs_estimation_mask_weights(B::Arr3Num, w::MatNum)

Return the eligibility mask of a cross-sectional regression history, and the weights that go with it.

An asset enters the fit of an observation when every one of its exposures is finite, and when the weight it carries is positive. The two answers travel together because every diagnostic of this file needs both: the mask counts the assets, and the weights scale the sums. An absent weight matrix is the dispatch rather than a branch, so no caller writes an `isnothing` test.

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
    Tf = float(real(eltype(B)))
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
    Tf = promote_type(float(real(eltype(B))), float(real(eltype(w))))
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

The slice of observation ``t`` is the normal matrix of the weighted design, and every diagnostic that reads the geometry of the cross-section is a function of it: the variance inflation factors, the condition number and the standard errors of the factor returns all read this one history. The design is masked first, so an asset whose exposures are not all finite contributes nothing, and so does an asset whose weight is zero.

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
 2. For each observation, scale the exposure rows by the square root of their weights, which is the weighted design.
 3. Multiply the weighted design by its own transpose, giving the slice of that observation.

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

[`cs_gram`](@ref) resolves the mask itself and calls this method. A diagnostic that has already built its own weights — one that also excludes a pair whose residual is not finite — calls this method instead, so the mask is resolved once per diagnostic rather than twice.

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
    Tf = promote_type(float(real(eltype(B))), float(real(eltype(u))))
    G = Array{Tf, 3}(undef, T, K, K)
    A = Matrix{Tf}(undef, N, K)
    for t in 1:T
        for k in 1:K, i in 1:N
            s = sqrt(Tf(u[t, i]))
            A[i, k] = iszero(s) ? zero(Tf) : s * Tf(B[t, i, k])
        end
        Gt = transpose(A) * A
        for l in 1:K, k in 1:K
            G[t, k, l] = Gt[k, l]
        end
    end
    return G
end
"""
    cs_gram_inverse_diagonal(G::Arr3Num) -> Matrix{<:Real}

Return the diagonal of the inverse of every slice of a Gram history.

The inverse is taken slice by slice, through the singular value decomposition of the slice. That is one formulation for both cases a design presents: a full-rank slice gets its inverse, and a collinear slice gets its pseudo-inverse, because a direction whose singular value falls under the tolerance contributes nothing. A branch that inverts and falls back on failure answers the same, and it costs the analysis of this file the whole body of the pseudo-inverse.

The diagonal is the only part any diagnostic of this file reads: the variance inflation factor multiplies it by the diagonal of the slice itself, and the standard error of a factor return scales it by the residual variance.

# Mathematical definition

```math
(\\mathbf{G}_{t}^{+})_{kk} = \\sum_{i \\,:\\, \\sigma_{i} > \\tau} \\frac{u_{ki} \\, v_{ki}}{\\sigma_{i}}
```

Where:

  - ``\\sigma_{i}``, ``\\boldsymbol{u}_{i}``, ``\\boldsymbol{v}_{i}``: ``i``-th singular value and the two singular vectors of the slice.
  - ``\\tau = K \\, \\varepsilon \\, \\max_{i} \\sigma_{i}``: Tolerance below which a direction is dropped. It is the one the pseudo-inverse of the standard library applies.
  - $(math_dict[:K])

# Arguments

  - `G`: Gram history `observations × factors × factors`.

# Returns

  - `D::Matrix{<:Real}`: `observations × factors`. Row `t` is the diagonal of the inverse of slice `t`.

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
    Tf = float(real(eltype(G)))
    D = Matrix{Tf}(undef, T, K)
    Gt = Matrix{Tf}(undef, K, K)
    for t in 1:T
        cs_gram_slice!(Gt, G, t)
        cs_inverse_diagonal!(D, Gt, t)
    end
    return D
end
"""
    cs_gram_slice!(Gt::AbstractMatrix, G::Arr3Num, t::Integer)

Copy one slice of a Gram history into a working matrix.

The two verbs that read a slice — [`cs_gram_inverse_diagonal`](@ref) and [`exposure_condition_number`](@ref) — each need it as a matrix of one concrete element type, and each reuses one buffer across the observations rather than allocating per slice.

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
    cs_inverse_diagonal!(D::AbstractMatrix, Gt::AbstractMatrix, t::Integer)

Write the diagonal of the inverse of one Gram slice into the answer.

The inverse is the sum over the singular directions of the outer product of the two singular vectors, scaled by the reciprocal singular value. A direction whose singular value falls under the tolerance is dropped, which makes the answer the pseudo-inverse of a collinear slice and the inverse of a full-rank one.

# Arguments

  - `D`: Answer `observations × factors`, written in place.
  - `Gt`: One Gram slice `factors × factors`.
  - `t`: Observation to write.

# Returns

  - `nothing`.

# Related

  - [`cs_gram_inverse_diagonal`](@ref)
  - [`cs_gram_slice!`](@ref)
"""
function cs_inverse_diagonal!(D::AbstractMatrix, Gt::AbstractMatrix, t::Integer)::Nothing
    Tf = eltype(D)
    F = LinearAlgebra.svd(Gt)
    tol = minimum(size(Gt)) * eps(Tf) * maximum(F.S)
    for k in axes(D, 2)
        d = zero(Tf)
        for i in eachindex(F.S)
            if F.S[i] > tol
                d += Tf(F.U[k, i]) * Tf(F.V[k, i]) / Tf(F.S[i])
            end
        end
        D[t, k] = d
    end
    return nothing
end
"""
    cs_regression_data(csfm::CrossSectionalFactorModel)

Return the lag-aligned regression history a cross-sectional diagnostic reads off a factor model block.

Every level-2 diagnostic of this file starts here. The exposures are trimmed at the tail and the return-like histories at the head, so the exposures of observation ``t - \\ell`` line up with the returns of observation ``t``. When the block carries a family re-basis, the exposures and the factor returns are mapped onto the reduced axis, and every answer that carries a factor axis is then on the reduced axis too.

# Algorithm

 1. Refuse a block that carries no exposure history, or no cross-sectional fit.
 2. Trim the exposure history at the tail by `csfm.lag`, and the factor returns, the residuals and the regression weights at the head by the same count.
 3. When `csfm.fcb` is set, slice the basis to the trimmed observation axis, map the exposures through [`reduce_exposures`](@ref), and map the factor returns through [`reduce_factor_returns`](@ref).

# Arguments

  - `csfm`: A cross-sectional factor model block.

# Validation

  - `csfm.Ms` is not `nothing`, else an `IsNothingError` naming `Ms` is raised.
  - `csfm.csr` is not `nothing`, else an `IsNothingError` naming `csr` is raised.

# Returns

  - `data::NamedTuple`: `(; B, f, eps, w)`, the lag-aligned exposures, factor returns, residuals and regression weights. `w` is `nothing` when the block carries no regression weight history.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`reduce_exposures`](@ref)
  - [`reduce_factor_returns`](@ref)
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
    Br, fr = cs_reduce_regression(csfm.fcb, B, f, lag)
    return (; B = Br, f = fr, eps = eps, w = w)
end
"""
    cs_regression_lag(lag::Nothing)
    cs_regression_lag(lag::Integer)

Return the exposure lag of a factor model block as a count.

A block that states no lag lags by nothing, so the two cases answer `0` and the stated count. The shape is the dispatch, which keeps the `isnothing` test out of every caller.

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
    cs_reduce_regression(fcb::Nothing, B::Arr3Num, f::MatNum, lag::Integer)
    cs_reduce_regression(fcb::FactorFamilyBasis, B::Arr3Num, f::MatNum, lag::Integer)

Map a lag-aligned regression history onto the reduced factor axis of a family re-basis.

A block that carries no re-basis is already on its own axis, so that case returns the pair unchanged. A block that carries one has a rank-deficient design on the raw axis, because every constrained family sums to zero, so the diagnostics are answered on the reduced axis instead. The basis is sliced to the trimmed observation axis before it maps the exposures, because the exposures were trimmed at the tail.

# Arguments

  - `fcb`: The `fcb` field of a [`CrossSectionalFactorModel`](@ref), or `nothing`.
  - `B`: Lag-aligned exposure history `observations × assets × factors`.
  - `f`: Lag-aligned factor return matrix `observations × factors`.
  - `lag`: Number of observations by which the exposures lag the returns.

# Returns

  - `B::Arr3Num`: The exposure history, on the reduced axis when a re-basis is set.
  - `f::MatNum`: The factor returns, on the reduced axis when a re-basis is set.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_exposures`](@ref)
  - [`reduce_factor_returns`](@ref)
  - [`cs_regression_data`](@ref)
"""
function cs_reduce_regression(::Nothing, B::Arr3Num, f::MatNum, ::Integer)
    return B, f
end
function cs_reduce_regression(fcb::FactorFamilyBasis, B::Arr3Num, f::MatNum, lag::Integer)
    Tb = size(fcb.ratios, 1) - lag
    fcbB = factor_basis_slice(fcb, 1:Tb)
    return reduce_exposures(fcbB, B), reduce_factor_returns(fcb, f)
end
"""
    exposure_vif(G::Arr3Num) -> Matrix{<:Real}
    exposure_vif(B::Arr3Num, w::Option{<:MatNum}) -> Matrix{<:Real}
    exposure_vif(csfm::CrossSectionalFactorModel) -> Matrix{<:Real}

Return the variance inflation factor of every factor, one row per observation.

The factor measures how much the collinearity of the cross-sectional design inflates the variance of a factor return. A value of one says that the factor is orthogonal to the others of that observation, and a large value says that the factor is nearly a combination of them, so its estimated return is unstable rather than wrong.

# Mathematical definition

```math
\\mathrm{VIF}_{t,k} = (\\mathbf{G}_{t})_{kk} \\, (\\mathbf{G}_{t}^{-1})_{kk}
```

Where:

  - ``\\mathbf{G}_{t}``: Gram matrix of observation ``t``, which [`cs_gram`](@ref) defines.
  - ``\\mathrm{VIF}_{t,k}``: Variance inflation factor of factor ``k`` at observation ``t``.

# Algorithm

 1. Take the diagonal of the inverse of every slice with [`cs_gram_inverse_diagonal`](@ref).
 2. Multiply it entry by entry by the diagonal of the slice itself.
 3. On the method that reads a design, answer `NaN` at an observation whose eligible asset count does not exceed the factor count, because such a design determines no variance.

# Arguments

  - `G`: Gram history `observations × factors × factors`, which [`cs_gram`](@ref) returns.
  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block. The answer is on the reduced factor axis when the block carries a family re-basis.

# Validation

  - `!isempty(B)`, and `size(w) == (size(B, 1), size(B, 2))` when `w` is present.
  - On the block method, `csfm.Ms` and `csfm.csr` are not `nothing`.

# Returns

  - `vif::Matrix{<:Real}`: `observations × factors`.

# Examples

```jldoctest
julia> B = reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2);

julia> exposure_vif(cs_gram(B))
1×2 Matrix{Float64}:
 1.0  1.0
```

# Related

  - [`cs_gram`](@ref)
  - [`cs_gram_inverse_diagonal`](@ref)
  - [`exposure_condition_number`](@ref)
  - [`cs_regression_t_stats`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_vif(G::Arr3Num)
    T = size(G, 1)
    K = size(G, 2)
    D = cs_gram_inverse_diagonal(G)
    Tf = eltype(D)
    vif = Matrix{Tf}(undef, T, K)
    for k in 1:K, t in 1:T
        vif[t, k] = Tf(G[t, k, k]) * D[t, k]
    end
    return vif
end
function exposure_vif(B::Arr3Num, w::Option{<:MatNum})
    mask, u = cs_estimation_mask_weights(B, w)
    return cs_masked_vif(B, mask, u)
end
function exposure_vif(csfm::CrossSectionalFactorModel)
    data = cs_regression_data(csfm)
    mask, u = cs_diagnostic_mask_weights(data.B, data.eps, data.w)
    return cs_masked_vif(data.B, mask, u)
end
"""
    cs_masked_vif(B::Arr3Num, mask::AbstractMatrix{Bool}, u::MatNum)

Return the variance inflation factors of a design whose mask a caller has already resolved.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `mask`: Eligibility mask `observations × assets`.
  - `u`: Resolved weight history `observations × assets`, zero outside the mask.

# Returns

  - `vif::Matrix{<:Real}`: `observations × factors`, `NaN` at an observation with no degrees of freedom.

# Related

  - [`exposure_vif`](@ref)
"""
function cs_masked_vif(B::Arr3Num, mask::AbstractMatrix{Bool}, u::MatNum)
    G = cs_gram_from_weights(B, u)
    vif = exposure_vif(G)
    K = size(B, 3)
    for t in axes(vif, 1)
        if count(view(mask, t, :)) <= K
            for k in 1:K
                vif[t, k] = convert(eltype(vif), NaN)
            end
        end
    end
    return vif
end
"""
    exposure_condition_number(G::Arr3Num) -> Vector{<:Real}
    exposure_condition_number(B::Arr3Num, w::Option{<:MatNum}) -> Vector{<:Real}
    exposure_condition_number(csfm::CrossSectionalFactorModel) -> Vector{<:Real}

Return the two-norm condition number of the cross-sectional design, one entry per observation.

The condition number reads the whole design at once, where a variance inflation factor reads one factor of it. A value of one says that the weighted design is orthonormal, and a large value says that one direction of the factor space carries almost no weighted exposure, so the fit of that observation is sensitive to a small change in the returns. An exactly collinear design answers `Inf` or a value near the reciprocal of the machine epsilon, and which of the two it answers is a property of the singular value decomposition rather than of the design: the smallest singular value of a rank-deficient matrix rounds either to zero or to a value near the machine epsilon. Read both answers as one statement, that the design of that observation is singular.

# Mathematical definition

```math
\\kappa_{t} = \\frac{\\sigma_{\\max}(\\mathbf{G}_{t})}{\\sigma_{\\min}(\\mathbf{G}_{t})}
```

Where:

  - ``\\mathbf{G}_{t}``: Gram matrix of observation ``t``, which [`cs_gram`](@ref) defines.
  - ``\\sigma_{\\max}``, ``\\sigma_{\\min}``: Largest and smallest singular values.

# Arguments

  - `G`: Gram history `observations × factors × factors`, which [`cs_gram`](@ref) returns.
  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block. The answer is on the reduced factor axis when the block carries a family re-basis.

# Validation

  - `!isempty(B)`, and `size(w) == (size(B, 1), size(B, 2))` when `w` is present.
  - On the block method, `csfm.Ms` and `csfm.csr` are not `nothing`.

# Returns

  - `kappa::Vector{<:Real}`: One entry per observation. The methods that read a design answer `NaN` at an observation whose eligible asset count does not exceed the factor count.

# Examples

```jldoctest
julia> B = reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2);

julia> exposure_condition_number(cs_gram(B))
1-element Vector{Float64}:
 1.0
```

# Related

  - [`cs_gram`](@ref)
  - [`exposure_vif`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_condition_number(G::Arr3Num)
    T = size(G, 1)
    K = size(G, 2)
    Tf = float(real(eltype(G)))
    kappa = Vector{Tf}(undef, T)
    Gt = Matrix{Tf}(undef, K, K)
    for t in 1:T
        cs_gram_slice!(Gt, G, t)
        kappa[t] = LinearAlgebra.cond(Gt)
    end
    return kappa
end
function exposure_condition_number(B::Arr3Num, w::Option{<:MatNum})
    mask, u = cs_estimation_mask_weights(B, w)
    return cs_masked_condition_number(B, mask, u)
end
function exposure_condition_number(csfm::CrossSectionalFactorModel)
    data = cs_regression_data(csfm)
    mask, u = cs_diagnostic_mask_weights(data.B, data.eps, data.w)
    return cs_masked_condition_number(data.B, mask, u)
end
"""
    cs_masked_condition_number(B::Arr3Num, mask::AbstractMatrix{Bool}, u::MatNum)

Return the condition numbers of a design whose mask a caller has already resolved.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `mask`: Eligibility mask `observations × assets`.
  - `u`: Resolved weight history `observations × assets`, zero outside the mask.

# Returns

  - `kappa::Vector{<:Real}`: One entry per observation, `NaN` at an observation with no degrees of freedom.

# Related

  - [`exposure_condition_number`](@ref)
"""
function cs_masked_condition_number(B::Arr3Num, mask::AbstractMatrix{Bool}, u::MatNum)
    G = cs_gram_from_weights(B, u)
    kappa = exposure_condition_number(G)
    K = size(B, 3)
    for t in eachindex(kappa)
        if count(view(mask, t, :)) <= K
            kappa[t] = convert(eltype(kappa), NaN)
        end
    end
    return kappa
end
"""
    cs_diagnostic_mask_weights(B::Arr3Num, eps::MatNum, w::Option{<:MatNum})

Return the eligibility mask of a cross-sectional regression diagnostic, and the weights that go with it.

A diagnostic of the fit reads the residual of every pair it counts, so a pair whose residual is not finite leaves the mask on top of the rule [`cs_estimation_mask_weights`](@ref) states. This is the mask every level-2 verb of this file resolves before it calls its level-1 method, which is why a level-2 answer reproduces the fit rather than the exposure history alone.

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
        G::Option{<:Arr3Num} = nothing
    ) -> Matrix{<:Real}
    cs_regression_t_stats(csfm::CrossSectionalFactorModel) -> Matrix{<:Real}

Return the t-statistic of every factor return, one row per observation.

The statistic says how many standard errors a factor return of one observation sits from zero, so a rule of thumb reads an absolute value above two as significant at about the five per cent level. The fit stores no standard error, so this verb recomputes it from the Gram matrix and the residuals, and the fit result gains no field.

# Mathematical definition

```math
\\begin{align}
t_{t,k} &= \\frac{f_{t,k}}{\\mathrm{SE}_{t,k}}\\,,\\\\
\\mathrm{SE}_{t,k} &= \\sqrt{\\hat{\\sigma}^{2}_{t} \\, (\\mathbf{G}_{t}^{-1})_{kk}}\\,,\\\\
\\hat{\\sigma}^{2}_{t} &= \\frac{\\mathrm{RSS}_{t}}{n_{t} - K}\\,,\\\\
\\mathrm{RSS}_{t} &= \\sum_{i} u_{t,i} \\, \\varepsilon_{t,i}^{2}\\,.
\\end{align}
```

Where:

  - ``f_{t,k}``: Factor return of factor ``k`` at observation ``t``, which is the coefficient of the cross-sectional fit.
  - ``\\mathbf{G}_{t}``: Gram matrix of observation ``t``, which [`cs_gram`](@ref) defines.
  - ``u_{t,i}``: Resolved regression weight of asset ``i`` at observation ``t``, zero outside the mask.
  - ``\\varepsilon_{t,i}``: Residual of asset ``i`` at observation ``t``.
  - ``n_{t}``: Number of eligible assets at observation ``t``.
  - $(math_dict[:K])

# Algorithm

 1. Resolve the mask and the weights with [`cs_diagnostic_mask_weights`](@ref), which also excludes a pair whose residual is not finite.
 2. Build the Gram history with [`cs_gram_from_weights`](@ref), or take the one the caller supplied through `G`.
 3. Take the residual sum of squares of every observation, and divide it by the degrees of freedom.
 4. Scale the diagonal of the inverse Gram by that variance, and take the square root, which is the standard error.
 5. Divide the factor returns by the standard errors. Answer `NaN` where the degrees of freedom are not positive, where a factor return of that observation is not finite, or where the standard error is zero.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `f`: Factor return matrix `observations × factors`, already lagged.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.
  - `G`: Gram history `observations × factors × factors` the caller already holds, or `nothing` to build it. A caller that supplies one is stating that it was built from the same mask, which this verb does not check.
  - `csfm`: A cross-sectional factor model block. The answer is on the reduced factor axis when the block carries a family re-basis.

# Validation

  - `!isempty(B)`, and every history agrees with `B` on the axes it shares.
  - On the block method, `csfm.Ms` and `csfm.csr` are not `nothing`.

# Returns

  - `t::Matrix{<:Real}`: `observations × factors`.

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
                               G::Option{<:Arr3Num} = nothing)
    K = size(B, 3)
    @argcheck(size(f, 1) == size(B, 1) && size(f, 2) == K,
              DimensionMismatch("f ($(size(f, 1))×$(size(f, 2))) must match B ($(size(B, 1)) observations, $K factors)"))
    mask, u = cs_diagnostic_mask_weights(B, eps, w)
    Gh = cs_resolved_gram(G, B, u)
    D = cs_gram_inverse_diagonal(Gh)
    Tf = promote_type(eltype(D), float(real(eltype(f))), float(real(eltype(eps))))
    T = size(B, 1)
    # The answer starts absent, so an observation the loop skips needs no branch of its own.
    t = fill(convert(Tf, NaN), T, K)
    for tt in 1:T
        dof = count(view(mask, tt, :)) - K
        if dof > zero(dof) && cs_row_is_finite(f, tt)
            s2 = cs_weighted_rss(eps, u, mask, tt, Tf) / Tf(dof)
            cs_t_stat_row!(t, f, D, s2, tt)
        end
    end
    return t
end
function cs_regression_t_stats(csfm::CrossSectionalFactorModel)
    data = cs_regression_data(csfm)
    return cs_regression_t_stats(data.B, data.f, data.eps, data.w)
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

The weights are the resolved ones rather than the normalised ones, which is what the standard error of a factor return divides by its degrees of freedom. [`cs_regression_score_parts`](@ref) normalises instead, because a score compares observations of different sizes.

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

A factor whose standard error is zero keeps the absent answer the caller filled the row with, so this writes only the entries that have one.

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
        if se > zero(Tf)
            t[tt, k] = Tf(f[tt, k]) / se
        end
    end
    return nothing
end
"""
    cs_resolved_gram(G::Nothing, B::Arr3Num, u::MatNum)
    cs_resolved_gram(G::Arr3Num, B::Arr3Num, u::MatNum)

Return the Gram history a diagnostic reads, building it only when the caller supplied none.

# Arguments

  - `G`: Gram history the caller holds, or `nothing`.
  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `u`: Resolved weight history `observations × assets`, zero outside the mask.

# Returns

  - `G::Arr3Num`: The supplied history, or the one [`cs_gram_from_weights`](@ref) builds.

# Related

  - [`cs_gram`](@ref)
  - [`cs_regression_t_stats`](@ref)
"""
function cs_resolved_gram(::Nothing, B::Arr3Num, u::MatNum)
    return cs_gram_from_weights(B, u)
end
function cs_resolved_gram(G::Arr3Num, B::Arr3Num, ::MatNum)
    @argcheck(size(G, 1) == size(B, 1) &&
              size(G, 2) == size(B, 3) &&
              size(G, 3) == size(B, 3),
              DimensionMismatch("G ($(size(G, 1))×$(size(G, 2))×$(size(G, 3))) must match B ($(size(B, 1)) observations, $(size(B, 3)) factors)"))
    return G
end
"""
    cs_regression_t_stat_exceedance_rate(t::MatNum; threshold::Number = 2) -> Vector{<:Real}
    cs_regression_t_stat_exceedance_rate(
        csfm::CrossSectionalFactorModel;
        threshold::Number = 2
    ) -> Vector{<:Real}

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

  - `t`: T-statistic matrix `observations × factors`.
  - `csfm`: A cross-sectional factor model block. The answer is on the reduced factor axis when the block carries a family re-basis.
  - `threshold`: Absolute t-statistic above which an observation counts as significant.

# Validation

  - `!isempty(t)`.

# Returns

  - `rate::Vector{<:Real}`: One entry per factor. A factor with no finite t-statistic answers zero.

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
    Tf = float(real(eltype(t)))
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
function cs_regression_t_stat_exceedance_rate(csfm::CrossSectionalFactorModel;
                                              threshold::Number = 2)
    return cs_regression_t_stat_exceedance_rate(cs_regression_t_stats(csfm);
                                                threshold = threshold)
end
"""
    cs_regression_score_parts(B::Arr3Num, f::MatNum, eps::MatNum, w::Option{<:MatNum})

Return the pieces every cross-sectional regression score is built from.

The four scores read the same three quantities: the eligible asset count, the weight-normalised residual sum of squares, and the coefficient of determination. They are computed here once per score rather than held on a Result, because [#709](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/709) refused a grouped Result: the scores are independent statistics with no identity between them.

The mask of a score is the finiteness of the **asset return**, which the exposures, the factor returns and the residuals reconstruct. That differs from the mask of a Gram diagnostic, which reads the exposures and the residuals separately.

# Algorithm

 1. Reconstruct the asset returns as the systematic part plus the residual.
 2. Mask a pair whose reconstructed return is not finite, and a pair whose weight is not positive.
 3. Normalise the weights of every observation to sum to one.
 4. Take the weighted residual sum of squares, the weighted mean return and the weighted total sum of squares.
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

# Related

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
    Tf = promote_type(float(real(eltype(B))), float(real(eltype(f))),
                      float(real(eltype(eps))))
    u0 = cs_estimation_weights_only(w, T, N, Tf)
    n = zeros(Int, T)
    rss = Vector{Tf}(undef, T)
    r2 = Vector{Tf}(undef, T)
    r = Vector{Tf}(undef, N)
    q = Vector{Tf}(undef, N)
    nan = convert(Tf, NaN)
    for t in 1:T
        n[t] = cs_score_observation!(r, q, B, f, eps, u0, t)
        rss[t], tss = cs_weighted_score_sums(r, q, eps, t)
        r2[t] = iszero(tss) ? nan : one(Tf) - rss[t] / tss
    end
    return n, rss, r2
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

The mask of a cross-sectional regression score is the finiteness of the reconstructed **asset return**, which differs from the mask of a Gram diagnostic. This is where that mask is applied: an ineligible pair leaves with a zero return and a zero weight, so every sum the score takes over the cross-section ignores it.

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
    for i in eachindex(r)
        if q[i] > zero(Tf)
            rss += q[i] * Tf(eps[t, i])^2
        end
        mean += q[i] * r[i]
    end
    tss = zero(Tf)
    for i in eachindex(r)
        tss += q[i] * (r[i] - mean)^2
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

The score says what share of the weighted cross-sectional variance of the returns the factors of that observation explain. The weights are normalised to sum to one, so an observation with many assets and an observation with few are on one scale.

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

  - `r2::Vector{<:Real}`: One entry per observation, `NaN` where the weighted total sum of squares is zero.

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
    _, _, r2 = cs_regression_score_parts(B, f, eps, w)
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
        k::Integer = size(B, 3)
    ) -> Vector{<:Real}
    cs_regression_adjusted_r2(csfm::CrossSectionalFactorModel) -> Vector{<:Real}

Return the cross-sectional coefficient of determination adjusted for the regressor count, one entry per observation.

The adjustment charges the score for every regressor, so a factor that explains nothing lowers it rather than leaving it flat. The adjusted score therefore never exceeds [`cs_regression_r2`](@ref), and it is the score to read when two designs of different sizes are compared.

# Mathematical definition

```math
\\bar{R}^{2}_{t} = 1 - (1 - R^{2}_{t}) \\, \\frac{n_{t} - 1}{n_{t} - k - 1}
```

Where:

  - ``R^{2}_{t}``: Coefficient of determination of observation ``t``, which [`cs_regression_r2`](@ref) defines.
  - ``n_{t}``: Number of eligible assets at observation ``t``.
  - ``k``: Effective number of regressors.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `f`: Factor return matrix `observations × factors`, already lagged.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.
  - `k`: Effective number of regressors. The block method takes the factor count of the reduced axis, which is what the fit spent.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(B)`, and every history agrees with `B` on the axes it shares.
  - On the block method, `csfm.Ms` and `csfm.csr` are not `nothing`.

# Returns

  - `adj::Vector{<:Real}`: One entry per observation, `NaN` where the eligible asset count does not exceed `k + 1`.

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
                                   w::Option{<:MatNum} = nothing; k::Integer = size(B, 3))
    n, _, r2 = cs_regression_score_parts(B, f, eps, w)
    Tf = eltype(r2)
    adj = Vector{Tf}(undef, length(r2))
    nan = convert(Tf, NaN)
    for t in eachindex(adj)
        adj[t] = if n[t] > k + 1
            one(Tf) - (one(Tf) - r2[t]) * Tf(n[t] - 1) / Tf(n[t] - k - 1)
        else
            nan
        end
    end
    return adj
end
function cs_regression_adjusted_r2(csfm::CrossSectionalFactorModel)
    data = cs_regression_data(csfm)
    return cs_regression_adjusted_r2(data.B, data.f, data.eps, data.w)
end
"""
    cs_regression_aic(
        B::Arr3Num,
        f::MatNum,
        eps::MatNum,
        w::Option{<:MatNum} = nothing;
        k::Integer = size(B, 3)
    ) -> Vector{<:Real}
    cs_regression_aic(csfm::CrossSectionalFactorModel) -> Vector{<:Real}

Return the Akaike information criterion of every cross-sectional fit, one entry per observation.

The criterion trades the fit of an observation against the size of its design, and a lower value is the better trade. It shares its residual term with [`cs_regression_bic`](@ref) and differs only in the penalty, which is flat in the asset count here and grows with it there.

# Mathematical definition

```math
\\mathrm{AIC}_{t} = n_{t} \\ln (\\mathrm{RSS}_{t}) + 2 k
```

Where:

  - ``\\mathrm{RSS}_{t}``: Weight-normalised residual sum of squares of observation ``t``.
  - ``n_{t}``: Number of eligible assets at observation ``t``.
  - ``k``: Effective number of regressors.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `f`: Factor return matrix `observations × factors`, already lagged.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.
  - `k`: Effective number of regressors. The block method takes the factor count of the reduced axis, which is what the fit spent.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(B)`, and every history agrees with `B` on the axes it shares.
  - On the block method, `csfm.Ms` and `csfm.csr` are not `nothing`.

# Returns

  - `aic::Vector{<:Real}`: One entry per observation, `NaN` where the eligible asset count does not exceed `k`.

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
                           w::Option{<:MatNum} = nothing; k::Integer = size(B, 3))
    n, rss, _ = cs_regression_score_parts(B, f, eps, w)
    Tf = eltype(rss)
    aic = Vector{Tf}(undef, length(rss))
    nan = convert(Tf, NaN)
    for t in eachindex(aic)
        aic[t] = n[t] > k ? Tf(n[t]) * log(rss[t]) + Tf(2 * k) : nan
    end
    return aic
end
function cs_regression_aic(csfm::CrossSectionalFactorModel)
    data = cs_regression_data(csfm)
    return cs_regression_aic(data.B, data.f, data.eps, data.w)
end
"""
    cs_regression_bic(
        B::Arr3Num,
        f::MatNum,
        eps::MatNum,
        w::Option{<:MatNum} = nothing;
        k::Integer = size(B, 3)
    ) -> Vector{<:Real}
    cs_regression_bic(csfm::CrossSectionalFactorModel) -> Vector{<:Real}

Return the Bayesian information criterion of every cross-sectional fit, one entry per observation.

The criterion trades the fit of an observation against the size of its design, and a lower value is the better trade. Its penalty grows with the logarithm of the eligible asset count, so it charges a large cross-section more for a regressor than [`cs_regression_aic`](@ref) does.

# Mathematical definition

```math
\\mathrm{BIC}_{t} = n_{t} \\ln (\\mathrm{RSS}_{t}) + k \\ln (n_{t})
```

Where:

  - ``\\mathrm{RSS}_{t}``: Weight-normalised residual sum of squares of observation ``t``.
  - ``n_{t}``: Number of eligible assets at observation ``t``.
  - ``k``: Effective number of regressors.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, already lagged.
  - `f`: Factor return matrix `observations × factors`, already lagged.
  - `eps`: Residual matrix `observations × assets`, already lagged.
  - `w`: Regression weight history `observations × assets`, or `nothing` for equal weights.
  - `k`: Effective number of regressors. The block method takes the factor count of the reduced axis, which is what the fit spent.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - `!isempty(B)`, and every history agrees with `B` on the axes it shares.
  - On the block method, `csfm.Ms` and `csfm.csr` are not `nothing`.

# Returns

  - `bic::Vector{<:Real}`: One entry per observation, `NaN` where the eligible asset count does not exceed `k`.

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
                           w::Option{<:MatNum} = nothing; k::Integer = size(B, 3))
    n, rss, _ = cs_regression_score_parts(B, f, eps, w)
    Tf = eltype(rss)
    bic = Vector{Tf}(undef, length(rss))
    nan = convert(Tf, NaN)
    for t in eachindex(bic)
        bic[t] = n[t] > k ? Tf(n[t]) * log(rss[t]) + Tf(k) * log(Tf(n[t])) : nan
    end
    return bic
end
function cs_regression_bic(csfm::CrossSectionalFactorModel)
    data = cs_regression_data(csfm)
    return cs_regression_bic(data.B, data.f, data.eps, data.w)
end

"""
    cs_diagnostic_factor_names(csfm::CrossSectionalFactorModel)

Return the factor names of the axis a cross-sectional regression diagnostic answers on.

A diagnostic that carries a factor axis answers on the reduced axis when the block carries a family re-basis, so the names of the raw axis do not label it. This verb maps them, and it is what a plot reads to label its axis. A block that names no factor answers `nothing`, and the caller then labels the axis by position.

# Arguments

  - `csfm`: A cross-sectional factor model block.

# Returns

  - `nf::Option{<:Vector{String}}`: The names of the answer's factor axis, or `nothing` when the block names no factor.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`reduce_factor_names`](@ref)
  - [`cs_regression_t_stats`](@ref)
  - [`exposure_vif`](@ref)
"""
function cs_diagnostic_factor_names(csfm::CrossSectionalFactorModel)
    return cs_diagnostic_factor_names(csfm.fcb, csfm.nf)
end
function cs_diagnostic_factor_names(::Option{<:AbstractFactorFamilyBasis},
                                    ::Nothing)::Nothing
    return nothing
end
function cs_diagnostic_factor_names(::Nothing, nf::VecStr)::Vector{String}
    return String[String(n) for n in nf]
end
function cs_diagnostic_factor_names(fcb::FactorFamilyBasis, nf::VecStr)::Vector{String}
    return reduce_factor_names(fcb, nf)
end

export cs_gram, cs_regression_t_stats, cs_regression_t_stat_exceedance_rate, exposure_vif,
       exposure_condition_number, cs_regression_r2, cs_regression_adjusted_r2,
       cs_regression_aic, cs_regression_bic
