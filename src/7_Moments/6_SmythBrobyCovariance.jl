function sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
    # Zone of confusion.
    # If the return is not a significant proportion of the standard deviation, we classify it as noise.
    if abs(xi) < sigmai * c1 && abs(xj) < sigmaj * c1
        return zero(eltype(xi))
    end

    # Zone of indecision.
    # Center returns at mu = 0 and sigma = 1.
    ri = abs((xi - mui) / sigmai)
    rj = abs((xj - muj) / sigmaj)
    # If the return is less than c2 standard deviations, or greater than c3 standard deviations, we can't make a call since it may be noise, or overall market forces.
    if ri < c2 && rj < c2 || ri > c3 && rj > c3
        return zero(eltype(xi))
    end

    kappa = sqrt((1 + ri) * (1 + rj))
    gamma = abs(ri - rj)

    return kappa / (1 + gamma^n)
end
abstract type BaseSmythBrobyCovariance <: BaseGerberCovariance end
abstract type SmythBrobyCovarianceAlgorithm <: AbstractMomentAlgorithm end
abstract type UnNormalisedSmythBrobyCovarianceAlgorithm <: SmythBrobyCovarianceAlgorithm end
abstract type NormalisedSmythBrobyCovarianceAlgorithm <: SmythBrobyCovarianceAlgorithm end
struct SmythBroby0 <: UnNormalisedSmythBrobyCovarianceAlgorithm end
struct SmythBroby1 <: UnNormalisedSmythBrobyCovarianceAlgorithm end
struct SmythBroby2 <: UnNormalisedSmythBrobyCovarianceAlgorithm end
struct SmythBrobyGerber0 <: UnNormalisedSmythBrobyCovarianceAlgorithm end
struct SmythBrobyGerber1 <: UnNormalisedSmythBrobyCovarianceAlgorithm end
struct SmythBrobyGerber2 <: UnNormalisedSmythBrobyCovarianceAlgorithm end
struct NormalisedSmythBroby0 <: NormalisedSmythBrobyCovarianceAlgorithm end
struct NormalisedSmythBroby1 <: NormalisedSmythBrobyCovarianceAlgorithm end
struct NormalisedSmythBroby2 <: NormalisedSmythBrobyCovarianceAlgorithm end
struct NormalisedSmythBrobyGerber0 <: NormalisedSmythBrobyCovarianceAlgorithm end
struct NormalisedSmythBrobyGerber1 <: NormalisedSmythBrobyCovarianceAlgorithm end
struct NormalisedSmythBrobyGerber2 <: NormalisedSmythBrobyCovarianceAlgorithm end

struct SmythBrobyCovariance{T1 <: AbstractExpectedReturnsEstimator,
                            T2 <: StatsBase.CovarianceEstimator, T3 <: PosDefEstimator,
                            T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real, T8 <: Real,
                            T9 <: SmythBrobyCovarianceAlgorithm} <: BaseSmythBrobyCovariance
    me::T1
    ve::T2
    pdm::T3
    threshold::T4
    c1::T5
    c2::T6
    c3::T7
    n::T8
    alg::T9
    #! Add conditional threading.
end
function SmythBrobyCovariance(;
                              me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                              ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                              pdm::Union{Nothing, <:PosDefEstimator} = PosDefEstimator(),
                              threshold::Real = 0.5, c1::Real = 0.5, c2::Real = 0.5,
                              c3::Real = 4.0, n::Real = 2.0,
                              alg::SmythBrobyCovarianceAlgorithm = SmythBrobyGerber1())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2) && c3 > c2)
    return SmythBrobyCovariance{typeof(me), typeof(ve), typeof(pdm), typeof(threshold),
                                typeof(c1), typeof(c2), typeof(c3), typeof(n), typeof(alg)}(me,
                                                                                            ve,
                                                                                            pdm,
                                                                                            threshold,
                                                                                            c1,
                                                                                            c2,
                                                                                            c3,
                                                                                            n,
                                                                                            alg)
end
function factory(ce::SmythBrobyCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return SmythBrobyCovariance(; me = factory(ce.me, w), ve = factory(ce.ve, w),
                                pdm = ce.pdm, threshold = ce.threshold, c1 = ce.c1,
                                c2 = ce.c2, c3 = ce.c3, n = ce.n, alg = ce.alg)
end
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby0},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                end
            end
            den = (pos + neg)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:NormalisedSmythBroby0},
                    X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                end
            end
            den = (pos + neg)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby1},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                end
            end
            den = (pos + neg + nn)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:NormalisedSmythBroby1},
                    X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)), one(eltype(X)),
                                   one(eltype(X)), c1, c2, c3, n)
                end
            end
            den = (pos + neg + nn)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby2},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                end
            end
            rho[j, i] = rho[i, j] = pos - neg
        end
    end
    h = sqrt.(diag(rho))
    rho .= rho ⊘ (h * transpose(h))
    posdef!(ce.pdm, rho)
    return rho
end
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:NormalisedSmythBroby2},
                    X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                end
            end
            rho[j, i] = rho[i, j] = pos - neg
        end
    end
    h = sqrt.(diag(rho))
    rho .= Symmetric(rho ⊘ (h * transpose(h)), :U)
    posdef!(ce.pdm, rho)
    return rho
end
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber0},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cneg += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            den = (tpos + tneg)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:NormalisedSmythBrobyGerber0},
                    X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cneg += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            den = (tpos + tneg)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber1},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            cneg = 0
            cpos = 0
            cnn = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cneg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cnn += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            tnn = nn * cnn
            den = (tpos + tneg + tnn)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:NormalisedSmythBrobyGerber1},
                    X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            cneg = 0
            cpos = 0
            cnn = 0
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cneg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)), one(eltype(X)),
                                   one(eltype(X)), c1, c2, c3, n)
                    cnn += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            tnn = nn * cnn
            den = (tpos + tneg + tnn)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber2},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cneg += 1
                end
            end
            rho[j, i] = rho[i, j] = pos * cpos - neg * cneg
        end
    end
    h = sqrt.(diag(rho))
    rho .= rho ⊘ (h * transpose(h))
    posdef!(ce.pdm, rho)
    return rho
end
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:NormalisedSmythBrobyGerber2},
                    X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cneg += 1
                end
            end
            rho[j, i] = rho[i, j] = pos * cpos - neg * cneg
        end
    end
    h = sqrt.(diag(rho))
    rho .= Symmetric(rho ⊘ (h * transpose(h)), :U)
    posdef!(ce.pdm, rho)
    return rho
end
function StatsBase.cor(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                <:Any, <:Any,
                                                <:UnNormalisedSmythBrobyCovarianceAlgorithm},
                       X::AbstractMatrix; dims::Int = 1, mean = nothing)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = 1) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    return smythbroby(ce, X, mean_vec, std_vec)
end
function StatsBase.cov(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                <:Any, <:Any,
                                                <:UnNormalisedSmythBrobyCovarianceAlgorithm},
                       X::AbstractMatrix; dims::Int = 1, mean = nothing)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = 1) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    return smythbroby(ce, X, mean_vec, std_vec) ⊙ (std_vec ⊗ std_vec)
end
function StatsBase.cor(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                <:Any, <:Any,
                                                <:NormalisedSmythBrobyCovarianceAlgorithm},
                       X::AbstractMatrix; dims::Int = 1, mean = nothing)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = 1) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ⊘ std_vec
    return smythbroby(ce, X)
end
function StatsBase.cov(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                <:Any, <:Any,
                                                <:NormalisedSmythBrobyCovarianceAlgorithm},
                       X::AbstractMatrix; dims::Int = 1, mean = nothing)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? StatsBase.mean(ce.me, X; dims = 1) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ⊘ std_vec
    return smythbroby(ce, X) ⊙ (std_vec ⊗ std_vec)
end

export SmythBroby0, SmythBroby1, SmythBroby2, SmythBrobyGerber0, SmythBrobyGerber1,
       SmythBrobyGerber2, NormalisedSmythBroby0, NormalisedSmythBroby1,
       NormalisedSmythBroby2, NormalisedSmythBrobyGerber0, NormalisedSmythBrobyGerber1,
       NormalisedSmythBrobyGerber2, SmythBrobyCovariance
