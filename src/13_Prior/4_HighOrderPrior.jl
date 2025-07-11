function block_vec_pq(A, p, q)
    mp, nq = size(A)

    if !(mod(mp, p) == 0 && mod(nq, q) == 0)
        throw(DimensionMismatch("size(A) = $(size(A)), must be integer multiples of (p, q) = ($p, $q)"))
    end

    m = Int(mp / p)
    n = Int(nq / q)

    A_vec = Matrix{eltype(A)}(undef, m * n, p * q)
    for j in 0:(n - 1)
        Aj = Matrix{eltype(A)}(undef, m, p * q)
        for i in 0:(m - 1)
            Aij = vec(A[(1 + (i * p)):((i + 1) * p), (1 + (j * q)):((j + 1) * q)])
            Aj[i + 1, :] .= Aij
        end
        A_vec[(1 + (j * m)):((j + 1) * m), :] .= Aj
    end

    return A_vec
end
function duplication_matrix(n::Int, diag::Bool = true)
    m = div(n * (n + 1), 2)
    nsq = n^2
    v = zeros(Int, nsq)
    r = 1
    a = 1
    for i in 1:n
        b = i
        for j in 0:(i - 2)
            v[r] = b
            b += n - j - 1
            r += 1
        end

        for j in 0:(n - i)
            v[r] = a + j
            r += 1
        end
        a += n - i + 1
    end

    return if diag
        sparse(1:nsq, v, 1, nsq, m)
    else
        filtered_cols = Vector{Int}(undef, 0)
        filtered_rows = Vector{Int}(undef, 0)
        m = div(n * (n - 1), 2)
        rows = 1:nsq
        counts = Dict{Int, Int}()
        for i in v
            !haskey(counts, i) ? counts[i] = 1 : counts[i] += 1
        end
        repeated_elem = Set{Int}()
        for (key, value) in counts
            if value > 1
                push!(repeated_elem, key)
            end
        end
        repeated_elem = sort!(collect(repeated_elem))

        cols = Dict{Int, Int}()
        cntr = 0
        for col in repeated_elem
            cntr += 1
            cols[col] = cntr
        end

        for i in 1:nsq
            if !iszero(count(x -> x == v[i], repeated_elem))
                push!(filtered_rows, rows[i])
                push!(filtered_cols, cols[v[i]])
            end
        end

        sparse(filtered_rows, filtered_cols, 1, nsq, m)
    end
end
function elimination_matrix(n::Int, diag::Bool = true)
    nsq = n^2
    r = 1
    a = 1

    if diag
        m = div(n * (n + 1), 2)
        rg = 1:n
        b = 0
    else
        m = div(n * (n - 1), 2)
        rg = 2:n
        b = 1
    end

    v = zeros(Int, m)
    for i in rg
        for j in 0:(n - i)
            v[r] = a + j + b
            r += 1
        end
        a += n - i + 1
        b += i
    end

    return sparse(1:m, v, 1, m, nsq)
end
function summation_matrix(n::Int, diag::Bool = true)
    nsq = n^2
    r = 0
    a = 1
    v1 = zeros(Int, nsq)
    v2 = zeros(Int, nsq)
    rows2 = zeros(Int, nsq)

    if diag
        m = div(n * (n + 1), 2)
        b = 0
        rg = 1:n
    else
        m = div(n * (n - 1), 2)
        b = 1
        rg = 2:n
    end

    for i in rg
        r += i - 1
        for j in 0:(n - i)
            v1[r + j + 1] = a + j + b
        end
        for j in 1:(n - i)
            v2[r + j + 1] = a + j + b
            rows2[r + j + 1] = a + j
        end
        r += n - i + 1
        a += n - i + 1
        b += i
    end

    v1 = v1[.!iszero.(v1)]
    v2 = v2[.!iszero.(v2)]
    rows2 = rows2[.!iszero.(rows2)]

    return if diag
        a = sparse(1:m, v1, 1, m, nsq)
        b = sparse(rows2, v2, 1, m, nsq)
        a + b
    else
        sparse(1:m, v1, 2, m, nsq)
    end
end
function dup_elim_sum_matrices(n::Int)
    m = div(n * (n + 1), 2)
    nsq = n^2
    v1 = zeros(Int, nsq)
    v2 = zeros(Int, m)
    r1 = 1
    r2 = 1
    a = 1
    b2 = 0
    for i in 1:n
        b1 = i
        for j in 0:(i - 2)
            v1[r1] = b1
            b1 += n - j - 1
            r1 += 1
        end

        for j in 0:(n - i)
            v1[r1] = a + j
            v2[r2] = a + j + b2
            r1 += 1
            r2 += 1
        end
        a += n - i + 1
        b2 += i
    end

    d = sparse(1:nsq, v1, 1, nsq, m)
    l = sparse(1:m, v2, 1, m, nsq)
    s = transpose(d) * d * l

    return d, l, s
end
struct HighOrderPriorResult{T1 <: AbstractPriorResult,
                            T2 <: Union{Nothing, <:AbstractMatrix},
                            T3 <: Union{Nothing, <:AbstractMatrix},
                            T4 <: Union{Nothing, <:AbstractMatrix},
                            T5 <: Union{Nothing, <:AbstractMatrix},
                            T6 <: Union{Nothing, <:AbstractMatrix},
                            T7 <: Union{Nothing, <:AbstractMatrixProcessingEstimator}} <:
       AbstractPriorResult
    pr::T1
    kt::T2
    L2::T3
    S2::T4
    sk::T5
    V::T6
    skmp::T7
end
function HighOrderPriorResult(; pr::AbstractPriorResult,
                              kt::Union{Nothing, <:AbstractMatrix},
                              L2::Union{Nothing, <:AbstractMatrix},
                              S2::Union{Nothing, <:AbstractMatrix},
                              sk::Union{Nothing, <:AbstractMatrix},
                              V::Union{Nothing, <:AbstractMatrix},
                              skmp::Union{Nothing, <:AbstractMatrixProcessingEstimator})
    kt_flag = isa(kt, AbstractMatrix)
    L2_flag = isa(L2, AbstractMatrix)
    S2_flag = isa(S2, AbstractMatrix)
    if kt_flag || L2_flag || S2_flag
        @smart_assert(kt_flag && L2_flag && S2_flag)
        @smart_assert(!isempty(kt) && !isempty(L2) && !isempty(S2))
        assert_matrix_issquare(kt)
        N = length(pr.mu)
        @smart_assert(length(pr.mu)^2 == size(kt, 1))
        @smart_assert(size(L2) == size(S2) == (div(N * (N + 1), 2), N^2))
    end
    sk_flag = isa(sk, AbstractMatrix)
    V_flag = isa(V, AbstractMatrix)
    if sk_flag
        @smart_assert(!isempty(sk))
        @smart_assert(length(pr.mu)^2 == size(sk, 2))
    end
    if V_flag
        @smart_assert(!isempty(V))
        assert_matrix_issquare(V)
    end
    if sk_flag || V_flag
        @smart_assert(sk_flag && V_flag,
                      "If either sk or V, is nothing, both must be nothing.")
    end
    return HighOrderPriorResult{typeof(pr), typeof(kt), typeof(L2), typeof(S2), typeof(sk),
                                typeof(V), typeof(skmp)}(pr, kt, L2, S2, sk, V, skmp)
end
function dup_elim_sum_view(args...)
    return nothing, nothing, nothing
end
function dup_elim_sum_view(::AbstractMatrix, N)
    return dup_elim_sum_matrices(N)
end
function prior_view(pr::HighOrderPriorResult, i::AbstractVector)
    idx = fourth_moment_index_factory(length(pr.mu), i)
    kt = pr.kt
    L2, S2 = dup_elim_sum_view(kt, length(i))[2:3]
    sk = pr.sk
    skmp = pr.skmp
    sk = nothing_scalar_array_view_odd_order(sk, i, idx)
    V = __coskewness(sk, view(pr.X, :, i), skmp)
    return HighOrderPriorResult(; pr = prior_view(pr.pr, i),
                                kt = nothing_scalar_array_view(kt, idx), L2 = L2, S2 = S2,
                                sk = sk, V = V, skmp = skmp)
end
function Base.getproperty(obj::HighOrderPriorResult, sym::Symbol)
    return if sym == :X
        obj.pr.X
    elseif sym == :mu
        obj.pr.mu
    elseif sym == :sigma
        obj.pr.sigma
    elseif sym == :chol
        obj.pr.chol
    elseif sym == :w
        obj.pr.w
    elseif sym == :loadings
        obj.pr.loadings
    elseif sym == :f_mu
        obj.pr.f_mu
    elseif sym == :f_sigma
        obj.pr.f_sigma
    elseif sym == :f_w
        obj.pr.f_w
    else
        getfield(obj, sym)
    end
end
struct HighOrderPriorEstimator{T1 <: AbstractLowOrderPriorEstimatorMap_1o2_1o2,
                               T2 <: Union{Nothing, <:CokurtosisEstimator},
                               T3 <: Union{Nothing, <:CoskewnessEstimator}} <:
       AbstractHighOrderPriorEstimator
    pe::T1
    kte::T2
    ske::T3
end
function factory(pe::HighOrderPriorEstimator,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return HighOrderPriorEstimator(; pe = factory(pe.pe, w), kte = factory(pe.kte, w),
                                   ske = factory(pe.ske, w))
end
function HighOrderPriorEstimator(;
                                 pe::AbstractLowOrderPriorEstimatorMap_1o2_1o2 = EmpiricalPriorEstimator(),
                                 kte::Union{Nothing, <:CokurtosisEstimator} = Cokurtosis(;
                                                                                         alg = Full()),
                                 ske::Union{Nothing, <:CoskewnessEstimator} = Coskewness(;
                                                                                         alg = Full()))
    return HighOrderPriorEstimator{typeof(pe), typeof(kte), typeof(ske)}(pe, kte, ske)
end
function Base.getproperty(obj::HighOrderPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function prior(pe::HighOrderPriorEstimator, X::AbstractMatrix,
               F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    pr = prior(pe.pe, X, F; kwargs...)
    (; X, mu) = pr
    kt = cokurtosis(pe.kte, X; mean = transpose(mu), kwargs...)
    L2, S2 = !isnothing(kt) ? dup_elim_sum_matrices(length(mu))[2:3] : (nothing, nothing)
    sk, V = coskewness(pe.ske, X; mean = transpose(mu), kwargs...)
    return HighOrderPriorResult(; pr = pr, kt = kt, L2 = L2, S2 = S2, sk = sk, V = V,
                                skmp = isnothing(sk) ? nothing : pe.ske.mp)
end

export HighOrderPriorResult, HighOrderPriorEstimator
