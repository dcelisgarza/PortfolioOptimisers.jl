"""
    block_vec_pq(A::MatNum, p::Integer, q::Integer)

Block vectorisation operator.

`block_vec_pq` transforms a matrix `A` into a block vectorised form, partitioning `A` into blocks of size `(p, q)` and stacking the vectorised blocks row-wise. This is useful for higher-order moment computations and tensor manipulations in portfolio analytics.

# Arguments

  - `A`: Input matrix of size `(m * p, n * q)`, where `m` and `n` are integers.
  - `p`: Number of rows in each block.
  - `q`: Number of columns in each block.

# Returns

  - `A_vec::Matrix`: Block vectorised matrix of size `(m * n, p * q)`.

# Validation

  - `size(A, 1)` must be an integer multiple of `p`.
  - `size(A, 2)` must be an integer multiple of `q`.

# Examples

```jldoctest
julia> A = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16];

julia> PortfolioOptimisers.block_vec_pq(A, 2, 2)
4×4 Matrix{Int64}:
  1   5   2   6
  9  13  10  14
  3   7   4   8
 11  15  12  16
```

# Related

  - [`dup_elim_sum_matrices`](@ref)

# References

  - $(ref_dict[:cajas2025]) Appendix A.1, Equation A.14.
"""
function block_vec_pq(A::MatNum, p::Integer, q::Integer)
    mp, nq = size(A)
    @argcheck(mod(mp, p) == 0,
              DomainError("size(A, 1) = $mp must be an integer multiple of p = $p"))
    @argcheck(mod(nq, q) == 0,
              DomainError("size(A, 2) = $nq must be an integer multiple of q = $q"))
    m = Int(mp / p)
    n = Int(nq / q)
    A_vec = Matrix{eltype(A)}(undef, m * n, p * q)
    for j in 0:(n - 1)
        Aj = Matrix{eltype(A)}(undef, m, p * q)
        for i in 0:(m - 1)
            Aij = vec(A[(1 + (i * p)):((i + 1) * p), (1 + (j * q)):((j + 1) * q)])
            Aj[i + 1, :] = Aij
        end
        A_vec[(1 + (j * m)):((j + 1) * m), :] = Aj
    end
    return A_vec
end
# COV_EXCL_START
"""
    duplication_matrix(n::Int, diag::Bool = true)

Construct the duplication matrix for a symmetric matrix of size `n × n`.

The duplication matrix `D` maps the vech (half-vectorisation) of a symmetric matrix to its full vec. Used internally in coskewness and cokurtosis computation.

# Arguments

  - `n`: Size of the symmetric matrix.
  - `diag`: Whether to include the diagonal elements.

# Returns

  - Sparse duplication matrix.

# Related

  - [`elimination_matrix`](@ref)
  - [`summation_matrix`](@ref)
"""
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
        SparseArrays.sparse(1:nsq, v, 1, nsq, m)
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
        SparseArrays.sparse(filtered_rows, filtered_cols, 1, nsq, m)
    end
end
"""
    elimination_matrix(n::Int, diag::Bool = true)

Construct the elimination matrix for a symmetric matrix of size `n × n`.

The elimination matrix `L` extracts the unique (lower triangular) elements of a symmetric matrix. Used internally in coskewness and cokurtosis computation.

# Arguments

  - `n`: Size of the symmetric matrix.
  - `diag`: Whether to include the diagonal elements.

# Returns

  - Sparse elimination matrix.

# Related

  - [`duplication_matrix`](@ref)
  - [`summation_matrix`](@ref)
"""
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

    return SparseArrays.sparse(1:m, v, 1, m, nsq)
end
"""
    summation_matrix(n::Int, diag::Bool = true)

Construct the summation matrix for a symmetric matrix of size `n × n`.

The summation matrix `S` adds up contributions from both triangular halves of a symmetric matrix. Used internally in coskewness and cokurtosis computation.

# Arguments

  - `n`: Size of the symmetric matrix.
  - `diag`: Whether to include the diagonal elements.

# Returns

  - Sparse summation matrix.

# Related

  - [`duplication_matrix`](@ref)
  - [`elimination_matrix`](@ref)
"""
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
        a = SparseArrays.sparse(1:m, v1, 1, m, nsq)
        b = SparseArrays.sparse(rows2, v2, 1, m, nsq)
        a + b
    else
        SparseArrays.sparse(1:m, v1, 2, m, nsq)
    end
end
# COV_EXCL_STOP
"""
    dup_elim_sum_matrices(n::Int)

Construct duplication, elimination, and summation matrices for symmetric matrix vectorisation.

`dup_elim_sum_matrices` returns the duplication matrix `D`, elimination matrix `L`, and summation matrix `S` for symmetric matrices of size `N × N`. These matrices are used in higher-order moment computations, tensor manipulations, and efficient vectorisation of symmetric matrices in portfolio analytics.

For a symmetric `A` of size `n × n`, the three satisfy `D * vech(A) == vec(A)`, `L * vec(A) == vech(A)`, and `S == transpose(D) * D * L`, which makes `sum(S * vec(A)) == sum(vec(A))`: `S` reads the lower triangle and weights each off-diagonal entry by the two places it occupies in `vec(A)`.

# Arguments

  - `n`: Size of the symmetric matrix (integer).

# Returns

  - `D::SparseMatrixCSC{Int64, Int64}`: Duplication matrix (`n^2 × m`), where `m = n(n+1)/2`.
  - `L::SparseMatrixCSC{Int64, Int64}`: Elimination matrix (`m × n^2`).
  - `S::SparseMatrixCSC{Int64, Int64}`: Summation matrix (`m × n^2`).

# Validation

  - `n` must be a positive integer.

# Examples

```jldoctest
julia> D, L, S = PortfolioOptimisers.dup_elim_sum_matrices(3);

julia> D
9×6 SparseArrays.SparseMatrixCSC{Int64, Int64} with 9 stored entries:
 1  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅
 ⋅  1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  1  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  1

julia> L
6×9 SparseArrays.SparseMatrixCSC{Int64, Int64} with 6 stored entries:
 1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1

julia> S
6×9 SparseArrays.SparseMatrixCSC{Int64, Int64} with 6 stored entries:
 1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  2  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  2  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  2  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1
```

# Related

  - [`block_vec_pq`](@ref)
  - [`duplication_matrix`](@ref)
  - [`elimination_matrix`](@ref)
  - [`summation_matrix`](@ref)

# References

  - $(ref_dict[:cajas2025]) Appendix A.2, Equations A.25 to A.27.
"""
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

    d = SparseArrays.sparse(1:nsq, v1, 1, nsq, m)
    l = SparseArrays.sparse(1:m, v2, 1, m, nsq)
    s = transpose(d) * d * l

    return d, l, s
end
"""
    dup_elim_sum_view(args...)

Answer with three `nothing`s when the first argument is not a matrix.

The fallback of [`dup_elim_sum_view`](@ref). It builds nothing and reads none of its arguments; the matrix method is the one that calls [`dup_elim_sum_matrices`](@ref).

Its two call sites are in [`port_opt_view`](@ref) on a [`HighOrderPrior`](@ref), which passes `pr.kt` as the first argument. No estimator in the library builds a carrier that reaches this method: `sk` and `V` travel together, and `kt`, `L2` and `S2` do too, so a carrier holding any of them holds `kt`. A hand-built carrier can — `D2` is the one moment field the constructor accepts on its own — and it is the case this method answers.

# Arguments

  - `args...`: Any arguments. None is read.

# Returns

  - `(nothing, nothing, nothing)::Tuple{Nothing, Nothing, Nothing}`.

# Related

  - [`duplication_matrix`](@ref)
  - [`elimination_matrix`](@ref)
  - [`summation_matrix`](@ref)
"""
function dup_elim_sum_view(args...)
    return nothing, nothing, nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute duplication, elimination, and summation matrices for a matrix argument.

Overload of [`dup_elim_sum_view`](@ref) for a matrix argument. Returns the three matrices for dimension `n = size(M, 2)`.

# Related

  - [`duplication_matrix`](@ref)
  - [`elimination_matrix`](@ref)
  - [`summation_matrix`](@ref)
"""
function dup_elim_sum_view(::MatNum, n)
    return dup_elim_sum_matrices(n)
end
"""
$(DocStringExtensions.TYPEDEF)

High order prior estimator for asset returns.

`HighOrderPriorEstimator` is a composite estimator that computes high order moments (coskewness and cokurtosis) for asset returns, in addition to low order moments (mean and covariance). It combines a low order prior estimator, a cokurtosis estimator, and a coskewness estimator to produce a [`HighOrderPrior`](@ref) result containing all relevant moments for advanced portfolio analytics.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HighOrderPriorEstimator(;
        pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
        kte::Option{<:CokurtosisEstimator} = Cokurtosis(;
            alg = FullMoment()
        ),
        ske::Option{<:CoskewnessEstimator} = Coskewness(;
            alg = FullMoment()
        )
    ) -> HighOrderPriorEstimator

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `pe`: Recursively updated via [`factory`](@ref).
  - `kte`: Recursively updated via [`factory`](@ref).
  - `ske`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `kte`: Recursively viewed via [`port_opt_view`](@ref).
  - `ske`: Recursively viewed via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> HighOrderPriorEstimator()
HighOrderPriorEstimator
   pe ┼ EmpiricalPrior
      │        ce ┼ PortfolioOptimisersCovariance
      │           │   ce ┼ Covariance
      │           │      │    me ┼ SimpleExpectedReturns
      │           │      │       │   w ┴ nothing
      │           │      │    ce ┼ GeneralCovariance
      │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │           │      │       │    w ┴ nothing
      │           │      │   alg ┴ FullMoment()
      │           │   mp ┼ MatrixProcessing
      │           │      │     pdm ┼ Posdef
      │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │           │      │      dn ┼ nothing
      │           │      │      dt ┼ nothing
      │           │      │     alg ┼ nothing
      │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │        me ┼ SimpleExpectedReturns
      │           │   w ┴ nothing
      │   horizon ┴ nothing
  kte ┼ Cokurtosis
      │    me ┼ SimpleExpectedReturns
      │       │   w ┴ nothing
      │    mp ┼ MatrixProcessing
      │       │     pdm ┼ Posdef
      │       │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │       │      dn ┼ nothing
      │       │      dt ┼ nothing
      │       │     alg ┼ nothing
      │       │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │   alg ┼ FullMoment()
      │     w ┴ nothing
  ske ┼ Coskewness
      │    me ┼ SimpleExpectedReturns
      │       │   w ┴ nothing
      │    mp ┼ MatrixProcessing
      │       │     pdm ┼ Posdef
      │       │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │       │      dn ┼ nothing
      │       │      dt ┼ nothing
      │       │     alg ┼ nothing
      │       │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │   alg ┼ FullMoment()
      │     w ┴ nothing
```

# Related

  - [`AbstractHighOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_F_AF`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`CokurtosisEstimator`](@ref)
  - [`CoskewnessEstimator`](@ref)
  - [`Cokurtosis`](@ref)
  - [`Coskewness`](@ref)
  - [`prior`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.1.4, Equations 3.6 and 3.7.
  - $(ref_dict[:pkurt])
"""
@propagatable @concrete struct HighOrderPriorEstimator <: AbstractHighOrderPriorEstimator
    """
    $(field_dict[:pe])
    """
    @fprop @vprop pe
    """
    $(field_dict[:kte])
    """
    @fprop @vprop kte
    """
    $(field_dict[:ske])
    """
    @fprop @vprop ske
    function HighOrderPriorEstimator(pe::AbstractLowOrderPriorEstimator_A_F_AF,
                                     kte::Option{<:CokurtosisEstimator},
                                     ske::Option{<:CoskewnessEstimator})
        return new{typeof(pe), typeof(kte), typeof(ske)}(pe, kte, ske)
    end
end
function HighOrderPriorEstimator(;
                                 pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                                 kte::Option{<:CokurtosisEstimator} = Cokurtosis(;
                                                                                 alg = FullMoment()),
                                 ske::Option{<:CoskewnessEstimator} = Coskewness(;
                                                                                 alg = FullMoment()))::HighOrderPriorEstimator
    return HighOrderPriorEstimator(pe, kte, ske)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties HighOrderPriorEstimator begin
    forward(pe, me, ce)
end
"""
    prior(pe::HighOrderPriorEstimator, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Compute high order prior moments for asset returns using a composite estimator.

`prior` estimates the mean, covariance, coskewness, and cokurtosis of asset returns using the specified high order prior estimator. It first computes low order moments (mean and covariance) using the embedded prior estimator, then computes coskewness and cokurtosis tensors using the provided coskewness and cokurtosis estimators. Optionally, factor returns `F` can be provided for factor-based estimation. The result is returned as a [`HighOrderPrior`](@ref) object.

# Mathematical definition

In addition to the first and second moments, the high order estimator computes the coskewness matrix ``\\mathbf{M}_3`` and the **square** cokurtosis matrix ``\\mathbf{\\Sigma}_4``:

```math
\\begin{align}
\\mathbf{W} &= (\\mathbf{Z} \\otimes \\boldsymbol{1}_N^\\intercal) \\odot (\\boldsymbol{1}_N^\\intercal \\otimes \\mathbf{Z})\\,, \\\\
\\mathbf{M}_3 &= \\frac{1}{T} \\mathbf{Z}^\\intercal \\mathbf{W}\\,, \\\\
\\mathbf{\\Sigma}_4 &= \\frac{1}{T} \\mathbf{W}^\\intercal \\mathbf{W}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{Z}``: ``T \\times N`` matrix of returns with each column's mean removed.
  - ``\\mathbf{W}``: ``T \\times N^2`` matrix of pairwise products of the centred returns.
  - ``\\mathbf{M}_3``: ``N \\times N^2`` coskewness matrix, `sk`.
  - ``\\mathbf{\\Sigma}_4``: ``N^2 \\times N^2`` square cokurtosis matrix, `kt`.
  - ``\\boldsymbol{1}_N``: ``N \\times 1`` vector of ones.
  - $(math_dict[:T])
  - ``\\otimes``: Kronecker product.
  - ``\\odot``: Hadamard product.

``\\mathbf{\\Sigma}_4`` is the square form and is ``N^2 \\times N^2``. It is not the ``N \\times N^3`` cokurtosis matrix ``\\mathbf{M}_4`` of the same source, which the library never builds.

`pe.kte` computes `kt` and `pe.ske` computes `sk`, so a non-default `alg` — a semi-comoment, an exponentially weighted one — replaces the displays above rather than refining them. `V` is the negative spectral coskewness of `sk`, and either estimator set to `nothing` drops its moment from the result.

# Arguments

  - `pe`: High order prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor returns matrix (observations × factors).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to underlying estimators.

# Returns

  - `pr::HighOrderPrior`: Result object containing asset returns, mean vector, covariance matrix, coskewness tensor, cokurtosis tensor, and related quantities.

# Validation

  - `dims in (1, 2)`.

# Related

  - [`HighOrderPriorEstimator`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`prior`](@ref)
"""
function prior(pe::HighOrderPriorEstimator, X::MatNum, F::Option{<:MatNum} = nothing;
               dims::Int = 1, kwargs...)
    X, F = dims_oriented(dims, X, F)
    pr = prior(pe.pe, X, F; kwargs...)
    kt = cokurtosis(pe.kte, X; kwargs...)
    D2 = nothing
    L2 = nothing
    S2 = nothing
    sk, V = coskewness(pe.ske, X; kwargs...)
    if !isnothing(kt) && !isnothing(sk)
        D2, L2, S2 = dup_elim_sum_matrices(size(pr.X, 2))
    elseif !isnothing(kt) && isnothing(sk)
        L2, S2 = dup_elim_sum_matrices(size(pr.X, 2))[2:3]
    end
    return HighOrderPrior(; pr = pr, kt = kt, D2 = D2, L2 = L2, S2 = S2, sk = sk, V = V,
                          skmp = isnothing(sk) ? nothing : pe.ske.mp)
end

function factor_residual_config(pe::HighOrderPriorEstimator)
    # The low-order block of the result is the wrapped estimator's own, residual block and
    # all, so this estimator forwards rather than answering for itself (see
    # [`factor_residual_config`](@ref)).
    return factor_residual_config(pe.pe)
end

export HighOrderPriorEstimator
