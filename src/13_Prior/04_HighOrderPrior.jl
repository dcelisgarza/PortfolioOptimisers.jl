"""
    block_vec_pq(A::MatNum, p::Integer, q::Integer)

Block vectorisation operator.

`block_vec_pq` transforms a matrix `A` into a block vectorised form, partitioning `A` into blocks of size `(p, q)` and writing each vectorised block as one row. This is useful for higher-order moment computations and tensor manipulations in portfolio analytics.

# Mathematical definition

Partition ``\\mathbf{A}`` into ``m \\times n`` blocks ``\\mathbf{A}_{ij}`` of size ``p \\times q``. The block vectorisation writes each block as one row, in block-column order:

```math
\\mathcal{V}_{p,q}(\\mathbf{A}) = \\begin{bmatrix}
\\mathrm{vec}(\\mathbf{A}_{11})^\\intercal \\\\
\\vdots \\\\
\\mathrm{vec}(\\mathbf{A}_{m1})^\\intercal \\\\
\\mathrm{vec}(\\mathbf{A}_{12})^\\intercal \\\\
\\vdots \\\\
\\mathrm{vec}(\\mathbf{A}_{mn})^\\intercal
\\end{bmatrix}\\,.
```

Where:

  - ``\\mathbf{A}_{ij}``: block ``(i, j)`` of ``\\mathbf{A}``, holding rows ``(i-1)p+1`` to ``ip`` and columns ``(j-1)q+1`` to ``jq``.
  - ``\\mathrm{vec}``: column-major vectorisation.
  - ``m = \\mathrm{size}(\\mathbf{A}, 1) / p``, ``n = \\mathrm{size}(\\mathbf{A}, 2) / q``: the block counts.

Row ``(j-1)m + i`` of the result is ``\\mathrm{vec}(\\mathbf{A}_{ij})^\\intercal``, so the block **column** index runs slowest. A square partition, ``m = n`` and ``p = q``, cannot separate this order from the one that runs the block row index slowest, because the two differ by a permutation that is the identity there.

# Algorithm

 1. Read `size(A)` into `mp` and `nq`, and check both divisibility conditions.
 2. Take the block counts `m = mp / p` and `n = nq / q`.
 3. Allocate `A_vec`, of size `(m * n, p * q)`.
 4. For each block column `j`, build `Aj`, whose `i`-th row is the vectorisation of block `(i, j)` of `A`.
 5. Write `Aj` into rows `j * m + 1` to `(j + 1) * m` of `A_vec`.

# Arguments

  - `A`: Input matrix of size `(m * p, n * q)`, where `m` and `n` are integers.
  - `p`: Number of rows in each block.
  - `q`: Number of columns in each block.

# Validation

  - `size(A, 1)` must be an integer multiple of `p`.
  - `size(A, 2)` must be an integer multiple of `q`.

# Returns

  - `A_vec::Matrix`: Block vectorised matrix of size `(m * n, p * q)`.

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
"""
    duplication_matrix(n::Int, diag::Bool = true)

Construct the duplication matrix for a symmetric matrix of size `n × n`.

The duplication matrix `D` maps the vech (half-vectorisation) of a symmetric matrix to its full vec. Used internally in coskewness and cokurtosis computation.

# Mathematical definition

``\\mathbf{D}_n`` is defined by the identity it restores, for every symmetric ``\\mathbf{A}`` of size ``n \\times n``:

```math
\\mathbf{D}_n \\mathrm{vech}(\\mathbf{A}) = \\mathrm{vec}(\\mathbf{A})\\,.
```

With `diag = false` the half-vectorisation drops the diagonal, and the identity restores the hollow matrix:

```math
\\mathbf{D}_n^{-} \\mathrm{vech}^{-}(\\mathbf{A}) = \\mathrm{vec}(\\mathbf{A} - \\mathrm{diag}(\\mathbf{A}))\\,.
```

Where:

  - ``\\mathrm{vec}``: column-major vectorisation, of length ``n^2``.
  - ``\\mathrm{vech}``: half-vectorisation, the lower triangle read column by column, of length ``n(n+1)/2``.
  - ``\\mathrm{vech}^{-}``: the strictly lower triangle read the same way, of length ``n(n-1)/2``.
  - ``\\mathrm{diag}(\\mathbf{A})``: the diagonal of ``\\mathbf{A}`` held as a matrix.

Each row of ``\\mathbf{D}_n`` carries exactly one entry, so the matrix selects rather than sums. A diagonal entry of ``\\mathbf{A}`` is selected once and an off-diagonal entry twice, which is what makes ``\\mathbf{D}_n^\\intercal \\mathbf{D}_n`` the weight matrix that [`summation_matrix`](@ref) applies.

# Algorithm

 1. Take `m = n(n+1)/2` and `nsq = n^2`.
 2. Fill `v`, whose `r`-th entry is the position in ``\\mathrm{vech}(\\mathbf{A})`` of the entry that row `r` of ``\\mathrm{vec}(\\mathbf{A})`` holds. The inner loops walk the strictly upper part of a column first, then its lower part.
 3. When `diag` is `true`, return the sparse matrix carrying a one at each `(r, v[r])`, of size `nsq × m`.
 4. When `diag` is `false`, count how often each position occurs in `v`, giving `counts`.
 5. Keep the positions that occur more than once — the off-diagonal ones — and renumber them from one, giving `cols`.
 6. Keep every row of ``\\mathrm{vec}(\\mathbf{A})`` whose position survives step 5, giving `filtered_rows` and `filtered_cols`.
 7. Return the sparse matrix carrying a one at each kept pair, of size `nsq × n(n-1)/2`.

# Arguments

  - `n`: Size of the symmetric matrix.
  - `diag`: Whether to include the diagonal elements.

# Returns

  - Sparse duplication matrix.

# Related

  - [`elimination_matrix`](@ref)
  - [`summation_matrix`](@ref)
  - [`dup_elim_sum_matrices`](@ref): builds this matrix and its two siblings in one walk.

# References

  - $(ref_dict[:cajas2025]) Appendix A.2, Equation A.25.
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

# Mathematical definition

``\\mathbf{L}_n`` is defined by the identity it applies, for every ``\\mathbf{A}`` of size ``n \\times n``:

```math
\\mathbf{L}_n \\mathrm{vec}(\\mathbf{A}) = \\mathrm{vech}(\\mathbf{A})\\,.
```

With `diag = false` it drops the diagonal as well:

```math
\\mathbf{L}_n^{-} \\mathrm{vec}(\\mathbf{A}) = \\mathrm{vech}^{-}(\\mathbf{A})\\,.
```

Where:

  - ``\\mathrm{vec}``: column-major vectorisation, of length ``n^2``.
  - ``\\mathrm{vech}``: half-vectorisation, the lower triangle read column by column, of length ``n(n+1)/2``.
  - ``\\mathrm{vech}^{-}``: the strictly lower triangle read the same way, of length ``n(n-1)/2``.

The identity holds for any square ``\\mathbf{A}``, symmetric or not, because ``\\mathbf{L}_n`` only reads the lower triangle. Dropping the diagonal removes the ``n`` rows that read it, so the row count falls from ``n(n+1)/2`` to ``n(n-1)/2`` while the column count stays ``n^2``.

# Algorithm

 1. Take `nsq = n^2`.
 2. Read `diag`, and set the row count `m`, the column range `rg` and the offset `b` from it: `m = n(n+1)/2`, `rg = 1:n` and `b = 0` when `diag` is `true`, and `m = n(n-1)/2`, `rg = 2:n` and `b = 1` otherwise.
 3. Fill `v`, whose `r`-th entry is the position in ``\\mathrm{vec}(\\mathbf{A})`` of the `r`-th entry of the half-vectorisation. `b` carries the offset that skips the entries above the diagonal, and — under `diag = false` — the diagonal entry too.
 4. Return the sparse matrix carrying a one at each `(r, v[r])`, of size `m × nsq`.

# Arguments

  - `n`: Size of the symmetric matrix.
  - `diag`: Whether to include the diagonal elements.

# Returns

  - Sparse elimination matrix.

# Related

  - [`duplication_matrix`](@ref)
  - [`summation_matrix`](@ref)
  - [`dup_elim_sum_matrices`](@ref): builds this matrix and its two siblings in one walk.

# References

  - $(ref_dict[:cajas2025]) Appendix A.2, Equation A.26.
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

# Mathematical definition

``\\mathbf{S}_n`` is the elimination matrix reweighted by the multiplicity that ``\\mathbf{D}_n`` restores:

```math
\\mathbf{S}_n = \\mathbf{D}_n^\\intercal \\mathbf{D}_n \\mathbf{L}_n\\,,
```

so that, for every ``\\mathbf{A}`` of size ``n \\times n``,

```math
\\boldsymbol{1}^\\intercal \\mathbf{S}_n \\mathrm{vec}(\\mathbf{A}) = \\boldsymbol{1}^\\intercal \\mathrm{vec}(\\mathbf{A})\\,.
```

Where:

  - ``\\mathbf{D}_n``: the duplication matrix of [`duplication_matrix`](@ref).
  - ``\\mathbf{L}_n``: the elimination matrix of [`elimination_matrix`](@ref).
  - ``\\boldsymbol{1}``: vector of ones of the length its neighbour needs.

``\\mathbf{S}_n`` reads the lower triangle and weights each entry by the number of places it occupies in ``\\mathrm{vec}(\\mathbf{A})``: one for a diagonal entry and two for an off-diagonal one. The sum identity above follows, and it holds for any square ``\\mathbf{A}``, symmetric or not.

The same construction with `diag = false` gives ``\\mathbf{S}_n^{-} = (\\mathbf{D}_n^{-})^\\intercal \\mathbf{D}_n^{-} \\mathbf{L}_n^{-}``, which weights every one of its ``n(n-1)/2`` rows by two. Its sum identity therefore reaches the off-diagonal entries alone, ``\\boldsymbol{1}^\\intercal \\mathbf{S}_n^{-} \\mathrm{vec}(\\mathbf{A}) = \\boldsymbol{1}^\\intercal \\mathrm{vec}(\\mathbf{A} - \\mathrm{diag}(\\mathbf{A}))``.

# Algorithm

The body builds the product of the definition directly, without forming ``\\mathbf{D}_n``.

 1. Take `nsq = n^2`. Read `diag`, and set the row count `m`, the column range `rg` and the offset `b` from it, exactly as [`elimination_matrix`](@ref) does.
 2. Walk the columns in `rg`. Write into `v1` the ``\\mathrm{vec}`` position of every entry of the half-vectorisation, and into `v2` and `rows2` the ``\\mathrm{vec}`` position and the half-vectorisation row of each **strictly** lower entry.
 3. Drop the zero entries of `v1`, `v2` and `rows2`, which are the slots the walk never filled.
 4. When `diag` is `true`, return the sum of two sparse matrices: one carrying a one at each `(r, v1[r])`, which is ``\\mathbf{L}_n``, and one carrying a one at each `(rows2[k], v2[k])`, which adds the second unit to every off-diagonal row.
 5. When `diag` is `false`, every row is off-diagonal, so return the sparse matrix carrying a **two** at each `(r, v1[r])`. `v2` and `rows2` go unread on this branch.

# Arguments

  - `n`: Size of the symmetric matrix.
  - `diag`: Whether to include the diagonal elements.

# Returns

  - Sparse summation matrix.

# Related

  - [`duplication_matrix`](@ref)
  - [`elimination_matrix`](@ref)
  - [`dup_elim_sum_matrices`](@ref): builds this matrix and its two siblings in one walk, through the product of the definition rather than by the construction above.

# References

  - $(ref_dict[:cajas2025]) Appendix A.2, Equation A.27.
  - $(ref_dict[:pkurt])
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
"""
    dup_elim_sum_matrices(n::Int)

Construct duplication, elimination, and summation matrices for symmetric matrix vectorisation.

`dup_elim_sum_matrices` returns the duplication matrix `D`, elimination matrix `L`, and summation matrix `S` for symmetric matrices of size `N × N`. These matrices are used in higher-order moment computations, tensor manipulations, and efficient vectorisation of symmetric matrices in portfolio analytics.

The three are the `diag = true` matrices of [`duplication_matrix`](@ref), [`elimination_matrix`](@ref) and [`summation_matrix`](@ref), built in one walk of the columns rather than in three.

# Mathematical definition

For every ``\\mathbf{A}`` of size ``n \\times n``:

```math
\\begin{align}
\\mathbf{D}_n \\mathrm{vech}(\\mathbf{A}) &= \\mathrm{vec}(\\mathbf{A})\\,, \\\\
\\mathbf{L}_n \\mathrm{vec}(\\mathbf{A}) &= \\mathrm{vech}(\\mathbf{A})\\,, \\\\
\\mathbf{S}_n &= \\mathbf{D}_n^\\intercal \\mathbf{D}_n \\mathbf{L}_n\\,.
\\end{align}
```

Where:

  - ``\\mathrm{vec}``: column-major vectorisation, of length ``n^2``.
  - ``\\mathrm{vech}``: half-vectorisation, the lower triangle read column by column, of length ``n(n+1)/2``.

The third line makes ``\\boldsymbol{1}^\\intercal \\mathbf{S}_n \\mathrm{vec}(\\mathbf{A}) = \\boldsymbol{1}^\\intercal \\mathrm{vec}(\\mathbf{A})``: ``\\mathbf{S}_n`` reads the lower triangle and weights each off-diagonal entry by the two places it occupies in ``\\mathrm{vec}(\\mathbf{A})``.

# Algorithm

 1. Check that `n` is positive.
 2. Take `m = n(n+1)/2` and `nsq = n^2`.
 3. Walk the columns once, filling `v1` and `v2` together. `v1` is the column index vector of [`duplication_matrix`](@ref) and `v2` is that of [`elimination_matrix`](@ref).
 4. Build `d`, the sparse matrix carrying a one at each `(r, v1[r])`, of size `nsq × m`.
 5. Build `l`, the sparse matrix carrying a one at each `(r, v2[r])`, of size `m × nsq`.
 6. Build `s` as the product `transpose(d) * d * l` of the definition above, rather than by the direct construction of [`summation_matrix`](@ref).

# Arguments

  - `n`: Size of the symmetric matrix (integer).

# Validation

  - `n > 0`.

# Returns

  - `D::SparseMatrixCSC{Int64, Int64}`: Duplication matrix (`n^2 × m`), where `m = n(n+1)/2`.
  - `L::SparseMatrixCSC{Int64, Int64}`: Elimination matrix (`m × n^2`).
  - `S::SparseMatrixCSC{Int64, Int64}`: Summation matrix (`m × n^2`).

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
    @argcheck(n > 0, DomainError("n = $n must be a positive integer"))
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

This is a varargs fallback, so it also answers any call whose argument count the matrix method does not take. `dup_elim_sum_view(M, n)` reaches the matrix method for a matrix `M`; `dup_elim_sum_view(M, n, extra)` reaches this one.

# Algorithm

 1. Return `(nothing, nothing, nothing)`, reading no argument.

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

Compute duplication, elimination, and summation matrices at the dimension the caller names.

Overload of [`dup_elim_sum_view`](@ref) for a matrix first argument. **The matrix is read for dispatch alone**, and the dimension is the second argument `n`, not `size` of the matrix. [`port_opt_view`](@ref) on a [`HighOrderPrior`](@ref) relies on that: it passes the carrier's full `N^2 × N^2` cokurtosis and the asset count of the **subproblem**, so the three matrices come back rebuilt at the smaller dimension rather than cut from the larger ones.

# Algorithm

 1. Forward `n` to [`dup_elim_sum_matrices`](@ref), and return its three matrices.

# Arguments

  - The first argument: any matrix. It selects this method and is not read.
  - `n`: Size of the symmetric matrix the three matrices are built for.

# Returns

  - `(D, L, S)::Tuple{SparseMatrixCSC, SparseMatrixCSC, SparseMatrixCSC}`: The three matrices of [`dup_elim_sum_matrices`](@ref) at dimension `n`.

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
      │           │      │   alg ┼ FullMoment()
      │           │      │     w ┴ nothing
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
      │      me ┼ SimpleExpectedReturns
      │         │   w ┴ nothing
      │      mp ┼ MatrixProcessing
      │         │     pdm ┼ Posdef
      │         │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │         │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │         │      dn ┼ nothing
      │         │      dt ┼ nothing
      │         │     alg ┼ nothing
      │         │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │     alg ┼ FullMoment()
      │       w ┼ nothing
      │   cache ┴ nothing
  ske ┼ Coskewness
      │      me ┼ SimpleExpectedReturns
      │         │   w ┴ nothing
      │      mp ┼ MatrixProcessing
      │         │     pdm ┼ Posdef
      │         │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │         │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │         │      dn ┼ nothing
      │         │      dt ┼ nothing
      │         │     alg ┼ nothing
      │         │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │     alg ┼ FullMoment()
      │       w ┼ nothing
      │   cache ┴ nothing
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

# Algorithm

 1. Orient `X` and `F` to `observations × variables` with [`dims_oriented`](@ref).
 2. Compute the low order block `pr` with `pe.pe`, passing `F` through.
 3. Compute the square cokurtosis `kt` with `pe.kte`. A `nothing` estimator gives a `nothing` moment.
 4. Compute the coskewness `sk` and its negative spectral form `V` with `pe.ske`. A `nothing` estimator gives `nothing` for both.
 5. Build the structure matrices at the asset count `size(pr.X, 2)` with [`dup_elim_sum_matrices`](@ref). Take all three when steps 3 and 4 both produced a moment, take `L2` and `S2` alone when step 3 produced one and step 4 did not, and take none otherwise. `D2` serves `sk` and the pair `L2`, `S2` serves `kt`, which is why the second case leaves `D2` as `nothing`.
 6. Assemble the [`HighOrderPrior`](@ref) through its keyword constructor, carrying `pe.ske.mp` as `skmp` when step 4 produced an `sk`. Every `@argcheck` of the constructor runs on the shapes steps 3 to 5 produced.

# Arguments

  - `pe`: High order prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor returns matrix (observations × factors).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to underlying estimators.

# Validation

  - `dims in (1, 2)`.

# Returns

  - `pr::HighOrderPrior`: Result object containing asset returns, mean vector, covariance matrix, coskewness tensor, cokurtosis tensor, and related quantities.

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
