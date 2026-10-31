"""
    block_vec_pq(A::MatNum, p::Integer, q::Integer)

Block vectorisation operator.

`block_vec_pq` cuts `A` into blocks of size `(p, q)`, and writes the vectorisation of each block as one row of the result. [`get_kt_Akt_pm`](@ref) applies it to the square cokurtosis `kt` with `p = q = N`, and takes the eigendecomposition of the result for the kurtosis risk constraints.

# Mathematical definition

Cut ``\\mathbf{A}`` into ``m \\times n`` blocks ``\\mathbf{A}_{ij}`` of size ``p \\times q``. The block vectorisation writes each block as one row, in the order of the block columns:

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

  - ``\\mathbf{A}_{ij}``: Block ``(i, j)`` of ``\\mathbf{A}``. It holds rows ``(i-1)p+1`` to ``ip`` and columns ``(j-1)q+1`` to ``jq``.
  - ``\\mathrm{vec}``: Column-major vectorisation.
  - ``m = \\mathrm{size}(\\mathbf{A}, 1) / p``, ``n = \\mathrm{size}(\\mathbf{A}, 2) / q``: The counts of blocks.

Row ``(j-1)m + i`` of the result is ``\\mathrm{vec}(\\mathbf{A}_{ij})^\\intercal``, so the index of the block column changes slowest. The order that changes the index of the block row slowest puts ``\\mathrm{vec}(\\mathbf{A}_{12})^\\intercal`` in row 2 of the example below, where this order puts ``\\mathrm{vec}(\\mathbf{A}_{21})^\\intercal``. The two orders agree when every block ``\\mathbf{A}_{ij}`` equals ``\\mathbf{A}_{ji}``. The square cokurtosis is such a matrix up to rounding, because its entry ``\\mathbb{E}[r_i r_k r_j r_l]`` does not change when ``i`` and ``j`` change places.

# Algorithm

 1. Read `size(A)` into `mp` and `nq`, and check both divisibility conditions.
 2. Take the block counts `m = mp ÷ p` and `n = nq ÷ q`.
 3. Allocate `A_vec`, of size `(m * n, p * q)`.
 4. For each block column `j` from `0` to `n - 1`, build `Aj`. Row `i + 1` of `Aj` is the vectorisation of the block in block row `i` and block column `j` of `A`.
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

  - $(ref_dict[:cajas2025]) Appendix A.1, Equation A.14. $(ref_dict[:no_original_source])
"""
function block_vec_pq(A::MatNum, p::Integer, q::Integer)
    mp, nq = size(A)
    @argcheck(mod(mp, p) == 0,
              DomainError((mp, p),
                          "`size(A, 1)` is $mp and `p` is $p. The rows split into blocks of `p`, so `size(A, 1)` must be an integer multiple of `p`."))
    @argcheck(mod(nq, q) == 0,
              DomainError((nq, q),
                          "`size(A, 2)` is $nq and `q` is $q. The columns split into blocks of `q`, so `size(A, 2)` must be an integer multiple of `q`."))
    m = mp ÷ p
    n = nq ÷ q
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

The duplication matrix `D` maps the half-vectorisation of a symmetric matrix to its full vectorisation. A [`HighOrderPrior`](@ref) holds it as `D2` beside the coskewness `sk`.

# Mathematical definition

This identity defines ``\\mathbf{D}_n``, for every symmetric ``\\mathbf{A}`` of size ``n \\times n``:

```math
\\mathbf{D}_n \\mathrm{vech}(\\mathbf{A}) = \\mathrm{vec}(\\mathbf{A})\\,.
```

With `diag = false`, the half-vectorisation drops the diagonal, and the identity gives the matrix with a zero diagonal. The source defines the matrix with the diagonal. The library adds the form without it.

```math
\\mathbf{D}_n^{-} \\mathrm{vech}^{-}(\\mathbf{A}) = \\mathrm{vec}(\\mathbf{A} - \\mathrm{diag}(\\mathbf{A}))\\,.
```

Where:

  - ``\\mathrm{vec}``: Column-major vectorisation, of length ``n^2``.
  - ``\\mathrm{vech}``: Half-vectorisation, which reads the lower triangle column by column, of length ``n(n+1)/2``.
  - ``\\mathrm{vech}^{-}``: The strictly lower triangle, read in the same order, of length ``n(n-1)/2``.
  - ``\\mathrm{diag}(\\mathbf{A})``: The diagonal of ``\\mathbf{A}``, as a diagonal matrix.

Each row of ``\\mathbf{D}_n`` holds one entry, so the matrix copies entries and adds none. It copies a diagonal entry of ``\\mathbf{A}`` to one place and an off-diagonal entry to two. The product ``\\mathbf{D}_n^\\intercal \\mathbf{D}_n`` is the diagonal matrix of these counts, and [`summation_matrix`](@ref) applies it as a weight.

# Algorithm

 1. Take `m = n(n+1)/2` and `nsq = n^2`.
 2. Fill `v`. Entry `r` of `v` is the position in ``\\mathrm{vech}(\\mathbf{A})`` of the entry in row `r` of ``\\mathrm{vec}(\\mathbf{A})``. For each column, the inner loops read the strictly upper part first, and then the lower part.
 3. When `diag` is `true`, return the sparse matrix of size `nsq × m` that holds a one at each `(r, v[r])`.
 4. When `diag` is `false`, count how many times each position occurs in `v`, and put the counts in `counts`.
 5. Keep the positions that occur more than once. These are the off-diagonal positions. Number them again from one, and put the numbers in `cols`.
 6. Keep every row of ``\\mathrm{vec}(\\mathbf{A})`` whose position step 5 kept, and put the pairs in `filtered_rows` and `filtered_cols`.
 7. Return the sparse matrix of size `nsq × n(n-1)/2` that holds a one at each kept pair.

# Arguments

  - `n`: Size of the symmetric matrix.
  - `diag`: `true` to include the diagonal entries, `false` to drop them.

# Returns

  - Sparse duplication matrix.

# Related

  - [`elimination_matrix`](@ref)
  - [`summation_matrix`](@ref)
  - [`dup_elim_sum_matrices`](@ref): builds this matrix and the other two in one loop.

# References

  - $(ref_dict[:magnusneudecker1980]) Section 3, Definition 3.2a.
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

The elimination matrix `L` keeps the lower triangle of a matrix, which holds every distinct entry of a symmetric matrix. A [`HighOrderPrior`](@ref) holds it as `L2` beside the square cokurtosis `kt`.

# Mathematical definition

This identity defines ``\\mathbf{L}_n``, for every ``\\mathbf{A}`` of size ``n \\times n``:

```math
\\mathbf{L}_n \\mathrm{vec}(\\mathbf{A}) = \\mathrm{vech}(\\mathbf{A})\\,.
```

With `diag = false`, the matrix also drops the diagonal. The source defines the matrix with the diagonal. The library adds the form without it.

```math
\\mathbf{L}_n^{-} \\mathrm{vec}(\\mathbf{A}) = \\mathrm{vech}^{-}(\\mathbf{A})\\,.
```

Where:

  - ``\\mathrm{vec}``: Column-major vectorisation, of length ``n^2``.
  - ``\\mathrm{vech}``: Half-vectorisation, which reads the lower triangle column by column, of length ``n(n+1)/2``.
  - ``\\mathrm{vech}^{-}``: The strictly lower triangle, read in the same order, of length ``n(n-1)/2``.

The identity holds for every square ``\\mathbf{A}``, symmetric or not, because ``\\mathbf{L}_n`` reads the lower triangle alone. `diag = false` removes the ``n`` rows that read the diagonal. The row count falls from ``n(n+1)/2`` to ``n(n-1)/2``, and the column count stays ``n^2``.

# Algorithm

 1. Take `nsq = n^2`.
 2. Set the row count `m`, the column range `rg` and the offset `b` from `diag`. When `diag` is `true`, they are `m = n(n+1)/2`, `rg = 1:n` and `b = 0`. Otherwise they are `m = n(n-1)/2`, `rg = 2:n` and `b = 1`.
 3. Fill `v`. Entry `r` of `v` is the position in ``\\mathrm{vec}(\\mathbf{A})`` of entry `r` of the half-vectorisation. The offset `b` skips the entries above the diagonal. Under `diag = false`, it also skips the diagonal entry.
 4. Return the sparse matrix of size `m × nsq` that holds a one at each `(r, v[r])`.

# Arguments

  - `n`: Size of the symmetric matrix.
  - `diag`: `true` to include the diagonal entries, `false` to drop them.

# Returns

  - Sparse elimination matrix.

# Related

  - [`duplication_matrix`](@ref)
  - [`summation_matrix`](@ref)
  - [`dup_elim_sum_matrices`](@ref): builds this matrix and the other two in one loop.

# References

  - $(ref_dict[:magnusneudecker1980]) Section 3, Definition 3.1a.
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

The summation matrix `S` reads the lower triangle of a symmetric matrix, and adds each off-diagonal entry once for each of its two places. A [`HighOrderPrior`](@ref) holds it as `S2` beside the square cokurtosis `kt`, and the kurtosis risk constraints read `S2 * kt * transpose(S2)`.

# Mathematical definition

``\\mathbf{S}_n`` is the elimination matrix, with each row multiplied by the number of places that ``\\mathbf{D}_n`` copies its entry to:

```math
\\mathbf{S}_n = \\mathbf{D}_n^\\intercal \\mathbf{D}_n \\mathbf{L}_n\\,,
```

so that, for every symmetric ``\\mathbf{A}`` of size ``n \\times n``,

```math
\\boldsymbol{1}^\\intercal \\mathbf{S}_n \\mathrm{vec}(\\mathbf{A}) = \\boldsymbol{1}^\\intercal \\mathrm{vec}(\\mathbf{A})\\,.
```

Where:

  - ``\\mathbf{D}_n``: The duplication matrix of [`duplication_matrix`](@ref).
  - ``\\mathbf{L}_n``: The elimination matrix of [`elimination_matrix`](@ref).
  - ``\\boldsymbol{1}``: Vector of ones, of the length that the product next to it needs.

``\\mathbf{S}_n`` reads the lower triangle. It multiplies each entry by the number of places that the entry has in ``\\mathrm{vec}(\\mathbf{A})``, which is one for a diagonal entry and two for an off-diagonal entry. The sum identity above follows from this. It holds for a symmetric ``\\mathbf{A}`` alone, because ``\\mathbf{S}_n`` counts each lower entry in place of the upper entry that mirrors it.

The same construction with `diag = false` gives ``\\mathbf{S}_n^{-} = (\\mathbf{D}_n^{-})^\\intercal \\mathbf{D}_n^{-} \\mathbf{L}_n^{-}``. It multiplies each of its ``n(n-1)/2`` rows by two. For a symmetric ``\\mathbf{A}``, its sum identity thus adds the off-diagonal entries alone, ``\\boldsymbol{1}^\\intercal \\mathbf{S}_n^{-} \\mathrm{vec}(\\mathbf{A}) = \\boldsymbol{1}^\\intercal \\mathrm{vec}(\\mathbf{A} - \\mathrm{diag}(\\mathbf{A}))``. The source defines the matrix with the diagonal. The library adds the form without it.

# Algorithm

The function builds the product of the definition, and it does not build ``\\mathbf{D}_n``.

 1. Take `nsq = n^2`. Set the row count `m`, the column range `rg` and the offset `b` from `diag`, as [`elimination_matrix`](@ref) does.
 2. Read the columns in `rg`. Write the ``\\mathrm{vec}`` position of each entry of the half-vectorisation into `v1`. For each **strictly** lower entry, write its ``\\mathrm{vec}`` position into `v2` and its row of the half-vectorisation into `rows2`.
 3. Remove the zero entries of `v1`, `v2` and `rows2`. Step 2 did not write to these positions.
 4. When `diag` is `true`, return the sum of two sparse matrices. The first holds a one at each `(r, v1[r])`, and it is ``\\mathbf{L}_n``. The second holds a one at each `(rows2[k], v2[k])`, and it adds a second one to each off-diagonal row.
 5. When `diag` is `false`, every row is off-diagonal. Return the sparse matrix that holds a **two** at each `(r, v1[r])`. This branch does not read `v2` and `rows2`.

# Arguments

  - `n`: Size of the symmetric matrix.
  - `diag`: `true` to include the diagonal entries, `false` to drop them.

# Returns

  - Sparse summation matrix.

# Related

  - [`duplication_matrix`](@ref)
  - [`elimination_matrix`](@ref)
  - [`dup_elim_sum_matrices`](@ref): builds this matrix and the other two in one loop. It computes this matrix as the product of the definition, not by the construction above.

# References

  - $(ref_dict[:pkurt]) Section 2, footnote 5.
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

Construct the duplication, elimination and summation matrices for a symmetric matrix of size `n × n`.

`dup_elim_sum_matrices` returns the duplication matrix `D`, the elimination matrix `L` and the summation matrix `S`. The high order priors store them as `D2`, `L2` and `S2` of a [`HighOrderPrior`](@ref).

The three are the `diag = true` matrices of [`duplication_matrix`](@ref), [`elimination_matrix`](@ref) and [`summation_matrix`](@ref). This function builds them in one loop over the columns, not in three.

# Mathematical definition

For every symmetric ``\\mathbf{A}`` of size ``n \\times n``:

```math
\\begin{align}
\\mathbf{D}_n \\mathrm{vech}(\\mathbf{A}) &= \\mathrm{vec}(\\mathbf{A})\\,, \\\\
\\mathbf{L}_n \\mathrm{vec}(\\mathbf{A}) &= \\mathrm{vech}(\\mathbf{A})\\,, \\\\
\\mathbf{S}_n &= \\mathbf{D}_n^\\intercal \\mathbf{D}_n \\mathbf{L}_n\\,.
\\end{align}
```

Where:

  - ``\\mathrm{vec}``: Column-major vectorisation, of length ``n^2``.
  - ``\\mathrm{vech}``: Half-vectorisation, which reads the lower triangle column by column, of length ``n(n+1)/2``.

The third line gives ``\\boldsymbol{1}^\\intercal \\mathbf{S}_n \\mathrm{vec}(\\mathbf{A}) = \\boldsymbol{1}^\\intercal \\mathrm{vec}(\\mathbf{A})``. ``\\mathbf{S}_n`` reads the lower triangle, and it multiplies each off-diagonal entry by two, the number of its places in ``\\mathrm{vec}(\\mathbf{A})``.

# Algorithm

 1. Check that `n` is positive.
 2. Take `m = n(n+1)/2` and `nsq = n^2`.
 3. Read the columns once, and fill `v1` and `v2` in the same loop. `v1` is the vector of column indices of [`duplication_matrix`](@ref), and `v2` is the vector of column indices of [`elimination_matrix`](@ref).
 4. Build `d`, the sparse matrix of size `nsq × m` that holds a one at each `(r, v1[r])`.
 5. Build `l`, the sparse matrix of size `m × nsq` that holds a one at each `(r, v2[r])`.
 6. Build `s` as the product `transpose(d) * d * l` of the definition above. This is not the construction of [`summation_matrix`](@ref).

# Arguments

  - `n`: Size of the symmetric matrix.

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

  - $(ref_dict[:magnusneudecker1980]) Section 3, Definitions 3.1a and 3.2a.
  - $(ref_dict[:pkurt]) Section 2, footnote 5.
"""
function dup_elim_sum_matrices(n::Int)
    assert_gt0(n, :n)
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

Return three `nothing` values when the first argument is not a matrix.

This is the fallback of [`dup_elim_sum_view`](@ref). It builds no matrix and reads no argument. The matrix method calls [`dup_elim_sum_matrices`](@ref).

[`port_opt_view`](@ref) on a [`HighOrderPrior`](@ref) calls `dup_elim_sum_view` at two sites, with `pr.kt` as the first argument. It calls it only when `pr` holds `D2` or `S2`. The constructor of `HighOrderPrior` refuses `S2` without `kt`, and the estimators of the library set `D2` only together with `kt` and `sk`. So no estimator of the library makes a `HighOrderPrior` that reaches this method. A `HighOrderPrior` that a caller builds by hand with `D2` and no `kt` reaches it, because `D2` is the one moment field that the constructor accepts alone.

This method takes a variable number of arguments, so it also answers each call whose count of arguments the matrix method does not take. For a matrix `M`, `dup_elim_sum_view(M, n)` reaches the matrix method, and `dup_elim_sum_view(M, n, extra)` reaches this method.

# Algorithm

 1. Return `(nothing, nothing, nothing)`, and read no argument.

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

Compute the duplication, elimination and summation matrices at the dimension that the caller gives.

This is the method of [`dup_elim_sum_view`](@ref) for a matrix as the first argument. **The method uses the matrix for dispatch alone.** The dimension is the second argument `n`, not a size of the matrix. [`port_opt_view`](@ref) on a [`HighOrderPrior`](@ref) needs this. It passes the full `N^2 × N^2` cokurtosis of the prior result and the asset count of the **subproblem**. The method then builds the three matrices again at the smaller dimension, and does not cut them from the larger ones.

# Algorithm

 1. Pass `n` to [`dup_elim_sum_matrices`](@ref), and return its three matrices.

# Arguments

  - The first argument: Any matrix. It selects this method, and the method does not read it.
  - `n`: Size of the symmetric matrix for which the method builds the three matrices.

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

`HighOrderPriorEstimator` holds a low order prior estimator, a cokurtosis estimator and a coskewness estimator. Its [`prior`](@ref) fits the mean and the covariance with the low order prior estimator. It then fits the square cokurtosis and the coskewness of the same returns, and returns all of them in a [`HighOrderPrior`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HighOrderPriorEstimator(;
        pe::Onl{<:AbstractLowOrderPriorEstimator_A_F_AF} = EmpiricalPrior(),
        kte::Option{<:CokurtosisEstimator} = Cokurtosis(;
            alg = FullMoment()
        ),
        ske::Option{<:CoskewnessEstimator} = Coskewness(;
            alg = FullMoment()
        ),
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> HighOrderPriorEstimator

Keywords correspond to the struct's fields.

## Validation

  - `x_src` is `nothing`, `DataSource()` or `ExcessSource()`, see [`assert_returns_source`](@ref).
  - The `x_src` of `pe` is `nothing`, because the prior picks the returns that it reads, see [`assert_inner_reader`](@ref).

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `pe`: Recursively updated through [`factory`](@ref).
  - `kte`: Recursively updated through [`factory`](@ref).
  - `ske`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `pe`: Recursively viewed through [`port_opt_view`](@ref).
  - `kte`: Recursively viewed through [`port_opt_view`](@ref).
  - `ske`: Recursively viewed through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> HighOrderPriorEstimator()
HighOrderPriorEstimator
     pe ┼ EmpiricalPrior
        │           ce ┼ PortfolioOptimisersCovariance
        │              │   ce ┼ Covariance
        │              │      │    me ┼ SimpleExpectedReturns
        │              │      │       │   w ┴ nothing
        │              │      │    ce ┼ GeneralCovariance
        │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
        │              │      │       │    w ┴ nothing
        │              │      │   alg ┼ FullMoment()
        │              │      │     w ┴ nothing
        │              │   mp ┼ MatrixProcessing
        │              │      │     pdm ┼ Posdef
        │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │              │      │      dn ┼ nothing
        │              │      │      dt ┼ nothing
        │              │      │     alg ┼ nothing
        │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
        │           me ┼ SimpleExpectedReturns
        │              │   w ┴ nothing
        │      horizon ┼ nothing
        │   fill_limit ┼ nothing
        │        x_src ┴ nothing
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
  x_src ┴ nothing
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

  - $(ref_dict[:jondeau2004]) Section 2.3.
  - $(ref_dict[:pkurt]) Appendix A.
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
    """
    $(field_dict[:x_src_pe])
    """
    x_src
    function HighOrderPriorEstimator(pe::Onl{<:AbstractLowOrderPriorEstimator_A_F_AF},
                                     kte::Option{<:CokurtosisEstimator},
                                     ske::Option{<:CoskewnessEstimator},
                                     x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, DATA_RETURNS_SOURCES, :HighOrderPriorEstimator)
        assert_inner_reader(pe, :pe, :HighOrderPriorEstimator)
        return new{typeof(pe), typeof(kte), typeof(ske), typeof(x_src)}(pe, kte, ske, x_src)
    end
end
function HighOrderPriorEstimator(;
                                 pe::Onl{<:AbstractLowOrderPriorEstimator_A_F_AF} = EmpiricalPrior(),
                                 kte::Option{<:CokurtosisEstimator} = Cokurtosis(;
                                                                                 alg = FullMoment()),
                                 ske::Option{<:CoskewnessEstimator} = Coskewness(;
                                                                                 alg = FullMoment()),
                                 x_src::Option{<:AbstractReturnsSource} = nothing)::HighOrderPriorEstimator
    return HighOrderPriorEstimator(pe, kte, ske, x_src)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties HighOrderPriorEstimator begin
    forward(pe, me, ce)
end
"""
    prior(pe::HighOrderPriorEstimator, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)

Compute the high order prior of the asset returns.

`prior` fits the mean and the covariance with the embedded prior estimator `pe.pe`. It then fits the square cokurtosis with `pe.kte` and the coskewness with `pe.ske`, and returns all of them in a [`HighOrderPrior`](@ref). The method passes the factor returns `F` to `pe.pe` alone, and the co-moments read the asset returns.

# Mathematical definition

The estimator computes the first and the second moments, the coskewness matrix ``\\mathbf{M}_3`` and the **square** cokurtosis matrix ``\\mathbf{\\Sigma}_4``:

```math
\\begin{align}
\\mathbf{W} &= (\\mathbf{Z} \\otimes \\boldsymbol{1}_N^\\intercal) \\odot (\\boldsymbol{1}_N^\\intercal \\otimes \\mathbf{Z})\\,, \\\\
\\mathbf{M}_3 &= \\frac{1}{T} \\mathbf{Z}^\\intercal \\mathbf{W}\\,, \\\\
\\mathbf{\\Sigma}_4 &= \\frac{1}{T} \\mathbf{W}^\\intercal \\mathbf{W}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{Z}``: ``T \\times N`` matrix of returns, with the mean of each column removed.
  - ``\\mathbf{W}``: ``T \\times N^2`` matrix of the products of each pair of centred returns.
  - ``\\mathbf{M}_3``: ``N \\times N^2`` coskewness matrix, `sk`.
  - ``\\mathbf{\\Sigma}_4``: ``N^2 \\times N^2`` square cokurtosis matrix, `kt`.
  - ``\\boldsymbol{1}_N``: ``N \\times 1`` vector of ones.
  - $(math_dict[:T])
  - ``\\otimes``: Kronecker product.
  - ``\\odot``: Hadamard product.

``\\mathbf{\\Sigma}_4`` is the square form, of size ``N^2 \\times N^2``. It is not the ``N \\times N^3`` cokurtosis matrix ``\\mathbf{M}_4`` of the same source. The library does not build ``\\mathbf{M}_4``.

`pe.kte` computes `kt`, and `pe.ske` computes `sk`. A different `alg`, such as a semi-moment or an exponentially weighted moment, gives a different formula in place of the displays above. `V` is the negative spectral coskewness of `sk`. When an estimator is `nothing`, the result holds `nothing` for its moment.

The low order block can give the probabilities ``\\mathbf{w}`` of its scenarios. An entropy pooling prior and an opinion pooling prior do this. Then the co-moments are those of the same distribution. ``\\mathbf{Z}`` is centred on the mean under ``\\mathbf{w}``, and each ``\\frac{1}{T}`` above becomes ``\\frac{1}{\\boldsymbol{1}^\\intercal \\mathbf{w}} \\mathrm{diag}(\\mathbf{w})`` inside the product. ``\\mathbf{w}`` replaces the observation weights of `pe.kte` and `pe.ske`. When the low order block gives no ``\\mathbf{w}``, the observation weights do not change.

# Algorithm

 1. Orient `X` and `F` to `observations × variables` with [`dims_oriented`](@ref).
 2. Compute the low order block `pr` with `pe.pe`, and pass `F` to it.
 3. Read the probabilities `w = pr.w` of its scenarios. Cut `X` and `pnl` to the rows that `w` weighs, with [`scenario_rows`](@ref).
 4. Compute the square cokurtosis `kt` with `factory(pe.kte, w)`. A `nothing` estimator gives a `nothing` moment.
 5. Compute the coskewness `sk` and its negative spectral form `V` with `factory(pe.ske, w)`. A `nothing` estimator gives `nothing` for both.
 6. Build the structure matrices at the asset count `size(pr.X, 2)` with [`dup_elim_sum_matrices`](@ref). Take all three when steps 4 and 5 both gave a moment. Take `L2` and `S2` alone when step 4 gave a moment and step 5 did not. Take none in the other cases. `D2` goes with `sk`, and `L2` and `S2` go with `kt`. The second case has no `sk`, and its `D2` is `nothing`.
 7. Build the [`HighOrderPrior`](@ref) with its keyword constructor. When step 5 gave an `sk`, set `skmp` to the processor of `pe.ske`, from [`coskewness_processor`](@ref). Each `@argcheck` of the constructor checks the shapes that steps 4 to 6 gave.
 8. Warn with [`assert_matched_coverage`](@ref) when the co-moments hold fewer assets than the low order block. Steps 6 to 8 are [`assemble_high_order_prior`](@ref).

# Arguments

  - `pe`: High order prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor returns matrix (observations × factors).
  - $(arg_dict[:pnl_prior])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments, passed to the estimators that `pe` holds.

# Validation

  - `dims in (1, 2)`.
  - Each condition that [`scenario_rows`](@ref) refuses.

# Returns

  - `pr::HighOrderPrior`: Result that holds the low order prior result, the square cokurtosis, the coskewness, its negative spectral form and the structure matrices.

# Related

  - [`HighOrderPriorEstimator`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`scenario_rows`](@ref)
"""
function prior(pe::HighOrderPriorEstimator, X::MatNum, F::Option{<:MatNum} = nothing,
               pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)
    X, F = dims_oriented(dims, X, F)
    pr = prior(pe.pe, X, F, pnl; kwargs...)
    # The co-moments are fitted as the low order moments are: the panel travels as the
    # third positional argument, the tensor is fitted on the Coverage Universe, and it is
    # expanded onto the full asset universe. `D2`, `L2` and `S2` are then sized from the full
    # width of `pr.X`, which is what the expanded tensors carry.
    #
    # A prior result that answers `w` weighs its scenarios with it, so the co-moments are
    # those of the same distribution as `mu` and `sigma` (#1650).
    w = pr.w
    X, pnl = scenario_rows(w, X, pnl)
    kt = cokurtosis(factory(pe.kte, w), X, pnl; kwargs...)
    sk, V = coskewness(factory(pe.ske, w), X, pnl; kwargs...)
    eoc = high_order_moment_counts(pe.ske, pe.kte, w, X, size(pr.X, 1); kwargs...)
    return assemble_high_order_prior(pe, pr, kt, sk, V, eoc)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build a [`HighOrderPrior`](@ref) from a low order prior result and the co-moments of the same fit.

Each fit of a [`HighOrderPriorEstimator`](@ref) ends here. The batch method calls this function after it computes the two co-moments over the returns of the caller. `prior(pe)` with no data calls it after it reads the two co-moments from their states. The structure matrices that a consumer needs depend only on which co-moments are present, so this one function holds the rule for both callers.

The function sizes `D2`, `L2` and `S2` from the **full** width of `pr.X`, which is the width of the expanded co-moments.

# Algorithm

 1. Build `D2`, `L2` and `S2` when both co-moments are present. Build `L2` and `S2` alone when only the cokurtosis is present.
 2. Build the [`HighOrderPrior`](@ref) with the counts `eoc`. When the coskewness is present, set `skmp` to the processor of `pe.ske`, from [`coskewness_processor`](@ref).
 3. Warn with [`assert_matched_coverage`](@ref) when the co-moments hold fewer assets than the low order block.

# Arguments

  - `pe`: High order prior estimator.
  - `pr`: The low order prior result of the embedded estimator.
  - `kt`: The square cokurtosis matrix, or `nothing`.
  - `sk`: The coskewness matrix, or `nothing`.
  - `V`: The negative spectral coskewness of `sk`, or `nothing`.
  - `eoc`: The effective counts `(T_sk, T_kt)` of the two co-moments, from [`high_order_moment_counts`](@ref). It is `(nothing, nothing)` by default.

# Validation

  - Every check of the constructor of [`HighOrderPrior`](@ref).
  - [`assert_matched_coverage`](@ref) warns when the co-moments hold fewer assets than the low order block. It does not throw.

# Returns

  - `hop::HighOrderPrior`: The high order prior result.

# Related

  - [`HighOrderPriorEstimator`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`dup_elim_sum_matrices`](@ref)
  - [`assert_matched_coverage`](@ref)
"""
function assemble_high_order_prior(pe::HighOrderPriorEstimator, pr::AbstractPriorResult,
                                   kt::Option{<:MatNum}, sk::Option{<:MatNum}, V,
                                   eoc::Tuple = (nothing, nothing))
    D2 = nothing
    L2 = nothing
    S2 = nothing
    if !isnothing(kt) && !isnothing(sk)
        D2, L2, S2 = dup_elim_sum_matrices(size(pr.X, 2))
    elseif !isnothing(kt) && isnothing(sk)
        L2, S2 = dup_elim_sum_matrices(size(pr.X, 2))[2:3]
    end
    hop = HighOrderPrior(; pr = pr, kt = kt, D2 = D2, L2 = L2, S2 = S2, sk = sk, V = V,
                         skmp = isnothing(sk) ? nothing : coskewness_processor(pe.ske),
                         eoc = eoc)
    assert_matched_coverage(hop)
    return hop
end
"""
    high_order_moment_counts(ske, kte, w, X::MatNum, t::Integer; kwargs...)
        -> Tuple{Option{<:Real}, Option{<:Real}}

State the effective number of observations of the coskewness and of the cokurtosis of a high order prior, in the form of the field `eoc` of a [`HighOrderPrior`](@ref).

The co-moments are fitted by `factory(ske, w)` and `factory(kte, w)` over the rows `X`, so each is worth the count that its estimator states, through [`high_order_observation_count`](@ref). A weighted estimator is worth the count of its weights, which are the weights `w` of the prior when the prior states them. The denoising step of each co-moment reads the same count. An entry equal to the `t` rows of the result is written as `nothing`, through [`moment_count_entry`](@ref).

# Arguments

  - `ske`: The coskewness estimator, or `nothing`.
  - `kte`: The cokurtosis estimator, or `nothing`.
  - `w`: The observation weights of the prior result, or `nothing`.
  - `X`: The rows that the co-moments read, `observations × assets`.
  - `t`: The number of rows that the prior result carries in `X`.
  - `kwargs...`: The keywords that the prior gives to the fits of the co-moments.

# Returns

  - `eoc::Tuple`: The entries `(T_sk, T_kt)` of the field `eoc`.

# Related

  - [`HighOrderPrior`](@ref)
  - [`high_order_observation_count`](@ref)
  - [`moment_count_entry`](@ref)
"""
function high_order_moment_counts(ske, kte, w, X::MatNum, t::Integer; kwargs...)
    return (moment_count_entry(high_order_observation_count(ske, w, X; kwargs...), t),
            moment_count_entry(high_order_observation_count(kte, w, X; kwargs...), t))
end
"""
    high_order_observation_count(est::Nothing, w, X::MatNum; kwargs...) -> Integer
    high_order_observation_count(est::AbstractEstimator, w, X::MatNum; kwargs...) -> Real

Return the effective number of observations of a co-moment that `factory(est, w)` fits over the rows `X`.

A high order prior gives the weights `w` of its low order result to its co-moment estimators, so the count reads the estimator after that substitution, with [`effective_observation_count`](@ref). With no estimator the prior fits no co-moment, and the count is `size(X, 1)`.

# Arguments

  - `est`: The coskewness or cokurtosis estimator, or `nothing`.
  - `w`: The observation weights of the prior result, or `nothing`.
  - `X`: The rows that the co-moment reads, `observations × assets`.
  - `kwargs...`: The keywords of the fit of the co-moment, for [`effective_observation_count`](@ref).

# Returns

  - `T_e::Real`: The effective number of observations of the co-moment.

# Related

  - [`high_order_moment_counts`](@ref)
  - [`effective_observation_count`](@ref)
"""
function high_order_observation_count(::Nothing, w, X::MatNum; kwargs...)
    return size(X, 1)
end
function high_order_observation_count(est::AbstractEstimator, w, X::MatNum; kwargs...)
    return effective_observation_count(factory(est, w), X; kwargs...)
end
"""
    scenario_rows(w::Nothing, X::MatNum, pnl::Option{<:AssetPanel})
    scenario_rows(w::ObsWeights, X::MatNum, pnl::Option{<:AssetPanel})

Cut the rows of `X` and of its Asset Panel to the scenarios that the probabilities `w` of a prior result weigh.

A prior result gives `w` on the rows of its own `X`, and it can hold fewer rows than the caller passed. A Scenario Cap keeps the last rows. A [`CrossSectionalFactorPrior`](@ref) drops the first rows, over which its Descriptors warm up. In both cases, `w` weighs the last `length(w)` rows of `X`. A high order estimator fits its co-moments on the `X` of the caller under `w`, so it needs these rows. When a prior result gives no `w`, it weighs every row the same, and the co-moments read every row.

# Algorithm

 1. When `w` is `nothing`, return `X` and `pnl` with no change.
 2. Otherwise take `t = size(X, 1)` and `s = length(w)`, and refuse `s > t`.
 3. When `s == t`, return `X` and `pnl` with no change.
 4. Otherwise take the rows `(t - s + 1):t`. Return the view of `X` over them, and the view of `pnl` over them from [`asset_panel_view`](@ref).

# Arguments

  - `w`: The probabilities of the scenarios of the prior result, or `nothing`.
  - `X`: The returns that the caller passed (observations × assets), or the factor returns.
  - $(arg_dict[:pnl_prior])

# Validation

  - `length(w) <= size(X, 1)`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `X′::MatNum`: The rows of `X` that `w` weighs.
  - `pnl′::Option{<:AssetPanel}`: The Asset Panel over the same rows, or `nothing`.

# Related

  - [`HighOrderPriorEstimator`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`asset_panel_view`](@ref)
  - [`read_comoment`](@ref)
"""
function scenario_rows(::Nothing, X::MatNum, pnl::Option{<:AssetPanel})
    return X, pnl
end
function scenario_rows(w::ObsWeights, X::MatNum, pnl::Option{<:AssetPanel})
    t = size(X, 1)
    s = length(w)
    @argcheck(s <= t,
              DimensionMismatch("the prior result answers $s scenario probabilities, but the returns hold $t observations. A prior answers `w` on the last rows of the returns it was given, so `w` cannot be longer than the returns."))
    if s == t
        return X, pnl
    end
    idx = (t - s + 1):t
    return view(X, idx, :), asset_panel_view(pnl, idx, :, nothing)
end
"""
    read_comoment(f, est, w::Nothing, X::MatNum, pnl::Option{<:AssetPanel}; kwargs...)
    read_comoment(f, est, w::ObsWeights, X::MatNum, pnl::Option{<:AssetPanel}; kwargs...)

Read a co-moment of a [`HighOrderPriorEstimator`](@ref) out of its fold, under the probabilities of the scenarios of the embedded prior result.

When the prior result gives no `w`, it weighs every row the same. Then the function reads the member as [`read_member`](@ref) does: from its state when the member folded, or by a refit over the rows of the buffer. A prior result that gives a `w`, such as an entropy pooling prior or an opinion pooling prior, weighs its scenarios with it. A state folds the rows under equal weights, and it cannot read `w`. So the function refits the member under `w` in every case, as the batch fit does.

# Algorithm

 1. When `w` is `nothing`, read the member with [`read_member`](@ref).
 2. Otherwise cut `X` and `pnl` to the rows that `w` weighs, with [`scenario_rows`](@ref).
 3. Give `w` to the member with [`factory`](@ref), and call `f` on the result over the cut rows. The batch verb does not read the state of the member.

# Arguments

  - `f`: The batch verb of the member, [`cokurtosis`](@ref) or [`coskewness`](@ref).
  - `est`: The member, or `nothing`.
  - `w`: The probabilities of the scenarios of the embedded prior result, or `nothing`.
  - `X`: The rows that the embedded prior folded.
  - `pnl`: The Asset Panel of the rows, or `nothing`.
  - `kwargs...`: Additional keyword arguments, forwarded to the verb.

# Validation

  - Each condition that [`scenario_rows`](@ref) refuses.

# Returns

  - `val`: The co-moment, in the shape that `f` returns.

# Related

  - [`read_member`](@ref)
  - [`scenario_rows`](@ref)
  - [`HighOrderPriorEstimator`](@ref)
"""
function read_comoment(f::F, est, ::Nothing, X::MatNum, pnl::Option{<:AssetPanel};
                       kwargs...) where {F}
    return read_member(f, est, X, pnl; kwargs...)
end
function read_comoment(f::F, est, w::ObsWeights, X::MatNum, pnl::Option{<:AssetPanel};
                       kwargs...) where {F}
    X, pnl = scenario_rows(w, X, pnl)
    return f(factory(est, w), X, pnl; dims = 1, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the diagonal entry of each asset in every co-moment that a [`HighOrderPrior`](@ref) holds, and return `true` at each asset where all of them are finite.

This is the high order part of the Investable Mask. For asset `i`, take the pair column `j = (i - 1) * N + i`. The entry `sk[i, j]` is the third central moment of asset `i`, and the entry `kt[j, j]` is its fourth central moment. Each is finite when the fit gave a value for that asset. [`investable_mask`](@ref) applies the same rule to the diagonal of `sigma`. A moment that the prior result does not hold removes no asset, so for a `nothing` moment every asset is `true`.

# Arguments

  - `sk`: The coskewness matrix, `assets × assets²`, or `nothing`.
  - `kt`: The square cokurtosis matrix, `assets² × assets²`, or `nothing`.
  - `N`: Number of assets.

# Returns

  - `msk::BitVector`: `true` at each asset whose high order moments are finite.

# Related

  - [`HighOrderPrior`](@ref)
  - [`investable_mask`](@ref)
  - [`assert_matched_coverage`](@ref)
"""
function comoment_investable(sk::Option{<:MatNum}, kt::Option{<:MatNum}, N::Integer)
    msk = trues(N)
    for i in 1:N
        j = (i - 1) * N + i
        if !isnothing(sk)
            msk[i] &= isfinite(sk[i, j])
        end
        if !isnothing(kt)
            msk[i] &= isfinite(kt[j, j])
        end
    end
    return msk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Derive the Investable Mask of a [`HighOrderPrior`](@ref) from every order of moment that the prior result holds.

The [`AbstractPriorResult`](@ref) method reads `mu` and the diagonal of `sigma`. Under a [`CoveragePolicy`](@ref), an asset with a finite `mu` and a finite variance can have a co-moment that is not finite. Each estimator holds its own policy, so `pe.pe` can hold one while `pe.ske` and `pe.kte` do not. This method also reads the diagonal entry of each asset in `sk` and `kt`, with [`comoment_investable`](@ref). An asset that the co-moments did not estimate leaves the problem. Without this, the asset gets to a spectral step that throws.

This method does not warn when the mask removes such an asset. Each entry to an optimiser derives the mask again, and the derivation must have no side effect. [`assert_matched_coverage`](@ref) warns once, at the fit.

# Arguments

  - $(arg_dict[:pr])

# Validation

  - At least one asset must be investable.

# Returns

  - `imsk::Option{BitVector}`: `true` at every investable asset, or `nothing` when every asset is investable.

# Related

  - [`HighOrderPrior`](@ref)
  - [`investable_mask`](@ref)
  - [`comoment_investable`](@ref)
  - [`assert_matched_coverage`](@ref)
"""
function investable_mask(pr::HighOrderPrior)::Option{BitVector}
    imsk = isfinite.(pr.mu) .& isfinite.(LinearAlgebra.diag(pr.sigma))
    imsk .&= comoment_investable(pr.sk, pr.kt, length(pr.mu))
    @argcheck(any(imsk),
              IsEmptyError("no asset of the prior result is investable: every asset carries a NaN in `mu`, on the diagonal of `sigma`, or on the per-asset diagonal of `sk` or `kt`. Check that the prior estimator received enough observations, that the universe holds at least one active asset, and that `ske` and `kte` carry the same `cvg` as the low order estimators."))
    return all(imsk) ? nothing : imsk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Warn once when the co-moments of a fitted [`HighOrderPrior`](@ref) hold fewer assets than its low order block.

A [`CoveragePolicy`](@ref) is a field of one estimator. A caller can set one on the mean and the covariance of `pe.pe`, and leave `pe.ske` and `pe.kte` on the Coverage Universe. The library accepts this configuration. The low order moments then give a value for an asset that starts to list inside the window, and the co-moments do not. But the Investable Mask then removes these assets again, and the policy has no effect on the problem. The function does not refuse the configuration. It names the assets in a warning, as [`scenario_fill`](@ref) names the assets whose returns it fills with zeros.

The check runs once, at the fit. It does not run in [`investable_mask`](@ref), which a fold loop calls at each entry to an optimiser.

# Arguments

  - `pr`: The fitted high order prior.

# Validation

  - The co-moments hold every asset that the low order block holds. The function emits a warning that names the assets otherwise.

# Returns

  - `nothing`.

# Related

  - [`HighOrderPrior`](@ref)
  - [`comoment_investable`](@ref)
  - [`investable_mask`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function assert_matched_coverage(pr::HighOrderPrior)::Nothing
    lo = isfinite.(pr.mu) .& isfinite.(LinearAlgebra.diag(pr.sigma))
    dropped = findall(lo .& .!comoment_investable(pr.sk, pr.kt, length(pr.mu)))
    if !isempty(dropped)
        @warn("the low order block of this high order prior holds $(length(dropped)) asset(s) that its higher orders do not, so the Investable Mask narrows back to the assets the higher orders could estimate. Assets $(dropped) carry a finite `mu` and a finite diagonal of `sigma`, and a non-finite per-asset diagonal of `sk` or `kt`. A `CoveragePolicy` is a field of one estimator, so set the same one on `ske` and `kte` as on the low order estimators — `HighOrderPriorEstimator(; ske = Coskewness(; cvg = ...), kte = Cokurtosis(; cvg = ...))` — or clear it everywhere to fit the whole prior on the Coverage Universe.",
              maxlog = log_limit(:warn), _id = @log_id())
    end
    return nothing
end

function factor_residual_config(pe::HighOrderPriorEstimator)
    # The low-order block of the result is the wrapped estimator's own, residual block and
    # all, so this estimator forwards rather than answering for itself (see
    # [`factor_residual_config`](@ref)).
    return factor_residual_config(pe.pe)
end
"""
    partial_fit!(pe::HighOrderPriorEstimator, X, F = nothing; kwargs...)

Fold observations into a [`HighOrderPriorEstimator`](@ref), and pass them to its members.

The estimator keeps no buffer. It builds `HighOrderPrior(; pr = pr, ...)` around the result of its embedded prior. The rows that `prior(pe)` with no data needs are the rows that the embedded prior holds. A buffer in this estimator is a second copy of the sample.

The estimator always passes the fold to `pe.pe`, because that member holds the rows. An embedded prior that cannot fold refuses here, and names the wrapper that gives it a buffer. This refusal is correct, because the estimator has no rows of its own to give it. The factor observations go with the rows, as the batch verb passes `F` to the embedded prior. The tree of the embedded prior decides what to do with them. `pe.kte` and `pe.ske` go through [`fold_member`](@ref). `prior(pe)` with no data has rows for them, and it refits a co-moment that cannot fold. So `HighOrderPriorEstimator(; ske = Coskewness(; alg = SemiMoment()))` needs no wrapper, and it keeps one copy of the sample.

# Algorithm

 1. At the first step, when `pe.pe` holds no state, refuse a member that holds one, with [`assert_cold_carry_entry`](@ref).
 2. Fold the observations and `F` into `pe.pe` with [`partial_fit!`](@ref).
 3. Fold the observations into `pe.kte` and `pe.ske` with [`fold_member`](@ref).
 4. Rebuild the estimator with the three folded members through [`rebuild_estimator`](@ref), and return it.

# Arguments

  - `pe`: High order prior estimator.
  - `X`: Observations to fold, a matrix of rows or one observation.
  - `F`: The factor observations beside `X`, or `nothing`. The method forwards them to the embedded prior.
  - `kwargs...`: Additional keyword arguments, forwarded to the members.

# Validation

  - `pe.pe` folds. Otherwise the embedded prior refuses the fold, with an error that names the wrapper that gives it a buffer.
  - At the first step, no member holds a partial-fit state. The embedded prior holds the rows, and it starts with no rows. So a co-moment that holds a state at this step reads rows that the embedded prior does not hold. An `ArgumentError` that names the path of the state is thrown otherwise, see [`assert_cold_carry_entry`](@ref).

# Returns

  - `pe`: The estimator, with its members folded.

# Related

  - [`HighOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
  - [`fold_member`](@ref)
  - [`assert_cold_carry_entry`](@ref)
  - [`supports_partial_fit`](@ref)
"""
function partial_fit!(pe::HighOrderPriorEstimator, X::VecNum_MatNum,
                      F::Option{<:VecNum_MatNum} = nothing; kwargs...)
    assert_cold_carry_entry(pe, online_entry_state(pe.pe), "HighOrderPriorEstimator")
    return rebuild_estimator(pe,
                             (; pe = partial_fit!(pe.pe, X, F; kwargs...),
                              kte = fold_member(pe.kte, X; kwargs...),
                              ske = fold_member(pe.ske, X; kwargs...)))
end
"""
    fold_prior_step(pe::HighOrderPriorEstimator, rd::ReturnsResult,
                    gaps::Option{<:AbstractVector{Bool}})

Fold the observations of a [`ReturnsResult`](@ref) into a [`HighOrderPriorEstimator`](@ref). The method passes `rd` and its marks to the embedded prior.

This is the method of [`fold_prior_step`](@ref) for an estimator that passes the fold on. [`partial_fit!`](@ref) over a `ReturnsResult` calls `fold_prior_step` for each prior. The outer prior keeps no rows, and its embedded prior keeps them. The route of the embedded prior decides which inputs of the step it accepts. The estimation mask and the Panel Fields get to it with no change, and it refuses each input that its route cannot accept. The estimator also folds `pe.kte` and `pe.ske` with [`fold_member`](@ref). These folds are exact, and they read the active mask alone, which [`step_active_kwargs`](@ref) gives. They fold the observations that `gaps` does not mark, as the members of a carry do, see [`shared_series_member_rows`](@ref).

# Algorithm

 1. At the first step, when `pe.pe` holds no state, refuse a member that holds one, with [`assert_cold_carry_entry`](@ref).
 2. Fold `rd` and `gaps` into `pe.pe` with [`fold_prior_step`](@ref).
 3. Take the rows of `rd.X` that `gaps` does not mark, and the same rows of the active mask of `rd.pnl`, with [`shared_series_member_rows`](@ref). Fold them into `pe.kte` and `pe.ske` with [`fold_member`](@ref).
 4. Rebuild the estimator with its folded members, with `rebuild_estimator`, and return it.

# Arguments

  - `pe`: The high order prior estimator.
  - $(arg_dict[:rd])
  - `gaps`: One mark for each observation of `rd`, `true` at an observation with a Shared Series gap, or `nothing`.

# Validation

  - Every condition that the fold of `pe.pe` over `rd` checks.
  - The rule of [`assert_cold_carry_entry`](@ref) at the first step.

# Returns

  - `pe`: The estimator, with its members folded.

# Related

  - [`HighOrderPriorEstimator`](@ref)
  - [`fold_member`](@ref)
  - [`step_active_kwargs`](@ref)
  - [`shared_series_member_rows`](@ref)
"""
function fold_prior_step(pe::HighOrderPriorEstimator, rd::ReturnsResult,
                         gaps::Option{<:AbstractVector{Bool}})
    assert_cold_carry_entry(pe, online_entry_state(pe.pe), "HighOrderPriorEstimator")
    X, kw = shared_series_member_rows(rd.X, gaps; step_active_kwargs(rd.pnl)...)
    return rebuild_estimator(pe,
                             (; pe = fold_prior_step(pe.pe, rd, gaps),
                              kte = fold_member(pe.kte, X; kw...),
                              ske = fold_member(pe.ske, X; kw...)))
end
"""
    prior(pe::HighOrderPriorEstimator; kwargs...)

Read a [`HighOrderPriorEstimator`](@ref) out of its fold, with no data matrix.

The method reads the embedded prior first. A co-moment that folded gives its value from its state. The method refits a co-moment that did not fold, over the rows that the embedded prior folded, which [`prior_returns_buffer`](@ref) reads. When the embedded result gives the probabilities `w` of its scenarios, the method refits both co-moments under `w`, folded or not, because a state cannot read `w`. [`assemble_high_order_prior`](@ref) then builds the result, as in the batch method.

The refit reads the folded rows, and not `pr.X`. `pr.X` holds the scenarios of the result of the embedded prior, and these are not always the folded rows. A Scenario Cap on the embedded prior keeps only the last rows. The scenario fill writes zeros into the first rows of an asset that starts to list late. The batch method fits both co-moments over every row that the caller passes, with the `NaN` entries. When the embedded result gives `w`, it fits them over the last `length(pr.w)` rows, with [`scenario_rows`](@ref). So in both cases a refit over `pr.X` gives a different value. Under an [`Online`](@ref) window, the buffer holds the last `max_history` rows, and the equal batch fit of that route uses the same window. The estimator keeps no copy of the rows, because the embedded prior holds them.

# Mathematical definition

```math
\\begin{align}
\\mathcal{P}_T &= \\mathcal{P}(\\mathbf{X})\\,.
\\end{align}
```

Where:

  - $(math_dict[:P_fold_prior])
  - $(math_dict[:P_batch_prior])
  - $(math_dict[:X_returns])

The equality holds for the embedded result and for every co-moment, both the co-moments that fold and the co-moments that the method refits. In floating point, each co-moment differs from the batch co-moment by rounding.

# Algorithm

 1. Read the embedded result `pr` out of `pe.pe` with [`prior`](@ref).
 2. Read the rows that the embedded prior folded, with [`prior_returns_buffer`](@ref). Drop the rows of a Shared Series gap with [`shared_series_buffer`](@ref), as the embedded prior and the members that fold drop them. Read the rows that remain with [`sample_buffer`](@ref), and call them `X`.
 3. Rebuild the Asset Panel of the rows from the same buffer with [`buffer_asset_panel`](@ref), and call it `pnl`. The batch method gives `rd.pnl` to both co-moments in the same way.
 4. Read `kt` out of `pe.kte` with [`read_comoment`](@ref) under `pr.w`. It refits the member over `X` and `pnl` when the member did not fold, and under `pr.w` when `pr.w` is not `nothing`.
 5. Read `sk` and `V` out of `pe.ske` with [`read_comoment`](@ref), in the same way.
 6. Build the result with [`assemble_high_order_prior`](@ref).
 7. Record in the result the returns source that `pe` states, with [`stamp_returns_source`](@ref), and return it.

# Arguments

  - `pe`: High order prior estimator whose members hold states.
  - `kwargs...`: Additional keyword arguments, passed to the members.

# Validation

  - Each condition that `prior(pe.pe)` with no data checks.
  - Each condition that [`prior_returns_buffer`](@ref) refuses.
  - Each condition that [`scenario_rows`](@ref) refuses.

# Returns

  - `hop::HighOrderPrior`: Result that holds the low order prior result and the co-moments.

# Related

  - [`HighOrderPriorEstimator`](@ref)
  - [`assemble_high_order_prior`](@ref)
  - [`read_comoment`](@ref)
  - [`read_member`](@ref)
  - [`prior_returns_buffer`](@ref)
  - [`buffer_asset_panel`](@ref)
  - [`partial_fit!`](@ref)
  - [`shared_series_buffer`](@ref)
"""
function prior(pe::HighOrderPriorEstimator; kwargs...)
    pr = prior(pe.pe; kwargs...)
    buf = shared_series_buffer(prior_returns_buffer(pe.pe))
    X = sample_buffer(buf)
    pnl = buffer_asset_panel(buf)
    kt = read_comoment(cokurtosis, pe.kte, pr.w, X, pnl; kwargs...)
    sk, V = read_comoment(coskewness, pe.ske, pr.w, X, pnl; kwargs...)
    eoc = high_order_moment_counts(pe.ske, pe.kte, pr.w, scenario_rows(pr.w, X, pnl)[1],
                                   size(pr.X, 1); kwargs...)
    return stamp_returns_source(assemble_high_order_prior(pe, pr, kt, sk, V, eoc), pe)
end

export HighOrderPriorEstimator
