"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all detoning estimators.

A type that detones a matrix is a subtype of `AbstractDetoneEstimator`.

# Interfaces

To add a detoning estimator, subtype `AbstractDetoneEstimator`, keep its parameters in the fields of the struct, and add these methods:

  - `detone!(dt::AbstractDetoneEstimator, X::MatNum) -> MatNum`: Detones `X` in place.
  - `detone(dt::AbstractDetoneEstimator, X::MatNum) -> MatNum`: Optional. Returns the detoned `X` in a new matrix. A fallback method copies `X` and calls `detone!`, so add this method only when it avoids the copy.

## Arguments

  - $(arg_dict[:dt])
  - $(arg_dict[:sigrhoX])

## Returns

  - `X::MatNum`: The detoned input matrix `X`.

# Examples

The example below defines a minimal detoning estimator.

```jldoctest
julia> struct MyDetoneEstimator <: PortfolioOptimisers.AbstractDetoneEstimator end

julia> function PortfolioOptimisers.detone!(dt::MyDetoneEstimator, X::PortfolioOptimisers.MatNum)
           # Implement your in-place detoning estimator here.
           println(\"Detoning matrix in-place...\")
           return X
       end

julia> function PortfolioOptimisers.detone(dt::MyDetoneEstimator, X::PortfolioOptimisers.MatNum)
           X = copy(X)
           println(\"Copy X...\")
           detone!(dt, X)
           return X
       end

julia> detone!(MyDetoneEstimator(), [1.0 2.0; 2.0 1.0])
Detoning matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0

julia> detone(MyDetoneEstimator(), [1.0 2.0; 2.0 1.0])
Copy X...
Detoning matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0
```

# Related

  - [`Detone`](@ref)
  - [`detone!`](@ref)
  - [`detone`](@ref)
"""
abstract type AbstractDetoneEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Removes the `n` largest principal components, the market modes, from a covariance or correlation matrix. [`detone!`](@ref) and [`detone`](@ref) apply it.

In financial data, the largest principal components often show movements of the whole market, which can hide the signals of single assets. The field `n` sets how many of these components `Detone` removes, so the result shows the relations between assets that the market modes do not explain [mlp1](@cite).

A detoned matrix is singular before [`posdef!`](@ref) runs under `pdm`, because the subtraction sets the removed eigenvalues to zero. Section 2.6 of [mlp1](@cite) states that this is not a problem for a clustering method, which does not invert the matrix. That section also states that a mean-variance optimisation cannot use the detoned matrix directly.

# Mathematical definition

The algorithm subtracts the ``n`` largest eigenmodes, and rescales the remainder to a unit diagonal:

```math
\\begin{align}
\\mathbf{C} &= \\mathbf{X} - \\sum_{k=N-n+1}^{N} \\lambda_k \\boldsymbol{v}_k \\boldsymbol{v}_k^\\intercal\\,, \\\\
\\tilde{X}_{ij} &= \\frac{C_{ij}}{\\sqrt{C_{ii} C_{jj}}}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{C}``: The remainder after the subtraction of the market modes.
  - ``\\tilde{\\mathbf{X}}``: The detoned matrix.
  - ``\\mathbf{X}``: The original correlation or covariance matrix.
  - ``\\lambda_k``: The ``k``-th eigenvalue of ``\\mathbf{X}`` in ascending order, so the sum takes the ``n`` largest.
  - ``\\boldsymbol{v}_k``: The eigenvector of ``\\lambda_k``.
  - ``n``: The number of market modes to remove.
  - $(math_dict[:N])

When the algorithm subtracts a set of eigenmodes, the diagonal of ``\\mathbf{C}`` falls below one. So the rescaling changes every entry of the matrix. The subtraction sets the removed eigenvalues to zero and keeps the others, so ``\\mathbf{C}`` and ``\\tilde{\\mathbf{X}}`` are positive semidefinite and singular. [`posdef!`](@ref) then runs on ``\\tilde{\\mathbf{X}}`` under `pdm`.

Equation 5 of [kim2005](@cite) splits a correlation matrix into the market mode, the group modes and the noise modes, and the subtraction at ``n = 1`` removes the first of those parts. That work does not rescale. Section 2.6 of [mlp1](@cite) states detoning. It subtracts the market components, usually one but possibly more, and rescales the remainder to a unit diagonal. A search of the literature found no earlier work that states the rescale. That section detones a denoised matrix, and this estimator detones the matrix that it gets.

# Algorithm

[`detone!`](@ref) runs these steps under this estimator.

 1. Read `dt.n` into `n`, and check that `0 < n <= size(X, 2)`.
 2. When `X` has a zero variance, set its row and column to zero and detone the block of the positive variances with a recursive call, through [`zero_variance_block!`](@ref), and return `X`. A zero variance has no correlation, so step 4 cannot standardise it. The recursive call checks `dt.n` against the size of the block.
 3. Decrement `n` by one. `n` counts the modes to remove, and steps 6 and 7 slice `(end - n):end`, which is a window of the original `dt.n` columns. So `dt.n = 1` removes the single largest component, which is the market mode.
 4. Read the diagonal of `X` into `s`. When any entry of `s` is not one, `X` is a covariance matrix. Then replace `s` with its square roots and convert `X` to a correlation matrix with `StatsBase.cov2cor!`. The test is `any(!isone, s)`, so it is the value of the diagonal that decides, never the type of `X`.
 5. Eigendecompose `X` into the ascending eigenvalues `vals` and the eigenvectors `vecs`.
 6. Take the last block of `vals`, which holds the `dt.n` largest eigenvalues.
 7. Take the matching last columns of `vecs`.
 8. Subtract `vecs * vals * transpose(vecs)` from `X`. The result is the remainder ``\\mathbf{C}``.
 9. Rescale the remainder to unit diagonal with `StatsBase.cov2cor`.
10. Repair the rescaled matrix with [`posdef!`](@ref), under `dt.pdm`.
11. When step 4 converted a covariance matrix, convert `X` back with `StatsBase.cor2cov!`. The standard deviations are the ones read in step 4, so the original diagonal returns exactly.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Detone(;
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
        n::Integer = 1,
    ) -> Detone

Keywords correspond to the struct's fields.

## Validation

  - `n > 0`.

# Examples

```jldoctest
julia> Detone(; n = 2)
Detone
  pdm ┼ Posdef
      │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
    n ┴ Int64: 2
```

# Related

  - [`AbstractDetoneEstimator`](@ref)
  - [`detone!`](@ref)
  - [`detone`](@ref)

# References

  - $(ref_dict[:kim2005]) Equation 5.
  - $(ref_dict[:mlp1]) Section 2.6.
"""
@concrete struct Detone <: AbstractDetoneEstimator
    """
    $(field_dict[:opdm])
    """
    pdm
    """
    Number of largest principal components to remove.
    """
    n
    function Detone(pdm::Option{<:AbstractPosdefEstimator}, n::Integer)
        assert_gt0(n, :n)
        return new{typeof(pdm), typeof(n)}(pdm, n)
    end
end
function Detone(; pdm::Option{<:AbstractPosdefEstimator} = Posdef(), n::Integer = 1)
    return Detone(pdm, n)
end
"""
    detone!(dt::Option{<:AbstractDetoneEstimator}, X::MatNum) -> MatNum

Removes the `n` largest principal components, the market modes, from a covariance or correlation matrix in place.

When the diagonal of `X` is not all ones, the function converts `X` to a correlation matrix, applies the algorithm, and scales the result back to a covariance.

# Arguments

  - $(arg_dict[:odt])

      + `::Detone`: Removes the `dt.n` largest principal components from `X` in place.
      + `::Nothing`: Returns `X` unchanged.

  - $(arg_dict[:sigrhoX])

# Validation

  - `0 < dt.n <= size(X, 2)`. When `X` has a zero variance, `dt.n` is also at most the number of positive variances, which the call on their block checks.

# Returns

  - `X::MatNum`: The input matrix `X`, modified in place.

# Examples

```jldoctest
julia> using StableRNGs

julia> rng = StableRNG(123456789);

julia> X = rand(rng, 10, 5);

julia> X = X' * X
5×5 Matrix{Float64}:
 3.29494  2.0765   1.73334  2.01524  1.77493
 2.0765   2.46967  1.39953  1.97242  2.07886
 1.73334  1.39953  1.90712  1.17071  1.30459
 2.01524  1.97242  1.17071  2.24818  1.87091
 1.77493  2.07886  1.30459  1.87091  2.44414

julia> detone!(Detone(), X)
5×5 Matrix{Float64}:
  3.29494    -1.14673     0.0868439  -0.502106   -1.71581
 -1.14673     2.46967    -0.876289   -0.0864304   0.274663
  0.0868439  -0.876289    1.90712    -1.18851    -0.750345
 -0.502106   -0.0864304  -1.18851     2.24818    -0.0774753
 -1.71581     0.274663   -0.750345   -0.0774753   2.44414
```

# Related

  - [`detone`](@ref)
  - [`Detone`](@ref): the closed form this function applies, and the steps it runs.
  - [`zero_variance_block!`](@ref)
  - [`MatNum`](@ref)
  - [`Option`](@ref)
  - [`Posdef`](@ref)

# References

  - $(ref_dict[:kim2005]) Equation 5.
  - $(ref_dict[:mlp1]) Section 2.6.
"""
function detone!(::Nothing, X::MatNum)::MatNum
    return X
end
function detone!(dt::Detone, X::MatNum)
    n = dt.n
    @argcheck(zero(n) < n <= size(X, 2),
              DomainError(n,
                          "`Detone.n` is $n, and `X` holds $(size(X, 2)) assets. The detoner removes the `n` leading principal components, so `n` is at most the number of assets. State a value in `1:$(size(X, 2))`."))
    # A zero variance has no correlation, so only the positive block is detoned. The call on
    # the block checks `dt.n` against the size of the block.
    if zero_variance_block!((block, _) -> detone!(dt, block), X)
        return X
    end
    n -= 1
    s = LinearAlgebra.diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    vals, vecs = LinearAlgebra.eigen(X)
    vals = LinearAlgebra.Diagonal(vals)[(end - n):end, (end - n):end]
    vecs = vecs[:, (end - n):end]
    X .-= vecs * vals * transpose(vecs)
    X .= StatsBase.cov2cor(X)
    posdef!(dt.pdm, X)
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return X
end
"""
    detone(dt::Option{<:AbstractDetoneEstimator}, X::MatNum) -> MatNum

Out-of-place version of [`detone!`](@ref).

# Algorithm

 1. Copy `X`.
 2. Apply [`detone!`](@ref) to the copy, and return it. The function never changes the input.

# Arguments

  - $(arg_dict[:odt])
      + `::Detone`: Removes the `dt.n` largest principal components from a copy of `X`.
      + `::Nothing`: Returns `X` unchanged.
  - $(arg_dict[:sigrhoX])

# Returns

  - `X::MatNum`: A new matrix equal to the detoned version of the input.

# Examples

```jldoctest
julia> using StableRNGs

julia> rng = StableRNG(123456789);

julia> X = rand(rng, 10, 5);
       X = X' * X;

julia> Xd = detone(Detone(), X);

julia> size(Xd)
(5, 5)
```

# Related

  - [`detone!`](@ref)
  - [`Detone`](@ref)
  - [`MatNum`](@ref)
  - [`Option`](@ref)
  - [`Posdef`](@ref)

# References

  - $(ref_dict[:kim2005]) Equation 5.
  - $(ref_dict[:mlp1]) Section 2.6.
"""
function detone(::Nothing, X::MatNum)::MatNum
    return X
end
function detone(dt::AbstractDetoneEstimator, X::MatNum)
    X = copy(X)
    detone!(dt, X)
    return X
end

export Detone, detone, detone!
public AbstractDetoneEstimator
