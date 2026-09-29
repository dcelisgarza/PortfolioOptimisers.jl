"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all positive definite matrix estimator types.

All concrete and/or abstract types that implement positive definite matrix projection or estimation should be subtypes of `AbstractPosdefEstimator`.

# Interfaces

In order to implement a new positive definite matrix estimator which will work seamlessly with the library, subtype `AbstractPosdefEstimator` with all necessary parameters as part of the struct, and implement the following methods:

  - `posdef!(pdm::AbstractPosdefEstimator, X::MatNum) -> MatNum`: In-place projection of a matrix to the nearest positive definite matrix.
  - `posdef(pdm::AbstractPosdefEstimator, X::MatNum) -> MatNum`: Optional out-of-place projection of a matrix to the nearest positive definite matrix. A fallback method copies `X` and calls `posdef!`, so it is only needed if the copy can be avoided.

## Arguments

  - $(arg_dict[:pdm])
  - $(arg_dict[:sigrhoX])

## Returns

  - `X::MatNum`: The projected input matrix `X`.

# Examples

We can create a dummy positive definite estimator as follows:

```jldoctest
julia> struct MyPosdefEstimator <: PortfolioOptimisers.AbstractPosdefEstimator end

julia> function PortfolioOptimisers.posdef!(pdm::MyPosdefEstimator, X::PortfolioOptimisers.MatNum)
           # Implement your in-place PD projection logic here.
           println(\"Projecting to positive definite matrix in-place...\")
           return X
       end

julia> function PortfolioOptimisers.posdef(pdm::MyPosdefEstimator, X::PortfolioOptimisers.MatNum)
           X = copy(X)
           println(\"Copy X...\")
           posdef!(pdm, X)
           return X
       end

julia> posdef!(MyPosdefEstimator(), [1.0 2.0; 2.0 1.0])
Projecting to positive definite matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0

julia> posdef(MyPosdefEstimator(), [1.0 2.0; 2.0 1.0])
Copy X...
Projecting to positive definite matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0
```

# Related

  - [`AbstractEstimator`](@ref)
  - [`Posdef`](@ref)
  - [`posdef!`](@ref)
  - [`posdef`](@ref)
"""
abstract type AbstractPosdefEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Repairs a covariance matrix by a clip of the eigenvalues of its correlation matrix, and keeps its variances. It is an algorithm that [`Posdef`](@ref) takes, beside the algorithms of `NearestCorrelationMatrix.jl`.

The algorithm owns its acceptance test. A matrix that passes the Cholesky factorisation, whose covariance eigenvalues are positive, and whose smallest correlation eigenvalue is at least `tau / 2` is returned unchanged. Any other matrix is repaired: the correlation eigenvalues are clipped at `tau`, the diagonal is scaled back to one, and the variances are restored. When the result fails the same test, the clip runs once more at `10 * tau`.

`Newton`, the default of [`Posdef`](@ref), finds the nearest correlation matrix in the Frobenius norm. The clip is a spectral repair with a floor on the eigenvalues, so it is not the nearest matrix, but its smallest eigenvalue is bounded away from zero.

# Mathematical definition

The covariance ``\\mathbf{\\Sigma}`` is standardised, and the eigenvalues of the correlation matrix ``\\mathbf{C}`` are clipped at ``\\tau``:

```math
\\begin{align}
\\mathbf{C} &= \\mathbf{D}^{-1} \\mathbf{\\Sigma} \\mathbf{D}^{-1} = \\mathbf{V} \\mathrm{Diag}(\\boldsymbol{\\lambda}) \\mathbf{V}^\\intercal\\,, \\\\
\\mathbf{Y} &= \\mathbf{V} \\mathrm{Diag}(\\max(\\boldsymbol{\\lambda}, \\tau)) \\mathbf{V}^\\intercal\\,, \\\\
\\hat{\\mathbf{C}}_{ij} &= \\frac{Y_{ij}}{\\sqrt{Y_{ii} Y_{jj}}}\\,, \\\\
\\hat{\\mathbf{\\Sigma}}_{ij} &= \\hat{\\mathbf{C}}_{ij}\\, \\sigma_i \\sigma_j \\;\\; (i \\neq j)\\,, \\qquad \\hat{\\mathbf{\\Sigma}}_{ii} = \\mathbf{\\Sigma}_{ii}\\,.
\\end{align}
```

The repair is accepted when ``\\hat{\\mathbf{\\Sigma}}`` has a Cholesky factor, ``\\lambda_{\\min}(\\hat{\\mathbf{\\Sigma}}) > 0`` and ``\\lambda_{\\min}(\\hat{\\mathbf{C}}) \\geq \\tau / 2``. The half floor absorbs the round-off of the scaling, so a matrix that was repaired once is not repaired again. After the retry at ``10 \\tau``, the repair is also accepted when the Cholesky factor exists and ``\\lambda_{\\min}(\\hat{\\mathbf{\\Sigma}}) \\geq -n\\, \\varepsilon \\max_i |\\lambda_i(\\hat{\\mathbf{\\Sigma}})|``.

With `higham = true`, Higham's alternating projections with Dykstra's correction replace ``\\mathbf{C}`` before the clip. Each iteration projects onto the positive semidefinite cone with an eigenvalue floor of ``5 \\varepsilon``, and then onto the matrices of unit diagonal:

```math
\\begin{align}
\\mathbf{R}_k &= \\mathbf{Y}_{k-1} - \\Delta\\mathbf{S}_{k-1}\\,, \\\\
\\mathbf{X}_k &= \\mathbf{V}_k \\mathrm{Diag}(\\max(\\boldsymbol{\\lambda}(\\mathbf{R}_k), 5 \\varepsilon)) \\mathbf{V}_k^\\intercal\\,, \\\\
\\Delta\\mathbf{S}_k &= \\mathbf{X}_k - \\mathbf{R}_k\\,, \\\\
\\mathbf{Y}_k &= \\mathbf{X}_k \\text{ with a unit diagonal}\\,.
\\end{align}
```

The iterations start at ``\\mathbf{Y}_0 = \\mathbf{C}`` and ``\\Delta\\mathbf{S}_0 = \\mathbf{0}``. They stop when ``\\lambda_{\\min}(\\mathbf{Y}_k) \\geq -n\\, \\varepsilon \\max_i |\\lambda_i(\\mathbf{Y}_k)|``, or after `iter` iterations. The clip then runs on the last ``\\mathbf{Y}_k``, so a sequence that has not converged still gives a repaired matrix, and the call is never refused for it.

Where:

  - ``\\mathbf{\\Sigma}``: Input covariance matrix.
  - ``\\mathbf{D} = \\mathrm{Diag}(\\boldsymbol{\\sigma})``: Diagonal matrix of the standard deviations ``\\sigma_i = \\sqrt{\\mathbf{\\Sigma}_{ii}}``.
  - ``\\mathbf{C}``: Correlation matrix of ``\\mathbf{\\Sigma}``, or the last iterate of the alternating projections.
  - ``\\mathbf{V}``, ``\\boldsymbol{\\lambda}``: Eigenvectors and eigenvalues of ``\\mathbf{C}``.
  - ``\\tau``: Floor of the correlation eigenvalues, the field `tau`.
  - ``\\mathbf{Y}``: Correlation matrix with the clipped spectrum, before the scaling of its diagonal.
  - ``\\hat{\\mathbf{C}}``: Repaired correlation matrix.
  - ``\\hat{\\mathbf{\\Sigma}}``: Repaired covariance matrix.
  - ``\\lambda_{\\min}(\\cdot)``: Smallest eigenvalue of a matrix.
  - ``n``: Number of rows of ``\\mathbf{\\Sigma}``.
  - $(math_dict[:eps_machine])
  - ``\\mathbf{R}_k``, ``\\mathbf{X}_k``, ``\\mathbf{Y}_k``: Corrected input, positive semidefinite projection and unit-diagonal projection of iteration ``k``.
  - ``\\mathbf{V}_k``: Eigenvectors of ``\\mathbf{R}_k``.
  - ``\\Delta\\mathbf{S}_k``: Dykstra's correction of iteration ``k``.

# Algorithm

[`posdef_accepts`](@ref) and [`posdef_repair!`](@ref) run these steps for this algorithm. [`posdef!`](@ref) sets the zero variances aside before the repair.

 1. Refuse a matrix with a non-finite entry, a negative variance, or a correlation matrix that is not symmetric to `1e-8 + 1e-5 |C_ji|` in each entry.
 2. Standardise the matrix into `C` with [`clipped_correlation`](@ref).
 3. Return the matrix unchanged when [`clipped_repair_holds`](@ref) accepts it.
 4. When `higham` is `true`, replace `C` with the last iterate of [`higham_alternating_projections`](@ref).
 5. Take the eigen decomposition of `C` once.
 6. For `tau` and then `10 * tau`: clip the eigenvalues, rebuild `Y`, scale its diagonal to one, scale it back to a covariance, symmetrise it, restore the variances, and return it when [`clipped_repair_holds`](@ref) accepts it.
 7. Warn when the last result has no Cholesky factor, or when its smallest eigenvalue is below `-n * eps * max|λ|`. The last result is returned either way, as [`posdef!`](@ref) returns an unrepaired matrix.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ClippedNearestCorrelation(;
        tau::Number = 1e-13,
        higham::Bool = false,
        iter::Integer = 100,
    ) -> ClippedNearestCorrelation

Keywords correspond to the struct's fields.

## Validation

  - `0 < tau < 1`.
  - `iter > 0`.

# Examples

```jldoctest
julia> ClippedNearestCorrelation()
ClippedNearestCorrelation
     tau ┼ Float64: 1.0e-13
  higham ┼ Bool: false
    iter ┴ Int64: 100
```

The correlation matrix below passes the Cholesky factorisation, and its smallest eigenvalue is about `1e-15`. The default `Posdef()` accepts it unchanged. The clip lifts its smallest eigenvalue to about `tau`.

```jldoctest
julia> using LinearAlgebra

julia> X = [1.0 1.0-1e-15; 1.0-1e-15 1.0];

julia> isposdef(X)
true

julia> posdef(Posdef(), X) == X
true

julia> Y = posdef(Posdef(; alg = ClippedNearestCorrelation()), X);

julia> eigmin(Y) > 5e-14
true
```

# Related

  - [`Posdef`](@ref)
  - [`posdef!`](@ref)
  - [`posdef_accepts`](@ref)
  - [`posdef_repair!`](@ref)

# References

  - $(ref_dict[:higham2002])
"""
@concrete struct ClippedNearestCorrelation <: AbstractAlgorithm
    """
    Floor of the correlation eigenvalues, ``0 < \\tau < 1``. The acceptance test reads `tau / 2`, and the one retry clips at `10 * tau`.
    """
    tau
    """
    When `true`, Higham's alternating projections run before the clip.
    """
    higham
    """
    Largest number of iterations of the alternating projections. It is read only when `higham` is `true`.
    """
    iter
    function ClippedNearestCorrelation(tau::Number, higham::Bool, iter::Integer)
        assert_unit_interval(tau, :tau)
        assert_gt0(iter, :iter)
        return new{typeof(tau), typeof(higham), typeof(iter)}(tau, higham, iter)
    end
end
function ClippedNearestCorrelation(; tau::Number = 1e-13, higham::Bool = false,
                                   iter::Integer = 100)
    return ClippedNearestCorrelation(tau, higham, iter)
end
"""
$(DocStringExtensions.TYPEDEF)

Projects a matrix to the nearest positive definite matrix, typically used for co-moment matrices.

`Posdef` encapsulates all parameters required for positive definite matrix projection in [`posdef!`](@ref) and [`posdef`](@ref) to perform the nearest positive definite projection according to the estimator.

The algorithm `alg` is one of the algorithms of `NearestCorrelationMatrix.jl`, with `NearestCorrelationMatrix.Newton` as the default, or [`ClippedNearestCorrelation`](@ref), a clip of the correlation eigenvalues that keeps the variances. Each algorithm decides with [`posdef_accepts`](@ref) which matrix it leaves unchanged.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Posdef(;
        alg::Any = NearestCorrelationMatrix.Newton,
        kwargs::NamedTuple = (;),
    ) -> Posdef

Keywords correspond to the struct's fields.

## Validation

  - `kwargs` is empty when `alg` is a [`ClippedNearestCorrelation`](@ref). That algorithm reads its own fields, so a keyword would be ignored.

# Examples

```jldoctest
julia> Posdef()
Posdef
     alg ┼ UnionAll: NearestCorrelationMatrix.Newton
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractPosdefEstimator`](@ref)
  - [`ClippedNearestCorrelation`](@ref)
  - [`posdef!`](@ref)
  - [`posdef`](@ref)
  - [`NearestCorrelationMatrix.jl`](https://github.com/adknudson/NearestCorrelationMatrix.jl)

# References

  - $(ref_dict[:higham2002])
  - $(ref_dict[:qisun2006])
"""
@concrete struct Posdef <: AbstractPosdefEstimator
    """
    The algorithm used for the nearest correlation matrix projection.
    """
    alg
    """
    A named tuple of keyword arguments to be passed to the algorithm.
    """
    kwargs
    function Posdef(alg::Any, kwargs::NamedTuple)::Posdef
        @argcheck(!isa(alg, ClippedNearestCorrelation) || isempty(kwargs),
                  ArgumentError("kwargs must be empty when alg is a ClippedNearestCorrelation, because that algorithm reads only its own fields. Got\nkwargs => $kwargs"))
        return new{typeof(alg), typeof(kwargs)}(alg, kwargs)
    end
end
function Posdef(; alg::Any = NearestCorrelationMatrix.Newton,
                kwargs::NamedTuple = (;))::Posdef
    return Posdef(alg, kwargs)
end
"""
    posdef!(pdm::Option{<:AbstractPosdefEstimator}, X::MatNum) -> MatNum

In-place projection of a matrix to the nearest positive definite matrix using the specified estimator.

For matrices without unit diagonal, the function converts them into correlation matrices i.e. matrices with unit diagonal, applies the algorithm, and rescales them back.

A variance that is exactly zero belongs to a constant variable, and it keeps a zero row and column. The repair runs on the block of positive variances, so the result is positive semidefinite and not positive definite. No warning comes for the zero rows, because they are the correct answer.

# Mathematical definition

Solves the nearest correlation matrix problem:

```math
\\begin{align}
\\hat{\\mathbf{C}} &= \\underset{\\mathbf{Y} \\succeq 0,\\; Y_{ii} = 1}{\\arg\\min} \\lVert \\mathbf{C} - \\mathbf{Y} \\rVert_F\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{C}}``: Nearest positive semidefinite correlation matrix.
  - ``\\mathbf{C}``: Input correlation matrix.
  - ``\\mathbf{Y}``: Feasible correlation matrix (positive semidefinite, unit diagonal).
  - ``\\lVert \\cdot \\rVert_F``: Frobenius norm.

For covariance matrices, first standardise ``\\mathbf{C} = \\mathrm{diag}(\\mathbf{\\Sigma})^{-1/2} \\mathbf{\\Sigma}\\, \\mathrm{diag}(\\mathbf{\\Sigma})^{-1/2}``, project, then rescale back.

A zero variance has no correlation, so it cannot be standardised. The projection keeps the diagonal, and a positive semidefinite matrix with a zero diagonal entry has a zero row and column, as [`zero_variance_rows!`](@ref) states. The projection above then runs on the block of positive variances alone.

# Algorithm

 1. Check that `X` is square.
 2. Return `X` unchanged when [`posdef_accepts`](@ref) accepts it under `pdm.alg`. For an algorithm of `NearestCorrelationMatrix.jl` the test is `isposdef`, and [`ClippedNearestCorrelation`](@ref) brings its own test and its refusals.
 3. Set the zero-variance rows and columns of `X` to zero with [`zero_variance_rows!`](@ref). When there is such a row, repair the block of positive variances with a recursive call, write the block back, and return `X`. The recursive call warns only when the block stays not positive definite. A matrix with a zero variance never passes the test of step 2, so this step is always reached for it.
 4. Repair `X` with [`posdef_repair!`](@ref) under `pdm.alg` and the keyword arguments `pdm.kwargs`, and return `X`.

# Arguments

  - $(arg_dict[:opdm])

      + `::Posdef`: The algorithm specified in `pdm.alg` is used to project `X` to the nearest PD matrix. If `X` is already positive definite, it is left unchanged.
      + `::Nothing`: No-op.

  - $(arg_dict[:sigrhoX])

# Validation

  - `X` is validated with [`assert_matrix_issquare`](@ref) before any other step.
  - [`ClippedNearestCorrelation`](@ref) refuses a non-finite entry, a negative variance and a matrix that is not symmetric, as [`posdef_accepts`](@ref) states.

# Returns

  - `X::MatNum`: The input matrix `X` modified in-place.

# Examples

```jldoctest
julia> using LinearAlgebra

julia> est = Posdef();

julia> X = [1.0 0.9; 0.9 1.0];

julia> X[1, 2] = 2.0;  # Not PD

julia> posdef!(est, X)
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0

julia> LinearAlgebra.isposdef(X)
true
```

The second variable below is constant. Its row and column stay zero, and the other two variables are repaired as above.

```jldoctest
julia> X = [1.0 0.0 2.0; 0.0 0.0 0.0; 2.0 0.0 1.0];

julia> posdef!(Posdef(), X)
3×3 Matrix{Float64}:
 1.0  0.0  1.0
 0.0  0.0  0.0
 1.0  0.0  1.0
```

# Related

  - [`posdef`](@ref)
  - [`Posdef`](@ref)
  - [`zero_variance_rows!`](@ref)
  - [`MatNum`](@ref)
"""
function posdef!(::Nothing, X::MatNum)::MatNum
    return X
end
function posdef!(pdm::Posdef, X::MatNum)
    assert_matrix_issquare(X, :X)
    if posdef_accepts(pdm.alg, X)
        return X
    end
    # A zero variance forces a zero row and column, so only the positive block is repaired.
    # The recursive call warns when that block stays indefinite, never for the zero rows.
    p = zero_variance_rows!(X, trues(size(X, 1)))
    if !all(p)
        block = X[p, p]
        posdef!(pdm, block)
        X[p, p] = block
        return X
    end
    return posdef_repair!(pdm.alg, X; pdm.kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Sets the rows and columns of the zero variances of a co-moment matrix to zero, and returns the mask of the positive variances.

A variance that is exactly zero belongs to a constant variable. A positive semidefinite matrix with a zero diagonal entry has a zero row and column, so a repair that keeps the diagonal has no other answer for that row, and no correlation exists to standardise it. The test is `iszero`, with no tolerance: a variance of `1e-34` is positive and stays in the block.

# Mathematical definition

Every ``2 \\times 2`` principal minor of a positive semidefinite matrix is non-negative:

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{ii}\\,\\hat{\\mathbf{\\Sigma}}_{jj} - \\hat{\\mathbf{\\Sigma}}_{ij}^{2} &\\geq 0\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_hat_ii])
  - $(math_dict[:Sigma_hat_ij])

So ``\\hat{\\mathbf{\\Sigma}}_{ii} = 0`` forces ``\\hat{\\mathbf{\\Sigma}}_{ij} = 0`` for every ``j``.

# Algorithm

 1. Mark each row of `fin` whose diagonal entry in `X` is exactly zero.
 2. Set every entry of `X` whose row or column is marked, and whose other index is in `fin`, to zero.
 3. Return `fin` without the marked rows.

# Arguments

  - $(arg_dict[:sigrhoX])
  - `fin`: The rows that the caller repairs. An entry whose row or column is outside `fin` is never written, so a `NaN` frame around the block stays as it is.

# Returns

  - `blk::BitVector`: The rows of `fin` with a variance that is not zero. The caller repairs this block.

# Related

  - [`posdef!`](@ref)
  - [`matrix_processing_block!`](@ref)
"""
function zero_variance_rows!(X::MatNum, fin::AbstractVector{Bool})
    z = fin .& iszero.(LinearAlgebra.diag(X))
    if any(z)
        X[z, fin] .= zero(eltype(X))
        X[fin, z] .= zero(eltype(X))
    end
    return fin .& .!z
end
"""
    posdef_accepts(alg, X::MatNum) -> Bool

Return `true` when the algorithm `alg` of [`Posdef`](@ref) leaves `X` unchanged, so [`posdef!`](@ref) returns before any repair.

Each algorithm owns its test. An algorithm of `NearestCorrelationMatrix.jl` accepts a matrix that passes `isposdef`, which is the test [`posdef!`](@ref) always ran. [`ClippedNearestCorrelation`](@ref) accepts only a matrix whose smallest correlation eigenvalue is at least `tau / 2` as well, so it repairs a matrix that is positive definite but nearly singular.

# Algorithm

The method Julia selects on the type of `alg` is the algorithm.

 1. Any algorithm but [`ClippedNearestCorrelation`](@ref): return `isposdef(X)`.
 2. [`ClippedNearestCorrelation`](@ref):
     1. Return `false` when a variance is exactly zero. The zero-row rule of [`posdef!`](@ref) runs next, and it repairs the block of positive variances, which this test then reads.
     2. Standardise `X` into `C` with [`clipped_correlation`](@ref), which refuses the inputs the repair cannot read.
     3. Return the verdict of [`clipped_repair_holds`](@ref) on `X` and `C`.

# Arguments

  - `alg`: The algorithm of [`Posdef`](@ref).
  - $(arg_dict[:sigrhoX])

# Validation

  - [`ClippedNearestCorrelation`](@ref) refuses what [`clipped_correlation`](@ref) refuses.

# Returns

  - `flag::Bool`: `true` when `X` is left unchanged.

# Related

  - [`posdef!`](@ref)
  - [`posdef_repair!`](@ref)
  - [`ClippedNearestCorrelation`](@ref)
"""
function posdef_accepts(alg, X::MatNum)::Bool
    return LinearAlgebra.isposdef(X)
end
function posdef_accepts(alg::ClippedNearestCorrelation, X::MatNum)::Bool
    v = LinearAlgebra.diag(X)
    # A zero variance is set aside by `posdef!` after this test, so it must reach that step.
    if any(iszero, v)
        return false
    end
    return clipped_repair_holds(alg, X, clipped_correlation(X, v))
end
"""
    posdef_repair!(alg, X::MatNum; kwargs...) -> MatNum

Repair `X` in place under the algorithm `alg` of [`Posdef`](@ref). [`posdef!`](@ref) calls it after [`posdef_accepts`](@ref) has refused `X` and after the zero variances are set aside, so every variance that reaches it is not zero.

# Algorithm

The method Julia selects on the type of `alg` is the algorithm.

 1. Any algorithm but [`ClippedNearestCorrelation`](@ref), which is an algorithm of `NearestCorrelationMatrix.jl`:
     1. Read the diagonal of `X` into `s`. When any entry of `s` is not one, `X` is a covariance matrix: replace `s` with its square roots and convert `X` to a correlation matrix with `StatsBase.cov2cor!`. The test is `any(!isone, s)`, so it is the value of the diagonal that decides, never the type of `X`.
     2. Project `X` onto the nearest correlation matrix with `NearestCorrelationMatrix.nearest_cor!`, under `alg` and `kwargs`.
     3. Warn when the projected `X` is still not positive definite. `X` is returned either way, so the caller must check the result when it cannot tolerate an unrepaired matrix.
     4. When step 1 converted a covariance matrix, convert `X` back with `StatsBase.cor2cov!`. The standard deviations are the ones read in step 1, so the original diagonal returns up to rounding.
 2. [`ClippedNearestCorrelation`](@ref), whose `# Mathematical definition` states the formulas:
     1. Read the variances `v` and the standard deviations `s`, and standardise `X` into `C` with [`clipped_correlation`](@ref).
     2. When `alg.higham` is `true`, replace `C` with the result of [`higham_alternating_projections`](@ref).
     3. Take the eigen decomposition `E` of `C`.
     4. For `tau` equal to `alg.tau` and then `10 * alg.tau`: clip the eigenvalues of `E` at `tau` into `Y`, scale the diagonal of `Y` to one, write `S = Y .* s' .* s` symmetrised into `X`, set the diagonal of `X` to `v`, and return `X` when [`clipped_repair_holds`](@ref) accepts `X` and `Y`.
     5. Warn when the last `X` has no Cholesky factor, or when its smallest eigenvalue is below `-n * eps * max|λ|`, and return `X`.

# Arguments

  - `alg`: The algorithm of [`Posdef`](@ref).
  - $(arg_dict[:sigrhoX])
  - `kwargs...`: The keyword arguments of `NearestCorrelationMatrix.nearest_cor!`. [`ClippedNearestCorrelation`](@ref) reads none, and [`Posdef`](@ref) refuses them with it.

# Returns

  - `X::MatNum`: The input matrix `X` modified in-place.

# Related

  - [`posdef!`](@ref)
  - [`posdef_accepts`](@ref)
  - [`ClippedNearestCorrelation`](@ref)
"""
function posdef_repair!(alg, X::MatNum; kwargs...)
    s = LinearAlgebra.diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    NearestCorrelationMatrix.nearest_cor!(X, alg; kwargs...)
    if !LinearAlgebra.isposdef(X)
        @warn("Matrix could not be made positive definite.")
    end
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return X
end
function posdef_repair!(alg::ClippedNearestCorrelation, X::MatNum; kwargs...)
    v = LinearAlgebra.diag(X)
    s = sqrt.(v)
    C = clipped_correlation(X, v)
    if alg.higham
        C = higham_alternating_projections(C, alg.iter)
    end
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(C, :L))
    for tau in (alg.tau, 10 * alg.tau)
        Y = (E.vectors .* transpose(max.(E.values, tau))) * transpose(E.vectors)
        d = sqrt.(LinearAlgebra.diag(Y))
        Y ./= d .* transpose(d)
        Y[LinearAlgebra.diagind(Y)] .= one(eltype(Y))
        S = Y .* transpose(s) .* s
        X .= (S .+ transpose(S)) ./ 2
        X[LinearAlgebra.diagind(X)] = v
        if clipped_repair_holds(alg, X, Y)
            return X
        end
    end
    l = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(X, :L))
    chol = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(X, :L); check = false)
    if !(LinearAlgebra.issuccess(chol) &&
         first(l) >= -length(l) * eps(eltype(l)) * maximum(abs, l))
        @warn("Matrix could not be made positive definite.")
    end
    return X
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Standardise a covariance matrix into its correlation matrix, and refuse the inputs that [`ClippedNearestCorrelation`](@ref) cannot read.

The entry ``C_{ij} = \\Sigma_{ij} / (\\sigma_i \\sigma_j)`` divides by the product of the standard deviations, and the diagonal is set to one exactly. The symmetry is tested on `C` and not on `X`, so the test does not move with the units of `X`: a covariance in units a thousand times smaller passes or fails exactly as the original does.

# Algorithm

 1. Refuse a non-finite entry of `X` with [`assert_all_finite`](@ref).
 2. Refuse a variance that is not positive with [`assert_gt0`](@ref).
 3. Divide `X` by `s .* s'`, where `s = sqrt.(v)`, into `C`, and set the diagonal of `C` to one.
 4. Refuse `C` when an entry differs from its transpose by more than `1e-8 + 1e-5 * abs(C[j, i])`. The repair reads the lower triangle alone, so a larger difference is a data error and not round-off.
 5. Return `C`.

# Arguments

  - $(arg_dict[:sigrhoX])
  - `v`: The diagonal of `X`.

# Validation

  - `all(isfinite, X)`, which raises an `IsNonFiniteError`.
  - `all(v .> 0)`, which raises a `DomainError`.
  - `C` symmetric to the tolerance of step 4, which raises an `ArgumentError`.

# Returns

  - `C::Matrix`: The correlation matrix of `X`.

# Related

  - [`ClippedNearestCorrelation`](@ref)
  - [`posdef_accepts`](@ref)
  - [`posdef_repair!`](@ref)
"""
function clipped_correlation(X::MatNum, v::VecNum)
    assert_all_finite(X, :X)
    assert_gt0(v, Symbol("diag(X)"))
    s = sqrt.(v)
    C = X ./ (s .* transpose(s))
    C[LinearAlgebra.diagind(C)] .= one(eltype(C))
    @argcheck(all(abs(C[i, j] - C[j, i]) <= 1e-8 + 1e-5 * abs(C[j, i])
                  for j in axes(C, 2), i in axes(C, 1)),
              ArgumentError("the correlation matrix of X must be symmetric to 1e-8 + 1e-5 |C[j, i]| in each entry, because the repair reads its lower triangle alone. Got a largest difference of $(maximum(abs, C - transpose(C)))."))
    return C
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when [`ClippedNearestCorrelation`](@ref) accepts the covariance `X` with the correlation matrix `C`.

The test is the acceptance test of the clip, and the same test decides whether an input is left unchanged. Each factorisation reads the lower triangle alone.

# Algorithm

 1. Return `false` when `X` has no Cholesky factor.
 2. Return `false` when the smallest eigenvalue of `C` is below `alg.tau / 2`.
 3. Return whether the smallest eigenvalue of `X` is positive.

# Arguments

  - `alg`: The algorithm.
  - $(arg_dict[:sigrhoX])
  - `C`: The correlation matrix of `X`.

# Returns

  - `flag::Bool`: `true` when `X` is accepted.

# Related

  - [`ClippedNearestCorrelation`](@ref)
  - [`posdef_accepts`](@ref)
  - [`posdef_repair!`](@ref)
"""
function clipped_repair_holds(alg::ClippedNearestCorrelation, X::MatNum, C::MatNum)::Bool
    chol = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(X, :L); check = false)
    return LinearAlgebra.issuccess(chol) &&
           LinearAlgebra.eigmin(LinearAlgebra.Symmetric(C, :L)) >= alg.tau / 2 &&
           LinearAlgebra.eigmin(LinearAlgebra.Symmetric(X, :L)) > 0
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run Higham's alternating projections with Dykstra's correction on a correlation matrix, and return the last iterate. [`ClippedNearestCorrelation`](@ref) states the iteration under `# Mathematical definition`.

The iterations stop at the first iterate that is positive semidefinite to round-off, or after `iter` iterations. The last iterate is returned in both cases, and the clip that follows repairs the rest. An iterate that has not converged has a unit diagonal and a spectrum that is nearly positive semidefinite, so its clip is a valid correlation matrix, and it is nearer the input than the clip of the input alone.

# Algorithm

 1. Set `Y` to `C` and the correction `dS` to zero.
 2. For each of at most `iter` iterations:
     1. Subtract `dS` from `Y` into `R`.
     2. Clip the eigenvalues of `R` at `5 * eps` into `Y`.
     3. Set `dS` to `Y - R`.
     4. Set the diagonal of `Y` to one.
     5. Stop when the smallest eigenvalue of `Y` is at least `-n * eps * max|λ|`.
 3. Return `Y`.

# Arguments

  - `C`: The correlation matrix.
  - `iter`: The largest number of iterations.

# Returns

  - `Y::Matrix`: The last iterate, a matrix of unit diagonal.

# Related

  - [`ClippedNearestCorrelation`](@ref)
  - [`posdef_repair!`](@ref)

# References

  - $(ref_dict[:higham2002])
"""
function higham_alternating_projections(C::MatNum, iter::Integer)
    T = real(eltype(C))
    Y = C
    dS = zero(C)
    for _ in 1:iter
        R = Y - dS
        E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(R, :L))
        Y = (E.vectors .* transpose(max.(E.values, 5 * eps(T)))) * transpose(E.vectors)
        dS = Y - R
        Y[LinearAlgebra.diagind(Y)] .= one(T)
        l = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(Y, :L))
        if first(l) >= -size(Y, 1) * eps(T) * maximum(abs, l)
            break
        end
    end
    return Y
end
"""
    posdef(pdm::Option{<:AbstractPosdefEstimator}, X::MatNum) -> MatNum

Out-of-place version of [`posdef!`](@ref).

# Algorithm

 1. Copy `X`.
 2. Apply [`posdef!`](@ref) to the copy, and return it. The input is never modified.

# Arguments

  - $(arg_dict[:opdm])
  - $(arg_dict[:sigrhoX])

# Returns

  - `X::MatNum`: A new matrix equal to the nearest positive definite projection of the input.

# Examples

```jldoctest
julia> using LinearAlgebra

julia> X = [1.0 2.0; 2.0 1.0];

julia> Xpd = posdef(Posdef(), X);

julia> LinearAlgebra.isposdef(Xpd)
true
```

# Related

  - [`posdef!`](@ref)
  - [`Posdef`](@ref)
  - [`MatNum`](@ref)
"""
function posdef(::Nothing, X::MatNum)::MatNum
    return X
end
function posdef(pdm::AbstractPosdefEstimator, X::MatNum)
    X = copy(X)
    posdef!(pdm, X)
    return X
end

export Posdef, ClippedNearestCorrelation, posdef, posdef!
public AbstractPosdefEstimator
