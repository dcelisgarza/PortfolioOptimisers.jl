"""
    ⊗(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}

Tensor product of two arrays. Returns a matrix of size `(length(A), length(B))` where each element is the product of elements from `A` and `B`.

# Mathematical definition

```math
\\begin{align}
(\\boldsymbol{a} \\otimes \\boldsymbol{b})_{ij} &= a_{i} b_{j}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{a}``: Vectorised first array `A`, of length ``n``.
  - ``\\boldsymbol{b}``: Vectorised second array `B`, of length ``m``.
  - ``a_{i}``, ``b_{j}``: Entries of ``\\boldsymbol{a}`` and ``\\boldsymbol{b}`` in linear index order.

The result is the outer product ``\\boldsymbol{a} \\boldsymbol{b}^\\intercal``, an ``n \\times m`` matrix. `A` and `B` may carry any shape, because both are read in linear index order.

# Arguments

  - `A::ArrNum`: First array.
  - `B::ArrNum`: Second array.

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊗([1, 2], [3, 4])
2×2 Matrix{Int64}:
 3  4
 6  8
```

# Related

  - [`ArrNum`](@ref)
  - [`kron`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#Base.kron)
"""
⊗(A::ArrNum, B::ArrNum) = reshape(kron(B, A), (length(A), length(B)))
"""
    ⊙(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}
    ⊙(A::ArrNum, B) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊙(A, B::ArrNum) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊙(A, B) -> promote_type(eltype(A), eltype(B))

Elementwise (Hadamard) multiplication.

# Mathematical definition

```math
\\begin{align}
(\\boldsymbol{a} \\odot \\boldsymbol{b})_{i} &= a_{i} b_{i}\\,, \\\\
(\\boldsymbol{a} \\odot \\beta)_{i} &= a_{i} \\beta\\,, \\\\
(\\alpha \\odot \\boldsymbol{b})_{i} &= \\alpha b_{i}\\,, \\\\
\\alpha \\odot \\beta &= \\alpha \\beta\\,.
\\end{align}
```

Where:

  - $(math_dict[:ab_operands])
  - $(math_dict[:alpha_beta_scalars])
  - $(math_dict[:i_linear])

Both array operands must carry the same length. A scalar operand multiplies every entry of the array operand.

# Arguments

  - `A`: First operand (array or scalar).
  - `B`: Second operand (array or scalar).

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊙([1, 2], [3, 4])
2-element Vector{Int64}:
 3
 8

julia> PortfolioOptimisers.:⊙([1, 2], 2)
2-element Vector{Int64}:
 2
 4

julia> PortfolioOptimisers.:⊙(2, [3, 4])
2-element Vector{Int64}:
 6
 8

julia> PortfolioOptimisers.:⊙(2, 3)
6
```

# Related

  - [`⊗`](@ref)
  - [`⊘`](@ref)
  - [`⊕`](@ref)
  - [`⊖`](@ref)
  - [`ArrNum`](@ref)
"""
⊙(A::ArrNum, B::ArrNum) = A .* B
⊙(A::ArrNum, B) = A * B
⊙(A, B::ArrNum) = A * B
⊙(A, B) = A * B
"""
    ⊘(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}
    ⊘(A::ArrNum, B) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊘(A, B::ArrNum) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊘(A, B) -> promote_type(eltype(A), eltype(B))

Elementwise (Hadamard) division.

# Mathematical definition

```math
\\begin{align}
(\\boldsymbol{a} \\oslash \\boldsymbol{b})_{i} &= \\frac{a_{i}}{b_{i}}\\,, \\\\
(\\boldsymbol{a} \\oslash \\beta)_{i} &= \\frac{a_{i}}{\\beta}\\,, \\\\
(\\alpha \\oslash \\boldsymbol{b})_{i} &= \\frac{\\alpha}{b_{i}}\\,, \\\\
\\alpha \\oslash \\beta &= \\frac{\\alpha}{\\beta}\\,.
\\end{align}
```

Where:

  - $(math_dict[:ab_operands])
  - $(math_dict[:alpha_beta_scalars])
  - $(math_dict[:i_linear])

Both array operands must carry the same length. The division is not guarded, so a zero divisor gives an infinity or a `NaN`.

# Arguments

  - `A`: Dividend (array or scalar).
  - `B`: Divisor (array or scalar).

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊘([4, 9], [2, 3])
2-element Vector{Float64}:
 2.0
 3.0

julia> PortfolioOptimisers.:⊘([4, 6], 2)
2-element Vector{Float64}:
 2.0
 3.0

julia> PortfolioOptimisers.:⊘(8, [2, 4])
2-element Vector{Float64}:
 4.0
 2.0

julia> PortfolioOptimisers.:⊘(8, 2)
4.0
```

# Related

  - [`⊗`](@ref)
  - [`⊙`](@ref)
  - [`⊕`](@ref)
  - [`⊖`](@ref)
  - [`ArrNum`](@ref)
"""
⊘(A::ArrNum, B::ArrNum) = A ./ B
⊘(A::ArrNum, B) = A / B
⊘(A, B::ArrNum) = A ./ B
⊘(A, B) = A / B
"""
    ⊕(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}
    ⊕(A::ArrNum, B) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊕(A, B::ArrNum) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊕(A, B) -> promote_type(eltype(A), eltype(B))

Elementwise (Hadamard) addition.

# Mathematical definition

```math
\\begin{align}
(\\boldsymbol{a} \\oplus \\boldsymbol{b})_{i} &= a_{i} + b_{i}\\,, \\\\
(\\boldsymbol{a} \\oplus \\beta)_{i} &= a_{i} + \\beta\\,, \\\\
(\\alpha \\oplus \\boldsymbol{b})_{i} &= \\alpha + b_{i}\\,, \\\\
\\alpha \\oplus \\beta &= \\alpha + \\beta\\,.
\\end{align}
```

Where:

  - $(math_dict[:ab_operands])
  - $(math_dict[:alpha_beta_scalars])
  - $(math_dict[:i_linear])

Both array operands must carry the same length. A scalar operand is added to every entry of the array operand, which the built-in `+` refuses.

# Arguments

  - `A`: First summand (array or scalar).
  - `B`: Second summand (array or scalar).

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊕([1, 2], [3, 4])
2-element Vector{Int64}:
 4
 6

julia> PortfolioOptimisers.:⊕([1, 2], 2)
2-element Vector{Int64}:
 3
 4

julia> PortfolioOptimisers.:⊕(2, [3, 4])
2-element Vector{Int64}:
 5
 6

julia> PortfolioOptimisers.:⊕(2, 3)
5
```

# Related

  - [`⊗`](@ref)
  - [`⊙`](@ref)
  - [`⊘`](@ref)
  - [`⊖`](@ref)
  - [`ArrNum`](@ref)
"""
⊕(A::ArrNum, B::ArrNum) = A + B
⊕(A::ArrNum, B) = A .+ B
⊕(A, B::ArrNum) = A .+ B
⊕(A, B) = A + B
"""
    ⊖(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}
    ⊖(A::ArrNum, B) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊖(A, B::ArrNum) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊖(A, B) -> promote_type(eltype(A), eltype(B))

Elementwise (Hadamard) subtraction.

# Mathematical definition

```math
\\begin{align}
(\\boldsymbol{a} \\ominus \\boldsymbol{b})_{i} &= a_{i} - b_{i}\\,, \\\\
(\\boldsymbol{a} \\ominus \\beta)_{i} &= a_{i} - \\beta\\,, \\\\
(\\alpha \\ominus \\boldsymbol{b})_{i} &= \\alpha - b_{i}\\,, \\\\
\\alpha \\ominus \\beta &= \\alpha - \\beta\\,.
\\end{align}
```

Where:

  - $(math_dict[:ab_operands])
  - $(math_dict[:alpha_beta_scalars])
  - $(math_dict[:i_linear])

Both array operands must carry the same length. A scalar operand is subtracted from every entry of the array operand, which the built-in `-` refuses.

# Arguments

  - `A`: Minuend (array or scalar).
  - `B`: Subtrahend (array or scalar).

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊖([4, 6], [1, 2])
2-element Vector{Int64}:
 3
 4

julia> PortfolioOptimisers.:⊖([4, 6], 2)
2-element Vector{Int64}:
 2
 4

julia> PortfolioOptimisers.:⊖(8, [2, 4])
2-element Vector{Int64}:
 6
 4

julia> PortfolioOptimisers.:⊖(8, 2)
6
```

# Related

  - [`⊗`](@ref)
  - [`⊙`](@ref)
  - [`⊘`](@ref)
  - [`⊕`](@ref)
  - [`ArrNum`](@ref)
"""
⊖(A::ArrNum, B::ArrNum) = A - B
⊖(A::ArrNum, B) = A .- B
⊖(A, B::ArrNum) = A .- B
⊖(A, B) = A - B
"""
    dot_scalar(a::Union{<:Number, <:JuMP.AbstractJuMPScalar}, b::VecNum) -> Number
    dot_scalar(a::VecNum, b::Union{<:Number, <:JuMP.AbstractJuMPScalar}) -> Number
    dot_scalar(a::VecNum, b::VecNum) -> Number

Efficient scalar and vector dot product utility.

  - If one argument is a `Union{<:Number, <:JuMP.AbstractJuMPScalar}` and the other an `VecNum`, returns the scalar times the sum of the vector.
  - If both arguments are `VecNum`s, returns their `dot` product.

# Mathematical definition

```math
\\begin{align}
\\mathrm{dot\\_scalar}(\\alpha, \\boldsymbol{b}) &= \\alpha \\sum_{i=1}^{n} b_{i}\\,, \\\\
\\mathrm{dot\\_scalar}(\\boldsymbol{a}, \\beta) &= \\beta \\sum_{i=1}^{n} a_{i}\\,, \\\\
\\mathrm{dot\\_scalar}(\\boldsymbol{a}, \\boldsymbol{b}) &= \\boldsymbol{a}^\\intercal \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - ``\\alpha``, ``\\beta``: Scalar operand, a number or a `JuMP` scalar.
  - ``\\boldsymbol{a}``, ``\\boldsymbol{b}``: Vector operand of length ``n``.

The first two forms are the dot product of the vector with a constant vector of value ``\\alpha``, so the scalar stands for a uniform vector. The sum replaces that constant vector, so no ``n``-length array is built.

# Arguments

  - `a`: First operand, a scalar or a vector.
  - `b`: Second operand, a scalar or a vector.

# Returns

  - `res::Number`: The resulting scalar.

# Examples

```jldoctest
julia> PortfolioOptimisers.dot_scalar(2.0, [1.0, 2.0, 3.0])
12.0

julia> PortfolioOptimisers.dot_scalar([1.0, 2.0, 3.0], 2.0)
12.0

julia> PortfolioOptimisers.dot_scalar([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
32.0
```

# Related

  - [`VecNum`](@ref)
  - [`JuMP.AbstractJuMPScalar`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.JuMP.AbstractJuMPScalar)
"""
function dot_scalar(a::Union{<:Number, <:JuMP.AbstractJuMPScalar}, b::VecNum)
    return a * sum(b)
end
function dot_scalar(a::VecNum, b::Union{<:Number, <:JuMP.AbstractJuMPScalar})
    return sum(a) * b
end
function dot_scalar(a::VecNum, b::VecNum)
    return LinearAlgebra.dot(a, b)
end
