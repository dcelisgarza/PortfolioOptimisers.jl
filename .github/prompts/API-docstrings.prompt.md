---
agent: ask
description: Generate Julia docstrings.
---

Use the most appropriate example as a guide to generate API docs.

```julia
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all custom processes in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement a custom process should subtype `MyAbstractCustomProcess`.

# Interfaces

In order to implement a new custom process that can seamlessly work with the library, subtype `MyAbstractCustomProcess`, ensuring that the structure contains all necessary parameters for the custom process, and implement the following methods:

## Custom process interface

### Functions

- `do_process(pr::MyAbstractCustomProcess, b::Real, c::Integer)`: Performs the custom process.

#### Arguments

- `pr`: Custom process.
- `b`: First argument for the custom process.
- `c`: Second argument for the custom process.

### Examples

We can create a dummy custom process as follows:

```jldoctest
julia> struct MyNewCustomProcess{T1, T2} <: PortfolioOptimisers.MyAbstractCustomProcess
           alg::T1
           new_param::T2
           function MyNewCustomProcess(alg::MyAbstractCustomProcessAlgorithm, new_param::Symbol)
               return new{typeof(alg), typeof(new_param)}(alg, new_param)
           end
       end

julia> function MyNewCustomProcess(; alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1(), new_param::Symbol = :Foo)
           return MyNewCustomProcess(alg, new_param)
        end

julia> function PortfolioOptimisers.do_process(a::MyNewCustomProcess, b::Real, c::Integer)
          println("new custom process: $b $c $(a.sym)")
          do_algorithm(a.alg, c)
          return nothing
       end

julia> do_process(MyNewCustomProcess(), -0.5, 9)
new custom process: -0.5 9 Foo
algorithm 1: 9
```

# Related

- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
abstract type MyAbstractCustomProcess end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all custom process algorithms in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement a custom process algorithms should subtype `MyAbstractCustomProcessAlgorithm`.

# Interfaces

In order to implement a new custom process algorithms that can seamlessly work with the library, subtype `MyAbstractCustomProcessAlgorithm`, ensuring that the structure contains all necessary parameters for the custom process algorithm, and implement the following methods:

## Custom process algorithm interface

### Functions

- `do_algorithm(pra::MyAbstractCustomProcessAlgorithm, c::Integer) -> Integer`: Performs the custom process algorithm and returns the result.

#### Arguments

- `pra`: Custom process algorithm.
- `c`: Argument for the custom process algorithm.

### Examples

We can create a dummy custom process algorithm as follows:

```jldoctest
julia> struct MyNewCustomProcessAlgorithm{T} <: PortfolioOptimisers.MyAbstractCustomProcessAlgorithm
           new_param::T
           function MyNewCustomProcessAlgorithm(new_param::Symbol)
               return new{typeof(new_param)}(new_param)
           end
       end

julia> function MyNewCustomProcessAlgorithm(; new_param::Symbol = :Bar)
           return MyNewCustomProcessAlgorithm(new_param)
        end

julia> function PortfolioOptimisers.do_algorithm(alg::MyNewCustomProcessAlgorithm, c::Integer)
          println("new algorithm: $c $(alg.new_param)")
          return c + 1
       end

julia> do_algorithm(MyNewCustomProcessAlgorithm(), 3)
new algorithm: 3 Bar
4
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
abstract type MyAbstractCustomProcessAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements my custom process algorithm 1.

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
struct MyCustomProcessAlgorithm1 <: MyAbstractCustomProcessAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Performs the custom process algorithm 1.

# Arguments

- `alg::MyCustomProcessAlgorithm1`: The algorithm to perform.
- `c::Integer`: The input integer.

# Details

- Multiplies `c` by 2.
- Prints the result with a custom message.
- Returns the result.

```jldoctest
julia> do_algorithm(MyCustomProcessAlgorithm1(), 3)
algorithm 1: 6
6
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`do_process`](@ref)
"""
function do_algorithm(::MyCustomProcessAlgorithm1, c::Integer)
    c = c * 2
    println("algorithm 1: $c")
    return c
end
"""
$(DocStringExtensions.TYPEDEF)

Defines my custom process 1.

# Fields

- `alg::MyAbstractCustomProcessAlgorithm`: The algorithm to use.

# Constructors

    MyConcreteCustomProcess1(; alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1())

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> MyConcreteCustomProcess1()
MyConcreteCustomProcess1
  alg ┴ MyCustomProcessAlgorithm1()
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
struct MyConcreteCustomProcess1{T} <: MyAbstractCustomProcess
    alg::T
    function MyConcreteCustomProcess1(alg::MyAbstractCustomProcessAlgorithm)
        return new{typeof(alg)}(alg)
    end
end
function MyConcreteCustomProcess1(;
                                  alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1())
    return MyConcreteCustomProcess1(alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Performs the custom process 1.

# Arguments

- `a::MyConcreteCustomProcess1`: The custom process to perform.
- `b::Real`: The first argument.
- `c::Integer`: The second argument.

# Validation

- `b >= 0`: `b` must be non-negative.

# Details

- Checks `b >= 0` before performing the process.
- Prints a message using the arguments `b` and `c`.
- Calls `do_algorithm` with the algorithm from `a.alg` and `c`.

# Examples

```jldoctest
julia> do_process(MyConcreteCustomProcess1(), 1.0, 2)
Custom process 1: 1.0 + 2
algorithm 1: 4
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`MyConcreteCustomProcess1`](@ref)
- [`do_algorithm`](@ref)
"""
function do_process(a::MyConcreteCustomProcess1, b::Real, c::Integer)
    @argcheck(b >= 0, "b must be non-negative")
    println("Custom process 1: $b + $c")
    do_algorithm(a.alg, c)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validates that `val > 2`.

# Arguments

- `val::Real`: The value to validate.
"""
function assert_val_value(val::Real)
    @argcheck(val >= 2 * one(eltype(val)), "val must be non-negative")
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Defines my custom process 2.

# Fields

- `alg::MyAbstractCustomProcessAlgorithm`: The algorithm to use.
- `val::Real`: The value to use.

# Constructors

    MyConcreteCustomProcess2(;
                             alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1(),
                             val::Real = 2.0)

Keywords correspond to the struct's fields.

## Validation

- `val` is validated via [`assert_val_value`](@ref).

# Examples

```jldoctest
julia> MyConcreteCustomProcess2()
MyConcreteCustomProcess2
  alg ┼ MyCustomProcessAlgorithm1()
  val ┴ 2.0
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`assert_val_value`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
struct MyConcreteCustomProcess2{T1, T2} <: MyAbstractCustomProcess
    alg::T1
    val::T2
    function MyConcreteCustomProcess2(alg::MyAbstractCustomProcessAlgorithm, val::Real)
        return new{typeof(alg), typeof(val)}(alg, val)
    end
end
function MyConcreteCustomProcess2(;
                                  alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1(),
                                  val::Real = 2.0)
    assert_val_value(val)
    return MyConcreteCustomProcess2(alg, val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Performs the custom process 2.

# Arguments

- `a::MyConcreteCustomProcess2`: The custom process to perform.
- `b::Real`: The first argument.
- `c::Integer`: The second argument.

# Details

- Prints a message using the arguments `a.val`, `b` and `c`.
- Calls `do_algorithm` with the algorithm from `a.alg` and `c`.

# Examples

```jldoctest
julia> do_process(MyConcreteCustomProcess2(), 1.0, 2)
Custom process 2: 2.0 - 1.0 + 2
algorithm 1: 4
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`MyConcreteCustomProcess1`](@ref)
- [`do_algorithm`](@ref)
"""
function do_process(a::MyConcreteCustomProcess2, b::Real, c::Integer)
    println("Custom process 2: $(a.val) - $b + $c")
    do_algorithm(a.alg, c)
    return nothing
end

```
