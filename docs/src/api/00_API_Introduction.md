# Introduction to the API

This section explains `PortfolioOptimisers.jl` API in detail. The pages are organised in exactly the same way as the `src` folder itself. This means there should be a 1 to 1 correspondence between documentation and source files[^1].

## Design philosophy

There are three overarching design choices in `PortfolioOptimisers.jl`:

### 1. Well-defined type hierarchies

- Easily and quickly add new features by sticking to defined interfaces.

### 2. Strongly typed immutable structs

- All types are concrete and known at instantiation.
- Constants can be propagated if necessary.
- There is always a single immutable source of truth for every process.
- If needed, modifying values must be done via interface functions, which simplifies finding and fixing bugs. If the interface for modification is not provided the code will throw a missing method exception.
- Future developments may make use of [`Accessors.jl`](https://github.com/JuliaObjects/Accessors.jl) for certain things.

### 3. Compositional design

- `PortfolioOptimisers.jl` is a toolkit whose components can interact in complex, deeply nested ways.
- Separation of concerns lets us subdivide logical components into isolated, self-contained units. Leading to easier and fearless development and testing.
- Extensive and judicious data validation checks are performed at the earliest possible moment---mostly at variable instantiation---to ensure correctness.
- Turtles all the way down. Structures can be used, reused, and nested in many ways. This allows for efficient data reuse and arbitrary complexity.

## Design goals

This philosophy has three primary goals:

### 1. Maintainability and expandability

- The only way to break existing functionality should be by modifying APIs.
- Adding functionality should be a case of subtyping existing abstract types and implementing the correct interfaces.
- Avoid leaking side effects to other components unless completely necessary. An example of this is entropy pooling requiring the use of a vector of observation weights which must be taken into account in different, largely unrelated places.

### 2. Correctness and robustness

- Each subunit should perform its own data validation as early as possible unless it absolutely needs downstream data.

### 3. Performance

- Types and constants are always fully known at inference time.
- Immutability ensures smaller structs live in the stack.

[^1]: Except for a few cases, most of which are convenience function overloads. This means some links do not go to the exact method definition. Other than hard-coding links to specific lines of code, which is fragile, I haven't found an easy solution.
