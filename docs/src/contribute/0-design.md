```@meta
Description = "The design philosophy and the design goals of PortfolioOptimisers.jl: typed hierarchies, immutable estimators, composition, and what each choice buys."
```

# [Design philosophy and goals](@id design)

The design choices behind `PortfolioOptimisers.jl`, and the goals they serve. A contributor reads
this before adding a type or an interface; a user of the [API introduction](@ref) needs none of it.

## Design philosophy

There are three overarching design choices in `PortfolioOptimisers.jl`:

### 1. Well-defined type hierarchies

- Easily and quickly add new features by sticking to defined interfaces.

### 2. Strongly typed immutable structs

- All types are concrete and known at instantiation.
- Constants can be propagated if necessary.
- There is always a single immutable source of truth for every process.
- If needed, modifying values must be done via interface functions, which simplifies finding and fixing bugs. If the interface for modification is not provided the code will throw a missing method exception.

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
