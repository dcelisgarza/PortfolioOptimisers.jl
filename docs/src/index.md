```@meta
CurrentModule = PortfolioOptimisers
```

# PortfolioOptimisers

Documentation for [PortfolioOptimisers](https://github.com/dcelisgarza/PortfolioOptimisers.jl).

There are three overarching design choices in `PortfolioOptimisers.jl`:

 1. Well-defined type hierarchy. This lets us define interfaces and leverage multiple dispatch to easily and quickly add new features.
 2. Strongly typed immutable structs. Which reduce runtime dispatches and allow the compiler to aggressively optimise. And it means there is always a single source of truth for each process, minimising bugs and ensuring valid structures throughout program execution.
 3. Compositional design. There are many interactions within `PortfolioOptimisers.jl`, and we want to decouple them as much as possible. This is achieved by defining small, self-contained units that can be composed together to form more complex structures. This allows us to build complex workflows without introducing unnecessary complexity. It also makes development and testing easier and fearless, as each component can be tested in isolation. It also means the only way to break existing functionality is to modify an existing structure/function.

These design choices increase usage and development friction by raising the skill floor and lowering convenience, but ensure correctness and performance.
