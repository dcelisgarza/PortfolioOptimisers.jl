```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "PortfolioOptimisers.jl"
  text:
  tagline: 🦍🤝💪 💎🙌 🚀🌕.
  image:
    src: logo.svg
    alt: PortfolioOptimisers
  actions:
    - theme: brand
      text: Get Started
      link: examples/1-Getting-Started
    - theme: alt
      text: View on Github
      link: https://github.com/dcelisgarza/PortfolioOptimisers.jl
    - theme: alt
      text: API
      link: 001-Base
---
```

```@meta
CurrentModule = PortfolioOptimisers
```

# PortfolioOptimisers

Documentation for [PortfolioOptimisers](https://github.com/dcelisgarza/PortfolioOptimisers.jl).

## Design philosophy

There are three overarching design choices in `PortfolioOptimisers.jl`:

 1. Well-defined type hierarchies:
    
     1. lets us define interfaces and leverage multiple dispatch to easily and quickly add new features.

 2. Strongly typed immutable structs:
    
     1. reduces runtime dispatch;
     2. allows the compiler to perform aggressive optimisations via specialisation and constant propagation;
     3. there is always a single source of truth for every process, minimising bugs and ensuring valid structures throughout program execution.
 3. Compositional design:
    
     1. there are many interactions within `PortfolioOptimisers.jl`, by using composition we can decouple and compartmentalise processes into self-contained units;
     2. complexity arises by combining these logical subunits, their immutability means that performing assertions at variable instantiation ensures their correctness throughout the program lifetime;
     3. makes development and testing easier and fearless, as each component can be tested in isolation;
     4. ensures the only way to break existing functionality is to modify an existing structure/function;
     5. we try to keep the most basal parameters in the most basal data structures, improving code reusability and maintainability, reduces the memory footprint, and allows for more flexibility.

These design choices increase initial usage and development friction by raising the skill floor and lowering convenience, but ensures correctness, robustness, performance, and maintainability.

:gorilla: 