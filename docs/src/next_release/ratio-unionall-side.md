```@meta
Description = "A side of a ratio measure whose element type Julia cannot split narrows to the Union of its element types, so the bind of a data benchmark costs nothing."
```

# [A ratio narrows a side whose element type Julia cannot split](@id migration-next-ratio-unionall-side)

## [Fields that change](@id migration-next-ratio-unionall-side-fields)

- **A side of a `NonOptimisationRiskRatio`, a `MeanReturnRiskRatio` or an `ExpectedReturnRiskRatio` that is a vector of one measure type whose elements differ in a type parameter**, such as `[Variance(), Variance(; sigma = S)]`, or a vector whose element type is a `Union` with such a member, such as `Union{Variance, Range}[Variance(), Range()]`, is now a copy whose element type is the `Union` of the types of its elements. So `r.r1 == v` holds, and `r.r1 === v` does not. Before this release, the ratio held the vector itself. A side whose element type is concrete, abstract, or a `Union` of concrete types is the vector itself, as before.
