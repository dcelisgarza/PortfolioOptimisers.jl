# Introduction

This section explains `PortfolioOptimisers.jl` API in detail. The pages are organised in exactly the same way as the `src` folder itself. This means there should be a 1 to 1 correspondence between documentation and source files[^1].

## Design philosophy

There are three overarching design choices in `PortfolioOptimisers.jl`:

 1. Well-defined type hierarchies:
    
     1. Easily and quickly add new features by sticking to defined interfaces.

 2. Strongly typed immutable structs:
    
     1. All types are concrete and known at instantiation.
     2. Constants can be propagated if necessary.
     3. There is always a single immutable source of truth for every process.
     4. If needed, modifying values must be done via interface functions, which simplifies finding and fixing bugs. If the interface for modification is not provided the code will throw a missing method exception.
     5. Future developments may make use of [`Accessors.jl`](https://github.com/JuliaObjects/Accessors.jl) for certain things.
 3. Compositional design:
   
    1. `PortfolioOptimisers.jl` is a toolkit whose components can interact in complex, deeply nested ways.
    2. Separation of concerns lets us subdivide logical components into self-contained units.
    3. Extensive and judicious data validation checks are performed at the earliest possible moment---mostly at variable instantiation---to ensure correctness.
    4. Basal parameters live in the most basal structures, allowing for the possibility of efficient data reuse.

<!-- 1. there are many interactions within `PortfolioOptimisers.jl`, by using composition we can decouple and compartmentalise processes into self-contained units;
2. complexity arises by combining these logical subunits, their immutability means that performing assertions at variable instantiation ensures their correctness throughout the program lifetime;
3. makes development and testing easier and fearless, as each component can be tested in isolation;
4. ensures the only way to break existing functionality is to modify an existing structure/function;
5. we try to keep the most basal parameters in the most basal data structures, improving code reusability and maintainability, reduces the memory footprint, and allows for more flexibility. -->

These design choices increase initial usage and development friction by raising the skill floor and lowering convenience, but ensures correctness, robustness, performance, and maintainability.

## Contents

```@contents
Pages = ["01_Base.md", "02_Tools.md", "03_PosdefMatrix.md", "04_Denoise.md", "05_Detone.md", "06_MatrixProcessing.md", "09_JuMPModelOptimisation.md", "10_OWA.md", "07_Moments/01_Base_Moments.md", "07_Moments/02_SimpleExpectedReturns.md", "07_Moments/03_Covariance.md", "07_Moments/04_SimpleVariance.md", "07_Moments/05_GerberCovariances.md", "07_Moments/06_SmythBrobyCovariance.md", "07_Moments/07_DistanceCovariance.md", "07_Moments/08_LTDCovariance.md", "07_Moments/09_RankCovariance.md", "07_Moments/10_Histogram.md", "07_Moments/11_MutualInfoCovariance.md", "07_Moments/12_PortfolioOptimisersCovariance.md", "07_Moments/13_ShrunkExpectedReturns.md", "07_Moments/14_EquilibriumExpectedReturns.md", "07_Moments/15_ExcessExpectedReturns.md", "07_Moments/16_Coskewness.md", "07_Moments/17_Cokurtosis.md", "07_Moments/18_Base_Regression.md", "07_Moments/19_StepwiseRegression.md", "07_Moments/20_DimensionReductionRegression.md", "07_Moments/21_ImpliedVolatility.md", "08_Distance/1_Base_Distance.md", "08_Distance/2_Distance.md", "08_Distance/3_DistanceDistance.md", "08_Distance/4_GeneralDistance.md", "08_Distance/5_GeneralDistanceDistance.md", "11_Phylogeny/1_Base_Phylogeny.md", "11_Phylogeny/2_Clustering.md", "11_Phylogeny/3_Hierarchical.md", "11_Phylogeny/4_DBHT.md", "11_Phylogeny/5_Phylogeny.md", "12_ConstraintGeneration/1_Base_ConstraintGeneration.md", "12_ConstraintGeneration/2_LinearConstraintGeneration.md", "12_ConstraintGeneration/3_PhylogenyConstraintGeneration.md", "12_ConstraintGeneration/4_WeightBoundsConstraintGeneration.md", "12_ConstraintGeneration/5_ThresholdConstraintGeneration.md", "13_Prior/10_EntropyPoolingPrior.md", "14_UncertaintySets.md/1_Base_UncertaintySets.md", "18_RiskMeasures/1_Base_RiskMeasures.md", "19_Optimisation/15_NestedClustering.md"]
```

[^1]: Except for a small number of cases, most of which are convenience function overloads. This means some links will not take you to the exact method definition. Other than hard-coding links to specific lines of code, which is fragile, I haven't found an easy solution.
