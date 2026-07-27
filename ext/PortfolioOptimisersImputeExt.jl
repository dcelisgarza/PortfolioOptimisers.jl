module PortfolioOptimisersImputeExt

using PortfolioOptimisers, Impute

## Supplies the `Impute.Imputor` method of the `apply_impute_method` seam in
## `src/03_Preprocessing.jl`, keeping `Impute` (and its `DataDeps`/`HTTP = "1"` cap) out of the
## default dependency footprint. See `docs/adr/0042-impute-is-a-weak-dependency.md`.
function PortfolioOptimisers.apply_impute_method(X, impute_method::Impute.Imputor)
    return Impute.impute(X, impute_method)
end

end
