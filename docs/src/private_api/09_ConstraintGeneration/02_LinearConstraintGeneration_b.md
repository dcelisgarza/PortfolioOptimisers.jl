```@meta
Description = "Linear Constraints (b), private API of PortfolioOptimisers.jl: LcE_Lc, VecLcE, LcE_VecLcE, VecLcE_Lc, LcE_Lc_VecLcE_Lc, get_linear_constraints, …"
```

# Linear Constraints (b): private API

```@docs
LcE_Lc
VecLcE
LcE_VecLcE
VecLcE_Lc
LcE_Lc_VecLcE_Lc
get_linear_constraints(lcs::PR_VecPR, sets::UniverseSets, key::Option{<:AbstractString} = nothing; datatype::DataType = Float64, strict::Bool = false, rr::Option{<:AbstractRegressionResult} = nothing)
universe_axis
constraint_row_length
constraint_row_term
_parse_equation
has_invalid_plus
_expr_depth_exceeds
assert_investable_constraint_width
```
