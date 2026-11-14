"""
    brinson_attribution(X::TimeArray, w::VecNum, wb::VecNum,
                        asset_classes::DataFrame, col, date0 = nothing, date1 = nothing)

Compute the Brinson performance attribution of each asset class [brinson_attribution](@cite).

`brinson_attribution` returns a DataFrame with the Brinson attribution of each asset class. It splits the total excess return of the class into asset allocation, security selection and interaction. The function reads the asset weights, the benchmark weights and the class of each asset over the date range.

`date0` and `date1` are **positional** arguments after `col`. The filter applies only when the call gives both. A call that gives only one of them attributes the whole of `X`.

`X` holds **prices**, not returns. The period return of an asset is the ratio of its last value in the range to its first value, minus one. This ratio is a return only when the entries are prices.

The function attributes **one** evaluation period, between the two dates, and it reads no weight path. `w` and `wb` stay the same over that period. So the function attributes a fund whose holdings drifted in the period at its weights at the start. A multi-period attribution that follows the drift needs a rule that links the effects of the single periods into a cumulative effect. This library has no such rule, so it has no such method. Attribute each period alone, and link the results outside the library.

Both class returns divide by the weight of the class, and no guard stops a division by zero. A class with zero portfolio weight makes its ``r_{i}`` a `NaN`. A class with zero benchmark weight makes its ``r_{i}^{b}`` a `NaN`. The two cases spread differently, because ``\\mathrm{AA}_{i}`` reads only ``r_{i}^{b}``. A zero portfolio weight keeps ``\\mathrm{AA}_{i}`` finite, and it makes the other three rows of that class a `NaN`. A zero benchmark weight makes all four rows a `NaN`.

The `Total` column is a row sum, so its entry of a row is a `NaN` when one of the class entries of the row is a `NaN`. The columns of the other classes stay finite, because each column reads only its own class and ``r^{b}``. Do not remove the rows of the empty class from `asset_classes` alone. Then `asset_classes` and `w` have different lengths, and the function raises a `DimensionMismatch`.

# Mathematical definition

```math
\\begin{align}
r_{j} &= \\dfrac{P_{j,\\,1}}{P_{j,\\,0}} - 1\\\\
r^{b} &= \\sum_{j} w_{j}^{b}\\, r_{j}\\\\
r_{i} &= \\dfrac{\\sum_{j \\in \\mathcal{C}_{i}} w_{j}\\, r_{j}}{\\sum_{j \\in \\mathcal{C}_{i}} w_{j}}\\\\
r_{i}^{b} &= \\dfrac{\\sum_{j \\in \\mathcal{C}_{i}} w_{j}^{b}\\, r_{j}}{\\sum_{j \\in \\mathcal{C}_{i}} w_{j}^{b}}\\\\
\\mathrm{AA}_{i} &= \\left(w_{i} - w_{i}^{b}\\right) \\left(r_{i}^{b} - r^{b}\\right)\\\\
\\mathrm{SS}_{i} &= w_{i}^{b} \\left(r_{i} - r_{i}^{b}\\right)\\\\
\\mathrm{I}_{i} &= \\left(w_{i} - w_{i}^{b}\\right) \\left(r_{i} - r_{i}^{b}\\right)\\\\
\\mathrm{TER}_{i} &= \\mathrm{AA}_{i} + \\mathrm{SS}_{i} + \\mathrm{I}_{i}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{C}_{i}``: The ``i``-th asset class, the set of assets whose `col` entry is that class.
  - ``P_{j,\\,0}``, ``P_{j,\\,1}``: Price of asset ``j`` at the first and at the last observation of the range.
  - ``r_{j}``: Return of asset ``j`` over the range.
  - ``w_{j}``, ``w_{j}^{b}``: Portfolio and benchmark weight of asset ``j``.
  - ``w_{i} = \\sum_{j \\in \\mathcal{C}_{i}} w_{j}``, ``w_{i}^{b} = \\sum_{j \\in \\mathcal{C}_{i}} w_{j}^{b}``: Portfolio and benchmark weight of the class.
  - ``r_{i}``, ``r_{i}^{b}``: Portfolio and benchmark return of the class. Each is the weighted sum of the returns of its members, divided by the weight of the class. So a class return is a weighted average and not a weighted sum.
  - ``r^{b}``: Benchmark return of the whole universe.
  - ``\\mathrm{AA}_{i}``: Asset allocation of the class, the part of the excess return that comes from the weight of the class.
  - ``\\mathrm{SS}_{i}``: Security selection of the class, the part that comes from the choice of assets in the class.
  - ``\\mathrm{I}_{i}``: Interaction of the class, the product of the two differences.
  - ``\\mathrm{TER}_{i}``: Total excess return of the class, the sum of the three.

# Algorithm

 1. Filter the date range. When the call gives `date0` and `date1`, find the first and the last observation of `X` in the range. These are `idx1` and `idx2`. Otherwise take the whole of `X`, so `idx1` is `1` and `idx2` is `length(X)`.
 2. Make `ret`, the period return of each asset, from the values at `idx1` and `idx2`.
 3. Take the dot product of `ret` and `wb`. This is `ret_b`, the benchmark return of the whole universe.
 4. Read the class of each asset from column `col` of `asset_classes`. Build `sets_mat`, the membership matrix. Its entry is `true` when the asset belongs to the class.
 5. Take the next class. Sum the two weight vectors over its members, to get `w_i` and `wb_i`. Divide the two weighted sums of the returns of its members by those weights, to get `ret_i` and `ret_b_i`.
 6. Compute `AA_i`, `SS_i`, `I_i` and `TER_i` of that class. Write the four numbers as the column of the class in `df`.
 7. Repeat steps 5 and 6 for the other classes.
 8. Add the `Total` column, the sum of the class columns for each row.

# Arguments

  - `X`: TimeArray of asset prices.
  - `w`: Vector of portfolio weights.
  - `wb`: Vector of benchmark weights.
  - `asset_classes`: DataFrame that gives the class of each asset.
  - `col`: Column of `asset_classes` that holds the class of each asset.
  - `date0`: Optional start date of the attribution period.
  - `date1`: Optional end date of the attribution period.

# Returns

  - `df::DataFrame`: DataFrame with one row each for asset allocation, security selection, interaction and the total effect. It has one column for each asset class, and a `Total` column.

# Related

  - [`VecNum`](@ref)
  - [`performance_summary`](@ref): It summarises one realised return series. This function splits one period across the classes.
  - [`calc_net_returns`](@ref): It gives the net return series of a portfolio, and it charges fees. This function charges no fees.
  - [`AssetSetsMatrixEstimator`](@ref): It builds the group membership matrix of the library. Step 4 builds its own matrix from a DataFrame column.

# References

  - $(ref_dict[:brinson_attribution]) Pages 73-76. $(ref_dict[:no_original_source])
"""
function brinson_attribution(X::TimeSeries.TimeArray, w::VecNum, wb::VecNum,
                             asset_classes::DataFrames.DataFrame, col, date0 = nothing,
                             date1 = nothing)
    # Efficient filtering of date range
    idx1, idx2 = if !isnothing(date0) && !isnothing(date1)
        timestamps = TimeSeries.timestamp(X)
        idx = (Dates.DateTime(date0) .<= timestamps) .&
              (timestamps .<= Dates.DateTime(date1))
        findfirst(idx), findlast(idx)
    else
        1, length(X)
    end

    ret = vec(values(X[idx2]) ./ values(X[idx1]) .- 1)
    ret_b = LinearAlgebra.dot(ret, wb)

    classes = asset_classes[!, col]
    unique_classes = unique(classes)

    df = DataFrames.DataFrame(;
                              index = ["Asset Allocation", "Security Selection",
                                       "Interaction", "Total Excess Return"])

    # Precompute class membership matrix for efficiency
    sets_mat = [class_j == class_i for class_j in classes, class_i in unique_classes]

    for (i, class_i) in enumerate(unique_classes)
        sets_i = view(sets_mat, :, i)

        w_i = LinearAlgebra.dot(sets_i, w)
        wb_i = LinearAlgebra.dot(sets_i, wb)

        ret_i = LinearAlgebra.dot(ret .* sets_i, w) / w_i
        ret_b_i = LinearAlgebra.dot(ret .* sets_i, wb) / wb_i

        w_diff_i = w_i - wb_i
        ret_diff_i = ret_i - ret_b_i

        AA_i = w_diff_i * (ret_b_i - ret_b)
        SS_i = wb_i * ret_diff_i
        I_i = w_diff_i * ret_diff_i
        TER_i = AA_i + SS_i + I_i

        df[!, class_i] = [AA_i, SS_i, I_i, TER_i]
    end

    df[!, "Total"] = sum(eachcol(df[!, 2:end]))

    return df
end

export brinson_attribution
