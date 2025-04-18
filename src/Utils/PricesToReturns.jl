function prices_to_returns(X::TimeArray, F::TimeArray = TimeArray(TimeType[], []);
                           ret_method::Symbol = :simple, padding::Bool = false,
                           missing_col_percent::Real = 1.0, missing_row_percent::Real = 1.0,
                           collapse_args::Tuple = (),
                           map_func::Union{Nothing, Function} = nothing,
                           join_method::Symbol = :outer,
                           impute_method::Union{Nothing, <:Impute.Imputor} = nothing)
    @smart_assert(zero(missing_col_percent) <
                  missing_col_percent <=
                  one(missing_col_percent))
    if !isinf(missing_row_percent)
        @smart_assert(zero(missing_row_percent) <
                      missing_row_percent <=
                      one(missing_row_percent))
    end

    if !isempty(F)
        asset_names = string.(colnames(X))
        factor_names = string.(colnames(F))
        X = merge(X, F; method = join_method)
    else
        asset_names = string.(colnames(X))
        factor_names = String[]
    end
    if !isnothing(map_func)
        X = map(map_func, X)
    end
    if !isempty(collapse_args)
        X = collapse(X, collapse_args...)
    end
    X = DataFrame(X)
    f(x) =
        if isa(x, Number) && isnan(x)
            missing
        else
            x
        end
    DataFrames.transform!(X, 2:ncol(X) .=> ByRow((x) -> f(x)); renamecols = false)
    if !isnothing(impute_method)
        X = Impute.impute(X, impute_method)
    end
    missing_mtx = ismissing.(Matrix(X[!, 2:end]))
    missings_cols = vec(count(missing_mtx; dims = 2))
    keep_rows = missings_cols .<= (ncol(X) - 1) * missing_col_percent
    X = X[keep_rows, :]
    missings_rows = vec(count(missing_mtx; dims = 1))
    keep_cols = if !isinf(missing_row_percent)
        missings_rows .<= nrow(X) * missing_row_percent
    else
        missings_rows .== StatsBase.mode(missings_rows)
    end
    X = X[!, [true; keep_cols]]
    select!(X, Not(names(X, Missing)))
    dropmissing!(X)
    X = DataFrame(percentchange(TimeArray(X; timestamp = :timestamp), ret_method;
                                padding = padding))
    col_names = names(X)
    ac = intersect(col_names, asset_names)
    fc = intersect(col_names, factor_names)
    oc = setdiff(col_names, union(ac, fc))
    ts = isempty(oc) ? nothing : vec(Matrix(X[!, oc]))
    if isempty(fc)
        fc = nothing
        F = nothing
    else
        F = Matrix(X[!, fc])
    end
    if isempty(ac)
        ac = nothing
        X = nothing
    else
        X = Matrix(X[!, ac])
    end
    return ReturnsData(; ts = ts, nx = ac, X = X, nf = fc, F = F)
end

export prices_to_returns
