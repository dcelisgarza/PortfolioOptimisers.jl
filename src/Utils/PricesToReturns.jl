function prices_to_returns(X::TimeArray, F::TimeArray = TimeArray(TimeType[], []);
                           ret_method::Symbol = :simple, padding::Bool = false,
                           missing_col_percent::Real = 1.0, missing_row_percent::Real = 1.0,
                           collapse_args::Tuple = (),
                           map_func::Union{Nothing, Function} = nothing,
                           join_method::Symbol = :outer)
    @smart_assert(zero(missing_col_percent) <
                  missing_col_percent <=
                  one(missing_col_percent))
    if !isinf(missing_row_percent)
        @smart_assert(zero(missing_row_percent) <
                      missing_row_percent <=
                      one(missing_row_percent))
    end
    asset_names = colnames(X)
    factor_names = isempty(F) ? () : colnames(F)

    factor_names = if !isempty(F)
        X = merge(X, F; method = join_method)
        colnames(F)
    else
        ()
    end
    if !isnothing(map_func)
        X = map(map_func, X)
    end
    if !isempty(collapse_args)
        X = collapse(X, collapse_args...)
    end
    X = DataFrame(percentchange(X, ret_method; padding = padding))
    f(x) =
        if isa(x, Number) && (isnan(x) || x < zero(x))
            missing
        else
            x
        end
    DataFrames.transform!(X, 2:ncol(X) .=> ByRow((x) -> f(x)); renamecols = false)

    missings_cols = vec(count(ismissing.(Matrix(X[!, 2:end])); dims = 2))
    keep_rows = missings_cols .<= (ncol(X) - 1) * missing_col_percent
    X = X[keep_rows, 1:end]

    missings_rows = vec(count(ismissing.(Matrix(X[!, 2:end])); dims = 1))
    keep_cols = if !isinf(missing_row_percent)
        missings_rows .<= nrow(X) * missing_row_percent
    else
        missings_rows .== StatsBase.mode(missings_rows)
    end
    X = X[!, findall([true; keep_cols])]
    dropmissing!(X)
    return X
end

export prices_to_returns

#=
function join_ticker_prices(tickers, lopt::LoadOpt = LoadOpt())
    path = lopt.path
    select = lopt.select
    tickers = intersect(tickers, replace.(readdir(path), ".csv" => ""))
    prices = DataFrame(; timestamp = DateTime[])
    jtp_iter = ProgressBar(tickers)
    for ticker ∈ jtp_iter
        set_description(jtp_iter, "Generating master prices dataframe:")
        ticker_prices = load_ticker_prices(ticker, lopt)
        if isempty(ticker_prices)
            continue
        end
        DataFrames.rename!(ticker_prices,
                           setdiff(select, (:timestamp,))[1] => Symbol(ticker))
        prices = outerjoin(prices, ticker_prices; on = :timestamp)
    end

    f(x) =
        if (isa(x, Number) && (isnan(x) || x < zero(x)))
            missing
        else
            x
        end

    transform!(prices, setdiff(names(prices), ("timestamp",)) .=> ByRow((x) -> f(x));
               renamecols = false)

    missings = Int[]
    sizehint!(missings, ncol(prices))
    for col ∈ eachcol(prices)
        push!(missings, count(ismissing.(col)))
    end

    m = StatsBase.mode(missings)
    missings[1] = m
    prices = prices[!, missings .== m]
    dropmissing!(prices)
    return sort!(prices, :timestamp)
end
=#
