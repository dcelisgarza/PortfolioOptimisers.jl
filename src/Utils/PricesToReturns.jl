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

    asset_names, factor_names = if !isempty(F)
        X = merge(X, F; method = join_method)
        colnames(X), colnames(F)
    else
        colnames(X), ()
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
