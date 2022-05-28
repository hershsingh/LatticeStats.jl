module LatticeStats

using Statistics
using LsqFit
using Distributions

export jackknife, jackknife_samples, dumbstats, block_average, jackknife_stats
export linear_fit, linear_fit_full
export inverse_prediction, linear_prediction

function block_average(obs, blocksize)
    n = length(obs)
    num_blocks = n÷blocksize
    ob = zeros(num_blocks)
    # block_size = n/num_blocks
    for i in 1:num_blocks
        ob[i] = sum(obs[(i-1)*blocksize + 1: i*blocksize])/blocksize
    end
    return ob
end

function dumbstats(obs)
    val = mean(obs)
    err = std(obs)/sqrt(length(obs))
    return val, err
end

function jackknife_samples(obs; statistic = mean, blocksize=1)

    if blocksize > 1
        obs = block_average(obs, blocksize)
    end

    N = length(obs)
    samples = zeros(Float64, N)
    for i in 1:N
        samples[i] = statistic( [obs[1:i-1]; obs[i+1:end]] )
    end
    return samples
end

function jackknife(obs; statistic = mean, blocksize=1)

    if blocksize > 1
        obs = block_average(obs, blocksize)
    end
    
    samples = jackknife_samples(obs; statistic = statistic,
                                blocksize=1)
    stat_jk = statistic(samples)
    N = length(samples)
    # var_jk = (N-1) * std(samples; corrected=false)
    var_jk = ((N-1)/N) * sum( samples.^2 .- stat_jk^2 )
    return stat_jk, sqrt(var_jk)
end

function jackknife_stats(samples; statistic = mean, blocksize=1)

    # if blocksize > 1
    #     obs = block_average(obs, blocksize)
    # end
    # samples = jackknife_samples(obs; statistic = statistic)

    stat_jk = statistic(samples)
    N = length(samples)
    # var_jk = (N-1) * std(samples; corrected=false)
    var_jk = ((N-1)/N) * sum( samples.^2 .- stat_jk^2 )
    return stat_jk, sqrt(var_jk)
end

"""
    linear_fit(x, y, yerr)

Computes a linear fit using 
    y := model(x, p) = p[1] .+ p[2] .* x

Returns [p, perr] 
"""
function linear_fit(xf, yf, yferr)
    model(x, p) = p[1] .+ p[2] .* x
    fp = curve_fit(model, xf, yf, 1 ./ yferr.^2, [yf[1], 1.0] ) 
    p = coef(fp)
    perr = stderror(fp)
    return p, perr
end

chi_sq(x, y, y_err, model, p) = sum((y .- model(x,p)).^2 ./ (y_err).^2) / (length(y)-1)
linear_model(x, p) = p[1] .+ p[2] .* x

function linear_fit_full(xf, yf, yferr)
    model(x, p) = p[1] .+ p[2] .* x
    fp = curve_fit(model, xf, yf, 1 ./ yferr.^2, [yf[1], 1.0] ) 
    p = coef(fp)
    perr = stderror(fp)
    return p, perr, fp, chi_sq(xf, yf, yferr, model, p)
end

###
using Infiltrator

"""
Reference:
https://stats.stackexchange.com/questions/206531/error-bars-linear-regression-and-standard-deviation-for-point/206682#206682
"""
function inverse_prediction(x,y,yerr, y0; α=0.33)
    n = length(x)

    xm = mean(x)
    ym = mean(y)

    #' Sum of squared deviations (SS_{xx})
    SSxx = sum((x .- xm).^2)

    #' Sum of squared deviations (SS_{yy})
    SSyy = sum((y .- ym).^2)
    
    SSxy = sum((x .- xm) .* (y .- ym))

    #' OLS estimates: y = β_0 + β1*x
    b1 = SSxy/SSxx
    b0 = ym - b1 * xm

    #' OLS estimates: x = γ_0 + γ_1*y
    c1 = SSxy/SSyy
    c0 = xm - c1 * ym

    #' Estimators ŷ(x) and x̂(y)
    ŷ(x) = b0 + b1 * x
    x̂(y) = c0 + c1 * y

    # @exfiltrate
    #' SSE_{y|x}
    SSEy_x = sum((y .- ŷ.(x)).^2)
    s2_y_x = SSEy_x ./ (n-2) 

    #' Standard error = sqrt(residuals/#dof)
    s_y = sqrt( sum((y .- ŷ.(x)).^2) / (n-2 ) )
    s_x = sqrt( sum((x .- x̂.(y)).^2) / (n-2 ) )
    s = s_y

    #' Inverse regressions method
    var_x(x0) = s_y^2/(b1^2) * (1 + 1/n + (x0 .- xm)/(SSx))
    err_x(x0) = sqrt.(var_x(x0))

    #' Standard error of the estimator ŷ(x) 
    # s = sqrt(SSxx / (n-2))
    err_y(x) = s_y * sqrt(( 1/n + (x - xm).^2 / SSxx ))

    #' Confidence intervals (CI) on err_y
    t = quantile(TDist(n-2), 1-α)
    CI_upper(x) = ŷ(x) + t * err_y(x)
    CI_lower(x) = ŷ(x) - t * err_y(x)

    D0(y) =(ym - y )/b1
    g = t*s / (b1 * sqrt(SSxx))
    # CIx_upper(y) = xm + (D0(y) + g + sqrt(D0^2 + (1-g^2)*SSxx/n))/(1-g^2)
    d(y) = D0(y)^2 + (1-g^2)*SSxx/n
    CIx_upper(y) = xm + (D0(y) + g * sqrt( d(y) ))/(1-g^2)
    CIx_lower(y) = xm + (D0(y) - g * sqrt( d(y) ))/(1-g^2)
    xerr(y) = abs(CIx_upper(y) - CIx_lower(y))/2
    # xerr(1.0)

    # p = plot();
    # plot!(p, x, y, yerr=yerr, marker=:circle);
    # scatter!(p, x, y, yerr=yerr, marker=:circle, markersize=10);

    # xx = LinRange(0.9*minimum(x),1.1*maximum(x),100)
    # plot!(p, xx, ŷ.(xx) );

    # # #' 95% confidence intervals
    # plot!(p, xx, CI_upper.(xx, 0.5) );
    # plot!(p, xx, CI_lower.(xx, 0.5) );

    # scatter!(p, x̂.([1.0]), [1.0], xerr=[xerr(1.0)], marker=:cross, markersize=10);
    # scatter!(p, x=CIx_upper.([1.0]), y=[1.0], marker=:cross, markersize=10);
    
    # scatter!(p, x̂.([1.0]), [1.0], xerr=err_x.(x̂.([1.0])), marker=:cross, markersize=10);
    # gui()

    # err_x.(x̂.([1.0]))

    chisq = sum((y .- ŷ.(x)).^2 ./ (yerr).^2)
    x0err = 0.0
    if d(y0) <= 0
        @warn "Confidence interval could not be computed. The slope is likely consistent with zero."
        x0err = 0.0
    else
        x0err = xerr(y0)
    end

    return x̂(y0), x0err, chisq
end

function linear_prediction(x,y,yerr, x0; α=0.33)
    n = length(x)

    xm = mean(x)
    ym = mean(y)

    #' Sum of squared deviations (SS_{xx})
    SSxx = sum((x .- xm).^2)

    #' Sum of squared deviations (SS_{yy})
    SSyy = sum((y .- ym).^2)
    
    SSxy = sum((x .- xm) .* (y .- ym))

    #' OLS estimates: y = β_0 + β1*x
    b1 = SSxy/SSxx
    b0 = ym - b1 * xm

    #' Estimators ŷ(x) and x̂(y)
    ŷ(x) = b0 + b1 * x

    #' SSE_{y|x}
    SSEy_x = sum((y .- ŷ.(x)).^2)
    s2_y_x = SSEy_x ./ (n-2) 

    #' Standard error = sqrt(residuals/#dof)
    s_y = sqrt( sum((y .- ŷ.(x)).^2) / (n-2 ) )
    # s_x = sqrt( sum((x .- x̂.(y)).^2) / (n-2 ) )
    s = s_y

    #' Inverse regressions method
    var_x(x0) = s_y^2/(b1^2) * (1 + 1/n + (x0 .- xm)/(SSx))
    err_x(x0) = sqrt.(var_x(x0))

    #' Standard error of the estimator ŷ(x) 
    # s = sqrt(SSxx / (n-2))
    err_y(x) = s_y * sqrt(( 1/n + (x - xm).^2 / SSxx ))

    #' Confidence intervals (CI) on err_y
    t = quantile(TDist(n-2), 1-α)
    CI_upper(x) = ŷ(x) + t * err_y(x)
    CI_lower(x) = ŷ(x) - t * err_y(x)

    chisq = sum((y .- ŷ.(x)).^2 ./ (yerr).^2)

    return ŷ(x0), t*err_y(x0), chisq, [b0, b1]
end

end
