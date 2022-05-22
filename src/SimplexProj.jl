"""
Implementation of Algorithms 5.1, 5.2, and 5.3
"""

"""
    project(y::Array{Float64}, a::Float64, p::Float64, n::Int64, ζ_guess::Number=NaN)

Calculates the projection of the vector `y` onto the bounded probability simplex, i.e.
   argmin(1/2*‖x - `y`‖²) s.t. xᵀ1 = `a` and 0 <= x <= `p`
where `p` is a positive real number. It is possible to pass a guess for the value of `ζ`.

# Arguments
- `y`: input vector
- `a`: right-hand side of equality condition
- `p`: upper bound of inequality condition
- `n`: length of `y`
- `ζ_guess`: index used as starting point
"""
function project(y::Array{Float64}, a::Float64, p::Float64, n::Int64, ζ_guess::Number=NaN)

    tol = 1e-12

    ## Verify input.
    p = min(a, p)
    y = vec(y)
    if isempty(y)
        error("y is empty.")
    end
    if p < a/n
        if abs(p - a/n) > tol
            @show p, a, n
            error("p must be greater than or equal to a/n.")
        else
            a = n*p
        end
    end
    if any(x -> x == Inf, y)
        error("y contains at least one entry that is Inf.")
    end

    ## Handle special case.
    if n == 1
        return [a], NaN
    end

    ## Compute the projection.

    # Sort y.
    u = sort(y, rev = true)

    # Calculate ζ.
    λ_found = false
    ζ, z_ζ  = find_ζ(u, a, p, ζ_guess)
    # ζ, z_ζ  = find_ζ_simple(u, a, p) # used only for performance comparison

    if z_ζ ≈ a
        λ = p - u[ζ]
        λ_found = true
    end

    # Calculate ρ.
    if !λ_found
        @views u_cum = a - ζ*p .- cumsum(u[ζ+1:n])
        if size(u, 1) == 1 && size(u, 2) == 1
            ρ = 1
        else
            z = u[ζ+1:n] + u_cum[1:n-ζ]./(1:n-ζ)
            @views ρ = ζ + findprev(x->(x>0), z, n - ζ)
        end
        λ = u_cum[ρ - ζ]/(ρ - ζ);
    end

    # Calculate x and return a column vector.
    x = max.(0, min.(y .+ λ, p));

    return x, ζ
end

"""
    find_ζ(q::Function, a::Float64, m::Int64, n::Int64, qₘ::AbstractFloat, qₙ::AbstractFloat)

Algorithm 5.2

Calculates the greatest index `ζ` in {`m`,...,`n`} such that `q`(`ζ`) ≦ `a`.
In order to save computation time, the function values `qₘ` and `qₙ` of the
interval bounds `m` and `n` can be passed. However, they are not necessary.
If even `q`(`m`) is > 1, `ζ` will be `m` - 1.

# Arguments
- `q`: function
- `a`: upper bound
- `m, n`: lower and upper bound of the index ζ
- `qₘ, qₙ`: function value of q at m and n
"""
function find_ζ(q::Function, a::Float64, m::Int64, n::Int64, qₘ::AbstractFloat,
                qₙ::AbstractFloat)

    ## Verify input and handle trivial cases.
    if m > n
        error("m cannot be greater than n.")
    end
    if isnan(qₘ)
        qₘ = q(m)
    end
    if qₘ > a
        ζ = m - 1
        q_ζ = NaN
        return ζ, q_ζ
    end
    if isnan(qₙ)
        qₙ = q(n)
    end
    if qₙ <= a
        ζ = n
        q_ζ = qₙ
        return ζ, q_ζ
    end
    if m + 1 == n
        ζ = m
        q_ζ = qₘ
        return ζ, q_ζ
    end

    ## Cut the domain of 'q' into two halves and rerun the algorithm.
    j = convert(Int64, m + ceil((n - m)/2))
    k, qₖ = find_ζ(q, a, m, j, qₘ, NaN)
    if k == j
        l, qₗ = find_ζ(q, a, j + 1, n, NaN, qₙ)
        if l == j
            ζ = k
            q_ζ = qₖ
            return ζ, q_ζ
        else
            ζ = l
            q_ζ = qₗ
            return ζ, q_ζ
        end
    else
        ζ = k
        q_ζ = qₖ
        return ζ, q_ζ
    end
end

"""
    find_ζ(u::Array{Float64}, a::Float64, p::Float64, ζ_guess::Number=NaN)

Algorithm 5.3

Calculates the greatest index `ζ` such that `q`(`ζ`) ≦ `a`, where
`q`(j) = sum(max(0, u(j + 1:`S`) + `p` - `u`(j))) + j*`p`. The vector `u` has
`S` entries and needs to be in a descending order. `ζ_guess` can be provided as
the starting point of the search in order to speed up the algorithm.

# Arguments
- `u`: vector in descending order
- `a`: upper bound
- `p`: positive real number
- `ζ_guess`: index used as starting point
"""
function find_ζ(u::Array{Float64}, a::Float64, p::Float64, ζ_guess::Number=NaN)

    ## Verify input.
    u = vec(u)

    ## Initialize variables.
    S = length(u)
    ζ, q_ζ, q_ζ_new = NaN, NaN, NaN
    q(j) = sum(max.(u[j+1:S] .+ p .- u[j], 0)) + j*p
    if !isnan(ζ_guess) && ζ_guess > 0
        ζ = ζ_guess
        q_ζ = q(ζ)
        q_ζ_new = copy(q_ζ)
    end
    flag = false

    ## Find ζ.
    if isnan(ζ)
        ζ, q_ζ = find_ζ(q, a, 1, S, NaN, NaN)
    else
        while true
            if q_ζ_new <= a
                # The final ζ must be >= the current ζ.
                q_ζ = q_ζ_new
                ζ += 1
                q_ζ_new = q(ζ)
                flag = true
            else
                ζ = ζ - 1
                if flag
                    break
                else
                    if ζ > 0
                        q_ζ_new = q(ζ)
                    else
                        # We always have q(0) = 0.
                        q_ζ = 0
                        break
                    end
                end
            end
        end
    end
    return ζ, q_ζ
end

"""
    find_ζ_simple(u::Array{Float64}, a::Float64, p::Float64)

Calculates the greatest index `ζ` such that `q`(`ζ`) ≦ `a` using `findfirst`, where
`q`(j) = sum(max(0, u(j + 1:`S`) + `p` - `u`(j))) + j*`p`. The vector `u` has
`S` entries and needs to be in a descending order.

# Arguments
- `u`: vector in descending order
- `a`: upper bound
- `p`: positive real number
"""
function find_ζ_simple(u::Array{Float64}, a::Float64, p::Float64)

    n = length(u)
    q(j) = sum(max.(u[j+1:n] .+ p .- u[j], 0)) + j*p
    t = findfirst(j -> q(j) > a, 1:n)
    ζ = isnothing(t) ? n : t - 1
    qζ = ζ > 0 ? q(ζ) : 0.0 # We always have q(0) = 0.

    return ζ, qζ
end
