"""
Implementation of Algorithm 4.1
"""

"""
    solve(prox_TG::Function, prox_CVaR⃰::Function, K::Function, K′⃰::Function,
          y::Function, u::Array{Float64}, v::Array{Float64}, F::Function, stg::Settings)

Implementation of Algorithm 4.1 to solve the optimality condition
        u = prox_TG(u - T*K'⃰(y(u))*v),
        v = prox_CVaR⃰(v + Σ*K(y(u))).

# Arguments
- `prox_TG, prox_CVaR⃰, K, K′⃰, y`: functions
- `u, v`: starting vectors
- `F`: objective function
- `stg`: settings structure
"""
function solve(prox_TG::Function, prox_CVaR⃰::Function, K::Function, K′⃰::Function,
               y::Function, u::Array{Float64}, v::Array{Float64}, F::Function,
               stg::Settings)

    ## Set up algorithm.
    Random.seed!(234)
    u_new = similar(u)
    v_new = similar(v)
    v_bar = copy(v)
    last_ζ_proj = NaN
    A = collect(1:stg.S) # Index set for CGF
    T, Σ = Array{Float64}(undef, stg.N), Array{Float64}(undef, stg.S) # Step size operators
    yu = Array{Float64}(undef, stg.N, stg.S)    # Storage for the PDE solution
    Kyu = Array{Float64}(undef, stg.S)          # Storage for K(u)
    K′⃰yu = Array{Float64}(undef, stg.N, stg.S) # Storage for K′(u)⃰
    K′⃰yu_old = []

    ## Initialize variables for additional information.
    num_ind = 0                             # Counter for the number of selected indices
    it = Array{Int64}(undef, 0)             # Iterations in which the stopping criterion was
                                            # checked
    pde_counter = 0                         # Current total number of solved PDEs
    it_pde_counter = Array{Int64}(undef, 0) # Number of solved PDEs for each element of `it`
    diff_u = 0.0                            # ‖uₖ-uₖ₋₁‖²
    it_diff_u = Array{Float64}(undef, 0)    # Storage for ‖uₖ-uₖ₋₁‖² for each element of `it`
    diff_v = 0.0                            # ‖vₖ-vₖ₋₁‖²
    it_diff_v = Array{Float64}(undef, 0)    # Storage for ‖vₖ-vₖ₋₁‖² for each element of `it`
    it_Fval = Array{Float64}(undef, 0)      # Storage for objective function values for each
                                            # element of `it`

    ## Set scalar (accelerated) step sizes.
    K′⃰yu = K′⃰(y(u))
    op_norm² = estimate_largest_eigenvalue(K′⃰yu' * K′⃰yu)
    Σ[:] = fill(stg.σ, stg.S)
    T[:] = fill(0.99*(Σ[1]*op_norm²)^(-1), stg.N)
    @printf("Initial scalar step sizes: τ₀ = %.2f, σ = %.2f\n\n", T[1], Σ[1])
    if stg.step_size == "acc" @printf("Acceleration parameter: γ = %.2e\n\n", stg.γ) end

    k = 1
    while true
        # Select indices.
        if k > 1 && stg.CGF_rule ≠ 0
            A = select_indices(k, stg)
        end
        num_ind += length(A)

        # Compute the PDE solution for control u only for scenarios in A.
        yu[:, A] = y(u, A)
        pde_counter += length(A)

        # Compute K(u) only for scenarios in A.
        Kyu[A] = K(yu, A)

        # Dual step
        v_new, last_ζ_proj = prox_CVaR⃰(v + Σ.*Kyu, last_ζ_proj)
        diff_v = norm(v_new - v)

        # Extrapolation step
        v_bar = 2*v_new - v
        v = copy(v_new)

        if !stg.use_Bk
            # Compute K′(u)⃰ only for scenarios in A.
            K′⃰yu_old = K′⃰yu[:, A]
            K′⃰yu[:, A] = K′⃰(yu, A)
            pde_counter += length(A)
        else
            # Alternative with Bₖ
            B = setdiff(A, findall(v_bar .== 0))
            K′⃰yu_old = K′⃰yu[:, B]
            K′⃰yu[:, B] = K′⃰(yu, B)
            pde_counter += length(B)
        end

        # Primal step
        u_new = prox_TG(u - T.*K′⃰yu*v_bar, T)
        diff_u = norm(u_new - u)
        u = copy(u_new)

        # Check for failure of the algorithm.
        if any(isnan, [u; v])
            error("The algorithm failed!")
        end

        # Check stopping criterion.
        if diff_u < stg.tol && diff_v < stg.tol
            print_to_console!(it, k, it_diff_u, it_diff_v, diff_u, diff_v, it_Fval, F(u),
                              it_pde_counter, pde_counter)
            println("\nThe stopping criterion was satisfied in iteration $k.")
            if stg.step_size == "acc" println("Final primal step size: τ = $(T[1])") end

            # Compute the norm of the residuals (i.e. the optimality condition with all
            # indices selected and v instead of v_bar).
            yu_oc = y(u)
            Kyu_oc = K(yu_oc)
            v_oc = prox_CVaR⃰(v + Σ.*Kyu_oc, last_ζ_proj)[1]
            K′⃰yu_oc = K′⃰(yu_oc)
            u_oc = prox_TG(u - T.*K′⃰yu_oc*v, T)
            res_u = norm(u_oc - u) # Norm of the residual of u
            res_v = norm(v_oc - v) # Norm of the residual of v
            println("Norm of the residual of u: $res_u")
            println("Norm of the residual of u: $res_v")
            break
        end

        # Print output and save additional information.
        if mod(k, stg.it_out) == 0 || k == 1
            print_to_console!(it, k, it_diff_u, it_diff_v, diff_u, diff_v, it_Fval, F(u),
                              it_pde_counter, pde_counter)
        end

        # Update step size operators.
        set_step_size!(T, Σ, K′⃰yu, k, stg)

        # Check for maximal number of iterations.
        if k == stg.maxit
            println("\nMaximum number of iterations reached!")
            break
        else
            k += 1
        end
    end

    avg_num = num_ind/k
    if stg.CGF_rule ≠ 0
        println("Average number of selected indices: $avg_num")
    end
    println("Number of solved PDEs: $pde_counter")

    return u, v, it_diff_u, it_diff_v, it, it_Fval, it_pde_counter, avg_num
end

"""
    select_indices(k::Int64, stg::Settings)

Randomly select a subset of the index set {1,…,S}.

# Arguments
- `k`: current iteration
- `stg`: settings structure
"""
function select_indices(k::Int64, stg::Settings)

    if stg.CGF_rule == 1
        # Index Selection Rule 1
        M = 1e20
        p = k < M^(1/3) ? stg.q : max(stg.q, (1 - M*k^(-3))^(1/stg.S))
        bernoulli_dist = Distributions.Bernoulli(p)
        A = findall(x -> x == 1, [rand(bernoulli_dist) for j=1:stg.S])

    elseif stg.CGF_rule == 2
        # Index Selection Rule 2
        w = StatsBase.Weights(fill(1/stg.S, stg.S))
        a = log(k)/stg.q
        A = StatsBase.sample(1:stg.S, w, min(stg.S, max(1, round(Int, a*stg.S))),
                             replace=false)
    end

    return A
end

"""
    set_step_size!(T::Array{Float64}, Σ::Array{Float64}, M::Array{Float64}, k::Int64,
                   stg::Settings)

Computes the step size vectors `T` and `Σ` in dependence of the matrix `M`.

# Arguments
- `T, Σ`: step sizes to be modified
- `M`: matrix K′(u)⃰
- `k`: current iteration
- `stg`: settings structure
"""
function set_step_size!(T::Array{Float64}, Σ::Array{Float64}, M::Array{Float64}, k::Int64,
                        stg::Settings)

    if stg.step_size == "ssa"
        # Step size operators analogously to [PC11]
        T_new, Σ_new = similar(T), similar(Σ)
        α = 1
        for i = 1:stg.N
            for j = 1:stg.S
                a = abs(M[i, j])
                T_new[i] = j == 1 ? a^(2 - α) : T_new[i] + a^(2 - α)
                Σ_new[j] = i == 1 ? a^α : Σ_new[j] + a^α
            end
        end
        T[:] = 1 ./ T_new
        Σ[:] = 1 ./ Σ_new

    elseif stg.step_size == "acc"
        # Scalar step sizes with acceleration
        if k < stg.it_acc
            T[:] = fill(T[1]/(1 + 2*stg.γ*T[1]), stg.N)
            Σ[:] = fill(Σ[1]*(1 + 2*stg.γ*T[1]), stg.S)
            if k + 1 == stg.it_acc
                @printf("Acceleration stopped. New primal step size: τ = %.2f\n", T[1])
            end
        end
    end

    return
end

"""
    estimate_largest_eigenvalue(A::Array{Float64})

Algorithm 6.1

Performs the power method to find the largest eigenvalue of the matrix `A`.

# Arguments
- `A`: symmetric matrix
"""
function estimate_largest_eigenvalue(A::Array{Float64})

    maxit = 5
    d = size(A, 1)
    x = ones(d)./sqrt(d) # Starting vector with norm 1
    λ = 0

    for i = 1:maxit
        Ax = A*x
        n = norm(Ax)
        if n == 0
            error("Unsuitable starting vector for operator norm estimation.")
        end
        x = Ax./n # Converges to the eigenvector belonging to the largest eigenvalue.
        if i == maxit
            λ = x'*A*x # Rayleigh quotient
        else
        end
    end

    return λ
end
