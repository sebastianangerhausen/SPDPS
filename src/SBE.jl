"""
Steady Burgers' Equation (Section 6.4)
"""

"Structure holding scenario specific elements"
struct SBE_Scenario
    ν::Float64         # Viscosity parameter of PDE
    f::Array{Float64}  # Function values of f at the grid points
    d₀::Float64        # Dirichlet boundary condition for x=0
    d₁::Float64        # Dirichlet boundary condition for x=1
    ỹ::Array{Float64}  # Straigth line connecting boundary values d₀ and d₁
    ϕpu::Function      # Function of y₀ returns the left-hand side of the PDE in "=u"-form
    Jϕ::Function       # Function of y₀, returns the Jacobian of φ
end

"""
    solve_SBE(stg::Settings, tol_newton::Float64, maxit_newton::Int64)

Solves the problem described in Section 6.4.

# Arguments
- `stg`: settings structure
- `tol_newton`: tolerance for Newton's method
- `maxit_newton`: maximum number of iterations for Newton's method
"""
function solve_SBE(stg::Settings, tol_newton::Float64, maxit_newton::Int64)

    ## Set parameters.
    α = 1e-3
    u_a, u_b = -10, 10 # Bounds in U_{ad}

    ## Initialize variables.
    x, Δx, D₁, D₂ = setup_fdm(stg.N)
    scenarios = create_scenarios(stg.N, stg.S, D₁, D₂, x)
    y₀_prev = zeros(stg.N, stg.S) # Storage for the homogeneous part of the PDE
                                  # solution in the previous iteration
    p = 1/(stg.S*(1 - stg.β)) # Upper bound for the probability simplex
    u,v = zeros(stg.N), zeros(stg.S)

    ## Define mappings.
    function prox_TG(u::Array{Float64}, T::Array{Float64})
        return [max(min(u[i]/(1 + α*T[i]), u_b), u_a) for i = 1:stg.N]
    end
    function prox_CVaR⃰(v::Array{Float64}, ζ_proj=NaN)
        if stg.S == 1 # Deterministic control
            return [1], ζ_proj
        elseif stg.risk_neutral # Risk-neutral control
            return 1/stg.S*ones(stg.S), ζ_proj
        else
            return project(v, 1.0, p, stg.S, ζ_proj)
        end
    end

    function y(u::Array{Float64}, A = 1:stg.S) # y(u)[i,j] = y(ξʲ,xᵢ;u)
        yu =  Array{Float64}(undef, stg.N, length(A))
        for (j, k) in enumerate(A)
            ϕ(y₀) = scenarios[k].ϕpu(y₀) - u
            y₀_temp = newtons_method(ϕ, scenarios[k].Jϕ, y₀_prev[:, k], tol_newton,
                                     maxit_newton)
            y₀_prev[:, k] = y₀_temp
            yu[:, j] = y₀_temp + scenarios[k].ỹ
        end
        return yu
    end

    function K(yu::Array{Float64}, A = 1:stg.S) # K(yu)[j] = J(y(ξʲ,⋅;u)) = K(u)ⱼ
        sol = Array{Float64}(undef, length(A))
        for (j, k) in enumerate(A)
            sol[j] = Δx/4*((scenarios[k].d₀ - 1)^2 + (scenarios[k].d₁ - 1)^2 +
                           2*sum((yu[:,k] .- 1).^2))
        end
        return sol
    end

    function K′⃰(yu::Array{Float64}, A = 1:stg.S) # K′⃰(yu) = K′(u)⃰
        sol = Array{Float64}(undef, stg.N, length(A))
        for (j, k) in enumerate(A)
           sol[:,j] = reshape(scenarios[k].Jϕ(yu[:,k] - scenarios[k].ỹ)'\
                               (yu[:,k] .- 1), stg.N, 1)
        end
        return sol
    end

    ## Define objective function.
    function CVaR(z)
        z_sorted = sort(z)
        m = Int(ceil(stg.β*stg.S))
        VaR = z_sorted[m]
        return (m - stg.β*stg.S)/((1 - stg.β)*stg.S)*VaR +
                sum(z_sorted[m+1:stg.S])/((1 - stg.β)*stg.S)
    end
    G(u) = α/4*2*Δx*sum(u.^2)
    F(u) = CVaR(K(y(u))) + G(u)

    ## Solve the problem.
    u, v, it_diff_u, it_diff_v, it, it_Fval, it_pde_counter, avg_num =
        solve(prox_TG, prox_CVaR⃰, K, K′⃰, y, u, v, F, stg)

    ## Add boundary values.
    x = vcat([0], x, [1])
    y = vcat([scenarios[j].d₀ for j in 1:stg.S]', y(u), [scenarios[j].d₁ for j in 1:stg.S]')
    u = vcat([0], u, [0])

    ## Create output (plots and/or csv-files).
    create_output(x, u, y, it_diff_u, it_diff_v, it_pde_counter, it_Fval, stg)

    return [it[end], avg_num, it_pde_counter[end]]
end

"""
    create_scenarios(N::Int64, S::Int64, D₁::SparseMatrixCSC{Float64},
                     D₂::SparseMatrixCSC{Float64}, x::Array{Float64})

Returns an array of scenarios with their respective random variable values.

# Arguments
- `N`: number of grid points
- `S`: number of scenarios
- `D₁`: central difference discretization of the first order derivative
- `D₂`: central difference discretization of the second order derivative
- `x`: grid for the discretization
"""
function create_scenarios(N::Int64, S::Int64, D₁::SparseMatrixCSC{Float64},
                          D₂::SparseMatrixCSC{Float64}, x::Array{Float64})

    # Initialize random number generator.
    Random.seed!(123)
    # Generate random numbers in [-1, 1]^4.
    ξ = reshape(rand(Distributions.Uniform(-1, 1), 4*S), S, 4)
    scenarios = Array{SBE_Scenario}(undef, S)
    for j = 1:S
        if S == 1 # Used only for deterministic control.
            # Set the parameters to its respective expected values.
            ν = 1e-2
            f = fill(0, N)
            d₀ = 1.0
            d₁ = 0.0
        else
            ν = 10^(ξ[j, 1] - 2)
            f = fill(ξ[j, 2]/100, N)
            d₀ = 1 + ξ[j, 3]/1000
            d₁ = ξ[j, 4]/1000
        end
        l(i) = d₀ + (d₁ - d₀)*i
        ỹ = [l(i) for i in x]
        # Define functions for Newton's method (φpu = φ + u).
        ϕpu(y₀) = Tridiagonal(-ν*D₂ + spdiagm(0 => y₀ + ỹ)*D₁)*y₀ + (y₀ + ỹ)*(d₁ - d₀) - f
        Jϕ(y₀) = Tridiagonal(-ν*D₂ + spdiagm(0 => D₁*y₀) + spdiagm(0 => y₀ + ỹ)*D₁ +
                             (d₁ - d₀)*I)
        scenarios[j] = SBE_Scenario(ν, f, d₀, d₁, ỹ, ϕpu, Jϕ)
    end

    return scenarios
end

"""
    setup_fdm(N::Int64)

Returns a grid of N points between 0 and 1 with equal distance (without 0 and 1). The
matrices D₁ and D₂ are used in the finite difference method to compute the first (D₁) and
second order (D₂) central difference approximation.

# Arguments
- `N`: number of inner grid points
"""
function setup_fdm(N::Int64)

    ## Uniform grid.
    Δx = 1/(N + 1)
    x = collect(range(0 + Δx, stop = 1 - Δx, length = N))
    e = fill(1/(2*Δx), N)
    D₁ = spdiagm(-1 => -e[2:end],1 => e[2:end])
    f = fill(1/Δx^2, N)
    D₂ = spdiagm(-1 => f[2:end], 0 => -2*f, 1 => f[2:end])

    return x, Δx, D₁, D₂
end

"""
    newtons_method(F::Function, F′::Function, y::Array{Float64}, tol::Float64, maxit::Int64)

Attempts to solve the equation `F(x)` = 0 by using Newton's Method.

# Arguments
- `F`: the function of which a root is to be found
- `F′`: first derivative of F
- `y`: starting vector
- `tol`: tolerance which is used for the stopping criterion
- `maxit`: maximum number of iterations
"""
function newtons_method(F::Function, F′::Function, y::Array{Float64}, tol::Float64,
                        maxit::Int64)

    δy = 1
    k = 1

    while true
        δy = -F′(y)\F(y)
        y = y + δy

        # Check stopping criterion.
        if norm(δy) < tol
            break
        elseif k == maxit
            println("\nNewton's method reached the maximum number of iterations!")
            break
        else
            k += 1
        end
    end

    if any(isnan, y)
        error("Newton's method did not converge!")
    end

    return y
end
