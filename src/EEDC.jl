"""
Elliptic Equation with Discontinuous Coefficient (Section 6.3)
"""

"Finite element structure holding grid and assembled matrices"
struct FEM
    x::Vector{Float64}                      # x coordinates of inner nodes
    Δx::Vector{Float64}                     # Distances of nodes
    D::SuiteSparse.CHOLMOD.Factor{Float64}  # Cholesky decomposition of stiffness matrix
    M::SparseMatrixCSC{Float64}             # Mass matrix
end

"Structure holding scenario specific elements"
struct EEDC_Scenario
    fem::FEM            # FEM structure
    Δxs::Array{Float64} # Vector with entries Δx[i] + Δx[i+1]
    f::Array{Float64}   # Function values of f at the grid points
end

"""
    solve_EEDC(stg::Settings)

Solves the problem described in Section 6.3.

# Arguments
- `stg`: settings structure
"""
function solve_EEDC(stg::Settings)

    ## Set parameters.
    α = 1e-4
    u_a, u_b = -10, 10 # Bounds in U_{ad}

    ## Initialize variables.
    scenarios = create_scenarios(stg.N, stg.S)
    p = 1/(stg.S*(1 - stg.β)) # Upper bound for the probability simplex
    u, v = zeros(stg.N), zeros(stg.S)

    ## Define mappings.
    function prox_TG(u::Array{Float64}, T::Array{Float64})
        return [max(min(u[i]/(1 + α*T[i]), u_b), u_a) for i = 1:stg.N]
    end
    if stg.risk_neutral
        prox_CVaR⃰(v, ζ_proj) = (1/stg.S*ones(stg.S), ζ_proj)
    else
        function prox_CVaR⃰(v::Array{Float64}, ζ_proj=NaN)
            if stg.S == 1 # Used only for deterministic control.
                return [1], ζ_proj
            else
                fval, ζval = project(v, 1.0, p, stg.S, ζ_proj)
                return fval, ζval
            end
        end
    end

    function y(u::Array{Float64}, A=1:stg.S) # y(u)[i,j] = y(ξʲ,xᵢ;u)
        yu =  Array{Float64}(undef, stg.N, length(A))
        for (j, k) in enumerate(A)
            yu[:, j] = reshape(scenarios[k].fem.D\(scenarios[k].fem.M*
                               (scenarios[k].f + u)), stg.N, 1)
        end
        return yu
    end

    function K(yu, A=1:stg.S) # K(yu)[j] = J(y(ξʲ,⋅;u)) = K(u)ⱼ
        sol = Array{Float64}(undef, length(A))
        for (j, k) in enumerate(A)
            sol[j] = 1/4*(scenarios[k].fem.Δx[1] + scenarios[k].fem.Δx[stg.N + 1] +
                          scenarios[k].Δxs'*(yu[:,k] .- 1).^2)
        end
        return sol
    end

    function K′⃰(yu, A=1:stg.S) # K′⃰(yu) = K′(u)⃰
        sol = Array{Float64}(undef, stg.N, length(A))
        for (j, k) in enumerate(A)
            sol[:, j] = 1/2*reshape(scenarios[k].fem.M*(scenarios[k].fem.D\(
                                    scenarios[k].Δxs.*(yu[:,k] .- 1))), stg.N, 1)
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
    G(u) = α/4*scenarios[1].Δxs'*u.^2
    F(u) = CVaR(K(y(u))) + G(u)

    ## Solve the problem.
    u, v, it_diff_u, it_diff_v, it, it_Fval, it_pde_counter, avg_num =
        solve(prox_TG, prox_CVaR⃰, K, K′⃰, y, u, v, F, stg)

    ## Add boundary values.
    x = vcat([-1], scenarios[1].fem.x, [1])
    y = vcat(zeros(1,stg.S), y(u), zeros(1,stg.S))
    u = vcat([0], u, [0])

    ## Create output (plots and/or csv-files).
    create_output(x, u, y, it_diff_u, it_diff_v, it_pde_counter, it_Fval, stg)

    return [it[end], avg_num, it_pde_counter[end]]
end

"""
    create_scenarios(N::Int64, S::Int64)

Returns an array of scenarios with their respective random variable values and
finite elements grid.

# Arguments
- `N`: number of grid points
- `S`: number of scenarios
"""
function create_scenarios(N::Int64, S::Int64)

    scenarios = Array{EEDC_Scenario}(undef, S)

    if S == 1 # Used only for deterministic control.
        ξ₁ = [0]
        ξ₂ = [0]
    else
        Random.seed!(123) # Initialize random number generator.
        ξ₁ = rand(Distributions.Uniform(-0.1, 0.1), S)
        ξ₂ = rand(Distributions.Uniform(-0.5, 0.5), S)
    end

    for j = 1:S
        ϵ(x)::Float64 = x > ξ₁[j] ? 10.0 : 0.1
        fem = setup_fem(N, ϵ)
        Δxs = fem.Δx[1:end-1] + fem.Δx[2:end]
        f = exp.(-(fem.x .- ξ₂[j]).^2) # f[i] = exp(-(xᵢ-ξʲ₂)²)
        scenarios[j] = EEDC_Scenario(fem, Δxs, f)
    end

    return scenarios
end

"""
    setup_fem(N::Int64, ϵ::Function)

Generates a uniform grid -1 = x₀ < ... < xₙ₊₁ = 1 and assembles the matrices for
the finite element method.

# Arguments
- `N`: number of inner grid points
- `ϵ`: function describing the discontinuous coefficient
"""
function setup_fem(N::Int64, ϵ::Function)

    # Set up grid.
    dx = 2/(N + 1)
    xm = range(-1 + dx, stop = 1 - dx, length = N)
    x = reshape(xm, 1, N)[:] # Inner grid points
    Δx = fill(dx, N + 1)

    # Set up mass matrix.
    d₁ = Δx[2:end-1]./6
    M = spdiagm(-1 => d₁, 0 => (Δx[1:end-1] + Δx[2:end])./3, 1 => d₁)

    # Set up stiffness matrix.
    ϵxd = ϵ.(vcat(x, [1]))./Δx
    D = spdiagm(-1 => -ϵxd[2:end-1], 0 => ϵxd[1:end-1] + ϵxd[2:end],
                 1 => -ϵxd[2:end-1])

    return FEM(x, Δx, cholesky(D), M)
end
