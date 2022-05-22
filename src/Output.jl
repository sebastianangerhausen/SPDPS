"""
    print_to_console!(it::Array{Int64}, k::Int64, it_diff_u::Array{Float64},
                      it_diff_v::Array{Float64}, diff_u::Float64, diff_v::Float64,
                      it_Fval::Array{Float64}, Fval::Float64, it_pde_counter::Array{Int64},
                      pde_counter::Int64)

Prints output to the console and saves additional information.

# Arguments
- `it`: iterations in which the stopping criterion was checked
- `k`: current iteration
- `it_diff_u, it_diff_v`: storage for `diff_u` and `diff_v`
- `diff_u, diff_v`: norm of the difference of successive iterates
- `it_Fval`: storage for `Fval`
- `Fval`: value of objective function
- `it_pde_counter`: number of solved PDEs for each element of `it`
- `pde_counter`: current total number of solved PDEs
"""
function print_to_console!(it::Array{Int64}, k::Int64, it_diff_u::Array{Float64},
                           it_diff_v::Array{Float64}, diff_u::Float64, diff_v::Float64,
                           it_Fval::Array{Float64}, Fval::Float64,
                           it_pde_counter::Array{Int64}, pde_counter::Int64)

    # Print output.
    if k == 1
        @printf("Iteration k \t | \t diff_u \t | \t diff_v \n")
        @printf("------------------------------")
        @printf("-----------------------------------\n")
    end
    @printf("%-7i \t | \t %.4e \t | \t %.4e\n", k, diff_u, diff_v)

    # Save additional information.
    push!(it, k)
    push!(it_diff_u, diff_u)
    push!(it_diff_v, diff_v)
    push!(it_pde_counter, pde_counter)
    push!(it_Fval, Fval)

    return
end

"""
    create_output(x::Array{Float64}, u::Array{Float64}, y::Array{Float64},
                  it_diff_u::Array{Float64}, it_diff_v::Array{Float64},
                  it_pde_counter::Array{Int64}, it_Fval::Array{Float64}, stg::Settings)

Creates plots and csv-files of the solution.

# Arguments
- `x`: domain of `u` and `y`
- `u`, `y`: primal and dual solution vector
- `it_diff_u`, `it_diff_v`: ‖uₖ-uₖ₋₁‖² and ‖vₖ-vₖ₋₁‖² for some iterations
- `it_pde_counter`: number of solved PDEs for some iterations
- `it_Fval`: objective function value for some iterations
- `stg`: settings structure
"""
function create_output(x::Array{Float64}, u::Array{Float64}, y::Array{Float64},
                       it_diff_u::Array{Float64}, it_diff_v::Array{Float64},
                       it_pde_counter::Array{Int64}, it_Fval::Array{Float64}, stg::Settings)

    ##  Display plots.
    if stg.plot
        # Plot u and y for a few scenarios.
        PyPlot.figure(1)
        PyPlot.subplots_adjust(hspace = 0.5)
        PyPlot.subplot(211)
        PyPlot.plot(x[:], u[:], label = "u")
        PyPlot.grid(true)
        PyPlot.title("Optimal Control")
        PyPlot.xlabel("x")
        PyPlot.ylabel("u(x)")
        PyPlot.subplot(212)
        for j = 1:min(10, stg.S)
            PyPlot.plot(x, y[:,j], label = "y(xi_$j, ,u)")
        end
        PyPlot.grid(true)
        PyPlot.title("PDE-Solutions for some Scenarios")
        PyPlot.xlabel("x")
        PyPlot.ylabel("y(ξᵢ,x;u)")

        # Plot norm of the difference of successive iterates per number of solved PDEs.
        PyPlot.figure(2)
        PyPlot.plot(it_pde_counter, it_diff_u, label = "it_diff_u", marker = "x")
        PyPlot.plot(it_pde_counter, it_diff_v, label = "it_diff_v", marker = ".")
        PyPlot.yscale("log")
        PyPlot.xscale("log")
        PyPlot.grid(true)
        PyPlot.title("Difference of Successive Iterates")
        PyPlot.xlabel("Number of solved PDEs")
        PyPlot.legend()

        # Plot objective function value per of number of solved PDEs
        PyPlot.figure(3)
        PyPlot.plot(it_pde_counter, it_Fval)
        PyPlot.xscale("log")
        PyPlot.grid(true)
        PyPlot.title("Objective Function Value")
        PyPlot.xlabel("Number of solved PDEs")
    end

    ## Create csv-files.
    if stg.csv

        # Compute statistics and write optimal state.
        m = Statistics.mean(y, dims = 2)
        std = Statistics.std(y, mean = m, dims = 2)
        y_out = hcat(x, m, m + std, m - std, m + 2*std, m - 2*std)
        CSV.write(string(stg.path, "y.csv"),  DataFrame(y_out, :auto), writeheader=false)

        # Optimal control
        u_out = hcat(x,u)
        CSV.write(string(stg.path, "u.csv"),  DataFrame(u_out, :auto), writeheader=false)

        # Norm of the difference of successive iterates per number of solved PDEs
        pde_diff_u = hcat(it_pde_counter, it_diff_u)
        pde_diff_v = hcat(it_pde_counter, it_diff_v)
        CSV.write(string(stg.path, "pde_diff_u.csv"),  DataFrame(pde_diff_u, :auto),
                  writeheader=false)
        CSV.write(string(stg.path, "pde_diff_v.csv"),  DataFrame(pde_diff_v, :auto),
                  writeheader=false)

        # Objective function value per of number of solved PDEs
        pde_Fval = hcat(it_pde_counter, it_Fval)
        CSV.write(string(stg.path, "pde_Fval.csv"),  DataFrame(pde_Fval, :auto),
                  writeheader=false)

        println("CSV-files written to \"$(stg.path)\".")
    end

    return
end
