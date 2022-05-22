"""
Stochastic Primal-Dual Proximal Splitting Method for Risk-Averse
Optimal Control of PDEs

Wrapper module for using the developed algorithm to solve either the elliptic equation with
discontinuous coefficient (EEDC) or the steady Burgers' equation (SBE).
"""
module SPDPS

using DataFrames, LinearAlgebra, SparseArrays, SuiteSparse, Random, Printf
import Statistics, CSV, Distributions, PyPlot, StatsBase

"Structure holding the settings for the algorithm"
struct Settings
    prob::String        # Name of the problem ("EEDC" or "SBE")
    N::Int64            # Number of grid points
    S::Int64            # Number of scenarios
    β::Float64          # Probability level of CVaR
    risk_neutral::Bool  # Determines whether to compute a risk-neutral control or not
    tol::Float64        # Tolerance for the stopping criterion
    step_size::String   # Step size rule
    σ::Float64          # Initial dual step size
    γ::Float64          # Acceleration parameter
    it_acc::Int64       # Number of iterations with acceleration
    CGF_rule::Int64     # Determines the index selection rule (1 or 2), 0 for no CGF
    q::Float64          # Parameter for the index selection
    use_Bk::Bool        # Determines whether to use the index set Bₖ or not
    maxit::Int64        # Maximum number of iterations
    it_out::Int64       # Defines after how many iterations the output is printed
    plot::Bool          # Determines whether to display plots or not
    csv::Bool           # Determines whether to create csv-files or not
    path::String        # Path for output files
end

include("./EEDC.jl")
include("./SBE.jl")
include("./Algorithm.jl")
include("./SimplexProj.jl")
include("./Output.jl")

"""
    run_test(prob::String; N::Union{Int64,Array{Int64}}=256,
             S::Union{Int64,Array{Int64}}=100, β::Union{Float64,Array{Float64}}=0.9,                    risk_neutral::Bool=false, tol::Union{Float64,Array{Float64}}=1e-6,
             step_size::String="constant", σ::Float64=0.1, γ::Float64=0.0,
             it_acc::Int64=Int(1e4), CGF_rule::Int64=0,
             q::Union{Float64,Array{Float64}}=1.0, use_Bk::Bool=false,
             maxit::Int64=Int(1e9), it_out::Int64=Int(1e2), plot::Bool=false,
             csv::Bool=true, folder::String="results", tol_newton::Float64=1e-8,
             maxit_newton::Int64=Int(1e4))

Solves the problem specified by `prob` for a specific parameter setting or for a
combination of parameters (if `N`, `S`, `β`, `tol`, or `q` are provided as arrays).

# Arguments
- `prob`: name of the problem, either "EEDC" or "SBE"
- `N`: number of grid points
- `S`: number of scenarios
- `risk_neutral`: Determines whether to compute a risk-neutral control or not
- `β`: probability level of CVaR
- `tol`: tolerance for the stopping criterion
- `step_size`: step size rule (either "constant", "acc" for acceleration, or "ssa"
               for step size adaptation)
- `σ`: initial dual step size
- `γ`: acceleration parameter
- `it_acc`: number of iterations with acceleration
- `CGF_rule`: determines the index selection rule (1 or 2), 0 for no CGF
- `q`: parameter for the index selection
- `use_Bk`: determines whether to use the index set Bₖ or not
- `maxit`: maximum number of iterations
- `it_out`: defines after how many iterations the output is printed
- `plot`: determines whether to display plots or not
- `csv`: determines whether to create csv-files or not
- `folder`: name of the folder for output files
- `tol_newton`: tolerance for Newton's method (only for SBE)
- `maxit_newton`: maximum number of iterations for Newton's method (only for SBE)
"""
function run_test(prob::String; N::Union{Int64,Array{Int64}}=256,
                  S::Union{Int64,Array{Int64}}=100, β::Union{Float64,Array{Float64}}=0.9,
                  risk_neutral::Bool=false, tol::Union{Float64,Array{Float64}}=1e-6,
                  step_size::String="constant", σ::Float64=0.1, γ::Float64=0.0,
                  it_acc::Int64=Int(1e4), CGF_rule::Int64=0,
                  q::Union{Float64,Array{Float64}}=1.0, use_Bk::Bool=false,
                  maxit::Int64=Int(1e9), it_out::Int64=Int(1e2), plot::Bool=false,
                  csv::Bool=true, folder::String="results", tol_newton::Float64=1e-8,
                  maxit_newton::Int64=Int(1e4))

    # Define multiple parameters for loop.
    N_set = isa(N, Number) ? [N] : N
    S_set = isa(S, Number) ? [S] : S
    β_set = isa(β, Number) ? [β] : β
    tol_set = isa(tol, Number) ? [tol] : tol
    q_set = isa(q, Number) ? [q] : q

    # Empty array for output vectors.
    output = Array{Any}(undef, 0, 9)

    # Loop over vector parameters.
    for N in N_set
        for S in S_set
            for β in β_set
                for tol in tol_set
                    for q in q_set
                        # Define the path for output files.
                        q_str = CGF_rule == 0 ? "" : "q=$q\\"
                        path = string(pwd(), "\\$folder\\N=$N\\S=$S\\beta=$β\\tol=$tol\\",
                                      q_str)
                        mkpath(path)

                        # Create settings structure.
                        stg = Settings(prob, N, S, β, tol, step_size, σ, γ, it_acc,
                                       CGF_rule, q, use_Bk, maxit, it_out, plot, csv, path)

                        # Print to console.
                        cgf_str = CGF_rule == 0 ? ", without CGF\n" : ", q=$q\n"
                        println(string("\nN = $N, S = $S, β = $β, ε=$tol", cgf_str))

                        # Call the solver of the desired problem.
                        if prob == "EEDC"
                            out, t, bytes, ~, ~ = @timed solve_EEDC(stg)
                        elseif prob == "SBE"
                            out, t, bytes, ~, ~ = @timed solve_SBE(stg, tol_newton,
                                                                   maxit_newton)
                        end

                        println("\nElapsed time: $t s")
                        println("Allocated memory: $(bytes/1024^2) MiB\n")
                        out = vcat([N, S, β, q, tol, t], out)
                        output = vcat(output, out')
                    end
                end
            end
        end
    end

    # Write csv-file with the output of the executed solver calls.
    if csv
        path = string(pwd(), "\\$folder\\results.csv")
        CSV.write(path,  DataFrame(output, ["N", "S", "beta", "q", "tol", "time",
                                            "iterations", "average number of indices",
                                            "solved pdes"]), writeheader=true, delim=",")
        println("Results written to \"$path\".")
    end

    return
end

end # module
