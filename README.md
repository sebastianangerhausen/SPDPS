# SPDPS

`SPDPS` is a [Julia](https://julialang.org/) project providing the code for the PhD thesis "A Stochastic Primal-Dual Proximal Splitting Method for Risk-Averse Optimal Control of PDEs" by Sebastian Angerhausen, submitted in 2022 to the Faculty of Mathematics, University of Duisburg-Essen.

## Usage

In order to use the project, you can follow the steps explained [here](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project). You basically need to

- download the code, start the Julia REPL and `cd` to the project directory,
- enter the package manager by pressing `]`,
- activate the project by executing the command `activate .`,
- for the first time use: instantiate the project by executing the command `instantiate`,
- exit the package manager by pressing backspace,
- load the project by executing `using SPDPS`.

Note that, in order to use the plotting features, you need to have Python installed together with the library `matplotlib`. For further help, see the installation instructions of [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl#installation).

You can then solve one of the two exemplary problems presented in the aforementioned PhD thesis by executing

    run_test("EEDC")

for the elliptic equation with a discontinuous coefficient (see Section 6.3), and

    run_test("SBE")

for the steady Burgers’ equation (see Section 6.4). The method `run_test` moreover takes the following keyword arguments:

- `N`: number of grid points
- `S`: number of scenarios
- `β`: probability level of CVaR
- `risk_neutral`: determines whether to compute a risk-neutral control or not
- `tol`: tolerance for the stopping criterion
- `step_size`: step size rule (either "constant" or "acc" for acceleration)
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

The parameters `N`, `S`, `β`, `tol`, and `q` can be provided as arrays in order to compute solutions for all parameter combinations in a row.

If `csv` is `true` (which is the default value), then the results are automatically saved to csv-files that are located within a folder (with the name specified in `folder`) of the current working directory.
