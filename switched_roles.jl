# -----------------------------------------------------------------------------
# POSTERIOR SAMPLERS: IMPORTANCE SAMPLING
# Econometrics IV, Part 1
# Prof. Frank Schorfheide
#
# Exercise 2, PS1
# -----------------------------------------------------------------------------

# ---------
# Packages
# ---------
using Distributions
using Expectations
using Plots, LaTeXStrings

# --------
# Modules
# --------
include("IS.jl")
using .IS

# -------------
# Housekeeping
# -------------
N_run = 100                # number of times the algorithm is run
ν = 4                      # degrees of freedom for the student t distribution
f = TDist(ν)               # posterior dist
g = Normal(0,1)            # importance sampling density
N =[10, 100, 1000, 10000]  # number of draws

h1(x) = x
h2(x) = x^2

# ---------
# Solution
# ---------
vH1bias = Float64[]
vH2bias = Float64[]
vH1var = Float64[]
vH2var = Float64[]
vH1InEff = Float64[]
vH2InEff = Float64[]

for iDraws in 1:length(N)

   # 1. Pre-allocation
   vH1bar = zeros(N_run)
   mTheta = zeros(N[iDraws], N_run)
   mWeights = zeros(N[iDraws], N_run)

   # 2. Monte Carlo
   # 2.1. Mean approximation
   for iMC in 1:N_run

      vH1bar[iMC], mWeights[:,iMC], mTheta[:,iMC] = IS.mean_approx(h1, f, g, N[iDraws])

   end

   # Notice that θ and W are independent of the choice of h, thus...
   vH2bar = mean(mWeights .* h2.(mTheta), dims=1)

   # 2.2. Results
   # Mean
   h1_MCmean = mean(vH1bar)
   h2_MCmean = mean(vH2bar)

   # Bias
   E1 = expectation(h1, f)
   E2 = expectation(h2, f)
   h1_MCbias = (E1 - h1_MCmean)^2
   h2_MCbias = (E2 - h2_MCmean)^2

   # Variance
   h1_MCvar = var(vH1bar)
   h2_MCvar = var(vH2bar)

   # Ineficiency factor
   # h1_InEff = h1_MCvar/( /N[iDraws])
   # h2_InEff = h2_MCvar/(/N[iDraws])


   # 3. Saving results
   global vH1bias = push!(vH1bias, h1_MCbias)
   global vH2bias = push!(vH2bias, h2_MCbias)

   global vH1var = push!(vH1var, h1_MCvar)
   global vH2var = push!(vH2var, h2_MCvar)

   # global vH1InEff = push!(vH1InEff, h1_InEff)
   # global vH2InEff = push!(vH2InEff, h2_InEff)

   # 4. Graphs
   if N[iDraws] == 100

      # 1. Distribution of importance weights
      vWeights = mWeights[:,end]
      pHistogramWeightsSwitched = histogram(vWeights, normalize=:probability, xlabel = "weights", color=:white, legend=:none)

      savefig(pHistogramWeightsSwitched, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV/PS/PS1/LaTeX/pHistogramWeightsSwitched.pdf")

      # 2. Density functions
      vTheta = sort!(mTheta[:,end])

      pSamplingDensitySwitched = plot(vTheta, pdf.(f, vTheta), label = "f", linewidth = 1.5, linestyle=:solid, linecolor=:black, xlabel=L"\theta")
      plot!(vTheta, pdf.(g, vTheta), label = "g", linewidth = 1.5, linestyle=:dash, linecolor=:black)

      savefig(pSamplingDensitySwitched, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV/PS/PS1/LaTeX/pSamplingDensitySwitched.pdf")

   end

end
