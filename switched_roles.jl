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
using SpecialFunctions
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
vH1BiasSq = zeros(N_run, length(N))
vH2BiasSq = similar(vH1BiasSq)
vH1var = Float64[]
vH2var = Float64[]
vH1InEff = Float64[]
vH2InEff = Float64[]
vInEff_approx = Float64[]

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

   # 2.2. Bias
   E1 = 0.0
   E2 = ν/(ν-2)

   h1_MCbias = (vH1bar .- E1).^2
   h2_MCbias = (vH2bar .- E2).^2

   # 2.3. Variance
   h1_MCvar = var(vH1bar)
   h2_MCvar = var(vH2bar)

   # 2.4. Ineficiency factor
   # Small sample
   h1_InEff = h1_MCvar/( ν/(ν-2) / N[iDraws])
   #h2_InEff = h2_MCvar/(  / N[iDraws])

   # Approximated
   Z = (gamma((ν+1)/2)/sqrt(ν*π)*gamma(ν/2))
   InEff_approx = 1 + (Z^(-2)*var(mWeights[iDraws,:]))


   # 3. Saving results
   vH1BiasSq[:,iDraws] = h1_MCbias
   vH2BiasSq[:, iDraws] = h2_MCbias

   global vH1var = push!(vH1var, h1_MCvar)
   global vH2var = push!(vH2var, h2_MCvar)

   global vH1InEff = push!(vH1InEff, h1_InEff)
   # global vH2InEff = push!(vH2InEff, h2_InEff)
   global vInEff_approx = push!(vInEff_approx, InEff_approx)

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

# Mean Sq. Bias
vH1MeanSqBias = mean(vH1BiasSq, dims=1)
vH2MeanSqBias = mean(vH2BiasSq, dims=1)

# ------
# Graphs
# -------
pInefficiencyFactorsSwitched = plot(N, vInEff_approx, color=:black, linestyle=:dash, label="")
plot!(N, vH1InEff, linecolor=:black, linestyle=:solid, marker=:utriangle, markercolor=:white, label="")

savefig(pInefficiencyFactorsSwitched, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV/PS/PS1/LaTeX/pInefficiencyFactorsSwitched.pdf"")
