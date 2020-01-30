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
using Distributions, Random
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
f = Normal(0,1)            # posterior dist
ν = 4                      # degrees of freedom for the student t distribution
g = TDist(ν)               # importance sampling density
N =[10, 100, 500, 1000]    # number of draws

h1(x) = x
h2(x) = x^2

# ---------
# Solution
# ---------
vH1BiasSq = zeros(N_run, length(N))
vH2BiasSq = similar(vH1BiasSq)
vH1var = Float64[]
vH2var = Float64[]
vH1InEff_Large = Float64[]
vH2InEff_Large = Float64[]
vH1InEff_Small = Float64[]
vH2InEff_Small = Float64[]
vInEff_Approx = Float64[]

for iDraws in 1:length(N)

   Random.seed!(1234) # For deterministic results

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
   E1 = 0.0    # E[x] = μ, you can check using E1 = expectation(h1, f)
   E2 = 1.0    # E[x^2] = σ^2, you can check using E2 = expectation(h2, f)

   h1_MCbias = vH1bar .- E1
   h2_MCbias = vH2bar .- E2

   # 2.3. Variance
   h1_MCvar = var(vH1bar)
   h2_MCvar = var(vH2bar)

   # 2.4. Ineficiency factor
   # Auxiliary variables
   Z = sqrt(2*π)^(-1)
   Var1_π = E2
   Var2_π = 3.0 - E2^2  # Note: Kurtosis of standard normal is equal to 3
   Ω1 = var( Z^(-1)*mWeights[iDraws,:] .* h1_MCbias)
   Ω2 = var( Z^(-1)*mWeights[iDraws,:] .* h2_MCbias)

   # Large sample
   h1_InEff_L = Ω1 / Var1_π
   h2_InEff_L = Ω2 / Var2_π

   # Small sample
   h1_InEff_S = h1_MCvar/( Var1_π / N[iDraws])
   h2_InEff_S = h2_MCvar/( Var2_π / N[iDraws])

   # Approximated
   InEff_Approx = 1 + (Z^(-2)*var(mWeights[iDraws,:]))


   # 3. Saving results
   vH1BiasSq[:,iDraws] = h1_MCbias.^2
   vH2BiasSq[:, iDraws] = h2_MCbias.^2

   global vH1var = push!(vH1var, h1_MCvar)
   global vH2var = push!(vH2var, h2_MCvar)

   global vH1InEffLarge = push!(vH1InEff_Large, h1_InEff_L)
   global vH2InEffLarge = push!(vH2InEff_Large, h2_InEff_L)
   global vH1InEffSmall = push!(vH1InEff_Small, h1_InEff_S)
   global vH2InEffSmall = push!(vH2InEff_Small, h2_InEff_S)
   global vInEffApprox = push!(vInEff_Approx, InEff_Approx)


   # 4. Graphs
   if N[iDraws] == 100

      # 1. Distribution of importance weights
      vWeights = mWeights[:,end]
      pHistogramWeights = histogram(vWeights, normalize=:probability, xlabel = "weights", color=:white, legend=:none)

      savefig(pHistogramWeights, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV/PS/PS1/LaTeX/pHistogramWeights.pdf")

      # 2. Density functions
      vTheta = sort!(mTheta[:,end])

      pSamplingDensity = plot(vTheta, pdf.(f, vTheta), label = "f", linewidth = 1.5, linestyle=:solid, linecolor=:black, xlabel=L"\theta")
      plot!(vTheta, pdf.(g, vTheta), label = "g", linewidth = 1.5, linestyle=:dash, linecolor=:black)

      savefig(pSamplingDensity, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV/PS/PS1/LaTeX/pSamplingDensity.pdf")

   end

end

# Mean Sq. Bias
vH1MeanSqBias = mean(vH1BiasSq, dims=1)
vH2MeanSqBias = mean(vH2BiasSq, dims=1)

# ------
# Graphs
# -------
pInefficiencyFactors = plot(N, vInEff_Approx, color=:black, alpha=0.6, linestyle=:solid, label="", xlabel="N")
plot!(N, vH1InEffSmall, linecolor=:black, linestyle=:solid, marker=:utriangle, markercolor=:white, label="")
plot!(N, vH2InEffSmall, linecolor=:black, linestyle=:solid, marker=:o, markercolor=:white, label="")
plot!(N, vH1InEffLarge, linecolor=:black, linestyle=:dash, marker=:utriangle, markercolor=:white, label="")
plot!(N, vH2InEffLarge, linecolor=:black, linestyle=:dash, marker=:o, markercolor=:white, label="")

savefig(pInefficiencyFactors, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV/PS/PS1/LaTeX/pInefficiencyFactors.pdf")
