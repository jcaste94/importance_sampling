# -----------------------------------------------------------------------------
# POSTERIOR SAMPLERS: IMPORTANCE SAMPLING
# Econometrics IV, Part 1
# Prof. Frank Schorfheide
#
# Approximated mean
# -----------------------------------------------------------------------------

module IS

export mean_approx

# ---------
# Packages
# ---------
using Distributions
using Random

# ----------
# Algorithm
# ----------
function mean_approx(h::Function, f::Distribution, g::Distribution, N::Integer)

    # --------------------------------
    # 1. Unnormalized importance weights
    # --------------------------------
    # 1.1. Auxiliary function
    function weights(g::Distribution, f::Distribution)

        # Draws from the importance sampling density
        θ = rand(g,1)[1]

        # Unnormalized weight
        w = pdf(f,θ)/pdf(g,θ)

        return w, θ
    end

    # 1.2. Computation
    vWeights = Float64[]
    vTheta = Float64[]

    for i in 1:N

        # Evaluate function
        w_i, θ = weights(f,g)

        # Save results
        vWeights = push!(vWeights, w_i)
        vTheta = push!(vTheta, θ)

    end

    # ---------------------------------
    # 2. Normalized importance weights
    # ---------------------------------
    vWeightsNormalized = vWeights ./ mean(vWeights)

    # Check: sample average = 1
    @assert(mean(vWeightsNormalized) ≈ 1, "By construction, the sample average must be equal to 1!")

    # -----------------------------
    # 3. Approximation of the mean
    # -----------------------------
    h_bar = mean(vWeightsNormalized .* h.(vTheta))

    return h_bar, vWeightsNormalized, vTheta
end

end
