# src/Compute.jl

using Random

export decode, energy, hamming, update_neuron!, recover

# Decoding for visualization

"""
    decode(s)

Convert a 1D bipolar state vector `s` (entries -1 or +1) back into a
2D image array with values in [0, 1].

We assume the image is square: side length = √length(s).

The notebook uses this as:
    decode(s) |> img -> Gray.(img)
"""
function decode(s::AbstractVector{<:Real})
    N = length(s)
    side = round(Int, sqrt(N))
    side * side == N ||
        error("decode: state length $N is not a perfect square, cannot form a square image.")

    # Map -1 → 0, +1 → 1
    img_vec = (Float32.(s) .+ 1.0f0) ./ 2.0f0
    return reshape(img_vec, side, side)
end

# Energy and Hamming distance


"""
    energy(model, s)

Compute Hopfield energy of state vector `s` for the given model:

    E(s) = -1/2 * sᵀ W s - bᵀ s

with `b = 0` for the classical Hopfield case.
"""
function energy(model::MyClassicalHopfieldNetworkModel,
                s::AbstractVector{<:Real})
    N = size(model.W, 1)
    length(s) == N ||
        error("energy: state length $(length(s)) does not match network size $N.")

    v = Float32.(s)
    return -0.5f0 * dot(v, model.W * v) - dot(model.b, v)
end

"""
    hamming(a, b)

Return the Hamming distance between vectors `a` and `b`,
i.e., the number of positions that differ.

Used in the notebook as:
    hamming(best_state, s1)
    hamming(s0, s1)
"""
function hamming(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) ||
        error("hamming: vectors must have the same length.")

    d = 0
    @inbounds for i in eachindex(a, b)
        d += (a[i] != b[i]) ? 1 : 0
    end
    return d
end


# Asynchronous neuron update

"""
    update_neuron!(model, state, i)

Asynchronously update neuron `i` in-place:

    sᵢ ← sign(∑ⱼ Wᵢⱼ sⱼ + bᵢ)

with tie-breaking rule `sign(0) = +1`.

`state` is a `Vector{Int32}` with entries -1 or +1 and is modified in-place.
"""
function update_neuron!(model::MyClassicalHopfieldNetworkModel,
                        state::Vector{Int32},
                        i::Int)
    N = length(state)
    1 ≤ i ≤ N || error("update_neuron!: index $i out of bounds for state of length $N.")

    h = 0.0f0
    @inbounds for j in 1:N
        h += model.W[i, j] * float(state[j])
    end
    h += model.b[i]

    state[i] = h >= 0.0f0 ? Int32(1) : Int32(-1)
    return state
end


# Recovery algorithm


"""
    recover(model, s0, true_image_energy;
            maxiterations::Int = 1000,
            patience::Union{Int,Nothing} = 5,
            miniterations_before_convergence::Union{Int,Nothing} = nothing)

Run the asynchronous Hopfield recovery dynamics starting from corrupted
state `s0`.

Arguments
---------
- `model::MyClassicalHopfieldNetworkModel`
- `s0::Array{Int32,1}`           : corrupted initial state (bipolar -1/+1)
- `true_image_energy::Float32`   : energy of the target memorized pattern
                                   (used for comparison/plots only)
- `maxiterations`                : maximum number of update steps
- `patience`                     : number of *consecutive* identical states
                                   required to declare convergence
- `miniterations_before_convergence` :
      minimum number of iterations before we start checking for convergence.
      If `nothing`, defaults to `patience`.

Returns
-------
- `frames::Dict{Int64,Array{Int32,1}}`      : iteration index → state
- `energydictionary::Dict{Int64,Float32}`   : iteration index → energy
"""
function recover(model::MyClassicalHopfieldNetworkModel,
                 s0::Array{Int32,1},
                 true_image_energy::Float32;
                 maxiterations::Int = 1000,
                 patience::Union{Int,Nothing} = 5,
                 miniterations_before_convergence::Union{Int,Nothing} = nothing)

    N = size(model.W, 1)
    length(s0) == N ||
        error("recover: initial state length $(length(s0)) does not match network size $N.")

    pat = isnothing(patience) ? 5 : patience
    miniter = isnothing(miniterations_before_convergence) ? pat : miniterations_before_convergence

    # Output dictionaries with the exact types the notebook mentions
    frames = Dict{Int64,Array{Int32,1}}()
    energydictionary = Dict{Int64,Float32}()

    state = copy(s0)
    last_state = copy(s0)
    same_count = 0

    for iter in 1:maxiterations
        # Record state and energy at this iteration
        frames[iter] = copy(state)
        energydictionary[iter] = energy(model, state)

        # One asynchronous sweep
        for i in randperm(N)
            update_neuron!(model, state, i)
        end

        # Check for convergence
        if all(state .== last_state)
            same_count += 1
        else
            same_count = 0
        end

        if iter ≥ miniter && same_count ≥ pat
            break
        end

        last_state .= state
    end

    return frames, energydictionary
end
