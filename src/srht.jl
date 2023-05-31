using Hadamard
using Random
using TimerOutputs
using LinearAlgebra
using Dates

"""
Compute the Subsampled Randomized Hadamard Transform of A with a sampling size of l.
Return B = Ω × A where A is m×n, Ω is l×m and B is lxn
Omega = sqrt(m/l) Dr R H Dl
* Dl and Dr are diagonal matrices of random signs
* H is the Hadamard matrix
* R is a sampling matrix, selecting l rows out of m

A should be stored in column major format.

In the context of a block SRHT: (Ω_1 ... Ω_p) (A_1 ... A_p)^t, then the
random number generator seeds should be passed to all srht() calls.
param global_seed: same for all block
param block_seed: different for each block
"""
function srht(A::Array{Float64,2}, l::Int; to::TimerOutput=TimerOutput(), global_seed::UInt=rand(UInt), block_seed::UInt=rand(UInt))::Array{Float64,2}
  rng_global = MersenneTwister(global_seed)
  rng_block = MersenneTwister(block_seed)
  #@info "$(now()) SRHT $(size(A)) into $l"; flush(stdout); flush(stderr)
  @assert l <= size(A)[1]
  @timeit to "init rand" begin
    # Rademacher vector
    Dr::Array{Int} = rand(rng_block, (-1,1), size(A)[1])
    Dl::Array{Int} = rand(rng_block, (-1,1), l)

    # Sampling without replacement gives better results in practice for this usecase
    rplc::Bool = true
    if (haskey(ENV, "BSRHT_RPLC"))
        rplc = eval(Meta.parse(ENV["BSRHT_RPLC"]))
    end
    if(rplc)
      # Random sampling permutation with replacement
      P = rand(rng_global, 1:size(A)[1], l)
    else
      # Random sampling permutation without replacement
      P = randperm(rng_global, size(A)[1])[1:l]
    end

    # Rescaling
    scale::Float64 = size(A)[1]/sqrt(l)
  end

  # X = Dr A
  @timeit to "Dr" lmul!(Diagonal(Dr), A)

  # X1 = H X
  @timeit to "H" fwht_natural!(A, 1)

  # X2 = R X1
  @timeit to "R" B::Array{Float64,2} = scale .* A[P,:]

  # Compute B = R Dl H Dr A
  @timeit to "Dl" lmul!(Diagonal(Dl), B)
  
  #@info "$(now()) SRHT done"; flush(stdout); flush(stderr)
  return B
end
