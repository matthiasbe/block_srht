@everywhere using DistributedArrays
@everywhere using Random
@everywhere using Distributions
using LinearAlgebra

@everywhere include("reduce.jl")

@everywhere function sketch_right_gaussian(A::Array{Float64,2},l::Int; seed::UInt64=rand(UInt64))::Array{Float64,2}
  m,n = size(A)
  d = Normal(0, 1 / sqrt(l))
  Ω = rand(MersenneTwister(seed), d, (n, l))
  return A * Ω
end

@everywhere function sketch_left_gaussian(A::Array{Float64,2},l::Int; seed::UInt64=rand(UInt64))::Array{Float64,2}
  m,n = size(A)
  d = Normal(0, 1 / sqrt(l))
  Ω = rand(MersenneTwister(seed), d, (m, l))
  return Ω' * A
end

"""
Compute B = AΩ where A is a m×m column-distributed matrix

A = [A1 ... Ap]

and Ω is a m×l gaussian sketching matrix

We must have l ≤ m/p
"""
function sketch_right_gaussian(Ap::DArray{Float64,2},l::Int, to::TimerOutput; block_seeds::Array{UInt64}=rand(UInt64,l))::Array{Float64,2}
  @assert(size(procs(Ap))[1] == 1) # check Ap is column-distributed

  m::Int = size(Ap)[1]
  n::Int = size(Ap)[2]
  #@assert l <= m / size(procs(Ap))[2]

  nps = prod(size(Ap.pids))
  ps = reshape(Ap.pids, nps)

  # Sketch each block: compute A_i Ω_i
  
  @timeit to "local" begin
    res = [@spawnat p begin
             proc_coord = findall(x -> x == p, procs(Ap))[1]
             sketch_right_gaussian(Ap[:l], l, seed=block_seeds[proc_coord[2]])
           end for p in ps]
    [wait(r) for r in res]
    # Synchronization + error handling
    Ap_sketched::DArray{Float64,2} = DArray(reshape(res,(1,nps)))
  end

  # Sum reduction : sums all partitions of Atp_sketched
  # The result is on aggregator_worker
  @timeit to "reduce" aggregator_worker = reduce_sum(Ap_sketched)

  # Get the result on master process
  A_sketched = @fetchfrom aggregator_worker Ap_sketched[:l]

  close(Ap_sketched)

  return A_sketched
end

"""
Sketch a 2D distributed matrix from right and left.

Each local block is sketched, then all results are sum-reduced.
Perform B_ij = Θ_i A_ij Ω_j
Sum all B_ij
"""
function sketch_2D_gaussian(Ap::DArray{Float64,2}, l::Tuple{Int,Int}, to::TimerOutput)::Array{Float64,2}

  (Pr,Pc) = size(Ap.pids)
  nps = prod(size(Ap.pids))
  ps = reshape(Ap.pids, nps)

  # Initialize global Random Number Generators, for left and right sketching
  left_rngs = [MersenneTwister() for i in 1:Pr+1]
  right_rngs = [MersenneTwister() for i in 1:Pc+1]

  # Sketch each block: compute Θ_i A_i Ω_i
  @timeit to "local" begin
    res = [@spawnat p begin
             proc_coord = findall(x -> x == p, procs(A))[1]
             temp = sketch_right_gaussian(Ap[:l], l[1], rng_idpt=right_rngs[proc_coord[2]])
             temp = permutedims(temp)
             sketch_right_gaussian(temp, l[2], seed=left_rngs[proc_coord[1]])
           end for p in ps]
    [wait(r) for r in res]
    Atp_sketched::DArray{Float64,2} = DArray(reshape(res,size(procs(Ap))))
  end

  # Sum reduction : sums all partitions of Atp_sketched
  # The result is on aggregator_worker
  @timeit to "reduce" aggregator_worker = reduce_sum(Atp_sketched)

  # Get the result on master process
  A_sketched = @fetchfrom aggregator_worker Atp_sketched[:l]'

  close(Atp_sketched)

  return A_sketched
end

"""
Apply a left sketch B = Ω A where A is distributed on a 2D grid Pr × Pc
and Ω on 1×Pr
"""
function sketch_left_2D_gaussian(Ap::DArray{Float64,2}, l::Int, to::TimerOutput; block_seeds::Vector{UInt}=rand(UInt,length(procs(Ap))))::DArray{Float64,2}
  (Pr,Pc) = size(procs(Ap))
  nps = prod(size(procs(Ap)))
  ps = reshape(procs(Ap), nps)

  # Sketch each block: compute Θ_i^t A_ij Θ_j, Θ_i^t A_ij and A_ij Θ_j
  @timeit to "local" begin
    res = [@spawnat p begin
             proc_coord::Tuple{Int,Int} = findall(x -> x == p, procs(Ap))[1]
             # D_i = Θ_i^t A_ij
             sketch_left_gaussian(copy(localpart(Ap)), l, seed=block_seeds[proc_coord[1]])
           end for p in ps]
    [wait(r) for r in res]
    Dp_temp::DArray{Float64,2} = DArray(reshape(res,(Pr,Pc)))
  end

  # Sum all blocks of D_i in a column of procs
  @timeit to "reduce UtA" begin
    resD = [@spawnat procs(Ap)[1,j] begin
              # Select the submatrix of the column
              res = [@spawnat procs(Ap)[i,j] begin Dp_temp[:l] end for i in 1:Pr]
              [wait(r) for r in res]
              sub::DArray{Float64, 2} = DArray(reshape(res, (Pr,1)))
              # Sum all blocks
              agg = reduce_sum(sub)
              @assert(agg == myid()) # Check we were elected by the reduce
              localpart(sub)
            end for j in 1:Pc]
    [wait(r) for r in resD]
    Dp::DArray{Float64,2} = DArray(reshape(resD,(1,Pc)))
  end
  return Dp
end

function sketch_nystrom_cholesky_gaussian(Ap::DArray{Float64,2}, l::Int, to::TimerOutput)::Tuple{Matrix{Float64}, DArray{Float64,2}}
  (Pr,Pc) = size(procs(Ap))

  # Initialize global Random Number Generator
  seed = rand(UInt)
  seeds = rand(UInt, Pr)

  @timeit to "reduce UtA" Dp = sketch_left_2D_gaussian(Ap, l, to, block_seeds=seeds)
  @timeit to "reduce UtAU" B = sketch_right_gaussian(Dp, l, to, block_seeds=seeds)

  @timeit to "cholesky" C = cholesky(Symmetric(B))
  @timeit to "compute B = Y C-1" begin
    res = [@spawnat p (C.L \ Dp[:l]) for p in procs(Dp)]
    [wait(r) for r in res]
    CDp = DArray(reshape(res, (1, length(procs(Dp)))))
  end

  return I(l), CDp
end

function sketch_nystrom_gaussian(Ap::DArray{Float64,2}, l::Int, to::TimerOutput)Tuple{Vector{Float64}, DArray{Float64,2}, Matrix{Float64}}
  (Pr,Pc) = size(procs(Ap))

  # Initialize global Random Number Generator
  seed = rand(UInt)
  seeds = rand(UInt, Pr)

  @timeit to "reduce UtA" Dp = sketch_left_2D_gaussian(Ap, l, to, block_seeds=seeds)
  @timeit to "reduce UtAU" B = sketch_right_gaussian(Dp, l, to, block_seeds=seeds)

  @timeit to "cholesky" C = cholesky(Symmetric(B))
  @timeit to "compute B = Y C-1" begin
    res = [@spawnat p (C.L \ Dp[:l])' for p in procs(Dp)]
    [wait(r) for r in res]
    CDp = DArray(reshape(res, (length(procs(Dp)), 1)))
  end

  @timeit to "tsqr" R::Matrix{Float64} = tsqr(CDp)

  @timeit to "last step" begin 
    @timeit to "svd or R" _,S,V = svd(R)
    
  end
  close(Dp)

  return S,CDp, V
end

