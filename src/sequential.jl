using LinearAlgebra
#using MatrixMarket
using Plots
using Hadamard
using Random
using TSVD

function srht2(A, l)::Array{Float64,2}
  m = size(A)[1]
  n = size(A)[2]
  D = rand((-1,1), m)
  B = Array{Float64}(undef, l, n)
  P = randperm(size(A)[1])[1:l]
  for j = 1:size(A)[2]
    u = A[:,j]
    u = D .* u
    u = fwht(u)
    u_sample = (m/sqrt(l)) .* u[P]
    B[:,j] = u_sample
  end
  return B
end

"""
Generate a random matrix with an exponential decay of the singular values
The returned matrix is of size m×m and has roughly k meaningful singular values
"""
function generate_exp_decay(m, k)::Tuple{Array{Float64,1},Array{Float64,2}}
  α::Float64 = 5 / k
  `singular values`
  S::Array{Float64,1} = map(y -> exp(-α * y), 1:m)
  
  U::Array{Float64,2} = LinearAlgebra.qr(rand(Float64, (m, m))).Q
  V::Array{Float64,2} = LinearAlgebra.qr(rand(Float64, (m, m))).Q

  US::Array{Float64,2} = hcat(map(x -> S .* x, eachrow(U)))

  M = US * V

  return (S, US * V)
end

function compare_columns_norm(A::Array{Float64}, B::Array{Float64})::Float64
  @assert size(A)[2] == size(B)[2]
  normsA = map(norm, eachcol(A))
  normsB = map(norm, eachcol(B))
  return max(normsA - normsB) ./ normA
end

"""
Block approximation of a random input matrix using randomization
This function is one-threaded (for prototyping).

A = [ A1 ... Ap ] is a m×m random matrix with k meaningful singular values
(simulation of the input matrix)

Ω = [ Ω1 .. Ωp]^T

We compute B = AΩ where Ω is a random matrix of size m×l

B = Σ AiΩi pour i = 1 .. p
"""
function block_rand_seq(;m::Int=2^10, l::Int=2^6, k::Int=10, P::Int=4)
  #M = MatrixMarket.mmread("fidap001.mtx")

  @assert l <= m
  @assert k <= m / P

  println("Generating matrix...")
  S,At::Array{Float64,2} = generate_exp_decay(m,k)
  println("Transposed matrix size ", size(At))


  # Split the columns of A
  # A = [A1 A2 ... Ap]
  # partitions[i] = (first column index of Ai, last column index of Ai)
  partition_start_indices = floor.(Int, collect(range(0,size(At)[2],length=P+1)))
  "An array of tuples, each tuple defines a domain of A"
  partitions = [partition_start_indices[i]+1:partition_start_indices[i+1] for i in 1:length(partition_start_indices) - 1]
  println("Partitioning : ", partitions)

  At_sketched::Array{Float64,2} = zeros(Float64, l, size(At)[2])

  for p in partitions
    Atp::Array{Float64,2} = copy(At[p,:])
    At_sketched = At_sketched + @time srht(Atp, l)
  end
 
  #@time SVD_A = svd(At)
  #@time SVD_truncA::Tuple{Array{Float64,2}, Array{Float64,1}, Array{Float64,2}} = TSVD.tsvd(At, k)
  @time SVD_sketched::SVD = LinearAlgebra.svd(At_sketched)

  plot([S[1:k] SVD_sketched.S[1:k]], lab=["Original" "Sketched"])
end
