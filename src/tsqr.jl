## The implementation is not complete. There might be bugs.

using DistributedArrays
using LinearAlgebra
using Distributed
using Random
#using Test
include("reduce.jl")

@everywhere struct BinaryTree
	worker::Int
	value::Future
	left::Union{BinaryTree,Nothing}
	right::Union{BinaryTree,Nothing}
end

function reductor(A::BinaryTree, B::BinaryTree)
	fact = @spawnat A.worker qr([fetch(A.value).R; fetch(B.value).R])
	return BinaryTree(A.worker, fact, A, B)
end

function tsqr(A::DArray{Float64,2})::Array{Float64,2}
	# Only valid for power of two processes
  @assert(log2(length(procs(A))) == floor(log2(length(procs(A)))), "TSQR only implemented for power of two processes yet")

  # The matrix must be distributed on a single column of procs
  @assert(size(procs(A))[2] == 1, "TSQR requires a one-column distribution grid")

  # Blocs should not be flat
  @assert(size(A)[1] / size(procs(A))[1] >= size(A)[2], "TSQR requires square or tall blocks")

	pieces = [@spawnat p qr(A[:l]) for p in procs(A)]
	initial_value = [BinaryTree(p, pieces[p_i], nothing, nothing) for (p_i, p) in enumerate(procs(A))]
  return fetch(reduce(reductor, initial_value).value).R
end

#@testset "TSQR" begin
#  # Valid test
#  A = drand((1000,250), workers(), [nworkers(), 1])
#  R_tsqr = tsqr(A)
#  Q = Matrix(A) / R_tsqr
#  @test norm(I - Q'*Q) ≈ 0 atol=10e-10
#
#  # Valid test : tall blocks
#  A = drand((2000,250), workers(), [nworkers(), 1])
#  R_tsqr = tsqr(A)
#  Q = Matrix(A) / R_tsqr
#  @test norm(I - Q'*Q) ≈ 0 atol=10e-10
#
#  # Unvalid test : Square distribution grid
#  P = Int(floor(nworkers() / 2))
#  A = drand((1000,250), workers(), [P,P])
#  @test_throws AssertionError tsqr(A)
#
#  # Unvalid test : blocks are a tiny bit flat
#  A = drand((1000,251), workers(), [nworkers(), 1])
#  @test_throws AssertionError tsqr(A)
#end
