proc_count = 16 # Number of procs required for this test

using Distributed
if nworkers() < proc_count; addprocs(proc_count - nworkers()); end

include("../src/reduce.jl")
include("../src/darray.jl")

using DistributedArrays
using Test
using LinearAlgebra

@testset "binary_tree_reduce" begin
  a = rand(Int, 64)
  @test sum(a) == binarytree_reduce(+, a)
  @test prod(a) == binarytree_reduce(*, a)

  a = rand(Int, 8)
  @test sum(a) == binarytree_reduce(+, a)
  @test prod(a) == binarytree_reduce(*, a)
end

@testset "distributed reduction" begin

  A = drand((2^10,2^10), workers(), [1, nworkers()])

  B = @fetchfrom workers()[1] A[:l]
  for p in workers()[2:nworkers()]
    B += @fetchfrom p A[:l]
  end

  agg = reduce_sum(A)

  C = @fetchfrom agg A[:l]

  @test C ≈ B atol=10e-10


  # It should not work with non uniform partitioning
  A = drand((2^10,1000), workers(), [1, nworkers()])
  @test_throws AssertionError reduce_sum(A)
end
