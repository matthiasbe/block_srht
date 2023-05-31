# To be executed with `julia -p 16`
proc_count = 16
using Distributed
if nworkers() < proc_count; addprocs(proc_count - nworkers()); end

include("../src/io.jl")

### TEST
@testset "Read array from DArray" begin
  m = 4000
  n = 4000

  A = drand((m,n))

  @time write_darray(A, "test1.out")
  B = read_darray("test1.out", 3001:3800, 53:557)

  @test Matrix(A)[3001:3800,53:557] == B

  B = read_darray("test1.out", 1:4000, 1:4000)

  @test Matrix(A) == B

  rm("test1.out", recursive=true)
end





@testset "Read DArray" begin
  ### TEST square matrix created with 16 processes, read with 4,5,6
  m = 1000
  n = 1000

  A = drand((m,n))

  write_darray(A, "test2.out")
  B = read_darray("test2.out")

  @test Matrix(A) == Matrix(B)

  B = read_darray("test2.out", procs=workers()[1:4])
  @test Matrix(A) == Matrix(B)

  B = read_darray("test2.out", procs=workers()[1:5])
  @test Matrix(A) == Matrix(B)

  B = read_darray("test2.out", procs=workers()[1:6])
  @test Matrix(A) == Matrix(B)

  rm("test2.out", recursive=true)




  ### Rectangular matrix
  m = 2000
  n = 1000

  A = drand((m,n))

  write_darray(A, "test2.out")
  B = read_darray("test2.out")

  @test Matrix(A) == Matrix(B)

  B = read_darray("test2.out", procs=workers()[1:4])
  @test Matrix(A) == Matrix(B)
  rm("test2.out", recursive=true)



  ### Rectangular matrix
  m = 1000
  n = 2000

  A = drand((m,n))

  write_darray(A, "test2.out")
  B = read_darray("test2.out")

  @test Matrix(A) == Matrix(B)

  B = read_darray("test2.out", procs=workers()[1:4])
  @test Matrix(A) == Matrix(B)
  rm("test2.out", recursive=true)



  ### TEST square matrix created with 4 processes, read with 16
  m = 1000
  n = 1000

  A = drand((m,n), workers()[1:4])

  write_darray(A, "test2.out")
  B = read_darray("test2.out")

  @test Matrix(A) == Matrix(B)

  B = read_darray("test2.out", procs=workers()[1:4])
  @test Matrix(A) == Matrix(B)

  B = read_darray("test2.out", procs=workers()[1:5])
  @test Matrix(A) == Matrix(B)

  B = read_darray("test2.out", procs=workers()[1:6])
  @test Matrix(A) == Matrix(B)

  rm("test2.out", recursive=true)

    ### TEST square matrix created with 4 processes, read with 16
    m = 11654
    n = 1000

    A = drand((m,n), workers()[1:4], [1,4])
    write_darray(A, "test2.out")
    B = read_darray("test2.out")
    @test Matrix(A) == Matrix(B)

    rm("test2.out", recursive=true)

    A = drand((m,n), workers()[1:4], [4,1])
    write_darray(A, "test2.out")
    B = read_darray("test2.out")
    @test Matrix(A) == Matrix(B)

    rm("test2.out", recursive=true)

end
