excluded_files = ["all_tests.jl", "test_utils.jl"]

for i in readdir("tests")
  if i ∉ excluded_files
    @info "running test $i"
    include(i)

    # Clean memory
    d_closeall()
    [remotecall_wait(GC.gc, p) for p in workers()]
    GC.gc()
  end
end
