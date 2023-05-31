"""
Return the 2-exponent of the largest power of two < a

a must be stricly greater than zero
"""
function previous_pot(a::Int)::Int
  @assert a > 0

  b = a >> 1
  i = 0
  while b > 0
    b = b >> 1
    i = i+1
  end
  return i
end

"""
Return true if on only if a is a power of two

a must be stricly greater than zero
"""
function is_pot(a::Int)::Bool
  @assert a > 0
  b::UInt32 = a

  sum = 1 & b
  while b > 0
    b = b >> 1
    sum += (1 & b)
  end
  return sum == 1
end
