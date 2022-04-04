# Functions that represent (tensor) spherical harmonics
sphharm30(_) = MultilinearForm{0,3}(() -> true)
sphharm31(n̂) = MultilinearForm{1,3}((v) -> n̂⋅v)
sphharm32(n̂) = MultilinearForm{2,3}((v1, v2) -> (n̂⋅v1)*(n̂⋅v2) - (v1⋅v2)/3 )
sphharm33(n̂) = MultilinearForm{3,3}((v1, v2, v3) ->
    (n̂⋅v1)*(n̂⋅v2)*(n̂⋅v3) - ((v1⋅v2)*(n̂⋅v3) + (v3⋅v1)*(n̂⋅v2) + (v2⋅v3)*(n̂⋅v1))/5)
