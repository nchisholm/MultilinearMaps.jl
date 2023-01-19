using Test

@testset "MultilinearMaps" begin
    include("LinearFunctionSpaces.jl")
    include("StandardBasis.jl")
    include("./MultilinearMaps.jl")
end

# using MultilinearMaps

# import ArrayInterface as Arr
# import ArrayInterfaceStaticArrays
# using LinearAlgebra
# using StaticArrays
# using Static

# const MM = MultilinearMaps




# @testset "Vector Space Algebra" begin
#     @testset "Equality" begin
#         e = StdBasis{3}(Real)
#         @test inner == inner
#         @test inner != skew && skew != inner
#         @test skew(e[1], :, e[2], :) != inner && inner != skew(e[1], :, e[2], :)
#     end
#     @testset "Scalar Multiples" begin
#         @test MM.ScalarMultiple(inner, 0.5) == 0.5 * inner == inner / 2
#         @test inner !== inner / 2
#         @test MM.ScalarMultiple(inner, 1//2) == inner // 2 == 1//2 * inner
#     end
#     @testset "Sums" begin
#         # Associativity
#         @test (inner + inner) + inner == inner + (inner + inner) == inner + inner + inner
#         # Can't add maps of unequal sizes (should probably give a more helpful exception)
#         @test_throws DimensionMismatch inner + skew
#     end;
#     @testset "Linear Combinations" begin
#         @test all(==(0), skew - skew)
#         @test inner + inner == 2 * inner
#         @test inner + inner + inner == 2*inner + inner == inner + 2*inner == 3*inner
#         @test 2*(skew + skew) / 2 == 2*skew
#     end
# end;

# @testset "StaticArrays traits" begin
#     @test StaticArrays.Length(inner) == StaticArrays.Length(3^2)
#     @test StaticArrays.Length(skew) == StaticArrays.Length(3^4)
#     @test StaticArrays.Size(inner) == StaticArrays.Size(3,3)
#     @test StaticArrays.Size(skew) == StaticArrays.Size(3,3,3,3)
# end

# include("harmonics.jl")

# """Test (recursively) if an array is traceless in every pair of indices"""
# istraceless(A::AbstractArray{<:Any, 0}, _::Int) = true
# istraceless(A::AbstractArray{<:Any, 1}, _::Int) = true
# istraceless(A::AbstractArray{<:Any, 2}, _::Int) =
#     ≈(tr(A), 0, atol=√(eps(eltype(A))))
# istraceless(A::AbstractArray, dim::Int) =
#     all(istraceless(B) for B in eachslice(A, dims=dim))
#     # For dim = 1, does
#     # all(≈(tr(out[i,:,:]), 0, atol=eps(eltype(out))) for i ∈ axes(out, 1))
# istraceless(A::AbstractArray) = all(istraceless(A, dim) for dim ∈ 1:ndims(A))

# _issymmetric(A::AbstractArray{<:Any, 0}) = true
# _issymmetric(A::AbstractArray{<:Any, 1}) = true
# _issymmetric(A::AbstractArray{<:Any, 2}) =
#     all(≈(A[i,j] - A[j,i], 0, atol=√(eps(eltype(A)))) for i ∈ axes(A,1), j ∈ axes(A,2))
# # _issymmetric(A::AbstractArray, dim) = all(issymmetric(B) for B in eachslice(A, dims=dim))
# # _issymmetric(A::AbstractArray) = all(issymmetric(A, dim) for dim in 1:ndims(A))

# @testset "Harmonics" begin
#     x = normalize(rand(SVector{3,Float64}))
#     ê = StdBasis{3}(Real)
#     @testset "Traceless" begin
#         for formfield in (sphharm30, sphharm31, sphharm32, sphharm33)
#             form = formfield(x)
#             K = ndims(form)
#             D = Arr.size(form, 1)
#             out = SArray(form)
#             @test ndims(out) == K
#             @test all(==(D), size(out))
#             @test istraceless(out)
#         end
#     end
#     @testset "Symmetric" begin
#         @test issymmetric(SArray(sphharm32(x)))
#         for i ∈ 1:3
#             @test _issymmetric(SArray(sphharm33(x)(:,:, ê[i])))
#             @test _issymmetric(SArray(sphharm33(x)(:, ê[i], :)))
#             # Needed? I think implied by the previous two
#             @test _issymmetric(SArray(sphharm33(x)(ê[i], :, :)))
#         end
#     end
# end;
