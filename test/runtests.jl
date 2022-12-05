using Test

using MultilinearMaps

import ArrayInterface as Arr
import ArrayInterfaceStaticArrays
using LinearAlgebra
using StaticArrays
using Static

const MM = MultilinearMaps


@testset "Unit Vectors" begin
    e1 = StandardBasis(1)
    e2 = StandardBasis(2)
    e3 = StandardBasis(3)

    @testset "Construction" begin
        @test size(e2[1]) == (2,)
        @test length(e2[1]) == only(size(e2[1]))
        @test_throws DomainError e1[0]
        @test_throws DomainError e2[3]
        @test only(e1[1])
    end
    @testset "Equality" begin
        @test e2[1] == e2[1]
        @test e2[1] !== e2[2]
        @test e2[1] !== e3[1]
        @test e2[1] == Bool[true, false]
        @test e2[1] !== Bool[true, false, false]
    end
    @testset "Dot product" begin
        @test @inferred e1[1] ⋅ e1[1]
        @test e2[1] ⋅ e2[1]
        @test !(e2[1] ⋅ e2[2])
        @test !(e2[2] ⋅ e2[1])
        @test e2[1] ⋅ [1,2] == [1,2] ⋅ e2[1] == 1
        @test e2[2] ⋅ [1,2] == [1,2] ⋅ e2[2] == 2
        @test e2[1] ⋅ SVector(1,2) == SVector(1,2) ⋅ e2[1] == 1
        @test e2[2] ⋅ [1,2] == [1,2] ⋅ e2[2] == 2
        @test_throws DimensionMismatch e2[1] ⋅ e1[1]
        @test_throws DimensionMismatch SVector(1,2) ⋅ e1[1]
        @test_throws DimensionMismatch [1,2] ⋅ e1[1]
    end
    # Other
    @test @inferred(e2[2] + [1,0]) == ones(2)
    @test SVector{2}(e2[1] + e2[2]) === ones(SVector{2,eltype(true+true)})
    @test_broken SVector(e2[1] + e2[2]) === ones(SVector{2,eltype(true+true)})
end;

@inline _skew(v1, v2, v3, v4) = (v1⋅v3)*(v2⋅v4) - (v1⋅v4)*(v2⋅v3)

@testset "MultilinearMap Evaluation" begin
    e = StandardBasis(3)
    u = e[1]
    v = e[2]
    onlytrue = @inferred MultilinearMap(() -> true, ())
    inner = @inferred MultilinearMap(dot, (2,2))
    skew = @inferred MultilinearMap(_skew, (2,2,2,2))
    @test onlytrue() == true
    @test_throws ArgumentError onlytrue(u)
    @test inner(u,u) == 1
    @test inner(u,v) == 0
    @test inner(v,u) == 0
    @test_throws ArgumentError inner(u)
    @test skew(u,u,v,v) == 0
    @test skew(u,v,u,v) == 1
    @test skew(u,v,v,u) == -1
end;

@testset "Partial Evaluation" begin
    # @testset "Component equality" begin
    #     skew_components = SArray(skew)  # Materialize the whole tensor
    #     # Now, slice the component array and compare it to tensor contraction
    #     # with the unit vectors
    #     @test SArray(skew(ê{3}(1), :, ê{3}(2), :)) == skew_components[1,:,2,:]
    #     @test SArray(skew(:, :, ê{3}(3), ê{3}(2))) == skew_components[:,:,3,2]
    # end
    @testset "2d" begin
        e = StandardUnitVector(1, 2)
        inner = MultilinearMap(dot, (2,2))
        @test_throws ArgumentError inner(:,:,:)
        @test_throws ArgumentError inner(:)
        @test inner(:,:) === inner
        @inferred inner(e,:)
        @test_broken inner(:,e) == inner(e,:)
    end
    @testset "3d" begin
        (u,v,w,x) = ntuple(_ -> rand(SVector{3,Float64}), Val(4))
        inner = MultilinearMap(dot, (3,3))
        skew = MultilinearMap(_skew, (3,3,3,3))
        @inferred skew(u,v,w,:)
        @inferred skew(u,v,w,:)(x)
        @test_broken inner(u,v) == inner(u,:)(v) == inner(:,u)(v) == inner(:,:)(u,v)
        @test_broken skew(u,v,w,x) ≈ skew(u,v,w,:)(x) ≈ skew(u,v,:,:)(w,x) ≈
            skew(u,:,:,:)(v,w,x) ≈ skew(:,v,w,x)(u)
        # @test eltype(inner(e[1], :)) == eltype(e[1])
        # @test eltype(inner(u, :)) == eltype(u)
        # @test eltype(skew(e[1], e[2], :, e[3])) == Int
    end
end

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
