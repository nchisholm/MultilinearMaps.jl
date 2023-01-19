using Test
using MultilinearMaps
using LinearAlgebra
using StaticArrays

@testset "MultilinearMap" begin

    @inline _skew(v1, v2, v3, v4) = (v1⋅v3)*(v2⋅v4) - (v1⋅v4)*(v2⋅v3)

    @testset "Evaluation" begin
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

    @testset "Egality & equality" begin
        inner22 = MultilinearMap(dot, (2,2))
        skew = MultilinearMap(_skew, (2,2,2,2))
        u = rand(2)
        v = rand(2)

        @test MultilinearMap(dot, (2,2)) === inner22
        @test MultilinearMap((x,y) -> x⋅y, (2,2)) !== inner22
        @test MultilinearMap(dot, (3,3)) !== inner22
        @test MultilinearMap((x,y) -> x⋅y, (2,2)) == inner22  # same values
        @test skew != inner22
        @test skew(:,u,v,:) != inner22
    end

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

    @testset "Vector space algebra" begin
        inner = MultilinearMap(dot, (2,2))
        skew = MultilinearMap(_skew, (2,2,2,2))
        u = rand(2)
        v = rand(2)

        @test (skew + skew) == 2skew == skew/0.5
        @test 2(skew + skew) / 4 == skew
        @test all(==(0), skew - skew)
        @test !all(==(0), skew(:,:,u,v) - inner)
    end
end
