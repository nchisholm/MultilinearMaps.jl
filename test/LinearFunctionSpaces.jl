using Test

import MultilinearMaps.LinearFunctionSpaces:
    One, NegOne, Recip, _scale,
    Primative, Const, Add, Scale, Neg, Sub
import MultilinearMaps: @singleton


# Define some primatives using the singleton approach...
@singleton f1 isa Primative
f1(x) = x^2 + 1
@singleton f2 isa Primative
f2(x) = x^3 / 2 - x
@singleton f3 isa Primative
f3(x) = 2 - x
# NOTE: must make these def'ns at toplevel due to a limitation in @singleton

@testset "LinearFunctionSpaces" begin

    @testset "SpecialScalars" begin
        # One() and NegOne()
        @test -One() == NegOne() && -NegOne() == One()
        @test all(s -> _scale(One(), s) == s, (One(), NegOne())) &&
            all(s -> _scale(NegOne(), s) == -s, (One(), NegOne()))
        let s = rand()
            @test _scale(One(), s) === s
            @test _scale(NegOne(), s) === -s
        end
        @test all(s -> inv(s) === s, (One(), NegOne()))
        @test all(op -> Integer(op(One())) == op(1), (+, -))

        # Properties of Recip()
        @test all(s -> Recip(s) == s, (One(), -One()))
        for T in (Bool, Int16, Float16, Complex{Float16})
            let s = rand(T), x = rand(T), v = rand(T, 2)
                # TODO: Explicitly force zeros in s and x
                @test Recip(s) != Recip(x) || s == x
                @test Recip(s) == inv(s) && inv(s) == Recip(s)
                @test Recip(Recip(s)) === s
                @test _scale(Recip(s), One()) === _scale(One(), Recip(s)) === Recip(s)
                @test _scale(Recip(s), Recip(x)) == Recip(s*x)
                @test s==0 || _scale(Recip(s), v) == v * Recip(s) == v/s
                @test Recip(s)' == inv(s)'  # adjoint

                # if T <: Integer
                #     @test Recip(2) // 3 == 1//6
                #     @test 3 // Recip(2) == 6
                #     @test (3//2) * Recip(2) == 3//4
                # else
                #     @test_throws #...
                # end

                if T <: Real
                    @test (s < x ? (>=) : <)(Recip(s), Recip(x))
                    @test_broken (Recip(s) < 1 ? (>=) : <)(Recip(s), s)
                    # ^ need to define promote_type methods to get this to work
                end
            end
        end
    end

    # Use imprecise floats to make rounding errors more significant so we
    # can see where exact equalities fail and if it is expected
    # TODO: do repeated tests for a sufficient number of floats?
    T = Float16
    (a,b,c,x) = rand(T, 4)

    @testset "Primatives" begin
        wf = Primative(x -> x^2 + 1)
        k_val = [1,2,3]
        k = Const(k_val)

        @test wf(x) === wf.func(x)
        @test k(x) === k_val
    end
    @testset "Addition" begin
        @test +f1 === f1
        @test (f1 + f2)(x) == (f2 + f1)(x) == f1(x) + f2(x)
        @test (f1 + f2 + f3)(x) == f1(x) + f2(x) + f3(x) ≈ (f1 + (f2 + f3))(x)
        # XXX: order interchange can matter for floats?
    end

    @testset "Scaling" begin
        @test -f1 isa Neg{typeof(f1)}
        @test (-f1)(x) == -f1(x)

        @test (a*f1)(x) == (f1*a)(x) == a * f1(x)
        @test (a*b*f1)(x) == (a*(b*f1))(x) == a*b*f1(x)

        @test (f1/a)(x) == f1(x) / a
        @test_broken (a\f1)(x) == a \ f1(x)  # FIXME: needs adjoint of f1?
        @test (f1/a/b)(x) == f1(x) / (a*b)

        @test (a*f1/b)(x) == (a*(f1/b))(x) == (a/b) * f1(x) ≈ a * f1(x) / b
        # NOTE: The last equality is approximate (up to floating point error)
        # b/c the order of multiplication and division are interchanged as
        # the `Scale` expression is constructed.  The interchange occurs because
        # scalar coefficients on `f1` are eagerly evaluated, and we do not
        # construct nested `Scale` objects
        @test a*b*f1 isa Scale{T, typeof(f1)}
        @test a*f1/b isa Scale{T, typeof(f1)}
    end

    @testset "Linear Combination" begin
        # Subtraction is implemented as scaling by NegOne()
        @test (f1 - f2) isa Sub{typeof(f1), typeof(f2)}
        @test (f1 - f2)(x) == (-f2 + f1)(x) == -(f2 - f1)(x) == f1(x) - f2(x)

        @test (a*f1 + f2)(x) == (f2 + a*f1)(x) == a*f1(x) + f2(x)
        @test (a*f1 + b*f2)(x) == (b*f2 + a*f1)(x) == a*f1(x) + b*f2(x)
        @test (a*(f1 + f2))(x) == a*(f1(x) + f2(x))
        @test (a*(f1/b - c*f2) + f3)(x) == a*(f1(x)/b - c*f2(x)) + f3(x)
    end
end
