#ideal = #polynomial.int_polynomial<1 + x**10>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967296 : i64, polynomialModulus=#ideal>

func.func @test_monomial() -> !polynomial.polynomial<ring=#ring> {
  %poly = polynomial.constant int<1 + x**2> : !polynomial.polynomial<ring=#ring>
  return %poly : !polynomial.polynomial<ring=#ring>
}
