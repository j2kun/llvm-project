#ring = #polynomial.ring<coefficientType = i32>

func.func @test_int_monomial() -> !polynomial.polynomial<ring=#ring> {
  %poly = polynomial.constant int<1 + 1 x**2> : !polynomial.polynomial<ring=#ring>
  %poly2 = polynomial.constant int<1x + 1 x**2 + 6 x**6> : !polynomial.polynomial<ring=#ring>
  %poly3 = polynomial.constant int<2x + 1 x**5> : !polynomial.polynomial<ring=#ring>
  return %poly : !polynomial.polynomial<ring=#ring>
}

// #float_ring = #polynomial.ring<coefficientType = f32>
//
// func.func @test_float_monomial() -> !polynomial.polynomial<ring=#float_ring> {
//   %poly = polynomial.constant float<1.0 + 1.0 x**2> : !polynomial.polynomial<ring=#float_ring>
//   %poly2 = polynomial.constant float<1.0x + 1.0 x**2 + 6.0x**6> : !polynomial.polynomial<ring=#float_ring>
//   %poly3 = polynomial.constant float<2.0x + 1.0 x**5> : !polynomial.polynomial<ring=#float_ring>
//   return %poly : !polynomial.polynomial<ring=#float_ring>
// }
