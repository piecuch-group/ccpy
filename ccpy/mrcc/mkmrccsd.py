# The Mk-MRCC residual equations are
# R_K(p) = < K(p) | (H exp(T(p)))_C | 0(p) > * c(p)
#           + \sum_{q /= p} < K(p) | exp(-T(p)) * exp(T(q)) | 0(p) > < 0(p) | (H exp(T(p)))_C | 0(q) > c(q),
# where we have assumed a complete model space (CMS), thus ensuring the rigorous size-extensivity of the equations. This
# can be seen by considering the fact that all reference determinant | 0(p) > = E_K | 0(q) >, where E_K involves orbital
# indices that are entirely contained within the active space. As a result, all cluster operators T(p) must have at
# least one particle or hole index outside of the active space