kt2n_add_gtest(kt2n_render_tests LABEL cpu SOURCES render_state_test.cc LIBS kt2n_render)
# The portable 128-bit integer of the PCG64 state (MSVC's path; selected on gcc/clang in this test only).
kt2n_add_gtest(kt2n_u128_tests LABEL cpu SOURCES u128_test.cc)
