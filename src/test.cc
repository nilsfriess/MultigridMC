#include <gtest/gtest.h>
#include "lattice/test_lattice.hh"
#include "intergrid/test_intergrid.hh"
#include "smoother/test_smoother.hh"
#include "solver/test_solver.hh"
#include "sampler/test_sampler.hh"

int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
