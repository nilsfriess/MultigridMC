#include <gtest/gtest.h>
#include "config.h"
#include "auxilliary/test_cholesky_wrapper.hh"
#include "auxilliary/test_statistics.hh"
#include "lattice/test_lattice.hh"
#include "intergrid/test_intergrid.hh"
#include "smoother/test_smoother.hh"
#include "solver/test_solver.hh"
#include "sampler/test_sampler.hh"

int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  if (thorough_testing)
  {
    std::cout << std::endl;
    std::cout << "+-----------------------------------------------------------+" << std::endl;
    std::cout << "! Thorough testing enabled, tests might take longer to run. !" << std::endl;
    std::cout << "+-----------------------------------------------------------+" << std::endl;
    std::cout << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "+----------------------------+" << std::endl;
    std::cout << "! Running basic, fast tests. !" << std::endl;
    std::cout << "+----------------------------+" << std::endl;
    std::cout << std::endl;
  }
  return RUN_ALL_TESTS();
}
