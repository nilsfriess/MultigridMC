#include <gtest/gtest.h>
#include "test_lattice.hh"
#include "test_intergrid.hh"

int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
