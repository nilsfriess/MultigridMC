#include <gtest/gtest.h>
#include <memory>
#include "lattice.hh"

/** @brief test lattice classes */
class Lattice2dTest : public ::testing::Test
{
protected:
    /* @brief initialise tests */
    void SetUp() override
    {
        int nx = 4;
        int ny = 3;
        lattice = std::make_shared<Lattice2d>(nx, ny);
    }
    /** Underlying lattice */
    std::shared_ptr<Lattice> lattice;
};

/** @brief check that index conversion works */
TEST_F(Lattice2dTest, TestLinear2Euclidean2d)
{
    Eigen::Vector2i idx = lattice->idx_linear2euclidean(6);
    Eigen::Vector2i cart_idx = {2, 1};
    EXPECT_EQ(idx, cart_idx);
}
/** @brief check that index conversion works */
TEST_F(Lattice2dTest, TestEuclidean2Linear2d)
{
    Eigen::Vector2i idx = {1, 2};
    unsigned int ell = lattice->idx_euclidean2linear(idx);
    EXPECT_EQ(ell, 9);
}

/** @brief check that shifting indices works */
TEST_F(Lattice2dTest, TestShift)
{
    Eigen::Vector2i shift_north = {0, +1};
    EXPECT_EQ(lattice->shift_index(3, shift_north), 7);
    Eigen::Vector2i shift_south = {0, -1};
    EXPECT_EQ(lattice->shift_index(3, shift_south), 11);
    Eigen::Vector2i shift_east = {+1, 0};
    EXPECT_EQ(lattice->shift_index(3, shift_east), 0);
    Eigen::Vector2i shift_west = {-1, 0};
    EXPECT_EQ(lattice->shift_index(3, shift_west), 2);
}