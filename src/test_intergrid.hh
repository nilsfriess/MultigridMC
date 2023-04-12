#ifndef TEST_INTERGRID_HH
#define TEST_INTERGRID_HH TEST_INTERGRID_HH

#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include "intergrid_operator.hh"
#include "diffusion_operator_2d.hh"

/** @brief fixture class for intergrid tests */
class IntergridTest : public ::testing::Test
{
protected:
    /* @brief initialise tests */
    void SetUp() override
    {
        // Lattice sizes - use anisotropic lattice
        unsigned int nx = 4;
        unsigned int ny = 4;
        lattice = std::make_shared<Lattice2d>(nx, ny);
        coarse_lattice = std::make_shared<Lattice2d>(nx / 2, ny / 2);
        intergrid_operator_2davg = std::make_shared<IntergridOperator2dAvg>(lattice);
        intergrid_operator_2dlinear = std::make_shared<IntergridOperator2dLinear>(lattice);
    }

    /** @brief return a sample state
     *
     * @param[in] coarse Generate state on coarse level?
     * @param[in] random Initialise with random numbers?
     */
    std::shared_ptr<SampleState> get_state(const bool coarse = false, const bool random = false)
    {
        unsigned int ndof = coarse ? coarse_lattice->M : lattice->M;
        unsigned int seed = 1212417;
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::shared_ptr<SampleState> X = std::make_shared<SampleState>(ndof);
        if (random)
        {
            for (unsigned int ell = 0; ell < ndof; ++ell)
            {
                X->data[ell] = dist(rng);
            }
        }
        return X;
    }

    /** @brief Convert a fine level vertex index to a coarse level index */
    unsigned int fine2coarse_idx(const unsigned int ell)
    {
        unsigned int nx = lattice->nx;
        int i = ell % nx;
        int j = ell / nx;
        assert(i % 2 == 0);
        assert(j % 2 == 0);
        return (nx / 2) * (j / 2) + (i / 2);
    }

    /** @brief underlying lattice */
    std::shared_ptr<Lattice2d> lattice;
    /** @brief underlying coarse lattice */
    std::shared_ptr<Lattice2d> coarse_lattice;
    /** @brief intergrid operator for averaging */
    std::shared_ptr<IntergridOperator2dAvg> intergrid_operator_2davg;
    /** @brief intergrid operator for linear interpolation */
    std::shared_ptr<IntergridOperator2dLinear> intergrid_operator_2dlinear;
};

/** @brief check that prolongating then restricting will return the same field up to a factor */
TEST_F(IntergridTest, TestProlongRestrict2dAvg)
{
    // initial coarse level state
    std::shared_ptr<SampleState> X_coarse = get_state(true, true);
    // prolongated state
    std::shared_ptr<SampleState> X_prol = get_state(false, false);
    // prolongate and restricted state
    std::shared_ptr<SampleState> X_prol_restr = get_state(true, false);
    intergrid_operator_2davg->prolongate_add(X_coarse, X_prol);
    intergrid_operator_2davg->restrict(X_prol, X_prol_restr);
    double tolerance = 1.E-12;
    EXPECT_NEAR((X_prol_restr->data - 4. * X_coarse->data).norm(), 0.0, tolerance);
}

/** @brief check that prolongating a field will return the same result as manually interpolating */
TEST_F(IntergridTest, TestProlongRestrict2dLinear)
{
    // initial coarse level state
    std::shared_ptr<SampleState> X_coarse = get_state(true, true);
    // prolongated state
    std::shared_ptr<SampleState> X_prol = get_state(false, false);
    // prolongate and restricted state
    intergrid_operator_2dlinear->prolongate_add(X_coarse, X_prol);
    // Manually interpolate linearly
    std::shared_ptr<SampleState> X_linear = get_state(false, false);
    Eigen::Vector2i shift_north = {0, +1};
    Eigen::Vector2i shift_south = {0, -1};
    Eigen::Vector2i shift_east = {+1, 0};
    Eigen::Vector2i shift_west = {-1, 0};
    Eigen::Vector2i shift_north_east = {+1, +1};
    Eigen::Vector2i shift_south_east = {+1, -1};
    Eigen::Vector2i shift_north_west = {-1, +1};
    Eigen::Vector2i shift_south_west = {-1, -1};
    unsigned int nx = lattice->nx;
    unsigned int ny = lattice->ny;
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            unsigned int ell = nx * j + i;
            if ((i % 2 == 0) && (j % 2 == 0))
            {
                // Copy coarse level point
                unsigned int ell_coarse = fine2coarse_idx(ell);
                X_linear->data[ell] = X_coarse->data[ell_coarse];
            }
            if ((i % 2 == 1) && (j % 2 == 1))
            {
                // centre point
                unsigned int ell_ne = fine2coarse_idx(lattice->shift_index(ell, shift_north_east));
                unsigned int ell_se = fine2coarse_idx(lattice->shift_index(ell, shift_south_east));
                unsigned int ell_nw = fine2coarse_idx(lattice->shift_index(ell, shift_north_west));
                unsigned int ell_sw = fine2coarse_idx(lattice->shift_index(ell, shift_south_west));
                X_linear->data[ell] = 0.25 * (X_coarse->data[ell_ne] +
                                              X_coarse->data[ell_se] +
                                              X_coarse->data[ell_nw] +
                                              X_coarse->data[ell_sw]);
            }
            if ((i % 2 == 1) && (j % 2 == 0))
            {
                // horizontal facet
                unsigned int ell_e = fine2coarse_idx(lattice->shift_index(ell, shift_east));
                unsigned int ell_w = fine2coarse_idx(lattice->shift_index(ell, shift_west));
                X_linear->data[ell] = 0.5 * (X_coarse->data[ell_e] +
                                             X_coarse->data[ell_w]);
            }
            if ((i % 2 == 0) && (j % 2 == 1))
            {
                // vertical facet
                unsigned int ell_n = fine2coarse_idx(lattice->shift_index(ell, shift_north));
                unsigned int ell_s = fine2coarse_idx(lattice->shift_index(ell, shift_south));
                X_linear->data[ell] = 0.5 * (X_coarse->data[ell_n] +
                                             X_coarse->data[ell_s]);
            }
        }
    }

    double tolerance = 1.E-12;
    EXPECT_NEAR((4 * X_prol->data - X_linear->data).norm(), 0.0, tolerance);
}

/** @brief check that coarsening the operator works
 *
 * Coarsening the shifted Laplace operator should result in rescaling the
 * second- order and zero- order terms by constant factors
 */
TEST_F(IntergridTest, TestCoarsenOperator)
{
    DiffusionOperator2d linear_operator(lattice, 1.0, 0.0, 1.0, 0.0);
    DiffusionOperator2d coarse_operator(coarse_lattice, 8.0, 0.0, 4.0, 0.0);
    LinearOperator coarsened_operator = intergrid_operator_2davg->coarsen_operator(linear_operator);
    const double tolerance = 1.E-12;
    EXPECT_NEAR((coarse_operator.as_sparse() - coarsened_operator.as_sparse()).norm(), 0.0, tolerance);
}

#endif // TEST_INTERGRID_HH