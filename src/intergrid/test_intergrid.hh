#ifndef TEST_INTERGRID_HH
#define TEST_INTERGRID_HH TEST_INTERGRID_HH

#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include "lattice/lattice1d.hh"
#include "lattice/lattice2d.hh"
#include "intergrid/intergrid_operator_avg.hh"
#include "intergrid/intergrid_operator_1dlinear.hh"
#include "intergrid/intergrid_operator_2dlinear.hh"
#include "linear_operator/diffusion_operator_2d.hh"

/** @brief fixture class for intergrid tests */
class IntergridTest : public ::testing::Test
{
protected:
    /* @brief initialise tests */
    void SetUp() override
    {
        // Lattice sizes - use anisotropic lattice
        unsigned int n = 8;
        lattice_1d = std::make_shared<Lattice1d>(n);
        coarse_lattice_1d = std::make_shared<Lattice1d>(n / 2);
        unsigned int nx = 4;
        unsigned int ny = 4;
        lattice_2d = std::make_shared<Lattice2d>(nx, ny);
        coarse_lattice_2d = std::make_shared<Lattice2d>(nx / 2, ny / 2);
        intergrid_operator_2davg = std::make_shared<IntergridOperatorAvg>(lattice_2d);
        intergrid_operator_1dlinear = std::make_shared<IntergridOperator1dLinear>(lattice_1d);
        intergrid_operator_2dlinear = std::make_shared<IntergridOperator2dLinear>(lattice_2d);
    }

    /** @brief return a sample state
     *
     * @param[in] lattice Underlying lattice
     * @param[in] random Initialise with random numbers?
     */
    Eigen::VectorXd get_state(const std::shared_ptr<Lattice> lattice,
                              const bool random = false)
    {
        unsigned int ndof = lattice->M;
        unsigned int seed = 1212417;
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        Eigen::VectorXd X(ndof);
        if (random)
        {
            for (unsigned int ell = 0; ell < ndof; ++ell)
            {
                X[ell] = dist(rng);
            }
        }
        else
        {
            X.setZero();
        }
        return X;
    }

    /** @brief Convert a fine level vertex index to a coarse level index in 2d */
    unsigned int fine2coarse_idx_1d(const unsigned int ell)
    {
        unsigned int n = lattice_1d->n;
        int i = ell % n;
        assert(i % 2 == 0);
        return (i / 2);
    }

    /** @brief Convert a fine level vertex index to a coarse level index in 2d */
    unsigned int fine2coarse_idx_2d(const unsigned int ell)
    {
        unsigned int nx = lattice_2d->nx;
        int i = ell % nx;
        int j = ell / nx;
        assert(i % 2 == 0);
        assert(j % 2 == 0);
        return (nx / 2) * (j / 2) + (i / 2);
    }

    /** @brief underlying 1d lattice */
    std::shared_ptr<Lattice1d> lattice_1d;
    /** @brief underlying 1d coarse lattice */
    std::shared_ptr<Lattice1d> coarse_lattice_1d;
    /** @brief underlying 2d lattice */
    std::shared_ptr<Lattice2d> lattice_2d;
    /** @brief underlying 2d coarse lattice */
    std::shared_ptr<Lattice2d> coarse_lattice_2d;
    /** @brief intergrid operator for averaging */
    std::shared_ptr<IntergridOperatorAvg> intergrid_operator_2davg;
    /** @brief intergrid operator for linear interpolation in 1d*/
    std::shared_ptr<IntergridOperator1dLinear> intergrid_operator_1dlinear;
    /** @brief intergrid operator for linear interpolation in 2d */
    std::shared_ptr<IntergridOperator2dLinear> intergrid_operator_2dlinear;
};

/** @brief check that prolongating then restricting will return the same field up to a factor */
TEST_F(IntergridTest, TestProlongRestrict2dAvg)
{
    // initial coarse level state
    Eigen::VectorXd X_coarse = get_state(coarse_lattice_2d, true);
    // prolongated state
    Eigen::VectorXd X_prol = get_state(lattice_2d, false);
    // prolongate and restricted state
    Eigen::VectorXd X_prol_restr = get_state(coarse_lattice_2d, false);
    intergrid_operator_2davg->prolongate_add(X_coarse, X_prol);
    intergrid_operator_2davg->restrict(X_prol, X_prol_restr);
    double tolerance = 1.E-12;
    EXPECT_NEAR((X_prol_restr - 4. * X_coarse).norm(), 0.0, tolerance);
}

/** @brief check that prolongating a field in 1d will return the same result as manually interpolating */
TEST_F(IntergridTest, TestProlongRestrict1dLinear)
{
    // initial coarse level state
    Eigen::VectorXd X_coarse = get_state(coarse_lattice_1d, true);
    // prolongated state
    Eigen::VectorXd X_prol = get_state(lattice_1d, false);
    // prolongate and restricted state
    intergrid_operator_1dlinear->prolongate_add(X_coarse, X_prol);
    // Manually interpolate linearly
    Eigen::VectorXd X_linear = get_state(lattice_1d, false);
    Eigen::VectorXi shift_right(1);
    shift_right[0] = +1;
    Eigen::VectorXi shift_left(1);
    shift_left[0] = -1;

    for (int i = 0; i < lattice_1d->n; ++i)
    {
        unsigned int ell = i;
        if (i % 2 == 0)
        {
            // Copy coarse level point
            unsigned int ell_coarse = fine2coarse_idx_1d(ell);
            X_linear[ell] = X_coarse[ell_coarse];
        }
        else
        {
            // horizontal facet
            unsigned int ell_right = fine2coarse_idx_1d(lattice_1d->shift_index(ell, shift_right));
            unsigned int ell_left = fine2coarse_idx_1d(lattice_1d->shift_index(ell, shift_left));
            X_linear[ell] = 0.5 * (X_coarse[ell_right] +
                                   X_coarse[ell_left]);
        }
    }

    double tolerance = 1.E-12;
    EXPECT_NEAR((2 * X_prol - X_linear).norm(), 0.0, tolerance);
}

/** @brief check that prolongating a field in 2d will return the same result as manually interpolating */
TEST_F(IntergridTest, TestProlongRestrict2dLinear)
{
    // initial coarse level state
    Eigen::VectorXd X_coarse = get_state(coarse_lattice_2d, true);
    // prolongated state
    Eigen::VectorXd X_prol = get_state(lattice_2d, false);
    // prolongate and restricted state
    intergrid_operator_2dlinear->prolongate_add(X_coarse, X_prol);
    // Manually interpolate linearly
    Eigen::VectorXd X_linear = get_state(lattice_2d, false);
    Eigen::Vector2i shift_north = {0, +1};
    Eigen::Vector2i shift_south = {0, -1};
    Eigen::Vector2i shift_east = {+1, 0};
    Eigen::Vector2i shift_west = {-1, 0};
    Eigen::Vector2i shift_north_east = {+1, +1};
    Eigen::Vector2i shift_south_east = {+1, -1};
    Eigen::Vector2i shift_north_west = {-1, +1};
    Eigen::Vector2i shift_south_west = {-1, -1};
    unsigned int nx = lattice_2d->nx;
    unsigned int ny = lattice_2d->ny;
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            unsigned int ell = nx * j + i;
            if ((i % 2 == 0) && (j % 2 == 0))
            {
                // Copy coarse level point
                unsigned int ell_coarse = fine2coarse_idx_2d(ell);
                X_linear[ell] = X_coarse[ell_coarse];
            }
            if ((i % 2 == 1) && (j % 2 == 1))
            {
                // centre point
                unsigned int ell_ne = fine2coarse_idx_2d(lattice_2d->shift_index(ell, shift_north_east));
                unsigned int ell_se = fine2coarse_idx_2d(lattice_2d->shift_index(ell, shift_south_east));
                unsigned int ell_nw = fine2coarse_idx_2d(lattice_2d->shift_index(ell, shift_north_west));
                unsigned int ell_sw = fine2coarse_idx_2d(lattice_2d->shift_index(ell, shift_south_west));
                X_linear[ell] = 0.25 * (X_coarse[ell_ne] +
                                        X_coarse[ell_se] +
                                        X_coarse[ell_nw] +
                                        X_coarse[ell_sw]);
            }
            if ((i % 2 == 1) && (j % 2 == 0))
            {
                // horizontal facet
                unsigned int ell_e = fine2coarse_idx_2d(lattice_2d->shift_index(ell, shift_east));
                unsigned int ell_w = fine2coarse_idx_2d(lattice_2d->shift_index(ell, shift_west));
                X_linear[ell] = 0.5 * (X_coarse[ell_e] +
                                       X_coarse[ell_w]);
            }
            if ((i % 2 == 0) && (j % 2 == 1))
            {
                // vertical facet
                unsigned int ell_n = fine2coarse_idx_2d(lattice_2d->shift_index(ell, shift_north));
                unsigned int ell_s = fine2coarse_idx_2d(lattice_2d->shift_index(ell, shift_south));
                X_linear[ell] = 0.5 * (X_coarse[ell_n] +
                                       X_coarse[ell_s]);
            }
        }
    }

    double tolerance = 1.E-12;
    EXPECT_NEAR((4 * X_prol - X_linear).norm(), 0.0, tolerance);
}

/** @brief check that coarsening the operator works
 *
 * Coarsening the shifted Laplace operator should result in rescaling the
 * second- order and zero- order terms by constant factors
 */
TEST_F(IntergridTest, TestCoarsenOperator2d)
{
    DiffusionOperator2d linear_operator(lattice_2d, 1.0, 0.0, 1.0, 0.0);
    DiffusionOperator2d coarse_operator(coarse_lattice_2d, 8.0, 0.0, 4.0, 0.0);
    LinearOperator coarsened_operator = linear_operator.coarsen(intergrid_operator_2davg);
    const double tolerance = 1.E-12;
    EXPECT_NEAR((0.25 * coarse_operator.get_sparse() - coarsened_operator.get_sparse()).norm(), 0.0, tolerance);
}

#endif // TEST_INTERGRID_HH