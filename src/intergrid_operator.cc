#include "intergrid_operator.hh"

/** @file intergrid_operator.cc
 * @brief Implementation of intergrid_operator.hh
 */

/** @brief Coarsen a linear operator to the next-coarser level */
void AbstractIntergridOperator::coarsen_operator(const std::shared_ptr<AbstractLinearOperator> A, std::shared_ptr<AbstractLinearOperator> A_coarse) const
{
    // Construct offset sets
    std::vector<Eigen::VectorXi> A_stencil = A->get_stencil();
    std::vector<Eigen::VectorXi> A_coarse_stencil = A_coarse->get_stencil();
    std::vector<Eigen::VectorXi> P_stencil = get_stencil();
    // Comparison for two Eigen vectors
    auto cmpVectorXi = [](Eigen::VectorXi u, Eigen::VectorXi v)
    {
        for (int j = 0; j < u.size(); ++j)
        {
            if (u[j] < v[j])
            {
                return true;
            }
        }
        return false;
    };
    std::set<Eigen::VectorXi, decltype(cmpVectorXi)> A_stencil_set(A_stencil.begin(), A_stencil.end(), cmpVectorXi);
    std::vector<std::vector<std::pair<int, int>>> S_A;
    for (int k = 0; k < A_coarse_stencil.size(); ++k)
    { // Loop over all entries of the coarse stencil
        Eigen::VectorXi Delta = A_coarse_stencil[k];
        std::vector<std::pair<int, int>> S_A_Delta;
        for (int k1 = 0; k1 < P_stencil.size(); ++k1)
        { // Loop over all
            Eigen::VectorXi sigma = P_stencil[k1];
            for (int k2 = 0; k2 < P_stencil.size(); ++k2)
            {
                Eigen::VectorXi sigma_prime = P_stencil[k2];
                // Comparison function
                if (std::count(A_stencil_set.begin(), A_stencil_set.end(), 2 * Delta + sigma_prime - sigma))
                {
                    S_A_Delta.push_back(std::make_pair(k1, k2));
                }
            }
        }
        S_A.push_back(S_A_Delta);
    }

    std::shared_ptr<Lattice> lattice = A->get_lattice();
    std::shared_ptr<Lattice> coarse_lattice = A_coarse->get_lattice();
    int stencil_size = A_stencil.size();
    int coarse_stencil_size = A_coarse_stencil.size();
    const double *P_matrix = get_matrix();
    double *A_matrix = A->get_matrix();
    double *A_coarse_matrix = A_coarse->get_matrix();

    // Map from offset to corresponding index
    std::map<Eigen::VectorXi, int, decltype(cmpVectorXi)> offset_map(cmpVectorXi);
    for (int k = 0; k < stencil_size; ++k)
    {
        offset_map[A_stencil[k]] = k;
    }
    // Loop over unknowns on coarse lattice
    for (unsigned ell = 0; ell < coarse_lattice->M; ++ell)
    {
        for (int k = 0; k < S_A.size(); ++k)
        {
            Eigen::VectorXi Delta = A_coarse_stencil[k];
            auto S_A_Delta = S_A[k];
            for (auto sit = S_A_Delta.begin(); sit != S_A_Delta.end(); ++sit)
            {
                int k1 = sit->first;
                int k2 = sit->second;
                unsigned int ell_fine = lattice->idx_euclidean2linear(2 * coarse_lattice->idx_linear2euclidean(ell) + P_stencil[k1]);
                int k_fine = offset_map[2 * Delta + P_stencil[k2]];
                A_coarse_matrix[ell * coarse_stencil_size + k] = P_matrix[k1] * P_matrix[k2] * A_matrix[ell_fine * stencil_size + k_fine];
            }
        }
    }
};

/** @brief Matrix entries for averaging IntergridOperator in 2d */
template <>
const double BaseIntergridOperator2d<4, IntergridOperator2dAvg>::matrix[4] = {1.0, 1.0, 1.0, 1.0};
/** @brief Offsets in x-direction for averaging IntergridOperator in 2d */
template <>
const int BaseIntergridOperator2d<4, IntergridOperator2dAvg>::offset_x[4] = {0, 1, 0, 1};
/** @brief Offsets in y-direction for averaging IntergridOperator in 2d */
template <>
const int BaseIntergridOperator2d<4, IntergridOperator2dAvg>::offset_y[4] = {0, 0, 1, 1};

/** @brief Create a new instance */
IntergridOperator2dAvg::IntergridOperator2dAvg(const std::shared_ptr<Lattice2d> lattice_) : Base(lattice_)
{
    for (unsigned int j = 0; j < ny; ++j)
    {
        for (unsigned int i = 0; i < nx; ++i)
        {
            for (int k = 1; k < stencil_size; ++k)
            {
                unsigned int ell = ((j + offset_y[k] + ny) % ny) * nx + ((i + offset_x[k] + nx) % nx);
                colidx[ell * stencil_size + k] = ell;
            }
        }
    }
};