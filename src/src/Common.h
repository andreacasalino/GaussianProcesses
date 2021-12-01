/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <Eigen/Core>

namespace gauss::gp {
    struct MatrixIndices {
        const Eigen::Index start;
        const Eigen::Index end;
    };

    void set_matrix_portion(Eigen::MatrixXd& recipient, const Eigen::MatrixXd& portion, 
                            const MatrixIndices& portion_rows, const MatrixIndices& portion_cols);
}
