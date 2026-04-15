#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXd A(3, 3);
    A << 1, 2, 7,
         3, 4, 8,
         5, 6, 9;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::MatrixXd S = svd.singularValues();

    std::cout << "U:\n" << U << "\n\n";
    std::cout << "S:\n" << S << "\n\n";
    std::cout << "V:\n" << V << "\n";

    return 0;
}
