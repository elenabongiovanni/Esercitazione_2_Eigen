#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace std;
using namespace Eigen;

//controllo se è risolvibile
bool CheckSolvable(const MatrixXd& A)
{
    JacobiSVD<MatrixXd> svd(A);
    VectorXd singularValuesA = svd.singularValues();

    if( singularValuesA.minCoeff() < 1e-16)
    {
        cout << "System unsolvable" << endl;
        return false;
    }
    return true;
}

//risolvo con fattorizzazione PALU
VectorXd SolveSystemPALU(const MatrixXd& A, const VectorXd& b)
{
    VectorXd solveSystemPALU = A.fullPivLu().solve(b);
    return solveSystemPALU;
}


//risolvo con fattorizzazione QR
VectorXd SolveSystemQR(const MatrixXd& A,
                       const VectorXd& b)
{
    VectorXd solveSystemQR = A.colPivHouseholderQr().solve(b);
    return solveSystemQR;
}


//controllo correttezza delle soluzioni
void CheckSolution(const MatrixXd& A,
                   const VectorXd& b,
                   const VectorXd& solution,
                   double& errRelPALU,
                   double& errRelQR)
{
    errRelPALU = (SolveSystemPALU(A,b)-solution).norm()/solution.norm();
    errRelQR = (SolveSystemQR(A,b)-solution).norm()/solution.norm();
}


int main()
{
    Vector2d solution(-1.0e+0, -1.0e+00);
    Matrix2d A1;
    A1<<5.547001962252291e-01, -3.770900990025203e-02,8.320502943378437e-01,
        -9.992887623566787e-01;
    Vector2d v1 = {-5.169911863249772e-01, 1.672384680188350e-01};

    double errRelPALU1, errRelQR1;
    if(CheckSolvable(A1))
    {
        CheckSolution(A1, v1, solution, errRelPALU1, errRelQR1);
        cout << scientific << setprecision(16) << "1 - " << "errore relativo PALU: " << errRelPALU1 << "; errore relativo QR: " << errRelQR1 << endl;
    }

    Matrix2d A2;
    A2<<5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01,
        -8.324762492991313e-01;
    Vector2d v2 = {-6.394645785530173e-04, 4.259549612877223e-04};

    double errRelPALU2, errRelQR2;
    if(CheckSolvable(A2))
    {
        CheckSolution(A2, v2, solution, errRelPALU2, errRelQR2);
        cout << scientific << setprecision(16) << "2 - " << "errore relativo PALU: " << errRelPALU2 << "; errore relativo QR: " << errRelQR2 << endl;
    }

    Matrix2d A3;
    A3<<5.547001962252291e-01, -5.547001955851905e-01,8.320502943378437e-01,
        -8.320502947645361e-01;
    Vector2d v3 = {-6.400391328043042e-10, 4.266924591433963e-10};

    double errRelPALU3, errRelQR3;
    if(CheckSolvable(A3))
    {
        CheckSolution(A3, v3, solution, errRelPALU3, errRelQR3);
        cout << scientific << setprecision(16) << "3 - " << "errore relativo PALU: " << errRelPALU3 << "; errore relativo QR: " << errRelQR3 << endl;

    }

    return 0;
}

