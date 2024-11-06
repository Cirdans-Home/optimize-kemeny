#include "IpIpoptApplication.hpp"
#include "kemeny_nlp.hpp"
#include <armadillo>
#include <iostream>

using namespace Ipopt;
using namespace arma;

int main(int argv, char** argc){

  double Kor = 0.0; // Original Kemeny constant of the matrix P

  std::cout << "-------------------------------------------------------------------------" << std::endl;
  std::cout << "This is a simple example for the Kemeny problem using the C++ interface  " << std::endl;
  std::cout << "-------------------------------------------------------------------------" << std::endl;

  int n = 10; // Size of the random stochastic matrix
  mat P = randu<mat>(n,n); // Generate a random matrix
  P.each_row() /= sum(P,1).t(); // Normalize the matrix
  colvec one(n, fill::ones); // all ones vector

  std::cout << "The matrix P is: " << std::endl;
  P.print();

  // Compute the stationary distribution of P
  rowvec pi(n, fill::ones);
  rowvec pi_old(n, fill::zeros);
  while(norm(pi-pi_old,2) > 1e-13){
    pi_old = pi;
    pi = pi*P;
    pi = pi/sum(pi);
  }

  std::cout << "The stationary distribution is: " << std::endl;
  pi.print();
  std::cout << "Error on the stationary distribution is: " << norm(pi*P-pi,2) << std::endl;

  // Compute the fundamental matrix of P
  mat INV = inv(eye<mat>(n,n) - P + one*pi);
  Kor = trace(INV);

  std::cout << "The original Kemeny constant is: " << Kor << std::endl;

  std::cout << "-------------------------------------------------------------------------" << std::endl;
  std::cout << "Solving the Kemeny problem using Ipopt" << std::endl;
  std::cout << "-------------------------------------------------------------------------" << std::endl;

  // Create a new instance of your nlp
  SmartPtr<TNLP> mynlp = new kemeny_nlp(P,pi);

  // Create a new instance of IpoptApplication
  SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

  // Set some options
  app->Options()->SetNumericValue("tol", 3.82e-6);
  app->Options()->SetStringValue("mu_strategy", "adaptive");
  app->Options()->SetStringValue("output_file", "kemeny_ipopt.out");

  // Initialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = app->Initialize();
  if( status != Solve_Succeeded ){
    std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
    return (int) status;
  }

  if( status == Solve_Succeeded ){
    std::cout << std::endl << std::endl << "*** The problem has been solved!" << std::endl;
  }else{
    std::cout << std::endl << std::endl << "*** The solution has FAILED!" << std::endl;
  }
  
  return (int) status;
}
