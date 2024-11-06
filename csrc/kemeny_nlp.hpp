#ifndef __KEMENY_NLP_HPP__
#define __KEMENY_NLP_HPP__

#include "IpTNLP.hpp"
#include <armadillo>

using namespace Ipopt;
using namespace arma;

/** C++ NLP interface for the Kemeny problem.
 *
 */
class kemeny_nlp: public TNLP
{
public:
   /** Base row stochastic matrix **/
   mat P;   // Contains the base row stochastic matrix
   mat INV; // Contains the fundamental matrix
   mat G;   // Contains the latest computed gradient of the objective function (matrix form)
   rowvec pi;  // Contains the stationary distribution
   vec one; // Contains the all ones vector
   Number xnorm; // Contains the norm of the vector x
   SpMat<double> C; // Contains the sparse constraint matrix

   /** Constructor */
   kemeny_nlp(
      mat P, rowvec pi, bool printiterate = false   /**< whether to print the iterate at each iteration */
   ){
      this->P = P;   // Initialize the matrix P with the input matrix
      this->pi = pi; // Initialize the stationary distribution with the input vector
      // check if the stationary distribution is stationary
      double error = norm(this->pi*this->P-this->pi,2);
      if(error > 1e-13){
         std::cerr << "The input stationary distribution is not stationary" << std::endl;
         exit(1);
      }
      this->printiterate_ = printiterate;
      this->one = vec(P.n_rows, fill::ones);
      // The matrix C is built with the constraints of the problem
      // C = [kron(one,I);kron(I,pi');kron(I,Dpi) - kron(Dpi,I)*T];
      // where I is the sparse identity matrix, Dpi is the sparse matrix with pi on the diagonal
      // and T is the sparse matrix mapping the vectorization of a matrix to the matrix transpose
      int n = P.n_rows;
      SpMat<double> I = speye(n,n); // sparse identity matrix
      SpMat<double> Dpi(n,n); // sparse matrix with pi on the diagonal
      Dpi.diag() = pi;     // set the diagonal of Dpi to pi
      SpMat<double> matone(1,n); // sparse matrix with all ones
      SpMat<double> matpi(n,1); // sparse matrix with pi as a row
      matone = one.t(); // set the matrix matone to all ones
      matpi = pi.t(); // set the matrix matpi to pi as a row
      SpMat<double> T = build_sparseT(n);
      this->C = join_cols(join_cols(kron(matone.t(),I),kron(I,matpi)),kron(I,Dpi) - kron(Dpi,I)*T);
   };

   /** Destructor */
   virtual ~kemeny_nlp(){
      // clean up the armadillo matrices and vectors P, INV, G and pi

   };

   /**@name Overloaded from TNLP */
   //@{
   /** Method to return some info about the NLP */
   virtual bool get_nlp_info(
      Index&          n,
      Index&          m,
      Index&          nnz_jac_g,
      Index&          nnz_h_lag,
      IndexStyleEnum& index_style
   );

   /** Method to return the bounds for my problem */
   virtual bool get_bounds_info(
      Index   n,
      Number* x_l,
      Number* x_u,
      Index   m,
      Number* g_l,
      Number* g_u
   );

   /** Method to return the starting point for the algorithm */
   virtual bool get_starting_point(
      Index   n,
      bool    init_x,
      Number* x,
      bool    init_z,
      Number* z_L,
      Number* z_U,
      Index   m,
      bool    init_lambda,
      Number* lambda
   );

   /** Method to return the objective value */
   virtual bool eval_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number&       obj_value
   );

   /** Method to return the gradient of the objective */
   virtual bool eval_grad_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number*       grad_f
   );

   /** Method to return the constraint residuals */
   virtual bool eval_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Number*       g
   );

   /** Method to return:
    *   1) The structure of the jacobian (if "values" is NULL)
    *   2) The values of the jacobian (if "values" is not NULL)
    */
   virtual bool eval_jac_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Index         nele_jac,
      Index*        iRow,
      Index*        jCol,
      Number*       values
   );

   /** Method to return:
    *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
    *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
    */
   virtual bool eval_h(
      Index         n,
      const Number* x,
      bool          new_x,
      Number        obj_factor,
      Index         m,
      const Number* lambda,
      bool          new_lambda,
      Index         nele_hess,
      Index*        iRow,
      Index*        jCol,
      Number*       values
   );

   /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
   virtual void finalize_solution(
      SolverReturn               status,
      Index                      n,
      const Number*              x,
      const Number*              z_L,
      const Number*              z_U,
      Index                      m,
      const Number*              g,
      const Number*              lambda,
      Number                     obj_value,
      const IpoptData*           ip_data,
      IpoptCalculatedQuantities* ip_cq
   );
   //@}

   bool intermediate_callback(
      AlgorithmMode              mode,
      Index                      iter,
      Number                     obj_value,
      Number                     inf_pr,
      Number                     inf_du,
      Number                     mu,
      Number                     d_norm,
      Number                     regularization_size,
      Number                     alpha_du,
      Number                     alpha_pr,
      Index                      ls_trials,
      const IpoptData*           ip_data,
      IpoptCalculatedQuantities* ip_cq
   );

private:
   /** whether to print iterate to stdout in intermediate_callback() */
   bool printiterate_;

   /**@name Methods to block default compiler methods.
    *
    * The compiler automatically generates the following three methods.
    *  Since the default compiler implementation is generally not what
    *  you want (for all but the most simple classes), we usually
    *  put the declarations of these methods in the private section
    *  and never implement them. This prevents the compiler from
    *  implementing an incorrect "default" behavior without us
    *  knowing. (See Scott Meyers book, "Effective C++")
    */
   //@{
   kemeny_nlp(
      const kemeny_nlp&
   );

   kemeny_nlp& operator=(
      const kemeny_nlp&
   );
   //@}

   /** Method to build the sparse matrix T mapping the vectorization of a matrix to the matrix transpose */
   SpMat<double> build_sparseT(int n) {
      // Initialize arrays to store row, column indices and values
      Mat<uword> locations(2, n * n);
      vec values(n * n, fill::ones);

      // Loop over each element of the matrix P
      for (int i = 0; i < n; ++i) {
         for (int j = 0; j < n; ++j) {
               // Linear index for element (i, j) in matrix P(:)
               uword row = j * n + i;
               // Linear index for element (i, j) in matrix reshape(P',n^2,1)
               uword col = i * n + j;
               // Store indices in sparse matrix
               locations(0, row) = row;
               locations(1, row) = col;
         }
      }
      // Build the sparse matrix T using the template
      // sp_mat(locations, values, n_rows, n_cols, sort_locations = true, check_for_zeros = true)
      SpMat<double> T(locations, values, n * n, n * n, true, false);

      return T;
   }

};

#endif
