/*
See https://coin-or.github.io/Ipopt/INTERFACES.html for more information on the Ipopt C++ interface and the TNLP class.
The implementation contained here is done following the instruction there.
*/

#include "kemeny_nlp.hpp"

#include "IpTNLP.hpp"
#include <cassert>
#include <iostream>

using namespace Ipopt;

   /**@name Overloaded from TNLP */
   //@{
   /** Method to return some info about the NLP */
   bool kemeny_nlp::get_nlp_info(
      Index&          n,
      Index&          m,
      Index&          nnz_jac_g,
      Index&          nnz_h_lag,
      IndexStyleEnum& index_style
   ){
   
   // The number of variables is equal to the number of entries of the matrix
   n = this->P.n_elem;

   // The number of constraints can be computed from the number of variables
   // since we do not assume structure in this example
   m = 2*(this->P.n_rows) + n;

   // Number of nonzero entries of the Jacobian 
   nnz_jac_g = m*m;

   // Number of nonzero entries of the Hessian
   nnz_h_lag = n*n;   
   
   // use the C style indexing (0-based)
   index_style = TNLP::C_STYLE;

   return true;

   };

   /** Method to return the bounds for my problem */
   bool kemeny_nlp::get_bounds_info(
      Index   n,
      Number* x_l,
      Number* x_u,
      Index   m,
      Number* g_l,
      Number* g_u
   ){
      // The number of variables is equal to the number of entries of the matrix
      n = this->P.n_elem;

      // The lower bounds are minus the entries of the matrix P
      for (int i = 0; i < n; i++){
         x_l[i] = -this->P(i);
      }

      // There is no upper-bound on the variables
      for (int i = 0; i < n; i++){
         x_u[i] = 1e19;
      }

      // The number of constraints can be computed from the number of variables
      // since we do not assume structure in this example
      m = 2*(this->P.n_rows) + n;

      // The constraint is an equality constraint so we set every lower and upper bound to 0
      for (int i = 0; i < m; i++){
         g_l[i] = 0.0;
         g_u[i] = 0.0;
      }
   
	   return false;
   };

   /** Method to return the starting point for the algorithm */
   bool kemeny_nlp::get_starting_point(
      Index   n,
      bool    init_x,
      Number* x,
      bool    init_z,
      Number* z_L,
      Number* z_U,
      Index   m,
      bool    init_lambda,
      Number* lambda
   ){
      // The starting point is the zero matrix
      assert(init_x == true);
      assert(init_z == false);
      assert(init_lambda == false);

      for (int i = 0; i < n; i++){
         x[i] = 0.0;
      }
   
   	return false;
   };

   /** Method to return the objective value */
   bool kemeny_nlp::eval_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number&       obj_value
   ){
      int i;
      // The objective function is trace(I - P - Delta + 1*pi)^-1) + 0.5*norm(x,2)^2
      // where Delta is the n x n matrix built reshaping the vector x
      // and pi is the stationary distribution of P

      // Check if this is a new point, if so, compute the expensive inverse
      if (new_x){
         // Create the matrix Delta reshaping the C array x
         mat Delta = mat(x,n,n);
         // Compute the fundamental matrix of P
         this->INV = inv(eye<mat>(sqrt(n),sqrt(n)) - this->P + Delta + this->one*this->pi);
         // Compute the Frobenius norm of the matricization of x, i.e., the sum of the squares of the elements of x
         for (i = 0; i < n; i++){
            this->xnorm += x[i]*x[i];
         }
      }

      obj_value = trace(this->INV) + 0.5*this->xnorm;
   
   	return true;
   };

   /** Method to return the gradient of the objective */
   bool kemeny_nlp::eval_grad_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number*       grad_f
   ){
      int i;
      // The gradient of the objective function is the vectorization of the matrix transpose(INV*INV) + x
      // where INV is the inverse of the fundamental matrix of P

      // Check if this is a new point, if so, compute the expensive inverse
      if (new_x){
         // Create the matrix Delta reshaping the C array x
         mat Delta = mat(x,n,n);
         // Compute the fundamental matrix of P
         this->INV = inv(eye<mat>(sqrt(n),sqrt(n)) - this->P + Delta + this->one*this->pi);
      }
      // Compute the gradient of the objective function  
      mat grad;
      grad = this->INV*this->INV;
      vec grad_f_ = vectorise(grad.t()) + vec(x,n);
      #pragma omp parallel shared(grad_f) private(i)
      for(i = 0; i < n; i++){
         grad_f[i] = grad_f_(i);
      }   
   	return true;
   };

   /** Method to return the constraint residuals */
   bool kemeny_nlp::eval_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Number*       g
   ){
   
      // convert the vector x to an armadillo column vector
      vec x_ = vec(x,n);
      // Multiply this->C with x_ to get the constraint residuals
      vec g_ = this->C*x_;
      int i;
      #pragma omp parallel shared(g) private(i)
      for(i = 0; i < m; i++){
         g[i] = g_(i);
      }

   	return false;
   };

   /** Method to return:
    *   1) The structure of the jacobian (if "values" is NULL)
    *   2) The values of the jacobian (if "values" is not NULL)
    */
   bool kemeny_nlp::eval_jac_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Index         nele_jac,
      Index*        iRow,
      Index*        jCol,
      Number*       values
   ){

      // The Jacobian is the matrix C
      // The Jacobian is a sparse matrix so we need to return the structure of the Jacobian that is
      // stored in the Armadillo matrix this->C
      sp_mat::const_iterator it     = this->C.begin();
      sp_mat::const_iterator it_end = this->C.end();
      int i = 0;
      if (x == NULL){
         // Return the structure of the Jacobian
         for(; it != it_end; i++){
            iRow[i] = it.row();
            jCol[i] = it.col();
            values[i] = (double) NULL;
            i = i + 1;
         }
      } else {
         // Return the content of the Jacobian
         for(; it != it_end; i++){
            iRow[i] = it.row();
            jCol[i] = it.col();
            values[i] = (*it);
            i = i + 1;
         }
      }
   
   	return false;
   };

   /** Method to return:
    *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
    *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
    */
   bool kemeny_nlp::eval_h(
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
   ){
   
   	return false;
   };

   /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
   void kemeny_nlp::finalize_solution(
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
   ){};
   //@}

   bool kemeny_nlp::intermediate_callback(
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
   ){
   
   	return false;
   };
