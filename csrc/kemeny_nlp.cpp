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
   n = P.n_elem;

   // The number of constraints can be computed from the number of variables
   // since we do not assume structure in this example
   m = 2*P.n_rows + n;

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
   
   	return false;
   };

   /** Method to return the objective value */
   bool kemeny_nlp::eval_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number&       obj_value
   ){
   
   	return false;
   };

   /** Method to return the gradient of the objective */
   bool kemeny_nlp::eval_grad_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number*       grad_f
   ){
   
   	return false;
   };

   /** Method to return the constraint residuals */
   bool kemeny_nlp::eval_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Number*       g
   ){
   
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
