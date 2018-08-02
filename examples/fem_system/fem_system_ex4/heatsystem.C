#include "heatsystem.h"

#include "libmesh/parsed_function.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/dof_map.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fem_context.h"
#include "libmesh/getpot.h"
#include "libmesh/mesh.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/zero_function.h"
#include "libmesh/elem.h"

using namespace libMesh;

void HeatSystem::init_data ()
{
  T_var = this->add_variable("T", static_cast<Order>(_fe_order),
                             Utility::string_to_enum<FEFamily>(_fe_family));

  const unsigned int dim = this->get_mesh().mesh_dimension();

  std::vector<unsigned int> T_only(1, T_var);
  ZeroFunction<Number> zero;

  // Add dirichlet boundaries on all sides
  boundary_id_type all_ids[6] = {0, 1, 2, 3, 4, 5};

  GetPot infile("fem_system_ex4.in");
  const std::string  mesh_name = infile("mesh_name", "DIE!");
  // trelis doesnt like 0 counting sideset ids
  if ( !mesh_name.empty() && (mesh_name != "DIE!") )
    for (int i = 0; i < 6; i++)
      all_ids[i] += 1;

  // Adjust BC for circle/pacman meshes since they have less BC..
  // TODO: this is pretty janky
  if ( mesh_name.find("circle") != std::string::npos )
    std::fill(all_ids, all_ids+6 ,1 );
  else if ( mesh_name.find("pacman") != std::string::npos )
    std::fill(all_ids, all_ids+6 ,2323 );

  std::set<boundary_id_type> bndrys(all_ids, all_ids+(dim*2));


  const std::string exact_str = (dim == 2) ?
    "sin(pi*x)*sin(pi*y)" : "sin(pi*x)*sin(pi*y)*sin(pi*z)";
  ParsedFunction<Number> exact_func(exact_str);

  // Most DirichletBoundary users will want to supply a "locally
  // indexed" functor
  if ( mesh_name == "quad_tri_circle.e")
    this->get_dof_map().add_dirichlet_boundary
      (DirichletBoundary (bndrys, T_only, exact_func, LOCAL_VARIABLE_ORDER));
  else
    this->get_dof_map().add_dirichlet_boundary
      (DirichletBoundary (bndrys, T_only, zero, LOCAL_VARIABLE_ORDER));



  // Do the parent's initialization after variables are defined
  FEMSystem::init_data();

}

void HeatSystem::init_context(DiffContext & context)
{
  FEMContext & c = cast_ref<FEMContext &>(context);

  unsigned char dim = c.get_dim();

  FEBase * fe = nullptr;

  c.get_element_fe(T_var, fe, dim);

  fe->get_JxW();  // For integration
  fe->get_dphi(); // For bilinear form
  fe->get_xyz();  // For forcing
  fe->get_phi();  // For forcing

  FEMSystem::init_context(context);
}


bool HeatSystem::element_time_derivative (bool request_jacobian,
                                          DiffContext & context)
{

  FEMContext & c = cast_ref<FEMContext &>(context);

  // First we get some references to cell-specific data that
  // will be used to assemble the linear system.
  const unsigned short dim = c.get_elem().dim();
  FEBase * fe = nullptr;
  c.get_element_fe(T_var, fe, dim);

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> & JxW = fe->get_JxW();

  const std::vector<Point> & xyz = fe->get_xyz();

  const std::vector<std::vector<Real>> & phi = fe->get_phi();

  const std::vector<std::vector<RealGradient>> & dphi = fe->get_dphi();

  // The number of local degrees of freedom in each variable
  const unsigned int n_T_dofs = c.n_dof_indices(T_var);

  // The subvectors and submatrices we need to fill:
  DenseSubMatrix<Number> & K = c.get_elem_jacobian(T_var, T_var);
  DenseSubVector<Number> & F = c.get_elem_residual(T_var);

  // Now we will build the element Jacobian and residual.
  // Constructing the residual requires the solution.  This must be
  // calculated at each quadrature point by summing the
  // solution degree-of-freedom values by the appropriate
  // weight functions.

  unsigned int n_qpoints = c.get_element_qrule().n_points();

  for (unsigned int qp=0; qp != n_qpoints; qp++)
    {

      const Point & p = xyz[qp];

      Gradient grad_T = c.interior_gradient(T_var, qp);

      // solution + laplacian depend on problem dimension
      const Number u_exact = (dim == 2) ?
        std::sin(libMesh::pi*p(0)) * std::sin(libMesh::pi*p(1)) :
        std::sin(libMesh::pi*p(0)) * std::sin(libMesh::pi*p(1)) *
        std::sin(libMesh::pi*p(2));

      const Number forcing = - u_exact * (dim * libMesh::pi * libMesh::pi);

      //std::cout << "force: " << forcing << "\n";

      const Number JxWxNK = JxW[qp];

      for (unsigned int i=0; i != n_T_dofs; i++)
        F(i) += JxWxNK * ( grad_T*dphi[i][qp] + forcing * phi[i][qp]);

      if (request_jacobian)
        {
         for (unsigned int i=0; i != n_T_dofs; i++)
            for (unsigned int j=0; j != n_T_dofs; ++j)
              K(i,j) += JxWxNK * (dphi[i][qp] * dphi[j][qp]);
        }
} // end of the quadrature point qp-loop

  return request_jacobian;
}
