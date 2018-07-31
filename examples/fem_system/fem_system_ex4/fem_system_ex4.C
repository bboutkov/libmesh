// The libMesh Finite Element Library.
// Copyright (C) 2002-2019 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



// <h1>FEMSystem Example 4 - Mixed-dimension heat transfer equation
// with FEMSystem</h1>
// \author Roy Stogner
// \date 2015
//
// This example shows how elements of multiple dimensions can be
// linked and computed upon simultaneously within the
// DifferentiableSystem class framework

// C++ includes
#include <iomanip>

// Basic include files
#include "libmesh/elem.h"
#include "libmesh/equation_systems.h"
#include "libmesh/error_vector.h"
#include "libmesh/exact_solution.h"
#include "libmesh/getpot.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/mesh.h"
#include "libmesh/partitioner.h"
#include "libmesh/parallel_mesh.h"
#include "libmesh/serial_mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/parsed_function.h"
#include "libmesh/auto_ptr.h" // libmesh_make_unique
#include "libmesh/enum_solver_package.h"
#include "libmesh/enum_norm_type.h"

// The systems and solvers we may use
#include "heatsystem.h"
#include "libmesh/diff_solver.h"
#include "libmesh/petsc_diff_solver.h"
#include "libmesh/euler_solver.h"
#include "libmesh/steady_solver.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// The main program.
int main (int argc, char ** argv)
{
  // Initialize libMesh.
  LibMeshInit init (argc, argv);

  // This example requires a linear solver package.
  libmesh_example_requires(libMesh::default_solver_package() != INVALID_SOLVER_PACKAGE,
                           "--enable-petsc, --enable-trilinos, or --enable-eigen");

 // Parse the input file
  GetPot infile("fem_system_ex4.in");

  // Read in parameters from the input file
  const unsigned int coarsegridsize    = infile("coarsegridsize", 25);
  const unsigned int coarserefinements = infile("coarserefinements", 3);
  const unsigned int dim               = infile("dimension", 2);
  const std::string  mesh_name         = infile("mesh_name", "DIE!");
  const std::string  mesh_type         = infile("mesh_type", "distributed");
  const std::string  write_solution    = infile("write_solution", "false");

  // Skip higher-dimensional examples on a lower-dimensional libMesh build
  libmesh_example_requires(dim <= LIBMESH_DIM, "2D/3D support");

  // We have only defined 2 and 3 dimensional problems
  libmesh_assert (dim == 2 || dim == 3);

  // Create UnstructuredMesh object (defaults to dimension 1).
  std::shared_ptr<libMesh::UnstructuredMesh> mesh;

  // Parse mesh type from input file
  if (mesh_type == "distributed")
    {
      out << "Building a distributed mesh!" << std::endl;
      mesh.reset(new libMesh::ParallelMesh(init.comm()));
    }
  else if (mesh_type == "replicated")
    {
      out << "Building a replicated mesh!" << std::endl;
      mesh.reset(new libMesh::ReplicatedMesh(init.comm()));
    }
  else
    {
      std::string error = "ERROR: Invalid mesh type  " + mesh_type + " provided.\n";
      error += "       Valid choices are: replicated, distributed.\n";
      libmesh_error_msg(error);
    }

  if (mesh_name.empty() || (mesh_name == "DIE!") )
    {
      out << "Generating mesh.. " << std::endl;
      // Use the MeshTools::Generation mesh generator to create a uniform
      // grid on the square or cube.
      if (dim == 2)
        {
          MeshTools::Generation::build_square
            (*mesh,
             coarsegridsize, coarsegridsize,
             0., 1.,
             0., 1.,
             QUAD4);
        }
      else if (dim == 3)
        {
          MeshTools::Generation::build_cube
            (*mesh,
             coarsegridsize, coarsegridsize,coarsegridsize,
             0., 1.,
             0., 1.,
             0., 1.,
             //TET4);
             HEX8);
        }
    }
  else
    {
      out << "Reading in mesh: " << mesh_name << std::endl;
      mesh->read(mesh_name);
    }

  // Now, make an object to refine mesh
  MeshRefinement mesh_refinement(*mesh);

  // Refine as per the input file
  mesh->partitioner() = NULL;
  mesh_refinement.uniformly_refine(coarserefinements);

  // Print information about the mesh to the screen.
  mesh->print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (*mesh);

  // Declare the system "Heat" and its variables.
  HeatSystem & system =
    equation_systems.add_system<HeatSystem> ("Heat");

  // Solve this as a steady system
  system.time_solver = libmesh_make_unique<SteadySolver>(system);

  // Initialize the system
  equation_systems.init ();

  // And the nonlinear solver options
  system.get_time_solver().diff_solver() = libmesh_make_unique<PetscDiffSolver>(system);
  DiffSolver & solver = *(system.time_solver->diff_solver().get());
  solver.quiet = infile("solver_quiet", false);
  solver.verbose = !solver.quiet;
  solver.max_nonlinear_iterations = infile("max_nonlinear_iterations", 1);
  solver.relative_step_tolerance = infile("relative_step_tolerance", 1.e-3);
  solver.relative_residual_tolerance = infile("relative_residual_tolerance", 0.0);
  solver.absolute_residual_tolerance = infile("absolute_residual_tolerance", 0.0);

  // And the linear solver options
  solver.max_linear_iterations = infile("max_linear_iterations", 5);
  solver.initial_linear_tolerance = infile("initial_linear_tolerance", 1.e-3);

  solver.init();

  // Print information about the system to the screen.
  equation_systems.print_info();

  // solve the steady solution
  system.solve();

  // write solution out if requested from input file
#ifdef LIBMESH_HAVE_EXODUS_API
  if ( !write_solution.empty() && (write_solution != "false") )
    ExodusII_IO(*mesh).write_equation_systems
      (write_solution, equation_systems);
#endif

#ifdef LIBMESH_HAVE_FPARSER
  // Check that we got close to the analytic solution
  ExactSolution exact_sol(equation_systems);
  const std::string exact_str = (dim == 2) ?
    "sin(pi*x)*sin(pi*y)" : "sin(pi*x)*sin(pi*y)*sin(pi*z)";
  ParsedFunction<Number> exact_func(exact_str);
  exact_sol.attach_exact_value(0, &exact_func);
  exact_sol.compute_error("Heat", "T");

  Number err = exact_sol.l2_error("Heat", "T");

  // Print out the error value
  libMesh::out << "L2-Error is: " << err << std::endl;

  libmesh_assert_less(libmesh_real(err), 2e-3);

#endif // #ifdef LIBMESH_HAVE_FPARSER

  // All done.
  return 0;
}
