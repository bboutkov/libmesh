// The libMesh Finite Element Library.
// Copyright (C) 2002-2017 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

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

// Local Includes
#include "libmesh/boundary_info.h" // needed for dirichlet constraints
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/elem_range.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_type.h"
#include "libmesh/function_base.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/mesh_base.h"
#include "libmesh/mesh_inserter_iterator.h"
#include "libmesh/mesh_tools.h" // for libmesh_assert_valid_boundary_ids()
#include "libmesh/numeric_vector.h" // for enforce_constraints_exactly()
#include "libmesh/parallel.h"
#include "libmesh/parallel_algebra.h"
#include "libmesh/parallel_elem.h"
#include "libmesh/parallel_node.h"
#include "libmesh/periodic_boundaries.h"
#include "libmesh/periodic_boundary.h"
#include "libmesh/periodic_boundary_base.h"
#include "libmesh/point_locator_base.h"
#include "libmesh/quadrature.h" // for dirichlet constraints
#include "libmesh/raw_accessor.h"
#include "libmesh/sparse_matrix.h" // needed to constrain adjoint rhs
#include "libmesh/system.h" // needed by enforce_constraints_exactly()
#include "libmesh/threads.h"
#include "libmesh/tensor_tools.h"
#include "libmesh/utility.h" // Utility::iota()

// C++ Includes
#include <set>
#include <algorithm> // for std::count, std::fill
#include <sstream>
#include <cstdlib> // *must* precede <cmath> for proper std:abs() on PGI, Sun Studio CC
#include <cmath>


namespace libMesh
{

// ------------------------------------------------------------
// DofMap member functions

#ifdef LIBMESH_ENABLE_CONSTRAINTS


  template <typename MatType> void
    DofMap::constrain_element_matrix (MatType & matrix,
                                      std::vector<dof_id_type> & elem_dofs,
                                      bool asymmetric_constraint_rows) const
    {
      // what do we do here for non dense / sparse cases?
    }

  // helper specialization for the SparseMatrix DSNA case
  template <> void
    DofMap::constrain_element_matrix<SparseMatrix<Number>> (SparseMatrix<Number> & matrix,
                                                           std::vector<dof_id_type> & elem_dofs,
                                                           bool asymmetric_constraint_rows) const
    {
      libmesh_assert_equal_to (elem_dofs.size(), matrix.m());
      libmesh_assert_equal_to (elem_dofs.size(), matrix.n());

      // check for easy return
      if (this->_dof_constraints.empty())
        return;

      // The constrained matrix is built up as C^T K C.
      DenseMatrix<Number> C;


      this->build_constraint_matrix (C, elem_dofs);

      LOG_SCOPE("constrain_elem_matrix()", "DofMap");

      // It is possible that the matrix is not constrained at all.
      if ((C.m() == matrix.m()) &&
          (C.n() == elem_dofs.size())) // It the matrix is constrained
        {
          // Compute the matrix-matrix-matrix product C^T K C
          matrix.left_multiply_transpose  (C);
          matrix.right_multiply (C);


          libmesh_assert_equal_to (matrix.m(), matrix.n());
          libmesh_assert_equal_to (matrix.m(), elem_dofs.size());
          libmesh_assert_equal_to (matrix.n(), elem_dofs.size());

          // TODO BB: mat.set will be slowwww do these operations need to be batched
          for (std::size_t i=0; i<elem_dofs.size(); i++)
            // If the DOF is constrained
            if (this->is_constrained_dof(elem_dofs[i]))
              {
                for (unsigned int j=0; j<matrix.n(); j++)
                  matrix.set(i,j,0.);

                matrix.set(i,i,1.);

                if (asymmetric_constraint_rows)
                  {
                    DofConstraints::const_iterator
                      pos = _dof_constraints.find(elem_dofs[i]);

                    libmesh_assert (pos != _dof_constraints.end());

                    const DofConstraintRow & constraint_row = pos->second;

                    // This is an overzealous assertion in the presence of
                    // heterogenous constraints: we now can constrain "u_i = c"
                    // with no other u_j terms involved.
                    //
                    // libmesh_assert (!constraint_row.empty());

                    // TODO BB: mat.set and () will be slowwww do these operations need to be batched
                    for (DofConstraintRow::const_iterator
                           it=constraint_row.begin(); it != constraint_row.end();
                         ++it)
                      for (std::size_t j=0; j<elem_dofs.size(); j++)
                        if (elem_dofs[j] == it->first)
                          {
                            auto tmp = matrix(i,j);
                            matrix.set(i,j,tmp - it->second);
                          }
                  }
              }
        } // end if is constrained...
    }

  // helper specialization for the common (nonDSNA) case
  template <> void
    DofMap::constrain_element_matrix<DenseMatrix<Number>> (DenseMatrix<Number> & matrix,
                                                           std::vector<dof_id_type> & elem_dofs,
                                                           bool asymmetric_constraint_rows) const
    {
      libmesh_assert_equal_to (elem_dofs.size(), matrix.m());
      libmesh_assert_equal_to (elem_dofs.size(), matrix.n());

      // check for easy return
      if (this->_dof_constraints.empty())
        return;

      // The constrained matrix is built up as C^T K C.
      DenseMatrix<Number> C;


      this->build_constraint_matrix (C, elem_dofs);

      LOG_SCOPE("constrain_elem_matrix()", "DofMap");

      // It is possible that the matrix is not constrained at all.
      if ((C.m() == matrix.m()) &&
          (C.n() == elem_dofs.size())) // It the matrix is constrained
        {
          // Compute the matrix-matrix-matrix product C^T K C
          matrix.left_multiply_transpose  (C);
          matrix.right_multiply (C);


          libmesh_assert_equal_to (matrix.m(), matrix.n());
          libmesh_assert_equal_to (matrix.m(), elem_dofs.size());
          libmesh_assert_equal_to (matrix.n(), elem_dofs.size());


          for (std::size_t i=0; i<elem_dofs.size(); i++)
            // If the DOF is constrained
            if (this->is_constrained_dof(elem_dofs[i]))
              {
                for (unsigned int j=0; j<matrix.n(); j++)
                  matrix(i,j) = 0.;

                matrix(i,i) = 1.;

                if (asymmetric_constraint_rows)
                  {
                    DofConstraints::const_iterator
                      pos = _dof_constraints.find(elem_dofs[i]);

                    libmesh_assert (pos != _dof_constraints.end());

                    const DofConstraintRow & constraint_row = pos->second;

                    // This is an overzealous assertion in the presence of
                    // heterogenous constraints: we now can constrain "u_i = c"
                    // with no other u_j terms involved.
                    //
                    // libmesh_assert (!constraint_row.empty());

                    for (DofConstraintRow::const_iterator
                           it=constraint_row.begin(); it != constraint_row.end();
                         ++it)
                      for (std::size_t j=0; j<elem_dofs.size(); j++)
                        if (elem_dofs[j] == it->first)
                          matrix(i,j) = -it->second;
                  }
              }
        } // end if is constrained...
    }


#endif // LIBMESH_ENABLE_CONSTRAINTS



} // namespace libMesh
