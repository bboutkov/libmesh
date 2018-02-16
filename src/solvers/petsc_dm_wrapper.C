// The libMesh Finite Element Library.
// Copyright (C) 2002-2018 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

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

#include "libmesh/libmesh_common.h"

#ifdef LIBMESH_HAVE_PETSC

// PETSc includes
#include <petscsf.h>

#include "libmesh/petsc_dm_wrapper.h"

#include "libmesh/system.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_base.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/partitioner.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/getpot.h"

namespace libMesh
{

  //--------------------------------------------------------------------
  // Functions with C linkage to pass to PETSc.  PETSc will call these
  // methods as needed.
  //
  // Since they must have C linkage they have no knowledge of a namespace.
  // We give them an obscure name to avoid namespace pollution.
  //--------------------------------------------------------------------
  extern "C"
  {

    //! Help PETSc identify the finer DM given a dmc
    PetscErrorCode __libmesh_petsc_DMRefine(DM dmc, MPI_Comm comm, DM * dmf)
    {
      libmesh_assert(dmc);
      libmesh_assert(dmf);
      libmesh_assert(comm);

      PetscErrorCode ierr;

      // extract our context from the incoming dmc
      void * ctx_c = NULL;
      ierr = DMShellGetContext(dmc, & ctx_c);CHKERRABORT(comm, ierr);
      libmesh_assert(ctx_c);
      PetscDMContext * p_ctx = static_cast<PetscDMContext * >(ctx_c);

      // check / set the finer DM
      libmesh_assert(p_ctx->finer_dm);
      libmesh_assert(*(p_ctx->finer_dm));
      *(dmf) = *(p_ctx->finer_dm);

      return 0;
    }

    //! Help PETSc identify the coarser DM given a dmf
    PetscErrorCode __libmesh_petsc_DMCoarsen(DM dmf, MPI_Comm comm, DM * dmc)
    {
      libmesh_assert(dmc);
      libmesh_assert(dmf);
      libmesh_assert(comm);

      PetscErrorCode ierr;

      // extract our context from the incoming dmf
      void * ctx_f = NULL;
      ierr = DMShellGetContext(dmf, &ctx_f);CHKERRABORT(comm, ierr);
      libmesh_assert(ctx_f);
      PetscDMContext * p_ctx = static_cast<PetscDMContext*>(ctx_f);

      // check / set the coarser DM
      libmesh_assert(p_ctx->coarser_dm);
      libmesh_assert(*(p_ctx->coarser_dm));
      *(dmc) = *(p_ctx->coarser_dm);

      return 0;
    }

    //! Function to give PETSc that sets the Interpolation Matrix between two dms
    PetscErrorCode
    __libmesh_petsc_DMCreateInterpolation (DM dmc /*coarse*/, DM dmf /*fine*/,
                                         Mat * mat ,Vec * vec)
    {
      libmesh_assert(dmc);
      libmesh_assert(dmf);
      libmesh_assert(mat);
      libmesh_assert(vec); // optional scaling (not needed for mg)

      // extract our coarse context from the incoming dm
      void * ctx_c = NULL;
      DMShellGetContext(dmc, &ctx_c);
      libmesh_assert(ctx_c);
      PetscDMContext * p_ctx_c = static_cast<PetscDMContext*>(ctx_c);

      // check / give PETSc its matrix
      libmesh_assert(p_ctx_c->K_interp_ptr);
      *(mat) = p_ctx_c->K_interp_ptr->mat();
      *(vec) = PETSC_NULL;

      return 0;
    }

    //! Function to give PETSc that sets the Restriction Matrix between two dms
    PetscErrorCode
    __libmesh_petsc_DMCreateRestriction (DM dmc /*coarse*/, DM dmf/*fine*/, Mat * mat)
    {
      libmesh_assert(dmc);
      libmesh_assert(dmf);
      libmesh_assert(mat);

      // extract our fine context from the incoming dm
      void * ctx_f = NULL;
      DMShellGetContext(dmf, &ctx_f);
      libmesh_assert(ctx_f);
      PetscDMContext * p_ctx_f = static_cast<PetscDMContext*>(ctx_f);

      // check / give PETSc its matrix
      libmesh_assert(p_ctx_f->K_restrict_ptr);
      *(mat) = p_ctx_f->K_restrict_ptr->mat();

      return 0;
    }

  } // end extern C functions



PetscDMWrapper::~PetscDMWrapper()
{
  this->clear();
}

void PetscDMWrapper::clear()
{
  // This will also Destroy the attached PetscSection and PetscSF as well
  // Destroy doesn't free the memory, but just resets points internally
  // in the struct, so we'd still need to wipe out the memory on our side
  for( auto dm_it = _dms.begin(); dm_it < _dms.end(); ++dm_it )
    DMDestroy( dm_it->get() );

  _dms.clear();
  _sections.clear();
  _star_forests.clear();
  _ctx_vec.clear();
  _pmtx_vec.clear();
  _vec_vec.clear();

}

void PetscDMWrapper::init_and_attach_petscdm(const System & system, SNES & snes)
{
  START_LOG ("init_and_attach_petscdm", "PetscDMWrapper");

  PetscErrorCode ierr;

  MeshBase & mesh = system.get_mesh();   // Convenience
  MeshRefinement mesh_refinement(mesh); // Used for swapping between grids

  // First walk over the active local elements and see how many maximum MG levels we can construct
  unsigned int n_levels = 0;
  for ( auto & elem : mesh.active_local_element_ptr_range() )
    {
      if ( elem->level() > n_levels )
        n_levels = elem->level();
    }
  // On coarse grids some processors may have no active local elements,
  // these processors shouldnt make projections
  if (n_levels >= 1)
    n_levels += 1;

  // TODO: How many MG levels did the user request?
  // Create a GetPot object to parse the command line
  /*
  int requested_levels = 0;
  GetPot command_line (argc, argv);
  if (command_line.search("-pc_mg_levels"))
    requested_levels = command_line.next(0);
  else
    libmesh_error_msg("ERROR: -pc_mg_levels not specified!");
  */


  // Init data structures: data[0] ~ coarse grid, data[n_levels-1] ~ fine grid
  this->init_dm_data(n_levels, system.comm());

  // Step 1.  contract : all active elements have no children
  mesh.contract();

  // Start on finest grid. Construct DM datas and stash some info for
  // later projection_matrix and vec sizing
  for(unsigned int level = n_levels; level >= 1; level--)
    {
      // Save the n_fine_dofs before coarsening for later projection matrix sizing
      _mesh_dof_sizes[level-1] = system.get_dof_map().n_dofs();
      _mesh_dof_loc_sizes[level-1] = system.get_dof_map().n_local_dofs();

      // Get refs to things we will fill
      DM & dm = this->get_dm(level-1);
      PetscSection & section = this->get_section(level-1);
      PetscSF & star_forest = this->get_star_forest(level-1);

      // The shell will contain other DM info
      ierr = DMShellCreate(system.comm().get(), &dm);
      CHKERRABORT(system.comm().get(),ierr);

      // Build the PetscSection and attach it to the DM
      this->build_section(system, section);
      ierr = DMSetDefaultSection(dm, section);
      CHKERRABORT(system.comm().get(),ierr);

      // We only need to build the star forest if we're in a parallel environment
      if (system.n_processors() > 1)
        {
          // Build the PetscSF and attach it to the DM
          this->build_sf(system, star_forest);
          ierr = DMSetDefaultSF(dm, star_forest);
          CHKERRABORT(system.comm().get(),ierr);
        }

      // Set PETSC's Restriction, Interpolation, Coarsen and Refine functions for the current DM
      ierr = DMShellSetCreateInterpolation ( dm, __libmesh_petsc_DMCreateInterpolation );
      CHKERRABORT(system.comm().get(), ierr);

      // Not implemented. For now we rely on galerkin style restrictions
      bool supply_restriction = false;
      if (supply_restriction)
        {
        ierr = DMShellSetCreateRestriction ( dm, __libmesh_petsc_DMCreateRestriction  );
        CHKERRABORT(system.comm().get(), ierr);
        }

      ierr = DMShellSetCoarsen ( dm, __libmesh_petsc_DMCoarsen );
      CHKERRABORT(system.comm().get(), ierr);

      ierr = DMShellSetRefine ( dm, __libmesh_petsc_DMRefine );
      CHKERRABORT(system.comm().get(), ierr);

      // Uniformly coarsen if not the coarsest grid and distribute dof info.
      // We dont repartition because we are assuming an initially load balanced grid
      if ( level != 1 )
        {
          mesh.partitioner() = NULL;
          mesh_refinement.uniformly_coarsen(1);

          system.get_dof_map().distribute_dofs(mesh);
          system.reinit_constraints();
        }
    } // End PETSc data structure creation

  // Now fill the corresponding internal PetscDMContext for each created DM
  for( unsigned int i=1; i <= n_levels; i++ )
    {
      _ctx_vec[i-1] = libmesh_make_unique<PetscDMContext>();

      // Set pointers to surrounding dm levels to help PETSc refine/coarsen
      if ( i == 1 ) // were at the coarsest mesh
        {
          (*_ctx_vec[i-1]).coarser_dm = NULL;
          (*_ctx_vec[i-1]).finer_dm   = _dms[1].get();
        }
      else if( i == n_levels ) // were at the finest mesh
        {
          (*_ctx_vec[i-1]).coarser_dm = _dms[_dms.size() - 2].get();
          (*_ctx_vec[i-1]).finer_dm   = NULL;
        }
      else // were in the middle of the heirarchy
        {
          (*_ctx_vec[i-1]).coarser_dm = _dms[i-2].get();
          (*_ctx_vec[i-1]).finer_dm   = _dms[i].get();
        }

      // Create and attach a sized vector to the current ctx
      _vec_vec[i-1]= libmesh_make_unique<PetscVector<Real>>( system.comm(), _mesh_dof_sizes[i-1] );
      _ctx_vec[i-1]->current_vec = _vec_vec[i-1].get();

    } // End context creation

  // Attach a vector and context to each DM
  for ( unsigned int i = 1; i <= n_levels ; ++i)
    {
      DM & dm = this->get_dm(i-1);

      ierr = DMShellSetGlobalVector( dm, (*_ctx_vec[ i-1 ]).current_vec->vec() );
      CHKERRABORT(system.comm().get(), ierr);

      ierr = DMShellSetContext( dm, _ctx_vec[ i-1 ].get() );
      CHKERRABORT(system.comm().get(), ierr);
    }

  // DM structures created, now we need projection matrixes.
  // To prepare for projection creation go to second coarsest mesh so we can utilize
  // old_dof_indices information in the projection creation
  mesh_refinement.uniformly_refine(1);
  system.get_dof_map().distribute_dofs(mesh);;
  system.reinit_constraints();

  // Create the Interpolation Matrices between adjacent mesh levels
  for ( unsigned int i = 1 ; i < n_levels ; ++i )
    {
      if ( i != n_levels )
        {
          unsigned int ndofs_c = _mesh_dof_sizes[i-1];
          unsigned int ndofs_f = _mesh_dof_sizes[i];

          // Create the Interpolation matrix and set its pointer
          _ctx_vec[i-1]->K_interp_ptr = _pmtx_vec[i-1].get();

          unsigned int ndofs_local = system.get_dof_map().n_dofs_on_processor(system.processor_id());
          unsigned int ndofs_old_first = system.get_dof_map().first_old_dof(system.processor_id());
          unsigned int ndofs_old_end   = system.get_dof_map().end_old_dof(system.processor_id());
          unsigned int ndofs_old_size   = ndofs_old_end - ndofs_old_first;

          // Init and zero the matrix
          _ctx_vec[i-1]->K_interp_ptr->init(ndofs_f, ndofs_c, ndofs_local, ndofs_old_size);

          // TODO: Projection matrix sparsity pattern?
          //MatSetOption(_ctx_vec[i-1]->K_interp_ptr->mat(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

          // Compute the interpolation matrix and set K_interp_ptr
          system.projection_matrix(*_ctx_vec[i-1]->K_interp_ptr);

          // Always close matrix that contains altered data
          _ctx_vec[i-1]->K_interp_ptr->close();
        }

      // Move to next grid to make next projection
      if ( i != n_levels - 1 )
        {
          mesh.partitioner() = NULL;
          mesh_refinement.uniformly_refine(1);
          system.get_dof_map().distribute_dofs(mesh);
          system.reinit_constraints();
        }
    } // End create transfer operators. System back at the finest grid

  // Lastly, give SNES the finest level DM
  DM & dm = this->get_dm(n_levels-1);
  ierr = SNESSetDM(snes, dm);
  CHKERRABORT(system.comm().get(),ierr);

  STOP_LOG ("init_and_attach_petscdm", "PetscDMWrapper");
}

void PetscDMWrapper::build_section( const System & system, PetscSection & section )
{
  START_LOG ("build_section", "PetscDMWrapper");

  PetscErrorCode ierr;
  ierr = PetscSectionCreate(system.comm().get(),&section);
  CHKERRABORT(system.comm().get(),ierr);

  ierr = PetscSectionSetNumFields(section,system.n_vars());
  CHKERRABORT(system.comm().get(),ierr);

  // First, set the actual names of all the fields variables we are interested in
  for( unsigned int v = 0; v < system.n_vars(); v++ )
    {
      ierr = PetscSectionSetFieldName( section, v, system.variable_name(v).c_str() );
      CHKERRABORT(system.comm().get(),ierr);
    }

  // Set "points" count into the section. A "point" in PETSc nomenclature
  // is a geometric object that can have dofs associated with it, e.g.
  // Node, Edge, Face, Elem. First we tell the PetscSection about all of our
  // points.
  this->set_point_range_in_section(system, section);

  // Now build up the dofs per "point" in the PetscSection
  this->add_dofs_to_section(system, section);

  // Final setup of PetscSection
  ierr = PetscSectionSetUp(section);CHKERRABORT(system.comm().get(),ierr);

  // Sanity checking at least that total n_dofs match
#ifndef NDEBUG
  this->check_section_n_dofs_match(system, section);
#endif

  STOP_LOG ("build_section", "PetscDMWrapper");
}

void PetscDMWrapper::build_sf( const System & system, PetscSF & star_forest )
{
  START_LOG ("build_sf", "PetscDMWrapper");

  const DofMap & dof_map = system.get_dof_map();

  const std::vector<dof_id_type> & send_list = dof_map.get_send_list();

  // Number of ghost dofs that send information to this processor
  PetscInt n_leaves = send_list.size();

  // Number of local dofs, including ghosts dofs
  PetscInt n_roots = dof_map.n_local_dofs();
  n_roots += send_list.size();

  // This is the vector of dof indices coming from other processors
  // TODO: We do a stupid copy here since we can't convert an
  //       unsigned int* to PetscInt*
  std::vector<PetscInt> send_list_copy(send_list.size());
  for( unsigned int i = 0; i < send_list.size(); i++ )
    send_list_copy[i] = send_list[i];

  PetscInt * local_dofs = send_list_copy.data();

  // This is the vector of PetscSFNode's for the local_dofs.
  // For each entry in local_dof, we have to supply the rank from which
  // that dof stems and its local index on that rank.
  // PETSc documentation here:
  // http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PetscSF/PetscSFNode.html
  std::vector<PetscSFNode> sf_nodes(send_list.size());

  for( unsigned int i = 0; i < send_list.size(); i++ )
    {
      unsigned int incoming_dof = send_list[i];

      PetscInt rank = this->find_dof_rank( incoming_dof, dof_map );

      /* Dofs are sorted and continuous on the processor so local index
         is counted up from the first dof on the processor. */
      PetscInt index = incoming_dof - dof_map.first_dof(rank);

      sf_nodes[i].rank  = rank; /* Rank of owner */
      sf_nodes[i].index = index;/* Index of dof on rank */
    }

  PetscSFNode * remote_dofs = sf_nodes.data();

  PetscErrorCode ierr;
  ierr = PetscSFCreate(system.comm().get(), &star_forest);CHKERRABORT(system.comm().get(),ierr);

  // TODO: We should create pointers to arrays so we don't have to copy
  //       and case use PETSC_OWN_POINTER where PETSc will take ownership
  //       and delete the memory for us. But then we have to use PetscMalloc.
  ierr = PetscSFSetGraph(star_forest,
                         n_roots,
                         n_leaves,
                         local_dofs,
                         PETSC_COPY_VALUES,
                         remote_dofs,
                         PETSC_COPY_VALUES);
  CHKERRABORT(system.comm().get(),ierr);

  STOP_LOG ("build_sf", "PetscDMWrapper");
}

void PetscDMWrapper::set_point_range_in_section( const System & system, PetscSection & section)
{
  const MeshBase & mesh = system.get_mesh();

  unsigned int pstart = 2^30;
  unsigned int pend = 0;

  // Find minimum and maximum (inclusive) global id numbers on the current processor
  // to build PetscSection. Currently, we're using the global node number as the point index.
  // TODO: This is currently restricted to nodal dofs only!
  //       Need to generalize to the cases where there's dofs at edges, faces, and interiors
  //       as well as vertices. When we generalize, we must guarantee that each id() is unique!
  //       We'll then add each edge, face, and/or interior as a new "point" in the PetscSection.
  if (mesh.n_active_local_elem() > 0)
    {
      for (MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
           el != mesh.active_local_elements_end(); el++)
        {
          const Elem * elem = *el;

          if (mesh.query_elem_ptr(elem->id()))
            {
              for (unsigned int n = 0; n < elem->n_nodes(); n++)
                {
                  // get the global id number of local node n
                  dof_id_type node = elem->node(n);

                  if (node < pstart)
                    pstart = node;

                  if (node > pend)
                    pend = node;
                }
            }
        }

      // PetscSectionSetChart is expecting [pstart,pend), so pad pend by 1.
      pend +=1;
    }

  // If we're on a processor who coarsened the mesh to have no local elements,
  // we should make an empty PetscSection. An empty PetscSection is specified
  // by passing [0,0) to the PetscSectionSetChart call.
  else
    {
      pstart = 0;
      pend = 0;
    }

  PetscErrorCode ierr = PetscSectionSetChart(section, pstart, pend);
  CHKERRABORT(system.comm().get(),ierr);
}

void PetscDMWrapper::add_dofs_to_section( const System & system, PetscSection & section )
{
  const MeshBase & mesh = system.get_mesh();
  const DofMap & dof_map = system.get_dof_map();

  PetscErrorCode ierr;

  // Now we go through and add dof information for each point object.
  // We do this by looping through all the elements on this processor
  // and add associated dofs with that element
  // TODO: Currently, we're only adding the dofs at nodes! We need to generalize
  //       beyond nodal FEM!
  for (MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
       el != mesh.active_local_elements_end(); el++)
    {
      const Elem * elem = *el;

      if (mesh.query_elem_ptr(elem->id()))
        {
          // We need to keep a count of total dofs at each point
          // for the PetscSectionSetDof call at the end.
          std::map<unsigned int, unsigned int> global_dofs;

          // Now set the dof for each field
          for( unsigned int v = 0; v < system.n_vars(); v++ )
            {
              std::vector<dof_id_type> dof_indices;
              dof_map.dof_indices( elem, dof_indices, v );

              // Right now, we're assuming at most one dof per node
              // TODO:  Need to remove this assumption of at most one dof per node
              for( unsigned int n = 0; n < dof_indices.size(); n++ )
                {
                  dof_id_type index = dof_indices[n];

                  // Check if this dof index is on this processor. If so, we count it.
                  if( index >= dof_map.first_dof() && index < dof_map.end_dof() )
                    {
                      // TODO: Need to remove this assumption of at most one dof per node
                      PetscInt n_dofs_at_node = 1;

                      dof_id_type global_node = elem->node(n);

                      ierr = PetscSectionSetFieldDof( section, global_node, v, n_dofs_at_node );
                      CHKERRABORT(system.comm().get(),ierr);

                      global_dofs[global_node] += 1;
                    }
                }
            }

          // [PB]: This is redundant, but PETSc needed it at the time of writing. Perhaps
          //       it will be fixed upstream at some point.
          for( std::map<unsigned int, unsigned int>::const_iterator it = global_dofs.begin();
               it != global_dofs.end(); ++it )
            {
              unsigned int global_node = it->first;
              unsigned int total_n_dofs_at_node = it->second;

              ierr = PetscSectionSetDof( section, global_node, total_n_dofs_at_node );
              CHKERRABORT(system.comm().get(),ierr);
            }
        }
    }
}

void PetscDMWrapper::check_section_n_dofs_match( const System & system, PetscSection & section )
{
  PetscInt total_n_dofs = 0;

  // Grap the starting and ending points from the section
  PetscInt pstart, pend;
  PetscErrorCode ierr = PetscSectionGetChart(section, &pstart, &pend);
  CHKERRABORT(system.comm().get(),ierr);

  // Count up the n_dofs for each point from the section
  for( PetscInt p = pstart; p < pend+1; p++ )
    {
      PetscInt n_dofs;
      ierr = PetscSectionGetDof(section,p,&n_dofs);CHKERRABORT(system.comm().get(),ierr);
      total_n_dofs += n_dofs;
    }

  // That should match the n_local_dofs for our system
  libmesh_assert_equal_to(total_n_dofs,(PetscInt)system.n_local_dofs());
}

PetscInt PetscDMWrapper::find_dof_rank( unsigned int dof, const DofMap& dof_map ) const
{
  libmesh_assert_greater_equal( dof, dof_map.first_dof( 0 ) );
  libmesh_assert_less( dof, dof_map.end_dof(dof_map.comm().size()-1) );

  // dofs are in order on each processor starting from processor 0, so we can
  // use a binary search
  unsigned int max_rank = dof_map.comm().size();
  unsigned int current_rank = max_rank/2;
  unsigned int min_rank = 0;

  do
    {
      if( dof >= dof_map.first_dof(current_rank) )
        {
          min_rank = current_rank;
          current_rank = (max_rank + min_rank)/2;
        }
      else
        {
          max_rank = current_rank;
          current_rank = (max_rank + min_rank)/2;
        }
    }
  while( max_rank - min_rank > 1 );

  libmesh_assert_less( dof, dof_map.end_dof(current_rank) );
  libmesh_assert_greater_equal( dof, dof_map.first_dof(current_rank) );

  return current_rank;
}

void PetscDMWrapper::init_dm_data(unsigned int n_levels, const Parallel::Communicator & comm)
{
  _dms.resize(n_levels);
  _sections.resize(n_levels);
  _star_forests.resize(n_levels);
  _ctx_vec.resize(n_levels);
  _pmtx_vec.resize(n_levels);
  _vec_vec.resize(n_levels);
  _mesh_dof_sizes.resize(n_levels);
  _mesh_dof_loc_sizes.resize(n_levels);

  for( unsigned int i = 0; i < n_levels; i++ )
    {
      _dms[i] = libmesh_make_unique<DM>();
      _sections[i] = libmesh_make_unique<PetscSection>();
      _star_forests[i] = libmesh_make_unique<PetscSF>();
      _ctx_vec[i] = libmesh_make_unique<PetscDMContext>();
      _pmtx_vec[i]= libmesh_make_unique<PetscMatrix<Real>>( comm );

    }
}

} // end namespace libMesh

#endif // LIBMESH_HAVE_PETSC
