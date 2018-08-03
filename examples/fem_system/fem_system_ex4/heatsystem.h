#include "libmesh/enum_fe_family.h"
#include "libmesh/fem_system.h"

// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class HeatSystem : public libMesh::FEMSystem
{
public:
  // Constructor
  HeatSystem(libMesh::EquationSystems & es,
             const std::string & name,
             const unsigned int number) :
    libMesh::FEMSystem(es, name, number),
      _fe_family("LAGRANGE"), _fe_order(1)
      {
        // do nothing
      }

  std::string & fe_family() { return _fe_family; }
  unsigned int & fe_order() { return _fe_order; }



virtual bool element_time_derivative (bool request_jacobian,
                                      libMesh::DiffContext & context);


protected:
  // System initialization
  virtual void init_data ();

  // Context initialization
  virtual void init_context (libMesh::DiffContext & context);

  // The FE type to use
  std::string _fe_family;
  unsigned int _fe_order;

  // The variable index (yes, this will be 0...)
  unsigned int T_var;

  // Forcing function read in from input
  std::string forcing_function;

};
