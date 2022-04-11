#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

using VectorType       = LinearAlgebra::distributed::Vector<double>;
using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
using SparseMatrixType = TrilinosWrappers::SparseMatrix;

#include "include/operator.h"
#include "include/preconditioner.h"

struct Parameters
{
  unsigned int dim               = 3;
  unsigned int fe_degree         = 1;
  unsigned int n_refinements     = 3; // will be overriden
  unsigned int min_n_refinements = 3;
  unsigned int max_n_refinements = 20;

  std::string operator_type             = "MatrixFree";
  std::string block_preconditioner_type = "GMG";
};

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    (void)p;
    (void)component;

    return 1.0;
  }

private:
};

template <int dim>
void
test(const Parameters &params, ConvergenceTable &table)
{
  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(params.n_refinements);

  QGauss<dim> quadrature(params.fe_degree + 1);
  FE_Q<dim>   fe(params.fe_degree);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  IndexSet                  locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
  constraints.close();

  std::shared_ptr<const MassLaplaceOperator> mass_laplace_operator;

  if (params.operator_type == "MatrixBased")
    mass_laplace_operator =
      std::make_unique<MassLaplaceOperatorMatrixBased>(dof_handler,
                                                       constraints,
                                                       quadrature);
  else if (params.operator_type == "MatrixFree")
    mass_laplace_operator =
      std::make_unique<MassLaplaceOperatorMatrixFree<dim, double>>(dof_handler,
                                                                   constraints,
                                                                   quadrature);
  else
    AssertThrow(false, ExcNotImplemented());


  std::unique_ptr<PreconditionerBase<VectorType>> preconditioner;

  std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations;

  if (params.block_preconditioner_type == "AMG")
    {
      preconditioner =
        std::make_unique<PreconditionerAMG<MassLaplaceOperator, VectorType>>(
          *mass_laplace_operator);
    }
  else if (params.block_preconditioner_type == "GMG")
    {
      // tighten, since we want to use a subcommunicator on the coarse grid
      RepartitioningPolicyTools::DefaultPolicy<dim> policy(true);
      mg_triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          triangulation, policy);

      const unsigned int min_level = 0;
      const unsigned int max_level = mg_triangulations.size() - 1;

      MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers(
        min_level, max_level);
      MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
        mg_constraints(min_level, max_level);
      MGLevelObject<std::shared_ptr<const MassLaplaceOperator>> mg_operators(
        min_level, max_level);

      for (unsigned int l = min_level; l <= max_level; ++l)
        {
          auto dof_handler =
            std::make_shared<DoFHandler<dim>>(*mg_triangulations[l]);
          auto constraints = std::make_shared<AffineConstraints<double>>();

          dof_handler->distribute_dofs(fe);

          IndexSet locally_relevant_dofs;
          DoFTools::extract_locally_relevant_dofs(*dof_handler,
                                                  locally_relevant_dofs);
          constraints->reinit(locally_relevant_dofs);

          DoFTools::make_zero_boundary_constraints(*dof_handler,
                                                   0,
                                                   *constraints);

          constraints->close();

          if (params.operator_type == "MatrixBased")
            mg_operators[l] =
              std::make_unique<MassLaplaceOperatorMatrixBased>(*dof_handler,
                                                               *constraints,
                                                               quadrature);
          else if (params.operator_type == "MatrixFree")
            mg_operators[l] =
              std::make_unique<MassLaplaceOperatorMatrixFree<dim, double>>(
                *dof_handler, *constraints, quadrature);
          else
            AssertThrow(false, ExcNotImplemented());

          mass_laplace_operator->attach(*mg_operators[l]);

          mg_dof_handlers[l] = dof_handler;
          mg_constraints[l]  = constraints;
        }

      preconditioner = std::make_unique<
        PreconditionerGMG<dim, MassLaplaceOperator, VectorType>>(
        dof_handler, mg_dof_handlers, mg_constraints, mg_operators);
    }
  else
    AssertThrow(false, ExcNotImplemented());

  preconditioner->reinit();

  VectorType dst, src;

  mass_laplace_operator->initialize_dof_vector(dst);
  mass_laplace_operator->initialize_dof_vector(src);

  VectorTools::create_right_hand_side(
    dof_handler, quadrature, RightHandSide<dim>(), src, constraints);

  ReductionControl     solver_control(1000, 1e-20, 1e-12);
  SolverCG<VectorType> cg(solver_control);

  {
    dst = 0.0;
    cg.solve(*mass_laplace_operator, dst, src, *preconditioner);
  }

  double time = 0.0;

  const unsigned int n_repetitions = 10;

  for (unsigned int counter = 0; counter < n_repetitions; ++counter)
    {
      dst = 0.0;

      const auto temp = std::chrono::system_clock::now();
      cg.solve(*mass_laplace_operator, dst, src, *preconditioner);
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now() - temp)
                .count() /
              1e9;
    }

  time = Utilities::MPI::sum(time, MPI_COMM_WORLD) /
         solver_control.last_step() / n_repetitions;

  table.add_value("dim", dim);
  table.add_value("degree", params.fe_degree);
  table.add_value("n_procs", Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
  table.add_value("n_cells",
                  dof_handler.get_triangulation().n_global_active_cells());
  table.add_value("n_dofs", dof_handler.n_dofs());
  table.add_value("L", triangulation.n_global_levels());
  table.add_value("n_iterations", solver_control.last_step());
  table.add_value("time", time);
  table.set_scientific("time", true);
}

int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      dealii::ConditionalOStream pcout(std::cout,
                                       dealii::Utilities::MPI::this_mpi_process(
                                         MPI_COMM_WORLD) == 0);

#ifdef DEBUG
      pcout << "Running in debug mode!" << std::endl;
#endif

      ConvergenceTable table;

      Parameters params;

      for (unsigned int i = params.min_n_refinements;
           i < params.max_n_refinements;
           ++i)
        {
          params.n_refinements = i;

          if (params.dim == 2)
            test<2>(params, table);
          else if (params.dim == 3)
            test<3>(params, table);
          else
            AssertThrow(false, ExcNotImplemented());

          if (pcout.is_active())
            {
              pcout << std::endl;
              table.write_text(pcout.get_stream());
              pcout << std::endl;
            }
        }

      if (pcout.is_active())
        {
          pcout << std::endl;
          table.write_text(pcout.get_stream());
          pcout << std::endl;
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}