#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>

#include <deal.II/fe/fe_system.h>

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

template <int dim, int n_components_static = 1>
void
test(const Parameters & params,
     const MPI_Comm &   comm,
     const unsigned int n_components,
     ConvergenceTable & table)
{
  parallel::distributed::Triangulation<dim> triangulation(comm);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(params.n_refinements);

  QGauss<dim>   quadrature(params.fe_degree + 1);
  FESystem<dim> fe(FE_Q<dim>(params.fe_degree), n_components_static);

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
  std::unique_ptr<const BatchedMassLaplaceOperator>
    mass_laplace_operator_batched;

  if (params.operator_type == "MatrixBased")
    mass_laplace_operator =
      std::make_unique<MassLaplaceOperatorMatrixBased>(dof_handler,
                                                       constraints,
                                                       quadrature);
  else if (params.operator_type == "MatrixFree")
    mass_laplace_operator = std::make_unique<
      MassLaplaceOperatorMatrixFree<dim, double, n_components_static>>(
      dof_handler, constraints, quadrature);
  else
    AssertThrow(false, ExcNotImplemented());


  std::unique_ptr<PreconditionerBase<VectorType>>      preconditioner;
  std::shared_ptr<PreconditionerBase<BlockVectorType>> preconditioner_batch;

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
      MGLevelObject<std::shared_ptr<const BatchedMassLaplaceOperator>>
        mg_batched_operators(min_level, max_level);

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
            mg_operators[l] = std::make_unique<
              MassLaplaceOperatorMatrixFree<dim, double, n_components_static>>(
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

      if ((n_components_static != n_components) &&
          (n_components_static == 1 && n_components > 1))
        {
          Vector<double> coeff(n_components);

          for (unsigned int i = 0; i < n_components; ++i)
            coeff[i] = 1.0;

          mass_laplace_operator_batched =
            std::make_unique<BatchedMassLaplaceOperatorMatrixFree<dim, double>>(
              coeff,
              dynamic_cast<const MassLaplaceOperatorMatrixFree<dim, double> *>(
                mass_laplace_operator.get())
                ->get_matrix_free());

          mass_laplace_operator_batched->reinit(1.0);

          for (unsigned int l = min_level; l <= max_level; ++l)
            {
              mg_batched_operators[l] = std::make_shared<
                BatchedMassLaplaceOperatorMatrixFree<dim, double>>(
                coeff,
                dynamic_cast<const MassLaplaceOperatorMatrixFree<dim, double>
                               *>(mg_operators[l].get())
                  ->get_matrix_free());
              mg_batched_operators[l]->reinit(1.0);
            }

          preconditioner_batch =
            std::make_shared<PreconditionerGMG<dim,
                                               BatchedMassLaplaceOperator,
                                               BlockVectorType,
                                               VectorType>>(
              dof_handler,
              mg_dof_handlers,
              mg_constraints,
              mg_batched_operators);
        }
    }
  else
    AssertThrow(false, ExcNotImplemented());

  double             time = 0.0;
  ReductionControl   solver_control(1000, 1e-20, 1e-12);
  const unsigned int n_repetitions = 10;

  if (n_components_static == n_components)
    {
      preconditioner->reinit();

      VectorType dst, src;

      mass_laplace_operator->initialize_dof_vector(dst);
      mass_laplace_operator->initialize_dof_vector(src);

      // VectorTools::create_right_hand_side(
      //  dof_handler, quadrature, RightHandSide<dim>(), src, constraints);
      src = 1.0;

      SolverCG<VectorType> cg(solver_control);

      {
        dst = 0.0;
        cg.solve(*mass_laplace_operator, dst, src, *preconditioner);
      }

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
    }
  else if (n_components_static == 1 && n_components > 1)
    {
      preconditioner_batch->reinit();

      BlockVectorType dst, src;

      mass_laplace_operator_batched->initialize_dof_vector(dst);
      mass_laplace_operator_batched->initialize_dof_vector(src);

      src = 1.0;

      SolverCG<BlockVectorType> cg(solver_control);

      {
        dst = 0.0;
        cg.solve(*mass_laplace_operator_batched,
                 dst,
                 src,
                 *preconditioner_batch);
      }

      for (unsigned int counter = 0; counter < n_repetitions; ++counter)
        {
          dst = 0.0;

          const auto temp = std::chrono::system_clock::now();
          cg.solve(*mass_laplace_operator_batched,
                   dst,
                   src,
                   *preconditioner_batch);
          time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now() - temp)
                    .count() /
                  1e9;
        }
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  time = Utilities::MPI::sum(time, MPI_COMM_WORLD) /
         solver_control.last_step() / n_repetitions /
         Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  table.add_value("dim", dim);
  table.add_value("degree", params.fe_degree);
  table.add_value("n_procs", Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
  table.add_value("n_cells",
                  dof_handler.get_triangulation().n_global_active_cells());

  const unsigned int n_roots = Utilities::MPI::sum<unsigned int>(
    Utilities::MPI::this_mpi_process(comm) == 0, MPI_COMM_WORLD);
  table.add_value("n_dofs", dof_handler.n_dofs() * n_roots);
  table.add_value("L", triangulation.n_global_levels());
  table.add_value("n_iterations", solver_control.last_step());
  table.add_value("time", time);
  table.set_scientific("time", true);
}

template <int n_components_static>
void
test_components(const Parameters & params,
                const MPI_Comm     comm,
                const unsigned int n_components,
                ConvergenceTable & table)
{
  if (params.dim == 2)
    test<2, n_components_static>(params, comm, n_components, table);
  else if (params.dim == 3)
    test<3, n_components_static>(params, comm, n_components, table);
  else
    AssertThrow(false, ExcNotImplemented());
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

          constexpr unsigned int n_components = 8;

          if (true) // 1 component
            {
              test_components<1>(params, MPI_COMM_WORLD, 1, table);
            }

          if (true) // n components
            {
              test_components<n_components>(params,
                                            MPI_COMM_WORLD,
                                            n_components,
                                            table);
            }

          if (true) // n subgroups -> one for each component
            {
              int rank, size;
              MPI_Comm_size(MPI_COMM_WORLD, &size);
              MPI_Comm_rank(MPI_COMM_WORLD, &rank);
              MPI_Comm sub_comm;
              MPI_Comm_split(MPI_COMM_WORLD,
                             rank / (size / n_components),
                             rank,
                             &sub_comm);

              test_components<1>(params, sub_comm, 1, table);

              MPI_Comm_free(&sub_comm);
            }

          if (true) // n components -> batched
            {
              test_components<1>(params, MPI_COMM_WORLD, n_components, table);
            }

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