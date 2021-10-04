/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

using VectorType       = LinearAlgebra::distributed::Vector<double>;
using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
using SparseMatrixType = TrilinosWrappers::SparseMatrix;

namespace dealii
{
  namespace MatrixCreator
  {
    template <int dim, int spacedim, typename number>
    void
    create_mass_matrix(const DoFHandler<dim, spacedim> &dof_handler,
                       const Quadrature<dim> &          quad,
                       SparseMatrixType &               matrix,
                       const AffineConstraints<number> &constraints)
    {
      const auto &  fe = dof_handler.get_fe();
      FEValues<dim> fe_values(
        fe, quad, update_values | update_gradients | update_JxW_values);

      FullMatrix<double>                   cell_matrix;
      std::vector<types::global_dof_index> local_dof_indices;

      // loop over all cells
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned() == false)
            continue;

          fe_values.reinit(cell);

          const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
          cell_matrix.reinit(dofs_per_cell, dofs_per_cell);

          // loop over cell dofs
          for (const auto q : fe_values.quadrature_point_indices())
            {
              for (const auto i : fe_values.dof_indices())
                for (const auto j : fe_values.dof_indices())
                  cell_matrix(i, j) +=
                    (fe_values.shape_value(i, q) * fe_values.shape_value(j, q) *
                     fe_values.JxW(q));
            }

          local_dof_indices.resize(cell->get_fe().dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 matrix);
        }

      matrix.compress(VectorOperation::values::add);
    }


    template <int dim, int spacedim, typename number>
    void
    create_laplace_matrix(const DoFHandler<dim, spacedim> &dof_handler,
                          const Quadrature<dim> &          quad,
                          SparseMatrixType &               matrix,
                          const AffineConstraints<number> &constraints)
    {
      const auto &  fe = dof_handler.get_fe();
      FEValues<dim> fe_values(
        fe, quad, update_values | update_gradients | update_JxW_values);

      FullMatrix<double>                   cell_matrix;
      std::vector<types::global_dof_index> local_dof_indices;

      // loop over all cells
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned() == false)
            continue;

          fe_values.reinit(cell);

          const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
          cell_matrix.reinit(dofs_per_cell, dofs_per_cell);

          // loop over cell dofs
          for (const auto q : fe_values.quadrature_point_indices())
            {
              for (const auto i : fe_values.dof_indices())
                for (const auto j : fe_values.dof_indices())
                  cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) *
                     fe_values.JxW(q));
            }

          local_dof_indices.resize(cell->get_fe().dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 matrix);
        }

      matrix.compress(VectorOperation::values::add);
    }
  } // namespace MatrixCreator
} // namespace dealii

namespace TimeIntegrationSchemes
{
  /**
   * Interface class for time integration.
   */
  class Interface
  {
  public:
    virtual void
    solve(VectorType &       solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const = 0;
  };



  /**
   * One-step-theta method implementation according to step-26.
   */
  class OneStepTheta : public Interface
  {
  public:
    OneStepTheta(const SparseMatrixType &mass_matrix,
                 const SparseMatrixType &laplace_matrix,
                 const std::function<void(const double, VectorType &)>
                   &evaluate_rhs_function)
      : theta(0.5)
      , mass_matrix(mass_matrix)
      , laplace_matrix(laplace_matrix)
      , evaluate_rhs_function(evaluate_rhs_function)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {}

    void
    solve(VectorType &       solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const override
    {
      pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

      SparseMatrixType system_matrix;
      VectorType       system_rhs;
      VectorType       tmp;
      VectorType       forcing_terms;

      system_matrix.reinit(mass_matrix);
      system_rhs.reinit(solution);
      tmp.reinit(solution);
      forcing_terms.reinit(solution);

      // create right-hand-side vector
      // ... old solution
      mass_matrix.vmult(system_rhs, solution);
      laplace_matrix.vmult(tmp, solution);
      system_rhs.add(-(1 - theta) * time_step, tmp);

      // ... rhs function (new)
      evaluate_rhs_function(time, tmp);
      forcing_terms = tmp;
      forcing_terms *= time_step * theta;

      // ... rhs function (old)
      evaluate_rhs_function(time - time_step, tmp);
      forcing_terms.add(time_step * (1 - theta), tmp);

      system_rhs += forcing_terms;

      // setup system matrix
      system_matrix.copy_from(mass_matrix);
      system_matrix.add(theta * time_step, laplace_matrix);

      // solve system
      SolverControl        solver_control(1000, 1e-8 * system_rhs.l2_norm());
      SolverCG<VectorType> cg(solver_control);

      // ... create operator
      SystemMatrix sm(system_matrix);

      // ... create preconditioner
      Preconditioner preconditioner(system_matrix);

      // ... solve
      cg.solve(sm, solution, system_rhs, preconditioner);

      pcout << "     " << solver_control.last_step() << " CG iterations."
            << std::endl;
    }

  private:
    class SystemMatrix
    {
    public:
      SystemMatrix(const SparseMatrixType &system_matrix)
        : system_matrix(system_matrix)
      {}

      void
      vmult(VectorType &dst, const VectorType &src) const
      {
        system_matrix.vmult(dst, src);
      }

    private:
      const SparseMatrixType &system_matrix;
    };

    class Preconditioner
    {
    public:
      Preconditioner(const SparseMatrixType &system_matrix)
        : system_matrix(system_matrix)
      {
        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        precondition_amg.initialize(system_matrix, amg_data);
      }

      void
      vmult(VectorType &dst, const VectorType &src) const
      {
        precondition_amg.vmult(dst, src);
      }

    private:
      const SparseMatrixType &          system_matrix;
      TrilinosWrappers::PreconditionAMG precondition_amg;
    };

    const double theta;

    const SparseMatrixType &mass_matrix;
    const SparseMatrixType &laplace_matrix;

    const std::function<void(const double, VectorType &)> evaluate_rhs_function;

    ConditionalOStream pcout;
  };



  /**
   * IRK implementation.
   */
  class IRK : public Interface
  {
  public:
    IRK(const SparseMatrixType &mass_matrix,
        const SparseMatrixType &laplace_matrix,
        const std::function<void(const double, VectorType &)>
          &evaluate_rhs_function)
      : mass_matrix(mass_matrix)
      , laplace_matrix(laplace_matrix)
      , evaluate_rhs_function(evaluate_rhs_function)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {}

    void
    solve(VectorType &       solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const override
    {
      // TODO: create right-hand-side vector
      BlockVectorType system_rhs;

      Assert(false, ExcNotImplemented());
      (void)timestep_number;
      (void)time;
      (void)time_step;

      // TODO: create an initial guess
      BlockVectorType system_solution;

      Assert(false, ExcNotImplemented());

      // solve system
      SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
      SolverCG<BlockVectorType> cg(solver_control);

      // ... create operator
      SystemMatrix sm(mass_matrix, laplace_matrix);

      // ... create preconditioner
      Preconditioner preconditioner;

      // ... solve
      cg.solve(sm, system_solution, system_rhs, preconditioner);

      pcout << "     " << solver_control.last_step() << " CG iterations."
            << std::endl;

      // TODO: accumulate result in solution
      (void)system_solution;
      (void)solution;
    }

  private:
    class SystemMatrix
    {
    public:
      SystemMatrix(const SparseMatrixType &mass_matrix,
                   const SparseMatrixType &laplace_matrix)
        : mass_matrix(mass_matrix)
        , laplace_matrix(laplace_matrix)
      {}

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const
      {
        Assert(false, ExcNotImplemented()); // TODO
        (void)dst;
        (void)src;
      }

    private:
      const SparseMatrixType &mass_matrix;
      const SparseMatrixType &laplace_matrix;
    };

    class Preconditioner
    {
    public:
      Preconditioner()
      {
        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      }

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const
      {
        dst = src; // TODO: for the first try, we use the identity matrix
                   // as preconditioner
      }

    private:
    };

    const SparseMatrixType &mass_matrix;
    const SparseMatrixType &laplace_matrix;

    const std::function<void(const double, VectorType &)> evaluate_rhs_function;

    ConditionalOStream pcout;
  };
} // namespace TimeIntegrationSchemes



namespace HeatEquation
{
  struct Parameters
  {
    unsigned int fe_degree     = 1;
    unsigned int n_refinements = 4;

    std::string time_integration_scheme = "ost";
    double      end_time                = 0.5;
    double      time_step_size          = 1.0 / 500;

    void
    parse(const std::string file_name)
    {
      dealii::ParameterHandler prm;
      prm.add_parameter("FEDegree", fe_degree);
      prm.add_parameter("NRefinements", n_refinements);
      prm.add_parameter("TimeIntegrationScheme",
                        time_integration_scheme,
                        "",
                        Patterns::Selection("ost|irk"));
      prm.add_parameter("EndTime", end_time);
      prm.add_parameter("TimeStepSize", time_step_size);

      std::ifstream file;
      file.open(file_name);
      prm.parse_input_from_json(file, true);
    }
  };



  template <int dim>
  class Problem
  {
  public:
    Problem(const Parameters &params)
      : params(params)
      , triangulation(MPI_COMM_WORLD)
      , fe(params.fe_degree)
      , quadrature(params.fe_degree + 1)
      , dof_handler(triangulation)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {}

    void
    run()
    {
      GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(params.n_refinements);

      setup_system();

      double       time            = 0.0;
      unsigned int timestep_number = 0;

      // TODO: initial condition might need to be adjusted
      VectorTools::interpolate(dof_handler, AnalyticalSolution(), solution);

      output_results(time, timestep_number);

      const auto evaluate_rhs_function = [&](const double time,
                                             VectorType & tmp) -> void {
        RightHandSide rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(
          dof_handler, quadrature, rhs_function, tmp, constraints);
      };

      std::unique_ptr<TimeIntegrationSchemes::Interface>
        time_integration_scheme;

      if (params.time_integration_scheme == "ost")
        time_integration_scheme =
          std::make_unique<TimeIntegrationSchemes::OneStepTheta>(
            mass_matrix, laplace_matrix, evaluate_rhs_function);
      else if (params.time_integration_scheme == "irk")
        time_integration_scheme =
          std::make_unique<TimeIntegrationSchemes::IRK>(mass_matrix,
                                                        laplace_matrix,
                                                        evaluate_rhs_function);
      else
        Assert(false, ExcNotImplemented());

      while (time <= params.end_time)
        {
          time += params.time_step_size;
          ++timestep_number;

          time_integration_scheme->solve(solution,
                                         timestep_number,
                                         time,
                                         params.time_step_size);

          constraints.distribute(solution);

          output_results(time, timestep_number);
        }
    }

  private:
    void
    setup_system()
    {
      dof_handler.distribute_dofs(fe);

      pcout << std::endl
            << "===========================================" << std::endl
            << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;

      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      // note: program is limited to homogenous DBCs
      DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
      constraints.close();

      TrilinosWrappers::SparsityPattern sparsity_pattern(
        dof_handler.locally_owned_dofs(), dof_handler.get_communicator());
      DoFTools::make_sparsity_pattern(dof_handler,
                                      sparsity_pattern,
                                      constraints,
                                      false);
      sparsity_pattern.compress();

      mass_matrix.reinit(sparsity_pattern);
      laplace_matrix.reinit(sparsity_pattern);

      MatrixCreator::create_mass_matrix(dof_handler,
                                        quadrature,
                                        mass_matrix,
                                        constraints);
      MatrixCreator::create_laplace_matrix(dof_handler,
                                           quadrature,
                                           laplace_matrix,
                                           constraints);

      this->initialize_dof_vector(solution);
      system_rhs.reinit(system_rhs);
    }

    template <typename Number>
    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number> &vec,
      const unsigned int mg_level = numbers::invalid_unsigned_int)
    {
      IndexSet locally_relevant_dofs;

      if (mg_level == numbers::invalid_unsigned_int)
        DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                locally_relevant_dofs);
      else
        DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                      mg_level,
                                                      locally_relevant_dofs);

      const auto partitioner_dealii =
        std::make_shared<const Utilities::MPI::Partitioner>(
          mg_level == numbers::invalid_unsigned_int ?
            dof_handler.locally_owned_dofs() :
            dof_handler.locally_owned_mg_dofs(mg_level),
          locally_relevant_dofs,
          dof_handler.get_communicator());

      vec.reinit(partitioner_dealii);
    }

    void
    output_results(const double time, const unsigned int timestep_number) const
    {
      DataOut<dim> data_out;

      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "U");

      data_out.build_patches();

      data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

      data_out.write_vtu_with_pvtu_record("./",
                                          "result",
                                          timestep_number,
                                          triangulation.get_communicator(),
                                          3,
                                          1);

      {
        solution.update_ghost_values();
        Vector<float> norm_per_cell(triangulation.n_active_cells());
        VectorTools::integrate_difference(dof_handler,
                                          solution,
                                          AnalyticalSolution(),
                                          norm_per_cell,
                                          QGauss<dim>(fe.degree + 2),
                                          VectorTools::L2_norm);
        const double error_norm =
          VectorTools::compute_global_error(triangulation,
                                            norm_per_cell,
                                            VectorTools::L2_norm);
        pcout << "   Error in the L2 norm           :     " << error_norm
              << std::endl;
        solution.zero_out_ghost_values();
      }
    }

    const Parameters &params;

    parallel::distributed::Triangulation<dim> triangulation;
    FE_Q<dim>                                 fe;
    QGauss<dim>                               quadrature;
    DoFHandler<dim>                           dof_handler;

    AffineConstraints<double> constraints;

    TrilinosWrappers::SparsityPattern sparsity_pattern;
    SparseMatrixType                  mass_matrix;
    SparseMatrixType                  laplace_matrix;

    VectorType solution;
    VectorType system_rhs;

    ConditionalOStream pcout;

    class RightHandSide : public Function<dim>
    {
    public:
      RightHandSide()
        : Function<dim>()
        , a_x(2.0)
        , a_y(2.0)
        , a_t(0.5)
      {}

      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override
      {
        (void)component;

        AssertDimension(dim, 2);

        const double x = p[0];
        const double y = p[1];
        const double t = this->get_time();

        return std::sin(a_x * numbers::PI * x) *
               std::sin(a_y * numbers::PI * y) *
               (numbers::PI * std::cos(numbers::PI * t) -
                a_t * (std::sin(numbers::PI * t) + 1) +
                (a_x * a_x + a_y * a_y) * numbers::PI * numbers::PI *
                  (std::sin(numbers::PI * t) + 1)) *
               std::exp(-a_t * t);
      }

    private:
      const double a_x;
      const double a_y;
      const double a_t;
    };

    class AnalyticalSolution : public Function<dim>
    {
    public:
      AnalyticalSolution()
        : Function<dim>()
        , a_x(2.0)
        , a_y(2.0)
        , a_t(0.5)
      {}

      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override
      {
        (void)component;

        AssertDimension(dim, 2);

        const double x = p[0];
        const double y = p[1];
        const double t = this->get_time();

        return std::sin(a_x * numbers::PI * x) *
               std::sin(a_y * numbers::PI * y) *
               (1 + std::sin(numbers::PI * t)) * std::exp(-a_t * t);
      }

    private:
      const double a_x;
      const double a_y;
      const double a_t;
    };
  };
} // namespace HeatEquation



int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

      HeatEquation::Parameters params;

      if (argc == 2)
        params.parse(std::string(argv[1]));

      HeatEquation::Problem<2> heat_equation_solver(params);
      heat_equation_solver.run();
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
