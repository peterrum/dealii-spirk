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

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

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
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

namespace Step26
{
  /**
   * Interface class for time integration.
   */
  class TimeIntegrationScheme
  {
  public:
    virtual void
    solve(Vector<double> &   solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const = 0;
  };



  /**
   * One-step-theta method implementation according to step-26.
   */
  class TimeIntegrationSchemeOneStepTheta : public TimeIntegrationScheme
  {
  public:
    TimeIntegrationSchemeOneStepTheta(
      const SparseMatrix<double> &mass_matrix,
      const SparseMatrix<double> &laplace_matrix,
      const std::function<void(const double, Vector<double> &)>
        &evaluate_rhs_function)
      : theta(0.5)
      , mass_matrix(mass_matrix)
      , laplace_matrix(laplace_matrix)
      , evaluate_rhs_function(evaluate_rhs_function)
    {}

    void
    solve(Vector<double> &   solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const override
    {
      std::cout << "Time step " << timestep_number << " at t=" << time
                << std::endl;

      SparseMatrix<double> system_matrix;
      Vector<double>       system_rhs;
      Vector<double>       tmp;
      Vector<double>       forcing_terms;

      system_matrix.reinit(mass_matrix.get_sparsity_pattern());
      system_rhs.reinit(solution.size());
      tmp.reinit(solution.size());
      forcing_terms.reinit(solution.size());

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
      SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
      SolverCG<Vector<double>> cg(solver_control);

      // ... create operator
      SystemMatrix sm(system_matrix);

      // ... create preconditioner
      Preconditioner preconditioner(system_matrix);

      // ... solve
      cg.solve(sm, solution, system_rhs, preconditioner);

      std::cout << "     " << solver_control.last_step() << " CG iterations."
                << std::endl;
    }

  private:
    class SystemMatrix
    {
    public:
      SystemMatrix(const SparseMatrix<double> &system_matrix)
        : system_matrix(system_matrix)
      {}

      void
      vmult(Vector<double> &dst, const Vector<double> &src) const
      {
        system_matrix.vmult(dst, src);
      }

    private:
      const SparseMatrix<double> &system_matrix;
    };

    class Preconditioner
    {
    public:
      Preconditioner(const SparseMatrix<double> &system_matrix)
        : system_matrix(system_matrix)
      {}

      void
      vmult(Vector<double> &dst, const Vector<double> &src) const
      {
        system_matrix.precondition_SSOR(dst, src, 1.0);
      }

    private:
      const SparseMatrix<double> &system_matrix;
    };

    const double theta;

    const SparseMatrix<double> &mass_matrix;
    const SparseMatrix<double> &laplace_matrix;

    const std::function<void(const double, Vector<double> &)>
      evaluate_rhs_function;
  };



  /**
   * IRK implementation.
   */
  class TimeIntegrationSchemeIRK : public TimeIntegrationScheme
  {
  public:
    TimeIntegrationSchemeIRK(
      const SparseMatrix<double> &mass_matrix,
      const SparseMatrix<double> &laplace_matrix,
      const std::function<void(const double, Vector<double> &)>
        &evaluate_rhs_function)
      : theta(0.5)
      , mass_matrix(mass_matrix)
      , laplace_matrix(laplace_matrix)
      , evaluate_rhs_function(evaluate_rhs_function)
    {}

    void
    solve(Vector<double> &   solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const override
    {
      // TODO: create right-hand-side vector
      BlockVector<double> system_rhs;

      Assert(false, ExcNotImplemented());
      (void)timestep_number;
      (void)time;
      (void)time_step;

      // TODO: create an initial guess
      BlockVector<double> system_solution;

      Assert(false, ExcNotImplemented());

      // solve system
      SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
      SolverCG<BlockVector<double>> cg(solver_control);

      // ... create operator
      SystemMatrix sm(mass_matrix, laplace_matrix);

      // ... create preconditioner
      Preconditioner preconditioner;

      // ... solve
      cg.solve(sm, system_solution, system_rhs, preconditioner);

      std::cout << "     " << solver_control.last_step() << " CG iterations."
                << std::endl;

      // TODO: accumulate result in solution
      (void)solution;
    }

  private:
    class SystemMatrix
    {
    public:
      SystemMatrix(const SparseMatrix<double> &mass_matrix,
                   const SparseMatrix<double> &laplace_matrix)
        : mass_matrix(mass_matrix)
        , laplace_matrix(laplace_matrix)
      {}

      void
      vmult(BlockVector<double> &dst, const BlockVector<double> &src) const
      {
        Assert(false, ExcNotImplemented()); // TODO
        (void)dst;
        (void)src;
      }

    private:
      const SparseMatrix<double> &mass_matrix;
      const SparseMatrix<double> &laplace_matrix;
    };

    class Preconditioner
    {
    public:
      Preconditioner()
      {}

      void
      vmult(BlockVector<double> &dst, const BlockVector<double> &src) const
      {
        dst = src; // TODO: for the first try we use the identity matrix
                   // as preconditioner
      }

    private:
    };

    const double theta;

    const SparseMatrix<double> &mass_matrix;
    const SparseMatrix<double> &laplace_matrix;

    const std::function<void(const double, Vector<double> &)>
      evaluate_rhs_function;
  };



  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation()
      : fe(1)
      , dof_handler(triangulation)
      , time_step(1. / 500)
    {}

    void
    run()
    {
      const unsigned int n_refinements = 4;

      GridGenerator::hyper_L(triangulation);
      triangulation.refine_global(n_refinements);

      setup_system();

      time            = 0.0;
      timestep_number = 0;

      // TODO: initial condition might need to be ajdusted
      VectorTools::interpolate(dof_handler,
                               Functions::ZeroFunction<dim>(),
                               solution);

      output_results();

      const auto evaluate_rhs_function = [&](const double    time,
                                             Vector<double> &tmp) -> void {
        RightHandSide rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(
                                              dof_handler.get_fe().degree + 1),
                                            rhs_function,
                                            tmp,
                                            constraints);
      };

      std::unique_ptr<TimeIntegrationScheme> time_integration_scheme;

      if (true /*use one-step-theta method*/)
        time_integration_scheme =
          std::make_unique<TimeIntegrationSchemeOneStepTheta>(
            mass_matrix, laplace_matrix, evaluate_rhs_function);
      else /*use IRK method*/
        time_integration_scheme =
          std::make_unique<TimeIntegrationSchemeIRK>(mass_matrix,
                                                     laplace_matrix,
                                                     evaluate_rhs_function);

      while (time <= 0.5)
        {
          time += time_step;
          ++timestep_number;

          time_integration_scheme->solve(solution,
                                         timestep_number,
                                         time,
                                         time_step);

          constraints.distribute(solution);

          output_results();
        }
    }

  private:
    void
    setup_system()
    {
      dof_handler.distribute_dofs(fe);

      std::cout << std::endl
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

      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
      sparsity_pattern.copy_from(dsp);

      mass_matrix.reinit(sparsity_pattern);
      laplace_matrix.reinit(sparsity_pattern);

      MatrixCreator::create_mass_matrix<dim, dim, double>(dof_handler,
                                                          QGauss<dim>(
                                                            fe.degree + 1),
                                                          mass_matrix,
                                                          nullptr,
                                                          constraints);
      MatrixCreator::create_laplace_matrix<dim, dim>(dof_handler,
                                                     QGauss<dim>(fe.degree + 1),
                                                     laplace_matrix,
                                                     nullptr,
                                                     constraints);

      solution.reinit(dof_handler.n_dofs());
      system_rhs.reinit(dof_handler.n_dofs());
    }

    void
    output_results() const
    {
      DataOut<dim> data_out;

      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "U");

      data_out.build_patches();

      data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

      const std::string filename =
        "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
      std::ofstream output(filename);
      data_out.write_vtk(output);
    }

    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;

    double       time;
    double       time_step;
    unsigned int timestep_number;

    class RightHandSide : public Function<dim>
    {
    public:
      RightHandSide()
        : Function<dim>()
        , period(0.2)
      {}

      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override
      {
        // TODO: adjust right-hand-side function

        (void)component;
        AssertIndexRange(component, 1);
        Assert(dim == 2, ExcNotImplemented());

        const double time = this->get_time();
        const double point_within_period =
          (time / period - std::floor(time / period));

        if ((point_within_period >= 0.0) && (point_within_period <= 0.2))
          {
            if ((p[0] > 0.5) && (p[1] > -0.5))
              return 1;
            else
              return 0;
          }
        else if ((point_within_period >= 0.5) && (point_within_period <= 0.7))
          {
            if ((p[0] > -0.5) && (p[1] > 0.5))
              return 1;
            else
              return 0;
          }
        else
          return 0;
      }

    private:
      const double period;
    };
  };
} // namespace Step26

int
main()
{
  try
    {
      using namespace Step26;

      HeatEquation<2> heat_equation_solver;
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
