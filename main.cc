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


namespace Step26
{
  using namespace dealii;

  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation();
    void
    run();

  private:
    void
    setup_system();
    void
    solve_time_step();
    void
    output_results() const;

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
  };



  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>()
      , period(0.2)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

  private:
    const double period;
  };



  template <int dim>
  double
  RightHandSide<dim>::value(const Point<dim> & p,
                            const unsigned int component) const
  {
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



  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
  };



  template <int dim>
  double
  BoundaryValues<dim>::value(const Point<dim> & /*p*/,
                             const unsigned int component) const
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }



  template <int dim>
  class TimeIntegrationScheme
  {
  public:
    TimeIntegrationScheme(const DoFHandler<dim> &          dof_handler,
                          const AffineConstraints<double> &constraints,
                          const SparseMatrix<double> &     mass_matrix,
                          const SparseMatrix<double> &     laplace_matrix)
      : theta(0.5)
      , dof_handler(dof_handler)
      , constraints(constraints)
      , mass_matrix(mass_matrix)
      , laplace_matrix(laplace_matrix)
    {}

    void
    solve(Vector<double> &   solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const
    {
      std::cout << "Time step " << timestep_number << " at t=" << time
                << std::endl;

      SparseMatrix<double> system_matrix;
      Vector<double>       system_rhs;
      Vector<double>       tmp;
      Vector<double>       forcing_terms;

      system_matrix.reinit(mass_matrix.get_sparsity_pattern());
      system_rhs.reinit(dof_handler.n_dofs());
      tmp.reinit(dof_handler.n_dofs());
      forcing_terms.reinit(dof_handler.n_dofs());

      // create right-hand-side vector
      mass_matrix.vmult(system_rhs, solution);

      laplace_matrix.vmult(tmp, solution);
      system_rhs.add(-(1 - theta) * time_step, tmp);

      RightHandSide<dim> rhs_function;
      rhs_function.set_time(time);
      VectorTools::create_right_hand_side(dof_handler,
                                          QGauss<dim>(
                                            dof_handler.get_fe().degree + 1),
                                          rhs_function,
                                          tmp);
      forcing_terms = tmp;
      forcing_terms *= time_step * theta;

      rhs_function.set_time(time - time_step);
      VectorTools::create_right_hand_side(dof_handler,
                                          QGauss<dim>(
                                            dof_handler.get_fe().degree + 1),
                                          rhs_function,
                                          tmp,
                                          constraints);

      forcing_terms.add(time_step * (1 - theta), tmp);

      system_rhs += forcing_terms;

      // setup matrix
      system_matrix.copy_from(mass_matrix);
      system_matrix.add(theta * time_step, laplace_matrix);

      SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
      SolverCG<Vector<double>> cg(solver_control);

      PreconditionSSOR<SparseMatrix<double>> preconditioner;
      preconditioner.initialize(system_matrix, 1.0);

      cg.solve(system_matrix, solution, system_rhs, preconditioner);

      constraints.distribute(solution);

      std::cout << "     " << solver_control.last_step() << " CG iterations."
                << std::endl;
    }

  private:
    const double theta;

    const DoFHandler<dim> &          dof_handler;
    const AffineConstraints<double> &constraints;

    const SparseMatrix<double> &mass_matrix;
    const SparseMatrix<double> &laplace_matrix;
  };



  template <int dim>
  HeatEquation<dim>::HeatEquation()
    : fe(1)
    , dof_handler(triangulation)
    , time_step(1. / 500)
  {}



  template <int dim>
  void
  HeatEquation<dim>::setup_system()
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
    DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix<dim, dim, double>(dof_handler,
                                                        QGauss<dim>(fe.degree +
                                                                    1),
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



  template <int dim>
  void
  HeatEquation<dim>::output_results() const
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



  template <int dim>
  void
  HeatEquation<dim>::run()
  {
    const unsigned int n_refinements = 4;

    GridGenerator::hyper_L(triangulation);
    triangulation.refine_global(n_refinements);

    setup_system();

    time            = 0.0;
    timestep_number = 0;

    VectorTools::interpolate(dof_handler,
                             Functions::ZeroFunction<dim>(),
                             solution);

    output_results();

    auto time_integration_scheme = std::make_unique<TimeIntegrationScheme<dim>>(
      dof_handler, constraints, mass_matrix, laplace_matrix);

    while (time <= 0.5)
      {
        time += time_step;
        ++timestep_number;

        time_integration_scheme->solve(solution,
                                       timestep_number,
                                       time,
                                       time_step);

        output_results();
      }
  }
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
