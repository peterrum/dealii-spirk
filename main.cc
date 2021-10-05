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
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.templates.h>

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
  template <typename VT>
  class ReshapedVector : public VT
  {
  public:
    using Number = typename VT::value_type;

    virtual ReshapedVector<VT> &
    operator=(const Number s) override
    {
      VT::operator=(s);

      return *this;
    }

    void
    reinit(const ReshapedVector<VT> &V)
    {
      VT::reinit(V);
      this->row_comm = V.row_comm;
    }

    void
    reinit(const ReshapedVector<VT> &V, const bool omit_zeroing_entries)
    {
      VT::reinit(V, omit_zeroing_entries);
      this->row_comm = V.row_comm;
    }

    void
    reinit(const VT &V, const MPI_Comm &row_comm)
    {
      VT::reinit(V);
      this->row_comm = row_comm;
    }

    virtual Number
    l2_norm() const override
    {
      return Utilities::MPI::sum(VT::l2_norm(), row_comm);
    }

  private:
    MPI_Comm row_comm;
  };

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
                  cell_matrix(i, j) -=
                    (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) *
                     fe_values.JxW(q)); // TODO: make addition again
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
    OneStepTheta(const MPI_Comm          comm,
                 const SparseMatrixType &mass_matrix,
                 const SparseMatrixType &laplace_matrix,
                 const std::function<void(const double, VectorType &)>
                   &evaluate_rhs_function)
      : theta(0.5)
      , mass_matrix(mass_matrix)
      , laplace_matrix(laplace_matrix)
      , evaluate_rhs_function(evaluate_rhs_function)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
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
  class IRKBase : public Interface
  {
  public:
    IRKBase(const MPI_Comm          comm,
            const unsigned int      n_stages,
            const SparseMatrixType &mass_matrix,
            const SparseMatrixType &laplace_matrix,
            const std::function<void(const double, VectorType &)>
              &evaluate_rhs_function)
      : n_stages(n_stages)
      , A_inv(load_matrix_from_file(n_stages, "A_inv"))
      , T(load_matrix_from_file(n_stages, "T"))
      , T_inv(load_matrix_from_file(n_stages, "T_inv"))
      , b_vec(load_vector_from_file(n_stages, "b_vec_"))
      , c_vec(load_vector_from_file(n_stages, "c_vec_"))
      , d_vec(load_vector_from_file(n_stages, "D_vec_"))
      , mass_matrix(mass_matrix)
      , laplace_matrix(laplace_matrix)
      , evaluate_rhs_function(evaluate_rhs_function)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
    {}

  private:
    static FullMatrix<typename VectorType::value_type>
    load_matrix_from_file(const unsigned int n_stages, const std::string label)
    {
      FullMatrix<typename VectorType::value_type> result(n_stages, n_stages);

      std::ifstream fin(label + std::to_string(n_stages) + ".txt");

      unsigned int m, n;
      fin >> m >> n;

      AssertDimension(m, n_stages);
      AssertDimension(n, n_stages);

      for (unsigned int i = 0; i < n_stages; ++i)
        for (unsigned j = 0; j < n_stages; ++j)
          fin >> result[i][j];

      return result;
    }

    static Vector<typename VectorType::value_type>
    load_vector_from_file(const unsigned int n_stages, const std::string label)
    {
      Vector<typename VectorType::value_type> result(n_stages);

      std::ifstream fin(label + std::to_string(n_stages) + ".txt");

      unsigned int m, n;
      fin >> m >> n;

      AssertDimension(m, 1);
      AssertDimension(n, n_stages);

      for (unsigned int i = 0; i < n_stages; ++i)
        fin >> result[i];

      return result;
    }

  protected:
    const unsigned int                                n_stages;
    const FullMatrix<typename VectorType::value_type> A_inv;
    const FullMatrix<typename VectorType::value_type> T;
    const FullMatrix<typename VectorType::value_type> T_inv;
    const Vector<typename VectorType::value_type>     b_vec;
    const Vector<typename VectorType::value_type>     c_vec;
    const Vector<typename VectorType::value_type>     d_vec;

    const SparseMatrixType &mass_matrix;
    const SparseMatrixType &laplace_matrix;

    const std::function<void(const double, VectorType &)> evaluate_rhs_function;

    ConditionalOStream pcout;
  };



  /**
   * IRK implementation.
   */
  class IRK : public IRKBase
  {
  public:
    IRK(const MPI_Comm          comm,
        const unsigned int      n_stages,
        const SparseMatrixType &mass_matrix,
        const SparseMatrixType &laplace_matrix,
        const std::function<void(const double, VectorType &)>
          &evaluate_rhs_function)
      : IRKBase(comm,
                n_stages,
                mass_matrix,
                laplace_matrix,
                evaluate_rhs_function)
      , n_max_iterations(1000)
      , rel_tolerance(1e-8)
    {}

    void
    solve(VectorType &       solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const override
    {
      pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

      AssertThrow((this->time_step == 0 || this->time_step == time_step),
                  ExcNotImplemented());

      this->time_step = time_step;

      BlockVectorType system_rhs(n_stages);      // TODO
      BlockVectorType system_solution(n_stages); //
      VectorType      tmp;                       //

      for (unsigned int i = 0; i < n_stages; ++i)
        {
          system_rhs.block(i).reinit(solution);
          system_solution.block(i).reinit(solution);
        }
      tmp.reinit(solution);

      for (unsigned int i = 0; i < n_stages; ++i)
        evaluate_rhs_function(time + (c_vec[i] - 1.0) * time_step,
                              system_rhs.block(i));

      laplace_matrix.vmult(tmp, solution);

      for (unsigned int i = 0; i < n_stages; ++i)
        system_rhs.block(i).add(1.0, tmp);

      {
        std::vector<typename VectorType::value_type> values(n_stages);

        for (const auto e : solution.locally_owned_elements())
          {
            for (unsigned int j = 0; j < n_stages; ++j)
              values[j] = system_rhs.block(j)[e];

            for (unsigned int i = 0; i < n_stages; ++i)
              {
                system_rhs.block(i)[e] = 0.0;
                for (unsigned int j = 0; j < n_stages; ++j)
                  system_rhs.block(i)[e] += A_inv[i][j] * values[j];
              }
          }
      }

      // solve system
      SolverControl solver_control(n_max_iterations,
                                   rel_tolerance *
                                     system_rhs.l2_norm() /*TODO*/);

      SolverFGMRES<BlockVectorType> cg(solver_control);

      // ... create operator and preconditioner
      if (system_matrix == nullptr)
        {
          this->system_matrix  = std::make_unique<SystemMatrix>(A_inv,
                                                               time_step,
                                                               mass_matrix,
                                                               laplace_matrix);
          this->preconditioner = std::make_unique<Preconditioner>(
            d_vec, T, T_inv, time_step, mass_matrix, laplace_matrix);
        }

      // ... solve
      cg.solve(*system_matrix, system_solution, system_rhs, *preconditioner);

      pcout << "     " << solver_control.last_step() << " CG iterations."
            << std::endl;

      // accumulate result in solution
      for (unsigned int i = 0; i < n_stages; ++i)
        solution.add(time_step * b_vec[i], system_solution.block(i));
    }

  private:
    class SystemMatrix
    {
    public:
      SystemMatrix(const FullMatrix<typename VectorType::value_type> &A_inv,
                   const double                                       time_step,
                   const SparseMatrixType &mass_matrix,
                   const SparseMatrixType &laplace_matrix)
        : n_stages(A_inv.m())
        , A_inv(A_inv)
        , time_step(time_step)
        , mass_matrix(mass_matrix)
        , laplace_matrix(laplace_matrix)
      {}

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const
      {
        VectorType tmp;
        tmp.reinit(src.block(0));

        dst = 0;
        for (unsigned int i = 0; i < n_stages; ++i)
          for (unsigned int j = 0; j < n_stages; ++j)
            {
              mass_matrix.vmult(tmp, src.block(j));
              dst.block(i).add(A_inv(i, j), tmp);

              if (i == j)
                {
                  laplace_matrix.vmult(tmp, src.block(j));
                  dst.block(i).add(-time_step, tmp);
                }
            }
      }

    private:
      const unsigned int                                 n_stages;
      const FullMatrix<typename VectorType::value_type> &A_inv;
      const double                                       time_step;
      const SparseMatrixType &                           mass_matrix;
      const SparseMatrixType &                           laplace_matrix;
    };

    class Preconditioner
    {
    public:
      Preconditioner(const Vector<typename VectorType::value_type> &    d_vec,
                     const FullMatrix<typename VectorType::value_type> &T,
                     const FullMatrix<typename VectorType::value_type> &T_inv,
                     const double            time_step,
                     const SparseMatrixType &mass_matrix,
                     const SparseMatrixType &laplace_matrix)
        : n_max_iterations(100)
        , abs_tolerance(1e-6)
        , cut_off_tolerance(1e-12)
        , n_stages(d_vec.size())
        , d_vec(d_vec)
        , T_mat(T)
        , T_mat_inv(T_inv)
        , tau(time_step)
        , mass_matrix(mass_matrix)
        , laplace_matrix(laplace_matrix)
      {
        operators.resize(n_stages);
        preconditioners.resize(n_stages);

        for (unsigned int i = 0; i < n_stages; ++i)
          {
            operators[i].copy_from(laplace_matrix);
            operators[i] *= -tau;
            operators[i].add(d_vec[i], mass_matrix);

            TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
            preconditioners[i].initialize(operators[i], amg_data);
          }
      }

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const
      {
        BlockVectorType tmp_vectors; // TODO
        tmp_vectors.reinit(src);     //

        dst = 0;
        for (unsigned int i = 0; i < n_stages; ++i)
          for (unsigned int j = 0; j < n_stages; ++j)
            if (std::abs(T_mat_inv(i, j)) > cut_off_tolerance)
              dst.block(i).add(T_mat_inv(i, j), src.block(j));

        for (unsigned int i = 0; i < n_stages; ++i)
          {
            SolverControl solver_control(n_max_iterations, abs_tolerance);
            SolverFGMRES<VectorType> solver(solver_control);

            solver.solve(operators[i],
                         tmp_vectors.block(i),
                         dst.block(i),
                         preconditioners[i]);
          }

        dst = 0;
        for (unsigned int i = 0; i < n_stages; ++i)
          for (unsigned int j = 0; j < n_stages; ++j)
            if (std::abs(T_mat(i, j)) > cut_off_tolerance)
              dst.block(i).add(T_mat(i, j), tmp_vectors.block(j));
      }

    private:
      const unsigned int n_max_iterations;
      const double       abs_tolerance;
      const double       cut_off_tolerance;

      const unsigned int                                 n_stages;
      const Vector<typename VectorType::value_type> &    d_vec;
      const FullMatrix<typename VectorType::value_type> &T_mat;
      const FullMatrix<typename VectorType::value_type> &T_mat_inv;

      const double tau;

      const SparseMatrixType &mass_matrix;
      const SparseMatrixType &laplace_matrix;

      std::vector<SparseMatrixType>                  operators;
      std::vector<TrilinosWrappers::PreconditionAMG> preconditioners;
    };

    const unsigned int n_max_iterations;
    const double       rel_tolerance;

    mutable double time_step = 0.0;

    mutable std::unique_ptr<SystemMatrix>   system_matrix;
    mutable std::unique_ptr<Preconditioner> preconditioner;
  };



  /**
   * IRK implementation.
   */
  class IRKStageParallel : public IRKBase
  {
  public:
    using ReshapedVectorType = ReshapedVector<VectorType>;

    IRKStageParallel(const MPI_Comm          comm_global,
                     const MPI_Comm          comm_row,
                     const MPI_Comm          comm_column,
                     const unsigned int      n_stages,
                     const SparseMatrixType &mass_matrix,
                     const SparseMatrixType &laplace_matrix,
                     const std::function<void(const double, VectorType &)>
                       &evaluate_rhs_function)
      : IRKBase(comm_global,
                n_stages,
                mass_matrix,
                laplace_matrix,
                evaluate_rhs_function)
      , comm_row(comm_row)
      , comm_column(comm_column)
      , n_max_iterations(1000)
      , rel_tolerance(1e-8)
    {}

    void
    solve(VectorType &       solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const override
    {
      pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

      AssertThrow((this->time_step == 0 || this->time_step == time_step),
                  ExcNotImplemented());

      this->time_step = time_step;

      ReshapedVectorType system_rhs, system_solution;
      VectorType         tmp;

      system_rhs.reinit(solution, comm_row);
      system_solution.reinit(solution, comm_row);
      tmp.reinit(solution);

      const unsigned int my_stage = Utilities::MPI::this_mpi_process(comm_row);

      // setup right-hand-side vector
      evaluate_rhs_function(time + (c_vec[my_stage] - 1.0) * time_step,
                            system_rhs);
      laplace_matrix.vmult(tmp, solution);
      system_rhs.add(1.0, tmp);

      // ... perform basis change
      perform_basis_chance(comm_row, system_rhs, system_rhs, A_inv);

      // solve system
      SolverControl solver_control(n_max_iterations,
                                   rel_tolerance *
                                     system_rhs.l2_norm() /*TODO*/);

      SolverFGMRES<ReshapedVectorType> cg(solver_control);

      // ... create operator and preconditioner
      if (system_matrix == nullptr)
        {
          this->system_matrix = std::make_unique<SystemMatrix>(
            comm_row, A_inv, time_step, mass_matrix, laplace_matrix);
          this->preconditioner = std::make_unique<Preconditioner>(
            comm_row, d_vec, T, T_inv, time_step, mass_matrix, laplace_matrix);
        }

      // ... solve
      cg.solve(*system_matrix, system_solution, system_rhs, *preconditioner);

      pcout << "     " << solver_control.last_step() << " CG iterations."
            << std::endl;

      // accumulate result in solution
      if (my_stage == 0)
        solution += system_solution;
      else
        solution = system_solution;

      MPI_Allreduce(MPI_IN_PLACE,
                    solution.get_values(),
                    solution.locally_owned_size(),
                    MPI_DOUBLE,
                    MPI_SUM,
                    comm_row);
    }

  private:
    template <typename VectorType>
    static void
    matrix_vector_rol_operation(
      const MPI_Comm &  comm,
      VectorType &      dst,
      const VectorType &src,
      std::function<
        void(unsigned int, unsigned int, VectorType &, const VectorType &)> fu)
    {
      const unsigned int rank  = Utilities::MPI::this_mpi_process(comm);
      const unsigned int nproc = Utilities::MPI::n_mpi_processes(comm);

      VectorType temp;
      temp.reinit(src, true);
      temp.copy_locally_owned_data_from(src);

      for (unsigned int k = 0; k < nproc; ++k)
        {
          if (k != 0)
            {
              const auto ierr = MPI_Sendrecv_replace(temp.get_values(),
                                                     temp.locally_owned_size(),
                                                     MPI_DOUBLE,
                                                     (rank + nproc - 1) % nproc,
                                                     k,
                                                     (rank + nproc + 1) % nproc,
                                                     k,
                                                     comm,
                                                     MPI_STATUS_IGNORE);

              AssertThrowMPI(ierr);
            }

          fu(rank, (k + rank) % nproc, dst, temp);
        }
    }

    template <typename VectorType>
    static void
    perform_basis_chance(const MPI_Comm &                                  comm,
                         VectorType &                                      dst,
                         const VectorType &                                src,
                         const FullMatrix<typename VectorType::value_type> T)
    {
      const auto fu =
        [&T](const auto i, const auto j, auto &dst, const auto &src) {
          if (i == j)
            dst.equ(T[i][j], src);
          else
            dst.add(T[i][j], src);
        };

      matrix_vector_rol_operation<VectorType>(comm, dst, src, fu);
    }

    class SystemMatrix
    {
    public:
      SystemMatrix(const MPI_Comm &                                   comm_row,
                   const FullMatrix<typename VectorType::value_type> &A_inv,
                   const double                                       time_step,
                   const SparseMatrixType &mass_matrix,
                   const SparseMatrixType &laplace_matrix)
        : comm_row(comm_row)
        , n_stages(A_inv.m())
        , A_inv(A_inv)
        , time_step(time_step)
        , mass_matrix(mass_matrix)
        , laplace_matrix(laplace_matrix)
      {}

      void
      vmult(ReshapedVectorType &dst, const ReshapedVectorType &src) const
      {
        ReshapedVectorType temp;
        temp.reinit(src);

        matrix_vector_rol_operation<ReshapedVectorType>(
          comm_row,
          dst,
          src,
          [this,
           &temp](const auto i, const auto j, auto &dst, const auto &src) {
            if (i == j)
              {
                laplace_matrix.vmult(static_cast<VectorType &>(temp),
                                     static_cast<const VectorType &>(src));
                dst.equ(-time_step, temp);
              }

            mass_matrix.vmult(static_cast<VectorType &>(temp),
                              static_cast<const VectorType &>(src));
            dst.add(A_inv(i, j), temp);
          });
      }

    private:
      const MPI_Comm                                     comm_row;
      const unsigned int                                 n_stages;
      const FullMatrix<typename VectorType::value_type> &A_inv;
      const double                                       time_step;
      const SparseMatrixType &                           mass_matrix;
      const SparseMatrixType &                           laplace_matrix;
    };

    class Preconditioner
    {
    public:
      Preconditioner(const MPI_Comm &                               comm_row,
                     const Vector<typename VectorType::value_type> &d_vec,
                     const FullMatrix<typename VectorType::value_type> &T,
                     const FullMatrix<typename VectorType::value_type> &T_inv,
                     const double            time_step,
                     const SparseMatrixType &mass_matrix,
                     const SparseMatrixType &laplace_matrix)
        : n_max_iterations(100)
        , abs_tolerance(1e-6)
        , cut_off_tolerance(1e-12)
        , comm_row(comm_row)
        , n_stages(d_vec.size())
        , d_vec(d_vec)
        , T_mat(T)
        , T_mat_inv(T_inv)
        , tau(time_step)
        , mass_matrix(mass_matrix)
        , laplace_matrix(laplace_matrix)
      {
        operators.resize(n_stages);
        preconditioners.resize(n_stages);

        const unsigned int i = Utilities::MPI::this_mpi_process(comm_row);

        operators[i].copy_from(laplace_matrix);
        operators[i] *= -tau;
        operators[i].add(d_vec[i], mass_matrix);

        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        preconditioners[i].initialize(operators[i], amg_data);
      }

      void
      vmult(ReshapedVectorType &dst, const ReshapedVectorType &src) const
      {
        ReshapedVectorType temp; // TODO
        temp.reinit(src);        //

        const unsigned int i = Utilities::MPI::this_mpi_process(comm_row);

        perform_basis_chance(comm_row, dst, src, T_mat_inv);

        SolverControl solver_control(n_max_iterations, abs_tolerance);
        SolverFGMRES<VectorType> solver(solver_control);

        solver.solve(operators[i],
                     static_cast<VectorType &>(temp),
                     static_cast<const VectorType &>(dst),
                     preconditioners[i]);

        perform_basis_chance(comm_row, dst, temp, T_mat);
      }

    private:
      const unsigned int n_max_iterations;
      const double       abs_tolerance;
      const double       cut_off_tolerance;

      const MPI_Comm                                     comm_row;
      const unsigned int                                 n_stages;
      const Vector<typename VectorType::value_type> &    d_vec;
      const FullMatrix<typename VectorType::value_type> &T_mat;
      const FullMatrix<typename VectorType::value_type> &T_mat_inv;

      const double tau;

      const SparseMatrixType &mass_matrix;
      const SparseMatrixType &laplace_matrix;

      std::vector<SparseMatrixType>                  operators;
      std::vector<TrilinosWrappers::PreconditionAMG> preconditioners;
    };

    const MPI_Comm comm_row;
    const MPI_Comm comm_column;

    const unsigned int n_max_iterations;
    const double       rel_tolerance;

    mutable double time_step = 0.0;

    mutable std::unique_ptr<SystemMatrix>   system_matrix;
    mutable std::unique_ptr<Preconditioner> preconditioner;
  };
} // namespace TimeIntegrationSchemes



namespace HeatEquation
{
  struct Parameters
  {
    unsigned int fe_degree     = 4;
    unsigned int n_refinements = 5;

    std::string time_integration_scheme = "ost";
    double      end_time                = 0.5;
    double      time_step_size          = 0.1;

    unsigned int irk_stages = 3;

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
      prm.add_parameter("IRKStages", irk_stages);

      std::ifstream file;
      file.open(file_name);
      prm.parse_input_from_json(file, true);
    }
  };



  template <int dim>
  class Problem
  {
  public:
    Problem(const Parameters &params,
            const MPI_Comm    comm_global,
            const MPI_Comm    comm_row,
            const MPI_Comm    comm_column)
      : params(params)
      , comm_global(comm_global)
      , comm_row(comm_row)
      , comm_column(comm_column)
      , triangulation(comm_column)
      , fe(params.fe_degree)
      , quadrature(params.fe_degree + 1)
      , dof_handler(triangulation)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(comm_global) == 0)
    {}

    void
    run()
    {
      GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(params.n_refinements);

      setup_system();

      double       time            = 0.0;
      unsigned int timestep_number = 0;

      VectorTools::interpolate(dof_handler, AnalyticalSolution(), solution);

      output_results(time, timestep_number);

      const auto evaluate_rhs_function = [&](const double time,
                                             VectorType & tmp) -> void {
        RightHandSide rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(
          dof_handler, quadrature, rhs_function, tmp, constraints);
        tmp.compress(VectorOperation::values::add); // TODO: should be done by
        // deal.II
      };

      std::unique_ptr<TimeIntegrationSchemes::Interface>
        time_integration_scheme;

      if (params.time_integration_scheme == "ost")
        time_integration_scheme =
          std::make_unique<TimeIntegrationSchemes::OneStepTheta>(
            comm_global, mass_matrix, laplace_matrix, evaluate_rhs_function);
      else if (params.time_integration_scheme == "irk")
        time_integration_scheme =
          std::make_unique<TimeIntegrationSchemes::IRK>(comm_global,
                                                        params.irk_stages,
                                                        mass_matrix,
                                                        laplace_matrix,
                                                        evaluate_rhs_function);
      else if (params.time_integration_scheme == "spirk")
        time_integration_scheme =
          std::make_unique<TimeIntegrationSchemes::IRKStageParallel>(
            comm_global,
            comm_row,
            comm_column,
            params.irk_stages,
            mass_matrix,
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
                                          AnalyticalSolution(time),
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

    const MPI_Comm comm_global;
    const MPI_Comm comm_row;
    const MPI_Comm comm_column;

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
      AnalyticalSolution(const double time = 0.0)
        : Function<dim>(1, time)
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


namespace dealii
{
  namespace Utilities
  {
    namespace MPI
    {
      std::pair<unsigned int, unsigned int>
      lex_to_pair(const unsigned int rank,
                  const unsigned int size1,
                  const unsigned int size2)
      {
        AssertThrow(rank < size1 * size2, dealii::ExcMessage("Invalid rank."));
        return {rank % size1, rank / size1};
      }



      MPI_Comm
      create_row_comm(const MPI_Comm &   comm,
                      const unsigned int size1,
                      const unsigned int size2)
      {
        int size, rank;
        MPI_Comm_size(comm, &size);
        AssertThrow(static_cast<unsigned int>(size) == size1 * size2,
                    dealii::ExcMessage("Invalid communicator size."));

        MPI_Comm_rank(comm, &rank);

        MPI_Comm row_comm;
        MPI_Comm_split(comm,
                       lex_to_pair(rank, size1, size2).second,
                       rank,
                       &row_comm);
        return row_comm;
      }



      MPI_Comm
      create_column_comm(const MPI_Comm &   comm,
                         const unsigned int size1,
                         const unsigned int size2)
      {
        int size, rank;
        MPI_Comm_size(comm, &size);
        AssertThrow(static_cast<unsigned int>(size) == size1 * size2,
                    dealii::ExcMessage("Invalid communicator size."));

        MPI_Comm_rank(comm, &rank);

        MPI_Comm col_comm;
        MPI_Comm_split(comm,
                       lex_to_pair(rank, size1, size2).first,
                       rank,
                       &col_comm);
        return col_comm;
      }



      MPI_Comm
      create_rectangular_comm(const MPI_Comm &   comm,
                              const unsigned int size_x,
                              const unsigned int size_v)
      {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        AssertThrow((size_x * size_v) <= static_cast<unsigned int>(size),
                    dealii::ExcMessage("Not enough ranks."));

        MPI_Comm sub_comm;
        MPI_Comm_split(comm,
                       (static_cast<unsigned int>(rank) < (size_x * size_v)),
                       rank,
                       &sub_comm);

        if (static_cast<unsigned int>(rank) < (size_x * size_v))
          return sub_comm;
        else
          {
            MPI_Comm_free(&sub_comm);
            return MPI_COMM_NULL;
          }
      }
    } // namespace MPI
  }   // namespace Utilities
} // namespace dealii



int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

      HeatEquation::Parameters params;

      if (argc == 2)
        params.parse(std::string(argv[1]));

      const unsigned int size_x =
        params.time_integration_scheme == "spirk" ? params.irk_stages : 1;
      const unsigned int size_v =
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) / size_x;

      MPI_Comm comm_global =
        Utilities::MPI::create_rectangular_comm(MPI_COMM_WORLD, size_x, size_v);

      if (comm_global != MPI_COMM_NULL)
        {
          MPI_Comm comm_row =
            Utilities::MPI::create_row_comm(comm_global, size_x, size_v);
          MPI_Comm comm_column =
            Utilities::MPI::create_column_comm(comm_global, size_x, size_v);


          HeatEquation::Problem<2> heat_equation_solver(params,
                                                        comm_global,
                                                        comm_row,
                                                        comm_column);
          heat_equation_solver.run();

          MPI_Comm_free(&comm_column);
          MPI_Comm_free(&comm_row);
          MPI_Comm_free(&comm_global);
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
