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
#include <deal.II/base/convergence_table.h>
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
  namespace LinearAlgebra
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
        const Number temp = VT::l2_norm();
        return std::sqrt(Utilities::MPI::sum(temp * temp, row_comm));
      }

      virtual Number
      add_and_dot(const Number                     a,
                  const VectorSpaceVector<Number> &V,
                  const VectorSpaceVector<Number> &W) override
      {
        const Number temp = VT::add_and_dot(a, V, W);
        return Utilities::MPI::sum(temp, row_comm);
      }

      virtual Number
      operator*(const VectorSpaceVector<Number> &V) const override
      {
        const Number temp = VT::operator*(V);
        return Utilities::MPI::sum(temp, row_comm);
      }

      const MPI_Comm &
      get_row_mpi_communicator() const
      {
        return row_comm;
      }

    private:
      MPI_Comm row_comm;
    };
  } // namespace LinearAlgebra

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

    virtual void
    get_statistics(ConvergenceTable &table) const = 0;
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
      (void)timestep_number;

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
      system_rhs.add((1 - theta) * time_step, tmp);

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
      system_matrix.add(-(theta * time_step), laplace_matrix);

      // solve system
      SolverControl        solver_control(1000, 1e-8 * system_rhs.l2_norm());
      SolverCG<VectorType> cg(solver_control);

      // ... create operator
      SystemMatrix sm(system_matrix);

      // ... create preconditioner
      Preconditioner preconditioner(system_matrix);

      // ... solve
      cg.solve(sm, solution, system_rhs, preconditioner);

      pcout << "   " << solver_control.last_step() << " CG iterations."
            << std::endl;
    }

    void
    get_statistics(ConvergenceTable &table) const override
    {
      (void)table;
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


  class MassLaplaceOperator
  {
  public:
    MassLaplaceOperator(const SparseMatrixType &mass_matrix,
                        const SparseMatrixType &laplace_matrix)
      : mass_matrix(mass_matrix)
      , laplace_matrix(laplace_matrix)
    {}

    void
    vmult(VectorType &      dst,
          const VectorType &src,
          const double      mass_matrix_scaling,
          const double      laplace_matrix_scaling) const
    {
      dst = 0.0; // TODO
      this->vmult_add(dst, src, mass_matrix_scaling, laplace_matrix_scaling);
    }

    void
    vmult_add(VectorType &      dst,
              const VectorType &src,
              const double      mass_matrix_scaling,
              const double      laplace_matrix_scaling) const
    {
      tmp.reinit(src, true);

      if (mass_matrix_scaling == 0.0)
        {
          // nothing to do
        }
      else if (mass_matrix_scaling == 1.0)
        {
          mass_matrix.vmult_add(dst, src);
        }
      else
        {
          mass_matrix.vmult(tmp, src);
          dst.add(mass_matrix_scaling, tmp);
        }

      if (laplace_matrix_scaling == 0.0)
        {
          // nothing to do
        }
      else if (laplace_matrix_scaling == 0.0)
        {
          laplace_matrix.vmult_add(dst, src);
        }
      else
        {
          laplace_matrix.vmult(tmp, src);
          dst.add(laplace_matrix_scaling, tmp);
        }
    }

  private:
    const SparseMatrixType &mass_matrix;
    const SparseMatrixType &laplace_matrix;

    mutable VectorType tmp;
  };



  /**
   * IRK base class.
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

    void
    get_statistics(ConvergenceTable &table) const override
    {
      table.add_value("time", time_total / 1e9);
      table.set_scientific("time", true);
      table.add_value("time_rhs", time_rhs / 1e9);
      table.set_scientific("time_rhs", true);
      table.add_value("time_outer_solver", time_outer_solver / 1e9);
      table.set_scientific("time_outer_solver", true);
      table.add_value("time_solution_update", time_solution_update / 1e9);
      table.set_scientific("time_solution_update", true);
      table.add_value("time_system_vmult", time_system_vmult / 1e9);
      table.set_scientific("time_system_vmult", true);
      table.add_value("time_preconditioner_bc", time_preconditioner_bc / 1e9);
      table.set_scientific("time_preconditioner_bc", true);
      table.add_value("time_preconditioner_solver",
                      time_preconditioner_solver / 1e9);
      table.set_scientific("time_preconditioner_solver", true);
    }

  private:
    static FullMatrix<typename VectorType::value_type>
    load_matrix_from_file(const unsigned int n_stages, const std::string label)
    {
      FullMatrix<typename VectorType::value_type> result(n_stages, n_stages);

      std::string file_name = label + std::to_string(n_stages) + ".txt";

      std::ifstream fin(file_name);

      AssertThrow(fin.fail() == false,
                  ExcMessage("File with the name " + file_name +
                             " could not be found!"));

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

      std::string file_name = label + std::to_string(n_stages) + ".txt";

      std::ifstream fin(file_name);

      AssertThrow(fin.fail() == false,
                  ExcMessage("File with the name " + file_name +
                             " could not be found!"));

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

    mutable double time_total                 = 0.0;
    mutable double time_rhs                   = 0.0;
    mutable double time_outer_solver          = 0.0;
    mutable double time_solution_update       = 0.0;
    mutable double time_system_vmult          = 0.0;
    mutable double time_preconditioner_bc     = 0.0;
    mutable double time_preconditioner_solver = 0.0;
  };



  /**
   * A parallel IRK implementation.
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
      (void)timestep_number;

      AssertThrow((this->time_step == 0 || this->time_step == time_step),
                  ExcNotImplemented());

      this->time_step = time_step;

      if (system_matrix == nullptr)
        {
          this->system_matrix = std::make_unique<SystemMatrix>(
            A_inv, time_step, mass_matrix, laplace_matrix, time_system_vmult);
          this->preconditioner =
            std::make_unique<Preconditioner>(d_vec,
                                             T,
                                             T_inv,
                                             time_step,
                                             mass_matrix,
                                             laplace_matrix,
                                             time_preconditioner_bc,
                                             time_preconditioner_solver);
        }

      const auto time_total = std::chrono::system_clock::now();
      const auto time_rhs   = std::chrono::system_clock::now();

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

      this->time_rhs += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now() - time_rhs)
                          .count();

      const auto time_outer_solver = std::chrono::system_clock::now();

      // solve system
      SolverControl solver_control(n_max_iterations,
                                   rel_tolerance *
                                     system_rhs.l2_norm() /*TODO*/);

      SolverFGMRES<BlockVectorType> cg(solver_control);

      cg.solve(*system_matrix, system_solution, system_rhs, *preconditioner);

      this->time_outer_solver +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now() - time_outer_solver)
          .count();

      pcout << "   " << solver_control.last_step()
            << " outer FGMRES iterations and "
            << preconditioner->get_n_iterations_and_clear()
            << " inner CG iterations." << std::endl;

      const auto time_solution_update = std::chrono::system_clock::now();

      // accumulate result in solution
      for (unsigned int i = 0; i < n_stages; ++i)
        solution.add(time_step * b_vec[i], system_solution.block(i));

      this->time_solution_update +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now() - time_solution_update)
          .count();

      this->time_total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now() - time_total)
                            .count();
    }

  private:
    class SystemMatrix
    {
    public:
      SystemMatrix(const FullMatrix<typename VectorType::value_type> &A_inv,
                   const double                                       time_step,
                   const SparseMatrixType &mass_matrix,
                   const SparseMatrixType &laplace_matrix,
                   double &                time)
        : n_stages(A_inv.m())
        , A_inv(A_inv)
        , time_step(time_step)
        , op(mass_matrix, laplace_matrix)
        , time(time)
      {}

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const
      {
        const auto time = std::chrono::system_clock::now();

        VectorType tmp;
        tmp.reinit(src.block(0));

        dst = 0;
        for (unsigned int i = 0; i < n_stages; ++i)
          for (unsigned int j = 0; j < n_stages; ++j)
            {
              const unsigned int k = (j + i) % n_stages;
              if (j == 0) // first process diagonal
                op.vmult(dst.block(i), src.block(k), A_inv(i, k), -time_step);
              else // proceed with off-diagonals
                op.vmult_add(dst.block(i), src.block(k), A_inv(i, k), 0.0);
            }


        this->time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now() - time)
                        .count();
      }

    private:
      const unsigned int                                 n_stages;
      const FullMatrix<typename VectorType::value_type> &A_inv;
      const double                                       time_step;
      const MassLaplaceOperator                          op;

      double &time;
    };

    class Preconditioner
    {
    public:
      Preconditioner(const Vector<typename VectorType::value_type> &    d_vec,
                     const FullMatrix<typename VectorType::value_type> &T,
                     const FullMatrix<typename VectorType::value_type> &T_inv,
                     const double            time_step,
                     const SparseMatrixType &mass_matrix,
                     const SparseMatrixType &laplace_matrix,
                     double &                time_bc,
                     double &                time_solver)
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
        , time_bc(time_bc)
        , time_solver(time_solver)
        , n_iterations(0)
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
        const auto time_bc_0 = std::chrono::system_clock::now();

        dst = 0;
        for (unsigned int i = 0; i < n_stages; ++i)
          for (unsigned int j = 0; j < n_stages; ++j)
            if (std::abs(T_mat_inv(i, j)) > cut_off_tolerance)
              dst.block(i).add(T_mat_inv(i, j), src.block(j));

        this->time_bc += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now() - time_bc_0)
                           .count();

        const auto time_solver = std::chrono::system_clock::now();

        BlockVectorType tmp_vectors; // TODO
        tmp_vectors.reinit(src);     //

        for (unsigned int i = 0; i < n_stages; ++i)
          {
            SolverControl solver_control(n_max_iterations, abs_tolerance);
            SolverCG<VectorType> solver(solver_control);

            solver.solve(operators[i],
                         tmp_vectors.block(i),
                         dst.block(i),
                         preconditioners[i]);

            n_iterations += solver_control.last_step();
          }

        this->time_solver +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now() - time_solver)
            .count();

        const auto time_bc_1 = std::chrono::system_clock::now();

        dst = 0;
        for (unsigned int i = 0; i < n_stages; ++i)
          for (unsigned int j = 0; j < n_stages; ++j)
            if (std::abs(T_mat(i, j)) > cut_off_tolerance)
              dst.block(i).add(T_mat(i, j), tmp_vectors.block(j));

        this->time_bc += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now() - time_bc_1)
                           .count();
      }

      unsigned
      get_n_iterations_and_clear()
      {
        const unsigned int temp = n_iterations;
        this->n_iterations      = 0;
        return temp;
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

      double &time_bc;
      double &time_solver;

      mutable unsigned int n_iterations;
    };

    const unsigned int n_max_iterations;
    const double       rel_tolerance;

    mutable double time_step = 0.0;

    mutable std::unique_ptr<SystemMatrix>   system_matrix;
    mutable std::unique_ptr<Preconditioner> preconditioner;
  };



  /**
   * A stage-parallel IRK implementation.
   */
  class IRKStageParallel : public IRKBase
  {
  public:
    using ReshapedVectorType = LinearAlgebra::ReshapedVector<VectorType>;

    IRKStageParallel(const MPI_Comm          comm_global,
                     const MPI_Comm          comm_row,
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
      , n_max_iterations(1000)
      , rel_tolerance(1e-8)
    {}

    void
    solve(VectorType &       solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const override
    {
      (void)timestep_number;

      AssertThrow((this->time_step == 0 || this->time_step == time_step),
                  ExcNotImplemented());

      this->time_step = time_step;

      // ... create operator and preconditioner
      if (system_matrix == nullptr)
        {
          this->system_matrix = std::make_unique<SystemMatrix>(
            A_inv, time_step, mass_matrix, laplace_matrix, time_system_vmult);
          this->preconditioner =
            std::make_unique<Preconditioner>(comm_row,
                                             d_vec,
                                             T,
                                             T_inv,
                                             time_step,
                                             mass_matrix,
                                             laplace_matrix,
                                             time_preconditioner_bc,
                                             time_preconditioner_solver);
        }

      const auto time_total = std::chrono::system_clock::now();
      const auto time_rhs   = std::chrono::system_clock::now();

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
      perform_basis_change(system_rhs, system_rhs, A_inv);

      this->time_rhs += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now() - time_rhs)
                          .count();

      const auto time_outer_solver = std::chrono::system_clock::now();

      // solve system
      SolverControl solver_control(n_max_iterations,
                                   rel_tolerance *
                                     system_rhs.l2_norm() /*TODO*/);

      SolverFGMRES<ReshapedVectorType> cg(solver_control);

      cg.solve(*system_matrix, system_solution, system_rhs, *preconditioner);

      this->time_outer_solver +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now() - time_outer_solver)
          .count();

      const double n_inner_iterations =
        preconditioner->get_n_iterations_and_clear();
      const auto n_inner_iterations_min_max_avg =
        Utilities::MPI::min_max_avg(n_inner_iterations, comm_row);

      pcout << "   " << solver_control.last_step()
            << " outer FGMRES iterations and "
            << static_cast<unsigned int>(n_inner_iterations_min_max_avg.min)
            << "/" << n_inner_iterations_min_max_avg.avg << "/"
            << static_cast<unsigned int>(n_inner_iterations_min_max_avg.max)
            << " inner CG iterations." << std::endl;

      const auto time_solution_update = std::chrono::system_clock::now();

      // accumulate result in solution
      if (my_stage == 0)
        solution.add(time_step * b_vec[my_stage], system_solution);
      else
        solution.equ(time_step * b_vec[my_stage], system_solution);

      MPI_Allreduce(MPI_IN_PLACE,
                    solution.get_values(),
                    solution.locally_owned_size(),
                    MPI_DOUBLE,
                    MPI_SUM,
                    comm_row);

      this->time_solution_update +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now() - time_solution_update)
          .count();

      this->time_total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now() - time_total)
                            .count();
    }

  private:
    template <typename VectorType>
    static void
    matrix_vector_rol_operation(
      LinearAlgebra::ReshapedVector<VectorType> &      dst,
      const LinearAlgebra::ReshapedVector<VectorType> &src,
      std::function<void(unsigned int,
                         unsigned int,
                         LinearAlgebra::ReshapedVector<VectorType> &,
                         const LinearAlgebra::ReshapedVector<VectorType> &)> fu)
    {
      const auto         comm  = src.get_row_mpi_communicator();
      const unsigned int rank  = Utilities::MPI::this_mpi_process(comm);
      const unsigned int nproc = Utilities::MPI::n_mpi_processes(comm);

      LinearAlgebra::ReshapedVector<VectorType> temp;
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
    perform_basis_change(LinearAlgebra::ReshapedVector<VectorType> &       dst,
                         const LinearAlgebra::ReshapedVector<VectorType> & src,
                         const FullMatrix<typename VectorType::value_type> T)
    {
      const auto fu =
        [&T](const auto i, const auto j, auto &dst, const auto &src) {
          if (i == j)
            dst.equ(T[i][j], src);
          else
            dst.add(T[i][j], src);
        };

      matrix_vector_rol_operation<VectorType>(dst, src, fu);
    }

    class SystemMatrix
    {
    public:
      SystemMatrix(const FullMatrix<typename VectorType::value_type> &A_inv,
                   const double                                       time_step,
                   const SparseMatrixType &mass_matrix,
                   const SparseMatrixType &laplace_matrix,
                   double &                time)
        : A_inv(A_inv)
        , time_step(time_step)
        , mass_matrix(mass_matrix)
        , laplace_matrix(laplace_matrix)
        , time(time)
      {}

      void
      vmult(ReshapedVectorType &dst, const ReshapedVectorType &src) const
      {
        const auto time = std::chrono::system_clock::now();

        ReshapedVectorType temp;
        temp.reinit(src);

        matrix_vector_rol_operation<VectorType>(
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

        this->time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now() - time)
                        .count();
      }

    private:
      const FullMatrix<typename VectorType::value_type> &A_inv;
      const double                                       time_step;
      const SparseMatrixType &                           mass_matrix;
      const SparseMatrixType &                           laplace_matrix;

      double &time;
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
                     const SparseMatrixType &laplace_matrix,
                     double &                time_bc,
                     double &                time_solver)
        : n_max_iterations(100)
        , abs_tolerance(1e-6)
        , d_vec(d_vec)
        , T_mat(T)
        , T_mat_inv(T_inv)
        , tau(time_step)
        , mass_matrix(mass_matrix)
        , laplace_matrix(laplace_matrix)
        , time_bc(time_bc)
        , time_solver(time_solver)
        , n_iterations(0)
      {
        const auto my_stage = Utilities::MPI::this_mpi_process(comm_row);

        linear_operator.copy_from(laplace_matrix);
        linear_operator *= -tau;
        linear_operator.add(d_vec[my_stage], mass_matrix);

        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        preconditioners.initialize(linear_operator, amg_data);
      }

      void
      vmult(ReshapedVectorType &dst, const ReshapedVectorType &src) const
      {
        const auto time_bc_0 = std::chrono::system_clock::now();

        perform_basis_change(dst, src, T_mat_inv);

        this->time_bc += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now() - time_bc_0)
                           .count();

        const auto time_solver = std::chrono::system_clock::now();

        ReshapedVectorType temp; // TODO
        temp.reinit(src);        //

        SolverControl        solver_control(n_max_iterations, abs_tolerance);
        SolverCG<VectorType> solver(solver_control);

        solver.solve(linear_operator,
                     static_cast<VectorType &>(temp),
                     static_cast<const VectorType &>(dst),
                     preconditioners);

        n_iterations += solver_control.last_step();

        this->time_solver +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now() - time_solver)
            .count();

        const auto time_bc_1 = std::chrono::system_clock::now();

        perform_basis_change(dst, temp, T_mat);

        this->time_bc += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now() - time_bc_1)
                           .count();
      }

      unsigned
      get_n_iterations_and_clear()
      {
        const unsigned int temp = this->n_iterations;
        this->n_iterations      = 0;
        return temp;
      }

    private:
      const unsigned int n_max_iterations;
      const double       abs_tolerance;

      const Vector<typename VectorType::value_type> &    d_vec;
      const FullMatrix<typename VectorType::value_type> &T_mat;
      const FullMatrix<typename VectorType::value_type> &T_mat_inv;

      const double tau;

      const SparseMatrixType &mass_matrix;
      const SparseMatrixType &laplace_matrix;

      SparseMatrixType                  linear_operator;
      TrilinosWrappers::PreconditionAMG preconditioners;

      double &time_bc;
      double &time_solver;

      mutable unsigned int n_iterations;
    };

    const MPI_Comm comm_row;

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

    bool do_output_paraview = false;

    void
    parse(const std::string file_name)
    {
      dealii::ParameterHandler prm;
      prm.add_parameter("FEDegree", fe_degree);
      prm.add_parameter("NRefinements", n_refinements);
      prm.add_parameter("TimeIntegrationScheme",
                        time_integration_scheme,
                        "",
                        Patterns::Selection("ost|irk|spirk"));
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
            const MPI_Comm    comm_column,
            ConvergenceTable &table)
      : params(params)
      , comm_global(comm_global)
      , comm_row(comm_row)
      , comm_column(comm_column)
      , triangulation(comm_column)
      , fe(params.fe_degree)
      , quadrature(params.fe_degree + 1)
      , dof_handler(triangulation)
      , table(table)
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
      };

      // select time-integration scheme
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
            params.irk_stages,
            mass_matrix,
            laplace_matrix,
            evaluate_rhs_function);
      else
        Assert(false, ExcNotImplemented());

      // perform time loop
      while (time <= params.end_time)
        {
          pcout << std::endl
                << "Time step " << timestep_number << " at t=" << time
                << std::endl;

          time += params.time_step_size;
          ++timestep_number;

          time_integration_scheme->solve(solution,
                                         timestep_number,
                                         time,
                                         params.time_step_size);

          constraints.distribute(solution);

          output_results(time, timestep_number);
        }

      time_integration_scheme->get_statistics(table);
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

      table.add_value("n_levels", triangulation.n_global_levels());
      table.add_value("n_cells", triangulation.n_global_active_cells());
      table.add_value("n_dofs", dof_handler.n_dofs());

      constraints.clear();

      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
      constraints.reinit(locally_relevant_dofs);

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
      if (params.do_output_paraview)
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
        }

      if (true)
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
          pcout << "   Error in the L2 norm : " << error_norm << std::endl;
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

    ConvergenceTable & table;
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
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      constexpr unsigned int dim = 2;

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
#ifdef DEBUG
          std::cout << "Running in debug mode!" << std::endl;
#endif
        }

      dealii::ConditionalOStream pcout(std::cout,
                                       dealii::Utilities::MPI::this_mpi_process(
                                         MPI_COMM_WORLD) == 0);

      if (argc == 1)
        {
          if (pcout.is_active())
            printf("ERROR: No .json parameter files has been provided!\n");

          return 1;
        }

      ConvergenceTable table;

      for (int i = 1; i < argc; ++i)
        {
          pcout << std::string(argv[i]) << std::endl;

          HeatEquation::Parameters params;
          params.parse(std::string(argv[i]));

          const unsigned int size_x =
            params.time_integration_scheme == "spirk" ? params.irk_stages : 1;
          const unsigned int size_v =
            Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) / size_x;

          AssertThrow(size_v > 0,
                      ExcMessage("Not enough ranks have been provided!"));

          MPI_Comm comm_global =
            Utilities::MPI::create_rectangular_comm(MPI_COMM_WORLD,
                                                    size_x,
                                                    size_v);

          if (comm_global != MPI_COMM_NULL)
            {
              MPI_Comm comm_row =
                Utilities::MPI::create_row_comm(comm_global, size_x, size_v);
              MPI_Comm comm_column =
                Utilities::MPI::create_column_comm(comm_global, size_x, size_v);


              HeatEquation::Problem<dim> heat_equation_solver(
                params, comm_global, comm_row, comm_column, table);
              heat_equation_solver.run();

              MPI_Comm_free(&comm_column);
              MPI_Comm_free(&comm_row);
              MPI_Comm_free(&comm_global);
            }

          if (pcout.is_active())
            table.write_text(pcout.get_stream());
        }

      if (pcout.is_active())
        table.write_text(pcout.get_stream());
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
