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
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
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
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.templates.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace dealii;

using VectorType       = LinearAlgebra::distributed::Vector<double>;
using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
using SparseMatrixType = TrilinosWrappers::SparseMatrix;

#include "include/operator.h"
#include "include/preconditioner.h"

namespace dealii
{
  template <typename VectorType>
  class SolverGCR : public SolverBase<VectorType>
  {
  public:
    SolverGCR(SolverControl &solver_control, const unsigned int GCRmaxit = 40)
      : SolverBase<VectorType>(solver_control)
      , GCRmaxit(GCRmaxit)
    {
      solver_control.set_max_steps(GCRmaxit);
    }

    template <typename MatrixType, typename PreconditionerType>
    void
    solve(const MatrixType &        A,
          VectorType &              x,
          const VectorType &        b,
          const PreconditionerType &preconditioner)
    {
      using number = typename VectorType::value_type;

      SolverControl::State conv = SolverControl::iterate;

      typename VectorMemory<VectorType>::Pointer search_pointer(this->memory);
      typename VectorMemory<VectorType>::Pointer Asearch_pointer(this->memory);
      typename VectorMemory<VectorType>::Pointer p_pointer(this->memory);

      VectorType &search  = *search_pointer;
      VectorType &Asearch = *Asearch_pointer;
      VectorType &p       = *p_pointer;

      std::vector<typename VectorType::value_type> Hn_preloc;
      Hn_preloc.reserve(GCRmaxit);

      internal::SolverGMRESImplementation::TmpVectors<VectorType> H_vec(
        GCRmaxit, this->memory);
      internal::SolverGMRESImplementation::TmpVectors<VectorType> Hd_vec(
        GCRmaxit, this->memory);

      search.reinit(x);
      Asearch.reinit(x);
      p.reinit(x);

      preconditioner.vmult(search, b);
      double res = search.l2_norm();

      A.vmult(p, x);
      p.add(-1., b);
      preconditioner.vmult(search, p);

      unsigned int it = 0;

      conv = this->iteration_status(it, res, x);
      if (conv != SolverControl::iterate)
        return;

      while (conv == SolverControl::iterate)
        {
          it++;

          H_vec(it - 1, x);
          Hd_vec(it - 1, x);

          Hn_preloc.resize(it);

          A.vmult(Asearch, search);

          for (unsigned int i = 0; i < it - 1; ++i)
            {
              const double temptest = (H_vec[i] * Asearch) / Hn_preloc[i];
              Asearch.add(-temptest, H_vec[i]);
              search.add(-temptest, Hd_vec[i]);
            }

          const double nAsearch_new = Asearch.norm_sqr();
          Hn_preloc[it - 1]         = nAsearch_new;
          H_vec[it - 1]             = Asearch;
          Hd_vec[it - 1]            = search;

          Assert(std::abs(nAsearch_new) != 0., ExcDivideByZero());

          const double c_preloc = (Asearch * p) / nAsearch_new;
          x.add(-c_preloc, search);
          p.add(-c_preloc, Asearch);

          preconditioner.vmult(search, p);

          res = search.l2_norm();

          conv = this->iteration_status(it, res, x);
        }

      if (conv != SolverControl::success)
        AssertThrow(false, SolverControl::NoConvergence(it, res));
    }

  private:
    const unsigned int GCRmaxit;
  };

  class SPSolverControl : public SolverControl
  {
  public:
    SPSolverControl(const MPI_Comm     comm,
                    const unsigned int n           = 100,
                    const double       tol         = 1.e-10,
                    const bool         log_history = false,
                    const bool         log_result  = true)
      : SolverControl(n, tol, log_history, log_result)
      , comm(comm)
    {}


    State
    check(const unsigned int step, const double check_value) override
    {
      return SolverControl::check(step, Utilities::MPI::max(check_value, comm));
    }

  private:
    const MPI_Comm comm;
  };

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

      using VT::reinit;

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
        VT::reinit(V.get_partitioner(), row_comm);
        this->row_comm = row_comm;
      }

      Number
      norm_sqr() const
      {
        const Number temp = VT::l2_norm();
        return Utilities::MPI::sum(temp * temp, row_comm);
      }

      virtual Number
      l2_norm() const override
      {
        return std::sqrt(norm_sqr());
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

  namespace Utilities
  {
    namespace MPI
    {
      std::pair<unsigned int, unsigned int>
      lex_to_pair(const unsigned int rank,
                  const unsigned int size1,
                  const unsigned int size2,
                  const bool         do_row_major)
      {
        AssertThrow(rank < size1 * size2, dealii::ExcMessage("Invalid rank."));

        if (do_row_major)
          return {rank % size1, rank / size1};
        else
          return {rank / size2, rank % size2};
      }



      MPI_Comm
      create_row_comm(const MPI_Comm &   comm,
                      const unsigned int size1,
                      const unsigned int size2,
                      const bool         do_row_major)
      {
        int size, rank;
        MPI_Comm_size(comm, &size);
        AssertThrow(static_cast<unsigned int>(size) == size1 * size2,
                    dealii::ExcMessage("Invalid communicator size."));

        MPI_Comm_rank(comm, &rank);

        MPI_Comm row_comm;
        MPI_Comm_split(comm,
                       lex_to_pair(rank, size1, size2, do_row_major).second,
                       rank,
                       &row_comm);
        return row_comm;
      }



      MPI_Comm
      create_column_comm(const MPI_Comm &   comm,
                         const unsigned int size1,
                         const unsigned int size2,
                         const bool         do_row_major)
      {
        int size, rank;
        MPI_Comm_size(comm, &size);
        AssertThrow(static_cast<unsigned int>(size) == size1 * size2,
                    dealii::ExcMessage("Invalid communicator size."));

        MPI_Comm_rank(comm, &rank);

        MPI_Comm col_comm;
        MPI_Comm_split(comm,
                       lex_to_pair(rank, size1, size2, do_row_major).first,
                       rank,
                       &col_comm);
        return col_comm;
      }



      MPI_Comm
      trim_comm(const MPI_Comm &comm, const int size)
      {
        int rank;
        MPI_Comm_rank(comm, &rank);

        const unsigned int color = rank < size;

        MPI_Comm sub_comm;
        MPI_Comm_split(comm, color, rank, &sub_comm);

        if (color == 1)
          return sub_comm;
        else
          {
            MPI_Comm_free(&sub_comm);
            return MPI_COMM_NULL;
          }
      }



      MPI_Comm
      create_rectangular_comm(const MPI_Comm &   comm,
                              const unsigned int size_x,
                              const unsigned int padding_x)
      {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        const unsigned int type_1 =
          (rank % padding_x) < ((padding_x / size_x) * size_x);

        const unsigned int n_ranks = Utilities::MPI::sum(type_1, comm);

        unsigned int offset = 0;

        const int ierr =
          MPI_Exscan(&type_1,
                     &offset,
                     1,
                     Utilities::MPI::mpi_type_id_for_type<unsigned int>,
                     MPI_SUM,
                     comm);

        AssertThrowMPI(ierr);

        const unsigned int type_2 = (offset < ((n_ranks / size_x) * size_x));

        const unsigned int color = type_1 > 0 && type_2;

        MPI_Comm sub_comm;
        MPI_Comm_split(comm, color, rank, &sub_comm);

        if (color == 1)
          return sub_comm;
        else
          {
            MPI_Comm_free(&sub_comm);
            return MPI_COMM_NULL;
          }
      }



      MPI_Comm
      create_sm(const MPI_Comm &comm)
      {
        int rank;
        MPI_Comm_rank(comm, &rank);

        MPI_Comm comm_shared;
        MPI_Comm_split_type(
          comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &comm_shared);

        return comm_shared;
      }



      unsigned int
      n_procs_of_sm(const MPI_Comm &comm)
      {
        MPI_Comm comm_sm = create_sm(comm);

        // determine size of current shared memory communicator
        int size_shared;
        MPI_Comm_size(comm_sm, &size_shared);

        MPI_Comm_free(&comm_sm);

        // determine maximum, since some shared memory communicators might not
        // be filed completely
        int size_shared_max;
        MPI_Allreduce(
          &size_shared, &size_shared_max, 1, MPI_INT, MPI_MAX, comm);

        return size_shared_max;
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
    virtual ~Interface() = default;

    virtual void
    solve(VectorType &       solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const = 0;

    virtual void
    get_statistics(ConvergenceTable &table,
                   const double      scaling_factor) const = 0;
  };



  /**
   * One-step-theta method implementation according to step-26.
   */
  class OneStepTheta : public Interface
  {
  public:
    OneStepTheta(const MPI_Comm                        comm,
                 const MassLaplaceOperator &           system_matrix,
                 const PreconditionerBase<VectorType> &block_preconditioner,
                 const std::function<void(const double, VectorType &)>
                   &evaluate_rhs_function)
      : theta(0.5)
      , system_matrix(system_matrix)
      , block_preconditioner(block_preconditioner)
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

      VectorType system_rhs;
      VectorType tmp;
      VectorType forcing_terms;

      system_rhs.reinit(solution);
      tmp.reinit(solution);
      forcing_terms.reinit(solution);

      // create right-hand-side vector
      // ... old solution
      system_matrix.vmult(system_rhs, solution, 1.0, (1 - theta) * time_step);

      // ... rhs function (new)
      evaluate_rhs_function(time, tmp);
      forcing_terms = tmp;
      forcing_terms *= time_step * theta;

      // ... rhs function (old)
      evaluate_rhs_function(time - time_step, tmp);
      forcing_terms.add(time_step * (1 - theta), tmp);

      system_rhs += forcing_terms;

      // setup system matrix
      system_matrix.reinit(1.0, -(theta * time_step));

      // solve system
      SolverControl        solver_control(1000, 1e-8 * system_rhs.l2_norm());
      SolverCG<VectorType> cg(solver_control);

      // ... create operator
      SystemMatrix sm(system_matrix);

      // ... create preconditioner
      Preconditioner preconditioner(system_matrix, block_preconditioner);

      // ... solve
      cg.solve(sm, solution, system_rhs, preconditioner);
    }

    void
    get_statistics(ConvergenceTable &table,
                   const double      scaling_factor) const override
    {
      (void)table;
      (void)scaling_factor;
    }

  private:
    class SystemMatrix
    {
    public:
      SystemMatrix(const MassLaplaceOperator &system_matrix)
        : system_matrix(system_matrix)
      {}

      void
      vmult(VectorType &dst, const VectorType &src) const
      {
        system_matrix.vmult(dst, src);
      }

    private:
      const MassLaplaceOperator &system_matrix;
    };

    class Preconditioner
    {
    public:
      Preconditioner(const MassLaplaceOperator &           system_matrix,
                     const PreconditionerBase<VectorType> &precondition)
        : system_matrix(system_matrix)
        , precondition(precondition)
      {
        precondition.reinit();
      }

      void
      vmult(VectorType &dst, const VectorType &src) const
      {
        precondition.vmult(dst, src);
      }

    private:
      const MassLaplaceOperator &           system_matrix;
      const PreconditionerBase<VectorType> &precondition;
    };

    const double theta;

    const MassLaplaceOperator &           system_matrix;
    const PreconditionerBase<VectorType> &block_preconditioner;

    const std::function<void(const double, VectorType &)> evaluate_rhs_function;

    ConditionalOStream pcout;
  };



  FullMatrix<typename VectorType::value_type>
  load_matrix_from_file(const unsigned int n_stages, const std::string label)
  {
    FullMatrix<typename VectorType::value_type> result(n_stages, n_stages);

    std::string file_name = label + std::to_string(n_stages) + ".txt";

    std::ifstream fin;
    fin.open(file_name);

    if (fin.fail())
      fin.open("../" + file_name);

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

  Vector<typename VectorType::value_type>
  load_vector_from_file(const unsigned int n_stages, const std::string label)
  {
    Vector<typename VectorType::value_type> result(n_stages);

    std::string file_name = label + std::to_string(n_stages) + ".txt";

    std::ifstream fin;
    fin.open(file_name);

    if (fin.fail())
      fin.open("../" + file_name);

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



  /**
   * IRK base class.
   */
  class IRKBase : public Interface
  {
  public:
    IRKBase(const MPI_Comm                        comm,
            const unsigned int                    n_stages,
            const bool                            do_reduce_number_of_vmults,
            const MassLaplaceOperator &           op,
            const PreconditionerBase<VectorType> &block_preconditioner,
            const std::function<void(const double, VectorType &)>
              &evaluate_rhs_function)
      : comm(comm)
      , n_stages(n_stages)
      , do_reduce_number_of_vmults(do_reduce_number_of_vmults)
      , A_inv(load_matrix_from_file(n_stages, "A_inv"))
      , T(load_matrix_from_file(n_stages, "T"))
      , T_inv(load_matrix_from_file(n_stages, "T_inv"))
      , b_vec(load_vector_from_file(n_stages, "b_vec_"))
      , c_vec(load_vector_from_file(n_stages, "c_vec_"))
      , d_vec(load_vector_from_file(n_stages, "D_vec_"))
      , op(op)
      , block_preconditioner(block_preconditioner)
      , evaluate_rhs_function(evaluate_rhs_function)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
    {}

    virtual void
    get_statistics(ConvergenceTable &table,
                   const double      scaling_factor = 1.0) const override
    {
      table.add_value("n_outer", n_outer_iterations / scaling_factor);

      const auto n_inner_iterations_min_max_avg =
        Utilities::MPI::min_max_avg(n_inner_iterations / n_outer_iterations,
                                    comm);

      table.add_value("n_inner_min", n_inner_iterations_min_max_avg.min);
      table.add_value("n_inner_avg", n_inner_iterations_min_max_avg.avg);
      table.add_value("n_inner_max", n_inner_iterations_min_max_avg.max);

      const auto add_time = [&](const std::string label, const double value) {
        const auto stat = Utilities::MPI::min_max_avg(value, comm);
        table.add_value(label, stat.avg / 1e9);
        table.set_scientific(label, true);
      };

      add_time("t", time_total);
      add_time("t_rhs", time_rhs);
      add_time("t_solver", time_outer_solver);
      add_time("t_update", time_solution_update);
      add_time("t_vmult", time_system_vmult);
      add_time("t_prec_bc", time_preconditioner_bc);
      add_time("t_prec_solver", time_preconditioner_solver);
    }

  protected:
    void
    clear_timers() const
    {
      time_total                 = 0.0;
      time_rhs                   = 0.0;
      time_outer_solver          = 0.0;
      time_solution_update       = 0.0;
      time_system_vmult          = 0.0;
      time_preconditioner_bc     = 0.0;
      time_preconditioner_solver = 0.0;
    }

    const MPI_Comm     comm;
    const unsigned int n_stages;
    const bool         do_reduce_number_of_vmults;
    const FullMatrix<typename VectorType::value_type> A_inv;
    const FullMatrix<typename VectorType::value_type> T;
    const FullMatrix<typename VectorType::value_type> T_inv;
    const Vector<typename VectorType::value_type>     b_vec;
    const Vector<typename VectorType::value_type>     c_vec;
    const Vector<typename VectorType::value_type>     d_vec;

    const MassLaplaceOperator &           op;
    const PreconditionerBase<VectorType> &block_preconditioner;

    const std::function<void(const double, VectorType &)> evaluate_rhs_function;

    ConditionalOStream pcout;

    mutable double time_total                 = 0.0;
    mutable double time_rhs                   = 0.0;
    mutable double time_outer_solver          = 0.0;
    mutable double time_solution_update       = 0.0;
    mutable double time_system_vmult          = 0.0;
    mutable double time_preconditioner_bc     = 0.0;
    mutable double time_preconditioner_solver = 0.0;

    mutable double n_outer_iterations = 0;
    mutable double n_inner_iterations = 0;
  };



  /**
   * A parallel IRK implementation.
   */
  class IRK : public IRKBase
  {
  public:
    IRK(const MPI_Comm                        comm,
        const double                          outer_tolerance,
        const double                          inner_tolerance,
        const unsigned int                    n_stages,
        const bool                            do_reduce_number_of_vmults,
        const MassLaplaceOperator &           op,
        const PreconditionerBase<VectorType> &block_preconditioner,
        const std::function<void(const double, VectorType &)>
          &evaluate_rhs_function)
      : IRKBase(comm,
                n_stages,
                do_reduce_number_of_vmults,
                op,
                block_preconditioner,
                evaluate_rhs_function)
      , n_max_iterations(1000)
      , outer_tolerance(outer_tolerance)
      , inner_tolerance(inner_tolerance)
      , times_preconditioner_solver(n_stages, 0.0)
    {}

    virtual void
    get_statistics(ConvergenceTable &table,
                   const double      scaling_factor = 1.0) const override
    {
      IRKBase::get_statistics(table, scaling_factor);

      const auto add_time = [&](const std::string label, const double value) {
        const auto stat = Utilities::MPI::min_max_avg(value, comm);
        table.add_value(label, stat.avg / 1e9);
        table.set_scientific(label, true);
      };

      for (unsigned int i = 0; i < n_stages; ++i)
        add_time("t_prec_solver_" + std::to_string(i),
                 times_preconditioner_solver[i]);
    }

    void
    solve(VectorType &       solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const override
    {
      (void)timestep_number;

      if (this->time_step != time_step)
        {
          this->system_matrix.reset();
          this->preconditioner.reset();
        }

      this->time_step = time_step;

      if (system_matrix == nullptr)
        {
          this->system_matrix =
            std::make_unique<SystemMatrix>(do_reduce_number_of_vmults,
                                           A_inv,
                                           time_step,
                                           op,
                                           time_system_vmult);
          this->preconditioner =
            std::make_unique<Preconditioner>(d_vec,
                                             T,
                                             T_inv,
                                             inner_tolerance,
                                             time_step,
                                             op,
                                             block_preconditioner,
                                             time_preconditioner_bc,
                                             time_preconditioner_solver,
                                             times_preconditioner_solver);
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

      op.vmult(tmp, solution, 0.0, -1.0);

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
                                   outer_tolerance * n_stages *
                                     system_rhs.block(0).size());

      std::string solver_name = "";

      try
        {
          if (true)
            {
              solver_name = "GCR";

              SolverGCR<BlockVectorType> cg(solver_control);
              cg.solve(*system_matrix,
                       system_solution,
                       system_rhs,
                       *preconditioner);
            }
          else
            {
              solver_name = "FGMRES";

              SolverFGMRES<BlockVectorType> cg(solver_control);
              cg.solve(*system_matrix,
                       system_solution,
                       system_rhs,
                       *preconditioner);
            }
        }
      catch (const SolverControl::NoConvergence &e)
        {
          AssertThrow(false, ExcMessage(e.what()));
        }

      this->time_outer_solver +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now() - time_outer_solver)
          .count();

      this->n_outer_iterations += solver_control.last_step();

      const auto n_inner_iterations =
        preconditioner->get_n_iterations_and_clear();

      for (const auto i : n_inner_iterations)
        this->n_inner_iterations += i;

      pcout << "   " << solver_control.last_step() << " outer " << solver_name
            << " iterations and ";

      pcout << n_inner_iterations[0];

      for (unsigned int i = 1; i < n_inner_iterations.size(); ++i)
        pcout << "+" << n_inner_iterations[i];

      pcout << " inner CG iterations." << std::endl;

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

      if (timestep_number == 1)
        clear_timers(); // clear timers since preconditioner is setup in
                        // first time step
    }

  private:
    class SystemMatrix
    {
    public:
      SystemMatrix(const bool do_reduce_number_of_vmults,
                   const FullMatrix<typename VectorType::value_type> &A_inv,
                   const double                                       time_step,
                   const MassLaplaceOperator &                        op,
                   double &                                           time)
        : n_stages(A_inv.m())
        , do_reduce_number_of_vmults(do_reduce_number_of_vmults)
        , A_inv(A_inv)
        , time_step(time_step)
        , op(op)
        , time(time)
      {}

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const
      {
        const auto time = std::chrono::system_clock::now();

        if (do_reduce_number_of_vmults == false)
          {
            dst = 0;
            for (unsigned int i = 0; i < n_stages; ++i)
              for (unsigned int j = 0; j < n_stages; ++j)
                {
                  const unsigned int k = (j + i) % n_stages;
                  if (j == 0) // first process diagonal
                    op.vmult(dst.block(i),
                             src.block(k),
                             A_inv(i, k),
                             time_step);
                  else // proceed with off-diagonals
                    op.vmult_add(dst.block(i), src.block(k), A_inv(i, k), 0.0);
                }
          }
        else
          {
            VectorType tmp;
            tmp.reinit(src.block(0));
            for (unsigned int i = 0; i < n_stages; ++i)
              op.vmult(dst.block(i), src.block(i), 0.0, time_step);

            for (unsigned int i = 0; i < n_stages; ++i)
              {
                op.vmult(tmp, src.block(i), 1.0, 0.0);

                for (unsigned int j = 0; j < n_stages; ++j)
                  dst.block(j).add(A_inv(j, i), tmp);
              }
          }


        this->time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now() - time)
                        .count();
      }

    private:
      const unsigned int n_stages;
      const bool         do_reduce_number_of_vmults;
      const FullMatrix<typename VectorType::value_type> &A_inv;
      const double                                       time_step;
      const MassLaplaceOperator &                        op;

      double &time;
    };

    class Preconditioner
    {
    public:
      Preconditioner(const Vector<typename VectorType::value_type> &    d_vec,
                     const FullMatrix<typename VectorType::value_type> &T,
                     const FullMatrix<typename VectorType::value_type> &T_inv,
                     const double                          inner_tolerance,
                     const double                          time_step,
                     const MassLaplaceOperator &           op,
                     const PreconditionerBase<VectorType> &preconditioner,
                     double &                              time_bc,
                     double &                              time_solver,
                     std::vector<double> &                 times_solver)
        : n_max_iterations(100)
        , inner_tolerance(inner_tolerance)
        , cut_off_tolerance(1e-12)
        , n_stages(d_vec.size())
        , d_vec(d_vec)
        , T_mat(T)
        , T_mat_inv(T_inv)
        , tau(time_step)
        , op(op)
        , time_bc(time_bc)
        , time_solver(time_solver)
        , times_solver(times_solver)
      {
        preconditioners.resize(n_stages);

        for (unsigned int i = 0; i < n_stages; ++i)
          {
            op.reinit(d_vec[i], tau);

            preconditioners[i] = preconditioner.clone();
            preconditioners[i]->reinit();
          }

        n_iterations.assign(n_stages, 0);
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
            const auto time_block = std::chrono::system_clock::now();

            if (inner_tolerance > 0.0)
              {
                SolverControl solver_control(n_max_iterations, inner_tolerance);
                SolverCG<VectorType> solver(solver_control);

                op.reinit(d_vec[i], tau);

                solver.solve(op,
                             tmp_vectors.block(i),
                             dst.block(i),
                             *preconditioners[i]);

                n_iterations[i] += solver_control.last_step();
              }
            else
              {
                op.reinit(d_vec[i], tau);
                preconditioners[i]->vmult(tmp_vectors.block(i), dst.block(i));
                n_iterations[i] += 1;
              }

            times_solver[i] +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now() - time_block)
                .count();
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

      std::vector<unsigned int>
      get_n_iterations_and_clear()
      {
        const auto temp = n_iterations;
        n_iterations.assign(n_stages, 0);
        return temp;
      }

    private:
      const unsigned int n_max_iterations;
      const double       inner_tolerance;
      const double       cut_off_tolerance;

      const unsigned int                                 n_stages;
      const Vector<typename VectorType::value_type> &    d_vec;
      const FullMatrix<typename VectorType::value_type> &T_mat;
      const FullMatrix<typename VectorType::value_type> &T_mat_inv;

      const double tau;

      const MassLaplaceOperator &op;
      std::vector<std::unique_ptr<const PreconditionerBase<VectorType>>>
        preconditioners;

      double &             time_bc;
      double &             time_solver;
      std::vector<double> &times_solver;

      mutable std::vector<unsigned int> n_iterations;
    };

    const unsigned int n_max_iterations;
    const double       outer_tolerance;
    const double       inner_tolerance;

    mutable double time_step = 0.0;

    mutable std::vector<double> times_preconditioner_solver;

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

    IRKStageParallel(const MPI_Comm             comm_global,
                     const MPI_Comm             comm_row,
                     const double               outer_tolerance,
                     const double               inner_tolerance,
                     const unsigned int         n_stages,
                     const bool                 do_reduce_number_of_vmults,
                     const bool                 use_sm,
                     const MassLaplaceOperator &op,
                     const PreconditionerBase<VectorType> &block_preconditioner,
                     const std::function<void(const double, VectorType &)>
                       &evaluate_rhs_function)
      : IRKBase(comm_global,
                n_stages,
                do_reduce_number_of_vmults,
                op,
                block_preconditioner,
                evaluate_rhs_function)
      , comm_row(comm_row)
      , n_max_iterations(1000)
      , outer_tolerance(outer_tolerance)
      , inner_tolerance(inner_tolerance)
      , use_sm(use_sm)
    {}

    ~IRKStageParallel()
    {
      GrowingVectorMemory<ReshapedVectorType>::release_unused_memory();
    }

    virtual void
    get_statistics(ConvergenceTable &table,
                   const double      scaling_factor = 1.0) const override
    {
      IRKBase::get_statistics(table, scaling_factor);

      const auto temp = Utilities::MPI::min_max_avg(
        Utilities::MPI::all_gather(comm_row, time_preconditioner_solver),
        this->comm);

      for (unsigned int i = 0; i < n_stages; ++i)
        {
          const std::string label = "t_prec_solver_" + std::to_string(i);
          table.add_value(label, temp[i].avg / 1e9);
          table.set_scientific(label, true);
        }
    }

    void
    solve(VectorType &       solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const override
    {
      (void)timestep_number;

      if (this->time_step != time_step)
        {
          this->system_matrix.reset();
          this->preconditioner.reset();
        }

      this->time_step = time_step;

      // ... create operator and preconditioner
      if (system_matrix == nullptr)
        {
          this->system_matrix =
            std::make_unique<SystemMatrix>(comm_row,
                                           do_reduce_number_of_vmults,
                                           A_inv,
                                           time_step,
                                           op,
                                           time_system_vmult,
                                           use_sm);
          this->preconditioner =
            std::make_unique<Preconditioner>(comm_row,
                                             d_vec,
                                             T,
                                             T_inv,
                                             inner_tolerance,
                                             time_step,
                                             op,
                                             block_preconditioner,
                                             time_preconditioner_bc,
                                             time_preconditioner_solver,
                                             use_sm);
        }

      const auto time_total = std::chrono::system_clock::now();
      const auto time_rhs   = std::chrono::system_clock::now();

      VectorType tmp;

      system_rhs.reinit(solution, comm_row);
      system_solution.reinit(solution, comm_row);
      tmp.reinit(solution);

      const unsigned int my_stage = Utilities::MPI::this_mpi_process(comm_row);

      // setup right-hand-side vector
      evaluate_rhs_function(time + (c_vec[my_stage] - 1.0) * time_step,
                            system_solution);
      op.vmult(tmp, solution, 0.0, -1.0);
      system_solution.add(1.0, tmp);

      // ... perform basis change
      perform_basis_change(system_rhs, system_solution, A_inv, false, use_sm);
      system_solution = 0.0;

      this->time_rhs += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now() - time_rhs)
                          .count();

      const auto time_outer_solver = std::chrono::system_clock::now();

      // solve system
      SolverControl solver_control(n_max_iterations,
                                   outer_tolerance * n_stages *
                                     system_rhs.size());

      std::string solver_name = "";

      try
        {
          if (true)
            {
              solver_name = "GCR";

              SolverGCR<ReshapedVectorType> cg(solver_control);
              cg.solve(*system_matrix,
                       system_solution,
                       system_rhs,
                       *preconditioner);
            }
          else
            {
              solver_name = "FGMRES";

              SolverFGMRES<ReshapedVectorType> cg(solver_control);
              cg.solve(*system_matrix,
                       system_solution,
                       system_rhs,
                       *preconditioner);
            }
        }
      catch (const SolverControl::NoConvergence &e)
        {
          AssertThrow(false, ExcMessage(e.what()));
        }

      this->time_outer_solver +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now() - time_outer_solver)
          .count();

      this->n_outer_iterations += solver_control.last_step();

      const double n_inner_iterations =
        preconditioner->get_n_iterations_and_clear();

      this->n_inner_iterations += n_inner_iterations;

      const auto n_inner_iterations_min_max_avg =
        Utilities::MPI::min_max_avg(n_inner_iterations, comm_row);

      pcout << "   " << solver_control.last_step() << " outer " << solver_name
            << " iterations and "
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

      if (timestep_number == 1)
        clear_timers(); // clear timers since preconditioner is setup in
                        // first time step
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

      GrowingVectorMemory<LinearAlgebra::ReshapedVector<VectorType>> memory;
      typename VectorMemory<LinearAlgebra::ReshapedVector<VectorType>>::Pointer
        temp_pointer(memory);

      LinearAlgebra::ReshapedVector<VectorType> &temp = *temp_pointer;
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
                         const FullMatrix<typename VectorType::value_type> T,
                         const bool                                        add,
                         const bool use_sm)
    {
      if (use_sm == false)
        {
          const auto fu =
            [&T, add](const auto i, const auto j, auto &dst, const auto &src) {
              if ((add == false) && (i == j))
                dst.equ(T[i][j], src);
              else
                dst.add(T[i][j], src);
            };

          matrix_vector_rol_operation<VectorType>(dst, src, fu);
        }
      else
        {
          const auto         comm     = src.get_row_mpi_communicator();
          const unsigned int i        = Utilities::MPI::this_mpi_process(comm);
          const unsigned int n_stages = Utilities::MPI::n_mpi_processes(comm);
          const double       cut_off_tolerance = 1e-12; // TODO

          MPI_Barrier(comm);

          const auto sm_ptr = src.shared_vector_data();

          for (unsigned int e = 0; e < src.locally_owned_size(); ++e)
            {
              typename VectorType::value_type temp = 0.0;

              for (unsigned int j = 0; j < n_stages; ++j)
                if (std::abs(T(i, j)) > cut_off_tolerance)
                  temp += T(i, j) * sm_ptr[j][e];

              if (add)
                dst.local_element(e) += temp;
              else
                dst.local_element(e) = temp;
            }


          MPI_Barrier(comm);
        }
    }

    class SystemMatrix
    {
    public:
      SystemMatrix(const MPI_Comm &comm_row,
                   const bool      do_reduce_number_of_vmults,
                   const FullMatrix<typename VectorType::value_type> &A_inv,
                   const double                                       time_step,
                   const MassLaplaceOperator &                        op,
                   double &                                           time,
                   const bool                                         use_sm)
        : my_stage(Utilities::MPI::this_mpi_process(comm_row))
        , do_reduce_number_of_vmults(do_reduce_number_of_vmults)
        , A_inv(A_inv)
        , time_step(time_step)
        , op(op)
        , time(time)
        , use_sm(use_sm)
      {}

      void
      vmult(ReshapedVectorType &dst, const ReshapedVectorType &src) const
      {
        const auto time = std::chrono::system_clock::now();

        temp.reinit(src);

        if (do_reduce_number_of_vmults == false)
          {
            matrix_vector_rol_operation<VectorType>(
              dst,
              src,
              [this](const auto i, const auto j, auto &dst, const auto &src) {
                if (i == j)
                  op.vmult(static_cast<VectorType &>(dst),
                           static_cast<const VectorType &>(src),
                           A_inv(i, j),
                           time_step);
                else
                  op.vmult_add(static_cast<VectorType &>(dst),
                               static_cast<const VectorType &>(src),
                               A_inv(i, j),
                               0.0);
              });
          }
        else
          {
            op.vmult(static_cast<VectorType &>(dst),
                     static_cast<const VectorType &>(src),
                     0.0,
                     time_step);
            op.vmult(static_cast<VectorType &>(temp),
                     static_cast<const VectorType &>(src),
                     1.0,
                     0.0);

            perform_basis_change(dst, temp, A_inv, true, use_sm);
          }

        this->time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now() - time)
                        .count();
      }

    private:
      const unsigned int my_stage;
      const bool         do_reduce_number_of_vmults;
      const FullMatrix<typename VectorType::value_type> &A_inv;
      const double                                       time_step;
      const MassLaplaceOperator &                        op;

      double &time;

      const bool use_sm;

      mutable ReshapedVectorType temp;
    };

    class Preconditioner
    {
    public:
      Preconditioner(const MPI_Comm &                               comm_row,
                     const Vector<typename VectorType::value_type> &d_vec,
                     const FullMatrix<typename VectorType::value_type> &T,
                     const FullMatrix<typename VectorType::value_type> &T_inv,
                     const double                          inner_tolerance,
                     const double                          time_step,
                     const MassLaplaceOperator &           op,
                     const PreconditionerBase<VectorType> &preconditioners,
                     double &                              time_bc,
                     double &                              time_solver,
                     const bool                            use_sm)
        : comm_row(comm_row)
        , n_max_iterations(100)
        , inner_tolerance(inner_tolerance)
        , my_stage(Utilities::MPI::this_mpi_process(comm_row))
        , d_vec(d_vec)
        , T_mat(T)
        , T_mat_inv(T_inv)
        , tau(time_step)
        , op(op)
        , preconditioners(preconditioners)
        , time_bc(time_bc)
        , time_solver(time_solver)
        , n_iterations(0)
        , use_sm(use_sm)
      {
        op.reinit(d_vec[my_stage], tau);
        preconditioners.reinit();
      }

      void
      vmult(ReshapedVectorType &dst, const ReshapedVectorType &src) const
      {
        const auto time_bc_0 = std::chrono::system_clock::now();

        perform_basis_change(dst, src, T_mat_inv, false, use_sm);

        this->time_bc += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now() - time_bc_0)
                           .count();

        const auto time_solver = std::chrono::system_clock::now();

        temp.reinit(src); // TODO

        if (inner_tolerance > 0.0)
          {
            std::unique_ptr<SolverControl> solver_control;

            if (true)
              solver_control = std::make_unique<SolverControl>(n_max_iterations,
                                                               inner_tolerance);
            else
              solver_control =
                std::make_unique<SPSolverControl>(comm_row,
                                                  n_max_iterations,
                                                  inner_tolerance);

            SolverCG<VectorType> solver(*solver_control);

            op.reinit(d_vec[my_stage], tau);

            solver.solve(op,
                         static_cast<VectorType &>(temp),
                         static_cast<const VectorType &>(dst),
                         preconditioners);

            n_iterations += solver_control->last_step();
          }
        else
          {
            op.reinit(d_vec[my_stage], tau);
            preconditioners.vmult(static_cast<VectorType &>(temp),
                                  static_cast<const VectorType &>(dst));
            n_iterations += 1;
          }

        this->time_solver +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now() - time_solver)
            .count();

        const auto time_bc_1 = std::chrono::system_clock::now();

        perform_basis_change(dst, temp, T_mat, false, use_sm);

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
      const MPI_Comm comm_row;

      const unsigned int n_max_iterations;
      const double       inner_tolerance;

      const unsigned int my_stage;

      const Vector<typename VectorType::value_type> &    d_vec;
      const FullMatrix<typename VectorType::value_type> &T_mat;
      const FullMatrix<typename VectorType::value_type> &T_mat_inv;

      const double tau;

      const MassLaplaceOperator &op;

      const PreconditionerBase<VectorType> &preconditioners;

      double &time_bc;
      double &time_solver;

      mutable unsigned int n_iterations;

      const bool use_sm;


      mutable ReshapedVectorType temp; // TODO
    };

    const MPI_Comm comm_row;

    const unsigned int n_max_iterations;
    const double       outer_tolerance;
    const double       inner_tolerance;
    const bool         use_sm;

    mutable double time_step = 0.0;

    mutable std::unique_ptr<SystemMatrix>   system_matrix;
    mutable std::unique_ptr<Preconditioner> preconditioner;

    mutable ReshapedVectorType system_rhs;
    mutable ReshapedVectorType system_solution;
  };



  /**
   * Complex IRK base class.
   */
  class ComplexIRKBase : public Interface
  {
  public:
    ComplexIRKBase(const MPI_Comm                        comm,
                   const unsigned int                    n_stages,
                   const MassLaplaceOperator &           op,
                   const PreconditionerBase<VectorType> &block_preconditioner,
                   const std::function<void(const double, VectorType &)>
                     &evaluate_rhs_function)
      : comm(comm)
      , n_stages(n_stages)
      , A_inv(load_matrix_from_file(n_stages, "A_inv"))
      , T_re(load_matrix_from_file(n_stages, "T_re"))
      , T_im(load_matrix_from_file(n_stages, "T_im"))
      , T_inv_re(load_matrix_from_file(n_stages, "T_inv_re"))
      , T_inv_im(load_matrix_from_file(n_stages, "T_inv_im"))
      , b_vec(load_vector_from_file(n_stages, "b_vec_"))
      , c_vec(load_vector_from_file(n_stages, "c_vec_"))
      , d_vec_re(load_vector_from_file(n_stages, "D_vec_re_"))
      , d_vec_im(load_vector_from_file(n_stages, "D_vec_im_"))
      , op(op)
      , block_preconditioner(block_preconditioner)
      , evaluate_rhs_function(evaluate_rhs_function)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
    {}

    virtual void
    get_statistics(ConvergenceTable &table,
                   const double      scaling_factor = 1.0) const override
    {
      (void)table;
      (void)scaling_factor;
    }

  protected:
    void
    clear_timers() const
    {}

    const MPI_Comm                                    comm;
    const unsigned int                                n_stages;
    const FullMatrix<typename VectorType::value_type> A_inv;
    const FullMatrix<typename VectorType::value_type> T_re;
    const FullMatrix<typename VectorType::value_type> T_im;
    const FullMatrix<typename VectorType::value_type> T_inv_re;
    const FullMatrix<typename VectorType::value_type> T_inv_im;
    const Vector<typename VectorType::value_type>     b_vec;
    const Vector<typename VectorType::value_type>     c_vec;
    const Vector<typename VectorType::value_type>     d_vec_re;
    const Vector<typename VectorType::value_type>     d_vec_im;

    const MassLaplaceOperator &           op;
    const PreconditionerBase<VectorType> &block_preconditioner;

    const std::function<void(const double, VectorType &)> evaluate_rhs_function;

    ConditionalOStream pcout;
  };



  /**
   * A parallel IRK implementation.
   */
  class ComplexIRK : public ComplexIRKBase
  {
  public:
    ComplexIRK(const MPI_Comm                        comm,
               const double                          outer_tolerance,
               const double                          inner_tolerance,
               const unsigned int                    n_stages,
               const MassLaplaceOperator &           op,
               const ComplexMassLaplaceOperator &    op_complex,
               const PreconditionerBase<VectorType> &block_preconditioner,
               const std::function<void(const double, VectorType &)>
                 &evaluate_rhs_function)
      : ComplexIRKBase(comm,
                       n_stages,
                       op,
                       block_preconditioner,
                       evaluate_rhs_function)
      , n_max_iterations(1000)
      , outer_tolerance(outer_tolerance)
      , inner_tolerance(inner_tolerance)
      , op_complex(op_complex)
    {}

    virtual void
    get_statistics(ConvergenceTable &table,
                   const double      scaling_factor = 1.0) const override
    {
      (void)table;
      (void)scaling_factor;
    }

    void
    solve(VectorType &       solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const override
    {
      (void)timestep_number;

      if (this->time_step != time_step)
        {
          preconditioners.clear();
        }

      this->time_step = time_step;

      const unsigned int n_stages_reduced = (n_stages + 1) / 2;

      if (preconditioners.size() == 0)
        {
          preconditioners.resize(n_stages_reduced);

          for (unsigned int i = 0; i < n_stages_reduced; ++i)
            {
              op.reinit(d_vec_re[i * 2] + d_vec_im[i * 2], time_step);

              preconditioners[i] = this->block_preconditioner.clone();
              preconditioners[i]->reinit();
            }
        }

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

      op.vmult(tmp, solution, 0.0, -1.0);

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

      const PreconditionComplex outer_preconditioner(n_stages,
                                                     n_max_iterations,
                                                     outer_tolerance,
                                                     inner_tolerance,
                                                     this->time_step,
                                                     T_inv_re,
                                                     T_inv_im,
                                                     T_re,
                                                     T_im,
                                                     d_vec_re,
                                                     d_vec_im,
                                                     op,
                                                     op_complex,
                                                     preconditioners);

      outer_preconditioner.vmult(system_solution, system_rhs);

      const auto n_iterations =
        outer_preconditioner.get_n_iterations_and_clear();

      pcout << "   Solved in: " << std::get<0>(n_iterations[0]) << " ("
            << std::get<1>(n_iterations[0]) << "+"
            << std::get<2>(n_iterations[0]) << ")";
      for (unsigned int i = 1; i < n_iterations.size(); ++i)
        pcout << ", " << std::get<0>(n_iterations[0]) << " ("
              << std::get<1>(n_iterations[0]) << "+"
              << std::get<2>(n_iterations[0]) << ")";
      pcout << std::endl;


      // accumulate result in solution
      for (unsigned int i = 0; i < n_stages; ++i)
        solution.add(time_step * b_vec[i], system_solution.block(i));

      if (timestep_number == 1)
        clear_timers(); // clear timers since preconditioner is setup in
                        // first time step
    }

  private:
    class PreconditionComplex
    {
    public:
      PreconditionComplex(
        const unsigned int                                 n_stages,
        const unsigned int                                 n_max_iterations,
        const double                                       outer_tolerance,
        const double                                       inner_tolerance,
        const double                                       time_step,
        const FullMatrix<typename VectorType::value_type> &T_inv_re,
        const FullMatrix<typename VectorType::value_type> &T_inv_im,
        const FullMatrix<typename VectorType::value_type> &T_re,
        const FullMatrix<typename VectorType::value_type> &T_im,
        const Vector<typename VectorType::value_type> &    d_vec_re,
        const Vector<typename VectorType::value_type> &    d_vec_im,
        const MassLaplaceOperator &                        op,
        const ComplexMassLaplaceOperator &                 op_complex,
        std::vector<std::unique_ptr<const PreconditionerBase<VectorType>>>
          &preconditioners)
        : n_stages(n_stages)
        , n_max_iterations(n_max_iterations)
        , outer_tolerance(outer_tolerance)
        , inner_tolerance(inner_tolerance)
        , time_step(time_step)
        , T_inv_re(T_inv_re)
        , T_inv_im(T_inv_im)
        , T_re(T_re)
        , T_im(T_im)
        , d_vec_re(d_vec_re)
        , d_vec_im(d_vec_im)
        , op(op)
        , op_complex(op_complex)
        , preconditioners(preconditioners)
      {
        this->n_iterations.assign(
          (n_stages + 1) / 2,
          std::tuple<unsigned int, unsigned int, unsigned int>{0, 0, 0});
      }

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const
      {
        const unsigned int n_stages_reduced = (n_stages + 1) / 2;

        std::vector<BlockVectorType> src_block(n_stages_reduced);
        std::vector<BlockVectorType> dst_block(n_stages_reduced);

        for (unsigned int i = 0; i < n_stages_reduced; ++i) // sp
          {
            src_block[i].reinit(2);
            dst_block[i].reinit(2);

            for (unsigned int j = 0; j < 2; ++j)
              {
                src_block[i].block(j).reinit(src.block(0));
                dst_block[i].block(j).reinit(src.block(0));
              }
          }

        // apply Tinv
        for (unsigned int i = 0; i < n_stages_reduced; ++i) // sp
          for (unsigned int j = 0; j < n_stages; ++j)
            {
              src_block[i].block(0).add(T_inv_re(i * 2, j), src.block(j));
              src_block[i].block(1).add(T_inv_im(i * 2, j), src.block(j));
            }

        // solve blocks
        for (unsigned int i = 0; i < n_stages_reduced; ++i) // sp
          {
            SolverControl solver_control(n_max_iterations,
                                         outer_tolerance *
                                           src.block(i * 2).size());
            SolverFGMRES<LinearAlgebra::distributed::BlockVector<double>>
              solver(solver_control);

            op_complex.reinit(d_vec_re[i * 2],
                              d_vec_im[i * 2],
                              this->time_step);

            PreconditionPRESB presb(op,
                                    *preconditioners[i],
                                    inner_tolerance,
                                    d_vec_re[i * 2],
                                    d_vec_im[i * 2],
                                    this->time_step);

            solver.solve(op_complex, dst_block[i], src_block[i], presb);

            const auto n_iterations_presb = presb.get_n_iterations_and_clear();

            std::get<0>(this->n_iterations[i]) += solver_control.last_step();
            std::get<1>(this->n_iterations[i]) += n_iterations_presb.first;
            std::get<2>(this->n_iterations[i]) += n_iterations_presb.second;
          }

        // apply T
        dst = 0;
        for (unsigned int i = 0; i < n_stages; ++i)
          for (unsigned int j = 0; j < n_stages_reduced; ++j) // sp
            {
              const double scaling = (j < (n_stages / 2)) ? 2.0 : 1.0;
              dst.block(i).add(scaling * T_re(i, j * 2),
                               dst_block[j].block(0),
                               -scaling * T_im(i, j * 2),
                               dst_block[j].block(1));
            }
      }

      std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>
      get_n_iterations_and_clear() const
      {
        const auto temp = this->n_iterations;
        this->n_iterations.assign(
          (n_stages + 1) / 2,
          std::tuple<unsigned int, unsigned int, unsigned int>{0, 0, 0});
        return temp;
      }

    private:
      const unsigned int n_stages;

      const unsigned int n_max_iterations;
      const double       outer_tolerance;
      const double       inner_tolerance;
      const double       time_step;

      const FullMatrix<typename VectorType::value_type> &T_inv_re;
      const FullMatrix<typename VectorType::value_type> &T_inv_im;
      const FullMatrix<typename VectorType::value_type> &T_re;
      const FullMatrix<typename VectorType::value_type> &T_im;
      const Vector<typename VectorType::value_type> &    d_vec_re;
      const Vector<typename VectorType::value_type> &    d_vec_im;

      const MassLaplaceOperator &       op;
      const ComplexMassLaplaceOperator &op_complex;

      std::vector<std::unique_ptr<const PreconditionerBase<VectorType>>>
        &preconditioners;

      mutable std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>
        n_iterations;
    };

    class PreconditionPRESB
    {
    public:
      PreconditionPRESB(const MassLaplaceOperator &           op,
                        const PreconditionerBase<VectorType> &preconditioner,
                        const double                          inner_tolerance,
                        const double                          lambda_re,
                        const double                          lambda_im,
                        const double                          tau)
        : op(op)
        , preconditioner(preconditioner)
        , inner_tolerance(inner_tolerance)
        , lambda_re(lambda_re)
        , lambda_im(lambda_im)
        , tau(tau)
        , n_iterations(0, 0)
      {}

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const
      {
        VectorType temp_0, temp_1;
        temp_0.reinit(src.block(0));
        temp_1.reinit(src.block(0));

        temp_0 = src.block(0);
        temp_0 += src.block(1);

        if (inner_tolerance == 0.0)
          {
            op.reinit(lambda_re + lambda_im, tau);
            preconditioner.vmult(dst.block(0), temp_0);

            n_iterations.first += 1;
          }
        else
          {
            SolverControl        reduction_control(100, inner_tolerance);
            SolverCG<VectorType> solver(reduction_control);

            op.reinit(lambda_re + lambda_im, tau);
            solver.solve(op, dst.block(0), temp_0, preconditioner);

            n_iterations.first += reduction_control.last_step();
          }

        op.reinit(lambda_im, 0.0);
        op.vmult(temp_0, dst.block(0));
        temp_0 *= -1.0;
        temp_0 += src.block(1);

        if (inner_tolerance == 0.0)
          {
            op.reinit(lambda_re + lambda_im, tau);
            preconditioner.vmult(dst.block(1), temp_0);

            n_iterations.second += 1;
          }
        else
          {
            SolverControl        reduction_control(100, inner_tolerance);
            SolverCG<VectorType> solver(reduction_control);

            op.reinit(lambda_re + lambda_im, tau);
            solver.solve(op, dst.block(1), temp_0, preconditioner);

            n_iterations.second += reduction_control.last_step();
          }

        dst.block(0) -= dst.block(1);
      }

      std::pair<unsigned int, unsigned int>
      get_n_iterations_and_clear() const
      {
        const auto temp    = this->n_iterations;
        this->n_iterations = {0, 0};
        return temp;
      }

    private:
      const MassLaplaceOperator &           op;
      const PreconditionerBase<VectorType> &preconditioner;

      const double inner_tolerance;

      const double lambda_re;
      const double lambda_im;
      const double tau;

      mutable std::pair<unsigned int, unsigned int> n_iterations;
    };

    const unsigned int n_max_iterations;
    const double       outer_tolerance;
    const double       inner_tolerance;

    mutable double time_step = 0.0;

    const ComplexMassLaplaceOperator &op_complex;

    mutable std::vector<std::unique_ptr<const PreconditionerBase<VectorType>>>
      preconditioners;
  };



  /**
   * A parallel IRK implementation.
   */
  class ComplexSPIRK : public ComplexIRKBase
  {
  public:
    ComplexSPIRK(const MPI_Comm                        comm,
                 const MPI_Comm                        comm_row,
                 const double                          outer_tolerance,
                 const double                          inner_tolerance,
                 const unsigned int                    n_stages,
                 const MassLaplaceOperator &           op,
                 const ComplexMassLaplaceOperator &    op_complex,
                 const PreconditionerBase<VectorType> &block_preconditioner,
                 const std::function<void(const double, VectorType &)>
                   &evaluate_rhs_function)
      : ComplexIRKBase(comm,
                       n_stages,
                       op,
                       block_preconditioner,
                       evaluate_rhs_function)
      , comm_row(comm_row)
      , n_max_iterations(1000)
      , outer_tolerance(outer_tolerance)
      , inner_tolerance(inner_tolerance)
      , op_complex(op_complex)
    {}

    virtual void
    get_statistics(ConvergenceTable &table,
                   const double      scaling_factor = 1.0) const override
    {
      (void)table;
      (void)scaling_factor;
    }

    void
    solve(VectorType &       solution,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const override
    {
      (void)timestep_number;

      if (this->time_step != time_step)
        {
          preconditioners.reset();
        }

      this->time_step = time_step;

      const unsigned int n_stages_reduced = (n_stages + 1) / 2;

      const unsigned int my_block = Utilities::MPI::this_mpi_process(comm_row);

      if (preconditioners == nullptr)
        {
          op.reinit(d_vec_re[my_block * 2] + d_vec_im[my_block * 2], time_step);

          preconditioners = this->block_preconditioner.clone();
          preconditioners->reinit();
        }

      BlockVectorType system_rhs(2);      // TODO
      BlockVectorType system_solution(2); //

      {
        VectorType tmp;
        tmp.reinit(solution);

        system_rhs.reinit(2);
        system_rhs.block(0).reinit(solution);
        system_rhs.block(1).reinit(solution);

        system_solution.reinit(2);
        system_solution.block(0).reinit(solution);
        system_solution.block(1).reinit(solution);

        for (unsigned int i = my_block * 2;
             i < std::min(n_stages, (my_block + 1) * 2);
             ++i)
          evaluate_rhs_function(time + (c_vec[i] - 1.0) * time_step,
                                system_solution.block(i % 2));

        op.vmult(tmp, solution, 0.0, -1.0);

        for (unsigned int i = my_block * 2;
             i < std::min(n_stages, (my_block + 1) * 2);
             ++i)
          system_solution.block(i % 2).add(1.0, tmp);
      }

      for (unsigned int jj = 0; jj < n_stages_reduced; ++jj) // sp
        for (unsigned int i = my_block * 2;
             i < std::min(n_stages, (my_block + 1) * 2);
             ++i)
          for (unsigned int j = jj * 2; j < std::min(n_stages, (jj + 1) * 2);
               ++j)
            system_rhs.block(i % 2).add(A_inv[i][j],
                                        system_solution.block(j % 2));

      for (auto &i : system_solution)
        i = 0.0;

      const PreconditionComplex outer_preconditioner(comm_row,
                                                     n_stages,
                                                     n_max_iterations,
                                                     outer_tolerance,
                                                     inner_tolerance,
                                                     this->time_step,
                                                     T_inv_re,
                                                     T_inv_im,
                                                     T_re,
                                                     T_im,
                                                     d_vec_re,
                                                     d_vec_im,
                                                     op,
                                                     op_complex,
                                                     preconditioners);

      outer_preconditioner.vmult(system_solution, system_rhs);

      /*
      const auto n_iterations =
        outer_preconditioner.get_n_iterations_and_clear();

      pcout << "   Solved in: " << std::get<0>(n_iterations[0]) << " ("
            << std::get<1>(n_iterations[0]) << "+"
            << std::get<2>(n_iterations[0]) << ")";
      for (unsigned int i = 1; i < n_iterations.size(); ++i)
        pcout << ", " << std::get<0>(n_iterations[0]) << " ("
              << std::get<1>(n_iterations[0]) << "+"
              << std::get<2>(n_iterations[0]) << ")";
      pcout << std::endl;
       */


      // accumulate result in solution
      for (unsigned int i = my_block * 2;
           i < std::min(n_stages, (my_block + 1) * 2);
           ++i)
        if ((i == 0) || (i % 2 == 0))
          solution.add(time_step * b_vec[i], system_solution.block(i % 2));
        else
          solution.equ(time_step * b_vec[i], system_solution.block(i % 2));

      if (timestep_number == 1)
        clear_timers(); // clear timers since preconditioner is setup in
                        // first time step
    }

  private:
    class PreconditionComplex
    {
    public:
      PreconditionComplex(
        const MPI_Comm                                         comm_row,
        const unsigned int                                     n_stages,
        const unsigned int                                     n_max_iterations,
        const double                                           outer_tolerance,
        const double                                           inner_tolerance,
        const double                                           time_step,
        const FullMatrix<typename VectorType::value_type> &    T_inv_re,
        const FullMatrix<typename VectorType::value_type> &    T_inv_im,
        const FullMatrix<typename VectorType::value_type> &    T_re,
        const FullMatrix<typename VectorType::value_type> &    T_im,
        const Vector<typename VectorType::value_type> &        d_vec_re,
        const Vector<typename VectorType::value_type> &        d_vec_im,
        const MassLaplaceOperator &                            op,
        const ComplexMassLaplaceOperator &                     op_complex,
        std::unique_ptr<const PreconditionerBase<VectorType>> &preconditioners)
        : comm_row(comm_row)
        , n_stages(n_stages)
        , n_max_iterations(n_max_iterations)
        , outer_tolerance(outer_tolerance)
        , inner_tolerance(inner_tolerance)
        , time_step(time_step)
        , T_inv_re(T_inv_re)
        , T_inv_im(T_inv_im)
        , T_re(T_re)
        , T_im(T_im)
        , d_vec_re(d_vec_re)
        , d_vec_im(d_vec_im)
        , op(op)
        , op_complex(op_complex)
        , preconditioners(preconditioners)
      {
        this->n_iterations =
          std::tuple<unsigned int, unsigned int, unsigned int>{0, 0, 0};
      }

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const
      {
        const unsigned int n_stages_reduced = (n_stages + 1) / 2;
        const unsigned int my_block =
          Utilities::MPI::this_mpi_process(comm_row);

        BlockVectorType src_block(2);
        BlockVectorType dst_block(2);

        for (unsigned int j = 0; j < 2; ++j)
          {
            src_block.block(j).reinit(src.block(0));
            dst_block.block(j).reinit(src.block(0));
          }

        // apply Tinv
        for (unsigned int i = 0; i < n_stages_reduced; ++i)      // sp: CA
          for (unsigned int jj = 0; jj < n_stages_reduced; ++jj) // sp
            for (unsigned int j = jj * 2; j < std::min(n_stages, (jj + 1) * 2);
                 ++j)
              {
                src_block.block(0).add(T_inv_re(i * 2, j), src.block(j % 2));
                src_block.block(1).add(T_inv_im(i * 2, j), src.block(j % 2));
              }

        // solve blocks
        {
          SolverControl solver_control(n_max_iterations,
                                       outer_tolerance * src.block(0).size());
          SolverFGMRES<LinearAlgebra::distributed::BlockVector<double>> solver(
            solver_control);

          op_complex.reinit(d_vec_re[my_block * 2],
                            d_vec_im[my_block * 2],
                            this->time_step);

          PreconditionPRESB presb(op,
                                  *preconditioners,
                                  inner_tolerance,
                                  d_vec_re[my_block * 2],
                                  d_vec_im[my_block * 2],
                                  this->time_step);

          solver.solve(op_complex, dst_block, src_block, presb);

          const auto n_iterations_presb = presb.get_n_iterations_and_clear();

          std::get<0>(this->n_iterations) += solver_control.last_step();
          std::get<1>(this->n_iterations) += n_iterations_presb.first;
          std::get<2>(this->n_iterations) += n_iterations_presb.second;
        }

        // apply T
        for (auto &i : dst)
          i = 0;

        for (unsigned int ii = 0; ii < n_stages_reduced; ++ii) // sp: CA
          for (unsigned int j = 0; j < n_stages_reduced; ++j)  // sp
            for (unsigned int i = ii * 2; i < std::min(n_stages, (ii + 1) * 2);
                 ++i)
              {
                const double scaling = (j < (n_stages / 2)) ? 2.0 : 1.0;
                dst.block(i % 2).add(scaling * T_re(i, j * 2),
                                     dst_block.block(0),
                                     -scaling * T_im(i, j * 2),
                                     dst_block.block(1));
              }
      }

      std::tuple<unsigned int, unsigned int, unsigned int>
      get_n_iterations_and_clear() const
      {
        const auto temp = this->n_iterations;
        this->n_iterations =
          std::tuple<unsigned int, unsigned int, unsigned int>{0, 0, 0};
        return temp;
      }

    private:
      const MPI_Comm comm_row;

      const unsigned int n_stages;

      const unsigned int n_max_iterations;
      const double       outer_tolerance;
      const double       inner_tolerance;
      const double       time_step;

      const FullMatrix<typename VectorType::value_type> &T_inv_re;
      const FullMatrix<typename VectorType::value_type> &T_inv_im;
      const FullMatrix<typename VectorType::value_type> &T_re;
      const FullMatrix<typename VectorType::value_type> &T_im;
      const Vector<typename VectorType::value_type> &    d_vec_re;
      const Vector<typename VectorType::value_type> &    d_vec_im;

      const MassLaplaceOperator &       op;
      const ComplexMassLaplaceOperator &op_complex;

      std::unique_ptr<const PreconditionerBase<VectorType>> &preconditioners;

      mutable std::tuple<unsigned int, unsigned int, unsigned int> n_iterations;
    };

    class PreconditionPRESB
    {
    public:
      PreconditionPRESB(const MassLaplaceOperator &           op,
                        const PreconditionerBase<VectorType> &preconditioner,
                        const double                          inner_tolerance,
                        const double                          lambda_re,
                        const double                          lambda_im,
                        const double                          tau)
        : op(op)
        , preconditioner(preconditioner)
        , inner_tolerance(inner_tolerance)
        , lambda_re(lambda_re)
        , lambda_im(lambda_im)
        , tau(tau)
        , n_iterations(0, 0)
      {}

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const
      {
        VectorType temp_0, temp_1;
        temp_0.reinit(src.block(0));
        temp_1.reinit(src.block(0));

        temp_0 = src.block(0);
        temp_0 += src.block(1);

        if (inner_tolerance == 0.0)
          {
            op.reinit(lambda_re + lambda_im, tau);
            preconditioner.vmult(dst.block(0), temp_0);

            n_iterations.first += 1;
          }
        else
          {
            SolverControl        reduction_control(100, inner_tolerance);
            SolverCG<VectorType> solver(reduction_control);

            op.reinit(lambda_re + lambda_im, tau);
            solver.solve(op, dst.block(0), temp_0, preconditioner);

            n_iterations.first += reduction_control.last_step();
          }

        op.reinit(lambda_im, 0.0);
        op.vmult(temp_0, dst.block(0));
        temp_0 *= -1.0;
        temp_0 += src.block(1);

        if (inner_tolerance == 0.0)
          {
            op.reinit(lambda_re + lambda_im, tau);
            preconditioner.vmult(dst.block(1), temp_0);

            n_iterations.second += 1;
          }
        else
          {
            SolverControl        reduction_control(100, inner_tolerance);
            SolverCG<VectorType> solver(reduction_control);

            op.reinit(lambda_re + lambda_im, tau);
            solver.solve(op, dst.block(1), temp_0, preconditioner);

            n_iterations.second += reduction_control.last_step();
          }

        dst.block(0) -= dst.block(1);
      }

      std::pair<unsigned int, unsigned int>
      get_n_iterations_and_clear() const
      {
        const auto temp    = this->n_iterations;
        this->n_iterations = {0, 0};
        return temp;
      }

    private:
      const MassLaplaceOperator &           op;
      const PreconditionerBase<VectorType> &preconditioner;

      const double inner_tolerance;

      const double lambda_re;
      const double lambda_im;
      const double tau;

      mutable std::pair<unsigned int, unsigned int> n_iterations;
    };

    const MPI_Comm comm_row;

    const unsigned int n_max_iterations;
    const double       outer_tolerance;
    const double       inner_tolerance;

    mutable double time_step = 0.0;

    const ComplexMassLaplaceOperator &op_complex;

    mutable std::unique_ptr<const PreconditionerBase<VectorType>>
      preconditioners;
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

    unsigned int irk_stages                 = 3;
    bool         do_reduce_number_of_vmults = true;

    std::string operator_type             = "MatrixBased";
    std::string block_preconditioner_type = "AMG";

    bool use_sm       = false;
    bool do_row_major = true;
    int  padding      = -1; // -1: no padding; 0: use sm;
                            // else valid: padding > irk_stages
    unsigned int max_ranks = 0;

    double outer_tolerance = 1e-8;
    double inner_tolerance = 1e-6;

    bool do_output_paraview = true;

    void
    parse(const std::string file_name)
    {
      dealii::ParameterHandler prm;
      prm.add_parameter("FEDegree", fe_degree);
      prm.add_parameter("NRefinements", n_refinements);
      prm.add_parameter("TimeIntegrationScheme",
                        time_integration_scheme,
                        "",
                        Patterns::Selection(
                          "ost|irk|spirk|complex_irk|complex_spirk"));
      prm.add_parameter("EndTime", end_time);
      prm.add_parameter("TimeStepSize", time_step_size);
      prm.add_parameter("IRKStages", irk_stages);

      prm.add_parameter("OuterTolerance", outer_tolerance);
      prm.add_parameter("InnerTolerance", inner_tolerance);

      prm.add_parameter("OperatorType",
                        operator_type,
                        "",
                        Patterns::Selection("MatrixBased|MatrixFree"));
      prm.add_parameter("BlockPreconditionerType",
                        block_preconditioner_type,
                        "",
                        Patterns::Selection("AMG|GMG"));

      prm.add_parameter("UseSharedMemory", use_sm);
      prm.add_parameter("DoRowMajor", do_row_major);
      prm.add_parameter("Padding", padding);
      prm.add_parameter("MaxRanks", max_ranks);

      prm.add_parameter("DoOutputParaview", do_output_paraview);

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

      // select operator
      std::unique_ptr<MassLaplaceOperator>        mass_laplace_operator;
      std::unique_ptr<ComplexMassLaplaceOperator> complex_mass_laplace_operator;

      if (params.operator_type == "MatrixBased")
        mass_laplace_operator =
          std::make_unique<MassLaplaceOperatorMatrixBased>(dof_handler,
                                                           constraints,
                                                           quadrature);
      else if (params.operator_type == "MatrixFree")
        mass_laplace_operator =
          std::make_unique<MassLaplaceOperatorMatrixFree<dim, double>>(
            dof_handler, constraints, quadrature);
      else
        AssertThrow(false, ExcNotImplemented());

      // select preconditioner
      std::unique_ptr<PreconditionerBase<VectorType>> preconditioner;

      std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations;

      if (params.block_preconditioner_type == "AMG")
        {
          preconditioner = std::make_unique<
            PreconditionerAMG<MassLaplaceOperator, VectorType>>(
            *mass_laplace_operator);
        }
      else if (params.block_preconditioner_type == "GMG")
        {
          // tighten, since we want to use a subcommunicator on the coarse grid
          RepartitioningPolicyTools::DefaultPolicy<dim> policy(true);
          mg_triangulations = MGTransferGlobalCoarseningTools::
            create_geometric_coarsening_sequence(triangulation, policy);

          const unsigned int min_level = 0;
          const unsigned int max_level = mg_triangulations.size() - 1;

          MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers(
            min_level, max_level);
          MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
            mg_constraints(min_level, max_level);
          MGLevelObject<std::shared_ptr<const MassLaplaceOperator>>
            mg_operators(min_level, max_level);

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
            this->dof_handler, mg_dof_handlers, mg_constraints, mg_operators);
        }
      else
        AssertThrow(false, ExcNotImplemented());

      // select time-integration scheme
      std::unique_ptr<TimeIntegrationSchemes::Interface>
        time_integration_scheme;

      const auto evaluate_rhs_function = [&](const double time,
                                             VectorType & tmp) -> void {
        RightHandSide rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(
          dof_handler, quadrature, rhs_function, tmp, constraints);
      };

      if (params.time_integration_scheme == "ost")
        time_integration_scheme =
          std::make_unique<TimeIntegrationSchemes::OneStepTheta>(
            comm_global,
            *mass_laplace_operator,
            *preconditioner,
            evaluate_rhs_function);
      else if (params.time_integration_scheme == "irk")
        time_integration_scheme = std::make_unique<TimeIntegrationSchemes::IRK>(
          comm_global,
          params.outer_tolerance,
          params.inner_tolerance,
          params.irk_stages,
          params.do_reduce_number_of_vmults,
          *mass_laplace_operator,
          *preconditioner,
          evaluate_rhs_function);
      else if (params.time_integration_scheme == "spirk")
        time_integration_scheme =
          std::make_unique<TimeIntegrationSchemes::IRKStageParallel>(
            comm_global,
            comm_row,
            params.outer_tolerance,
            params.inner_tolerance,
            params.irk_stages,
            params.do_reduce_number_of_vmults,
            params.use_sm,
            *mass_laplace_operator,
            *preconditioner,
            evaluate_rhs_function);
      else if (params.time_integration_scheme == "complex_irk")
        {
          if (params.operator_type == "MatrixFree")
            complex_mass_laplace_operator = std::make_unique<
              ComplexMassLaplaceOperatorMatrixFree<dim, double>>(dof_handler,
                                                                 constraints,
                                                                 quadrature);
          else
            AssertThrow(false, ExcNotImplemented());

          if (false)
            complex_mass_laplace_operator->set_scalar_operator(
              *mass_laplace_operator);

          time_integration_scheme =
            std::make_unique<TimeIntegrationSchemes::ComplexIRK>(
              comm_global,
              params.outer_tolerance,
              params.inner_tolerance,
              params.irk_stages,
              *mass_laplace_operator,
              *complex_mass_laplace_operator,
              *preconditioner,
              evaluate_rhs_function);
        }
      else if (params.time_integration_scheme == "complex_spirk")
        {
          if (params.operator_type == "MatrixFree")
            complex_mass_laplace_operator = std::make_unique<
              ComplexMassLaplaceOperatorMatrixFree<dim, double>>(dof_handler,
                                                                 constraints,
                                                                 quadrature);
          else
            AssertThrow(false, ExcNotImplemented());

          if (false)
            complex_mass_laplace_operator->set_scalar_operator(
              *mass_laplace_operator);

          time_integration_scheme =
            std::make_unique<TimeIntegrationSchemes::ComplexSPIRK>(
              comm_global,
              comm_row,
              params.outer_tolerance,
              params.inner_tolerance,
              params.irk_stages,
              *mass_laplace_operator,
              *complex_mass_laplace_operator,
              *preconditioner,
              evaluate_rhs_function);
        }
      else
        Assert(false, ExcNotImplemented());

      mass_laplace_operator->initialize_dof_vector(solution);
      mass_laplace_operator->initialize_dof_vector(system_rhs);

      double       time            = 0.0;
      unsigned int timestep_number = 0;

      VectorTools::interpolate(dof_handler, AnalyticalSolution(), solution);

      auto error = output_results(time, timestep_number);

      double dx_local = std::numeric_limits<double>::max();
      for (const auto &cell : triangulation.active_cell_iterators())
        dx_local = std::min(dx_local, cell->minimum_vertex_distance());
      const double dx = Utilities::MPI::min(dx_local, comm_global);

      const double time_step_size =
        (params.time_step_size > 0.0) ?
          params.time_step_size :
          std::pow(dx,
                   (params.fe_degree + 1.0) / (2.0 * params.irk_stages - 1.0));

      pcout << std::endl
            << "Starting time loop with dt=" << time_step_size << std::endl;

      AssertThrow(time_step_size < params.end_time, ExcNotImplemented());

      // perform time loop
      while ((params.end_time - time) > (1e-4 * time_step_size))
        {
          double time_step_size_truncated = time_step_size;

          if (time + time_step_size > params.end_time)
            {
              const double time_old    = time;
              time                     = params.end_time;
              time_step_size_truncated = time - time_old;
            }
          else
            {
              time += time_step_size;
            }

          pcout << std::endl
                << "Time step " << timestep_number << " at t=" << time
                << std::endl;

          ++timestep_number;

          time_integration_scheme->solve(solution,
                                         timestep_number,
                                         time,
                                         time_step_size_truncated);

          constraints.distribute(solution);

          error = output_results(time, timestep_number);
        }

      table.add_value("n_t", timestep_number);
      table.add_value("final_t", time);
      table.set_scientific("final_t", true);
      table.add_value("dt", time_step_size);
      table.set_scientific("dt", true);
      table.add_value("error_L2", error.first);
      table.set_scientific("error_L2", true);
      table.add_value("error_Linf", error.second);
      table.set_scientific("error_Linf", true);

      time_integration_scheme->get_statistics(table, timestep_number);
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
      table.add_value("fe_degree", fe.degree);
      table.add_value("n_dofs", dof_handler.n_dofs());
      table.add_value("n_stages", params.irk_stages);
      table.add_value("n_procs",
                      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
      table.add_value("n_procs_global",
                      Utilities::MPI::n_mpi_processes(comm_global));
      table.add_value("n_procs_row", Utilities::MPI::n_mpi_processes(comm_row));
      table.add_value("n_procs_column",
                      Utilities::MPI::n_mpi_processes(comm_column));

      constraints.clear();

      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
      constraints.reinit(locally_relevant_dofs);

      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      // note: program is limited to homogenous DBCs
      DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
      constraints.close();
    }

    std::pair<double, double>
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
          const double error_L2_norm =
            VectorTools::compute_global_error(triangulation,
                                              norm_per_cell,
                                              VectorTools::L2_norm);

          VectorTools::integrate_difference(dof_handler,
                                            solution,
                                            AnalyticalSolution(time),
                                            norm_per_cell,
                                            QGauss<dim>(fe.degree + 2),
                                            VectorTools::Linfty_norm);
          const double error_Linfty_norm =
            VectorTools::compute_global_error(triangulation,
                                              norm_per_cell,
                                              VectorTools::Linfty_norm);

          pcout << "   Error in the L2/L\u221E norm : " << error_L2_norm << "/"
                << error_Linfty_norm << std::endl;
          solution.zero_out_ghost_values();

          return {error_L2_norm, error_Linfty_norm};
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
        , a_z(2.0)
        , a_t(0.5)
      {}

      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override
      {
        (void)component;

        const double x = p[0];
        const double y = dim >= 2 ? p[1] : 0.0;
        const double z = dim >= 3 ? p[2] : 0.0;
        const double t = this->get_time();

        if (dim == 2)
          return std::sin(a_x * numbers::PI * x) *
                 std::sin(a_y * numbers::PI * y) *
                 (numbers::PI * std::cos(numbers::PI * t) -
                  a_t * (std::sin(numbers::PI * t) + 1) +
                  (a_x * a_x + a_y * a_y) * numbers::PI * numbers::PI *
                    (std::sin(numbers::PI * t) + 1)) *
                 std::exp(-a_t * t);
        else if (dim == 3)
          return std::sin(a_x * numbers::PI * x) *
                 std::sin(a_y * numbers::PI * y) *
                 std::sin(a_z * numbers::PI * z) *
                 (numbers::PI * std::cos(numbers::PI * t) -
                  a_t * (std::sin(numbers::PI * t) + 1) +
                  (a_x * a_x + a_y * a_y + a_z * a_z) * numbers::PI *
                    numbers::PI * (std::sin(numbers::PI * t) + 1)) *
                 std::exp(-a_t * t);

        Assert(false, ExcNotImplemented());

        return 0.0;
      }

    private:
      const double a_x;
      const double a_y;
      const double a_z;
      const double a_t;
    };

    class AnalyticalSolution : public Function<dim>
    {
    public:
      AnalyticalSolution(const double time = 0.0)
        : Function<dim>(1, time)
        , a_x(2.0)
        , a_y(2.0)
        , a_z(2.0)
        , a_t(0.5)
      {}

      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override
      {
        (void)component;

        const double x = p[0];
        const double y = dim >= 2 ? p[1] : 0.0;
        const double z = dim >= 3 ? p[2] : 0.0;
        const double t = this->get_time();

        if (dim == 2)
          return std::sin(a_x * numbers::PI * x) *
                 std::sin(a_y * numbers::PI * y) *
                 (1 + std::sin(numbers::PI * t)) * std::exp(-a_t * t);
        else if (dim == 3)
          return std::sin(a_x * numbers::PI * x) *
                 std::sin(a_y * numbers::PI * y) *
                 std::sin(a_z * numbers::PI * z) *
                 (1 + std::sin(numbers::PI * t)) * std::exp(-a_t * t);

        Assert(false, ExcNotImplemented());

        return 0.0;
      }

    private:
      const double a_x;
      const double a_y;
      const double a_z;
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

      constexpr unsigned int dim = IRK_DIMENSION;

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

          MPI_Comm comm = MPI_COMM_WORLD;

          if (params.max_ranks != 0 &&
              Utilities::MPI::n_mpi_processes(comm) != params.max_ranks)
            {
              AssertThrow(Utilities::MPI::n_mpi_processes(comm) >
                            params.max_ranks,
                          ExcInternalError());

              comm = Utilities::MPI::trim_comm(comm, params.max_ranks);

              if (comm == MPI_COMM_NULL)
                continue;
            }

          const unsigned int size_x =
            params.time_integration_scheme == "spirk" ?
              params.irk_stages :
              (params.time_integration_scheme == "complex-spirk" ?
                 (params.irk_stages + 1) / 2 :
                 1);

          AssertThrow(size_x <= Utilities::MPI::n_mpi_processes(comm),
                      ExcMessage("Not enough ranks have been provided!"));

          AssertThrow(
            (params.do_row_major == true) || (params.padding == -1),
            ExcMessage(
              "Padding can only be turned if row-major ordering is enabled."));
          AssertThrow(
            (params.do_row_major == false) || (params.padding <= 0) ||
              size_x <= static_cast<unsigned int>(params.padding),
            ExcMessage(
              "Padding has to be at least as large as the number of stages."));

          const unsigned int padding =
            (params.padding == -1) ?
              size_x :
              ((params.padding == 0) ? Utilities::MPI::n_procs_of_sm(comm) :
                                       params.padding);

          MPI_Comm comm_global =
            Utilities::MPI::create_rectangular_comm(comm, size_x, padding);

          if (comm_global != MPI_COMM_NULL)
            {
              const unsigned int size_v =
                Utilities::MPI::n_mpi_processes(comm_global) / size_x;

              MPI_Comm comm_row = Utilities::MPI::create_row_comm(
                comm_global, size_x, size_v, params.do_row_major);
              MPI_Comm comm_column = Utilities::MPI::create_column_comm(
                comm_global, size_x, size_v, params.do_row_major);

#ifdef DEBUG
              auto ranks = Utilities::MPI::gather(
                comm_global,
                std::array<unsigned int, 3>{
                  {Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                   Utilities::MPI::this_mpi_process(comm_row),
                   Utilities::MPI::this_mpi_process(comm_column)

                  }},
                0);

              std::sort(ranks.begin(), ranks.end(), [](auto &a, auto &b) {
                if (a[2] != b[2])
                  return a[2] < b[2];
                return a[1] < b[1];
              });


              const unsigned int needed_digits =
                1 +
                Utilities::needed_digits(Utilities::MPI::n_mpi_processes(comm));

              pcout << "Virtual topology:" << std::endl;
              for (unsigned int i = 0, c = 0; i < size_v; ++i)
                {
                  if (i == 0)
                    {
                      pcout << std::setw(needed_digits) << " "
                            << " ";
                      for (unsigned int j = 0; j < size_x; ++j)
                        pcout << std::setw(needed_digits) << j << " ";
                      pcout << std::endl;
                    }

                  pcout << std::setw(needed_digits) << i << " ";
                  for (unsigned int j = 0; j < size_x; ++j, ++c)
                    pcout << std::setw(needed_digits) << ranks[c][0] << " ";
                  pcout << std::endl;
                }

#endif

              {
                HeatEquation::Problem<dim> heat_equation_solver(
                  params, comm_global, comm_row, comm_column, table);
                heat_equation_solver.run();
              }

              MPI_Comm_free(&comm_column);
              MPI_Comm_free(&comm_row);
              MPI_Comm_free(&comm_global);

              if (comm != MPI_COMM_WORLD)
                MPI_Comm_free(&comm);
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
