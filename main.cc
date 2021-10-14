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

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <vector>

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
                     Utilities::MPI::internal::mpi_type_id(&offset),
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



class MassLaplaceOperator : public Subscriptor
{
public:
  using Number = typename VectorType::value_type;

  MassLaplaceOperator()
    : mass_matrix_scaling(1.0)
    , laplace_matrix_scaling(1.0)
  {}

  void
  reinit(const double mass_matrix_scaling,
         const double laplace_matrix_scaling) const
  {
    this->mass_matrix_scaling    = mass_matrix_scaling;
    this->laplace_matrix_scaling = laplace_matrix_scaling;

    for (const auto &op : attached_operators)
      op->reinit(this->mass_matrix_scaling, this->laplace_matrix_scaling);
  }

  virtual void
  initialize_dof_vector(VectorType &vec) const = 0;

  virtual void
  vmult(VectorType &      dst,
        const VectorType &src,
        const double      mass_matrix_scaling,
        const double      laplace_matrix_scaling) const
  {
    this->reinit(mass_matrix_scaling, laplace_matrix_scaling);
    this->vmult(dst, src);
  }

  virtual types::global_dof_index
  m() const = 0;

  virtual Number
  el(unsigned int, unsigned int) const = 0;

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    this->vmult(dst, src);
  }

  virtual void
  vmult(VectorType &dst, const VectorType &src) const = 0;

  void
  vmult_add(VectorType &      dst,
            const VectorType &src,
            const double      mass_matrix_scaling,
            const double      laplace_matrix_scaling) const
  {
    this->reinit(mass_matrix_scaling, laplace_matrix_scaling);
    this->vmult_add(dst, src);
  }

  virtual void
  vmult_add(VectorType &dst, const VectorType &src) const = 0;

  virtual const SparseMatrixType &
  get_system_matrix() const = 0;

  virtual void
  compute_inverse_diagonal(VectorType &diagonal) const = 0;

  void
  attach(const MassLaplaceOperator &other) const
  {
    attached_operators.push_back(&other);
  }

protected:
  mutable double mass_matrix_scaling;
  mutable double laplace_matrix_scaling;

  mutable std::vector<const MassLaplaceOperator *> attached_operators;
};



class MassLaplaceOperatorMatrixBased : public MassLaplaceOperator
{
public:
  template <int dim, typename Number>
  MassLaplaceOperatorMatrixBased(const DoFHandler<dim> &          dof_handler,
                                 const AffineConstraints<Number> &constraints,
                                 const Quadrature<dim> &          quadrature)
  {
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

    IndexSet locally_relevant_dofs;

    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    this->partitioner = std::make_shared<const Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(),
      locally_relevant_dofs,
      dof_handler.get_communicator());
  }

  types::global_dof_index
  m() const override
  {
    return mass_matrix.m();
  }

  Number
  el(unsigned int i, unsigned int j) const
  {
    return mass_matrix_scaling * mass_matrix(i, j) +
           laplace_matrix_scaling * laplace_matrix(i, j);
  }

  void
  initialize_dof_vector(VectorType &vec) const override
  {
    vec.reinit(partitioner);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    dst = 0.0; // TODO
    this->vmult_add(dst, src);
  }

  void
  vmult_add(VectorType &dst, const VectorType &src) const override
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

  void
  compute_inverse_diagonal(VectorType &diagonal) const override
  {
    AssertThrow(false, ExcNotImplemented());
    (void)diagonal;
  }

  const SparseMatrixType &
  get_system_matrix() const override
  {
    tmp_matrix.copy_from(laplace_matrix);
    tmp_matrix *= laplace_matrix_scaling;
    tmp_matrix.add(mass_matrix_scaling, mass_matrix);

    return tmp_matrix;
  }


private:
  mutable SparseMatrixType                                   mass_matrix;
  mutable SparseMatrixType                                   laplace_matrix;
  mutable std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

  mutable VectorType       tmp;
  mutable SparseMatrixType tmp_matrix;
};



template <int dim, typename Number>
class MassLaplaceOperatorMatrixFree : public MassLaplaceOperator
{
public:
  MassLaplaceOperatorMatrixFree(const DoFHandler<dim> &          dof_handler,
                                const AffineConstraints<Number> &constraints,
                                const Quadrature<dim> &          quadrature)
  {
    this->constraints = &constraints;

    typename MatrixFree<dim, Number>::AdditionalData data;
    data.mapping_update_flags = update_values | update_gradients;
    matrix_free.reinit(
      MappingQ1<dim>(), dof_handler, constraints, quadrature, data);
  }

  types::global_dof_index
  m() const override
  {
    return matrix_free.get_dof_handler().n_dofs();
  }

  Number
  el(unsigned int, unsigned int) const override
  {
    Assert(false, ExcNotImplemented());
    return 0.0;
  }

  void
  initialize_dof_vector(VectorType &vec) const override
  {
    matrix_free.initialize_dof_vector(vec);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    this->matrix_free.cell_loop(
      &MassLaplaceOperatorMatrixFree::do_cell_integral_range,
      this,
      dst,
      src,
      true);
  }

  void
  vmult_add(VectorType &dst, const VectorType &src) const override
  {
    this->matrix_free.cell_loop(
      &MassLaplaceOperatorMatrixFree::do_cell_integral_range,
      this,
      dst,
      src,
      false);
  }

  const SparseMatrixType &
  get_system_matrix() const override
  {
    if (system_matrix.m() == 0 && system_matrix.n() == 0)
      {
        const auto &dof_handler = this->matrix_free.get_dof_handler();
        TrilinosWrappers::SparsityPattern dsp(
          dof_handler.locally_owned_dofs(),
          dof_handler.get_triangulation().get_communicator());
        DoFTools::make_sparsity_pattern(dof_handler, dsp, *constraints);
        dsp.compress();
        system_matrix.reinit(dsp);
      }

    MatrixFreeTools::compute_matrix(
      matrix_free,
      *constraints,
      system_matrix,
      &MassLaplaceOperatorMatrixFree::do_cell_integral,
      this);

    return system_matrix;
  }

  void
  compute_inverse_diagonal(VectorType &diagonal) const override
  {
    diagonal = 0.0;
    MatrixFreeTools::compute_diagonal(
      matrix_free,
      diagonal,
      &MassLaplaceOperatorMatrixFree::do_cell_integral,
      this);
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }


private:
  using FECellIntegrator = FEEvaluation<dim, -1, 0, 1, Number>;

  void
  do_cell_integral_range(
    const MatrixFree<dim, Number> &              matrix_free,
    VectorType &                                 dst,
    const VectorType &                           src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free, range);
    for (unsigned cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);

        if (mass_matrix_scaling != 0.0 && laplace_matrix_scaling != 0.0)
          integrator.gather_evaluate(src,
                                     EvaluationFlags::values |
                                       EvaluationFlags::gradients);
        else if (mass_matrix_scaling != 0.0)
          integrator.gather_evaluate(src, EvaluationFlags::values);
        else if (laplace_matrix_scaling != 0.0)
          integrator.gather_evaluate(src, EvaluationFlags::gradients);

        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            if (mass_matrix_scaling != 0.0)
              integrator.submit_value(mass_matrix_scaling *
                                        integrator.get_value(q),
                                      q);
            if (laplace_matrix_scaling != 0.0)
              integrator.submit_gradient(-laplace_matrix_scaling *
                                           integrator.get_gradient(q),
                                         q);
          }

        if (mass_matrix_scaling != 0.0 && laplace_matrix_scaling != 0.0)
          integrator.integrate_scatter(EvaluationFlags::values |
                                         EvaluationFlags::gradients,
                                       dst);
        else if (mass_matrix_scaling != 0.0)
          integrator.integrate_scatter(EvaluationFlags::values, dst);
        else if (laplace_matrix_scaling != 0.0)
          integrator.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  void
  do_cell_integral(FECellIntegrator &integrator) const
  {
    if (mass_matrix_scaling != 0.0 && laplace_matrix_scaling != 0.0)
      integrator.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    else if (mass_matrix_scaling != 0.0)
      integrator.evaluate(EvaluationFlags::values);
    else if (laplace_matrix_scaling != 0.0)
      integrator.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        if (mass_matrix_scaling != 0.0)
          integrator.submit_value(mass_matrix_scaling * integrator.get_value(q),
                                  q);
        if (laplace_matrix_scaling != 0.0)
          integrator.submit_gradient(-laplace_matrix_scaling *
                                       integrator.get_gradient(q),
                                     q);
      }

    if (mass_matrix_scaling != 0.0 && laplace_matrix_scaling != 0.0)
      integrator.integrate(EvaluationFlags::values |
                           EvaluationFlags::gradients);
    else if (mass_matrix_scaling != 0.0)
      integrator.integrate(EvaluationFlags::values);
    else if (laplace_matrix_scaling != 0.0)
      integrator.integrate(EvaluationFlags::gradients);
  }

  SmartPointer<const AffineConstraints<Number>> constraints;

  MatrixFree<dim, Number> matrix_free;

  mutable SparseMatrixType system_matrix;
};



template <typename VectorType>
class PreconditionerBase
{
public:
  virtual void
  reinit() const = 0;

  virtual void
  vmult(VectorType &dst, const VectorType &src) const = 0;

  virtual std::unique_ptr<const PreconditionerBase<VectorType>>
  clone() const = 0;


private:
};



template <typename MatrixType, typename VectorType>
class PreconditionerAMG : public PreconditionerBase<VectorType>
{
public:
  PreconditionerAMG(
    const MatrixType &op,
    const typename TrilinosWrappers::PreconditionAMG::AdditionalData
      additional_data =
        typename TrilinosWrappers::PreconditionAMG::AdditionalData())
    : op(op)
    , additional_data(additional_data)
  {}

  virtual void
  reinit() const override
  {
    amg.initialize(op.get_system_matrix(), additional_data);
  }

  virtual void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    amg.vmult(dst, src);
  }

  virtual std::unique_ptr<const PreconditionerBase<VectorType>>
  clone() const override
  {
    return std::make_unique<PreconditionerAMG<MatrixType, VectorType>>(
      this->op, this->additional_data);
  }

private:
  const MatrixType &op;

  const typename TrilinosWrappers::PreconditionAMG::AdditionalData
    additional_data;

  mutable TrilinosWrappers::PreconditionAMG amg;
};



struct PreconditionerGMGAdditionalData
{
  double       smoothing_range               = 20;
  unsigned int smoothing_degree              = 5;
  unsigned int smoothing_eig_cg_n_iterations = 20;

  unsigned int coarse_grid_smoother_sweeps = 1;
  unsigned int coarse_grid_n_cycles        = 1;
  std::string  coarse_grid_smoother_type   = "ILU";

  unsigned int coarse_grid_maxiter = 1000;
  double       coarse_grid_abstol  = 1e-20;
  double       coarse_grid_reltol  = 1e-4;
};



template <int dim, typename LevelMatrixType, typename VectorType>
class PreconditionerGMG : public PreconditionerBase<VectorType>
{
public:
  PreconditionerGMG(
    const DoFHandler<dim> &dof_handler,
    const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>
      &mg_dof_handlers,
    const MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
      &                                                          mg_constraints,
    const MGLevelObject<std::shared_ptr<const LevelMatrixType>> &mg_operators)
    : dof_handler(dof_handler)
    , mg_dof_handlers(mg_dof_handlers)
    , mg_constraints(mg_constraints)
    , mg_operators(mg_operators)
    , min_level(mg_dof_handlers.min_level())
    , max_level(mg_dof_handlers.max_level())
    , transfers(min_level, max_level)
    , transfer(transfers, [&](const auto l, auto &vec) {
      this->mg_operators[l]->initialize_dof_vector(vec);
    })
  {
    // setup transfer operators
    for (auto l = min_level; l < max_level; ++l)
      transfers[l + 1].reinit(*mg_dof_handlers[l + 1],
                              *mg_dof_handlers[l],
                              *mg_constraints[l + 1],
                              *mg_constraints[l]);
  }

  virtual void
  reinit() const override
  {
    PreconditionerGMGAdditionalData additional_data;

    // wrap level operators
    mg_matrix = std::make_unique<mg::Matrix<VectorType>>(mg_operators);

    // setup smoothers on each level
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
      min_level, max_level);

    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        smoother_data[level].preconditioner =
          std::make_shared<SmootherPreconditionerType>();
        mg_operators[level]->compute_inverse_diagonal(
          smoother_data[level].preconditioner->get_vector());
        smoother_data[level].smoothing_range = additional_data.smoothing_range;
        smoother_data[level].degree          = additional_data.smoothing_degree;
        smoother_data[level].eig_cg_n_iterations =
          additional_data.smoothing_eig_cg_n_iterations;
      }

    mg_smoother.initialize(mg_operators, smoother_data);

    // setup coarse-grid solver
    coarse_grid_solver_control =
      std::make_unique<ReductionControl>(additional_data.coarse_grid_maxiter,
                                         additional_data.coarse_grid_abstol,
                                         additional_data.coarse_grid_reltol,
                                         false,
                                         false);
    coarse_grid_solver =
      std::make_unique<SolverCG<VectorType>>(*coarse_grid_solver_control);

    precondition_amg = std::make_unique<TrilinosWrappers::PreconditionAMG>();

    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.smoother_sweeps = additional_data.coarse_grid_smoother_sweeps;
    amg_data.n_cycles        = additional_data.coarse_grid_n_cycles;
    amg_data.smoother_type = additional_data.coarse_grid_smoother_type.c_str();
    precondition_amg->initialize(mg_operators[min_level]->get_system_matrix(),
                                 amg_data);
    mg_coarse = std::make_unique<
      MGCoarseGridIterativeSolver<VectorType,
                                  SolverCG<VectorType>,
                                  LevelMatrixType,
                                  TrilinosWrappers::PreconditionAMG>>(
      *coarse_grid_solver, *mg_operators[min_level], *precondition_amg);

    // create multigrid algorithm (put level operators, smoothers, transfer
    // operators and smoothers together)
    mg = std::make_unique<Multigrid<VectorType>>(
      *mg_matrix, *mg_coarse, transfer, mg_smoother, mg_smoother);

    // convert multigrid algorithm to preconditioner
    preconditioner =
      std::make_unique<PreconditionMG<dim, VectorType, MGTransferType>>(
        dof_handler, *mg, transfer);
  }

  virtual void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    preconditioner->vmult(dst, src);
  }

  virtual std::unique_ptr<const PreconditionerBase<VectorType>>
  clone() const override
  {
    return std::make_unique<
      PreconditionerGMG<dim, MassLaplaceOperator, VectorType>>(dof_handler,
                                                               mg_dof_handlers,
                                                               mg_constraints,
                                                               mg_operators);
  }

private:
  using MGTransferType = MGTransferGlobalCoarsening<dim, VectorType>;
  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                             VectorType,
                                             SmootherPreconditionerType>;

  const DoFHandler<dim> &dof_handler;

  const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers;
  const MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
                                                              mg_constraints;
  const MGLevelObject<std::shared_ptr<const LevelMatrixType>> mg_operators;

  const unsigned int min_level;
  const unsigned int max_level;

  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
  MGTransferType                                     transfer;

  mutable std::unique_ptr<mg::Matrix<VectorType>> mg_matrix;

  mutable MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>
    mg_smoother;

  mutable std::unique_ptr<ReductionControl> coarse_grid_solver_control;

  mutable std::unique_ptr<SolverCG<VectorType>> coarse_grid_solver;

  mutable std::unique_ptr<TrilinosWrappers::PreconditionAMG> precondition_amg;

  mutable std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  mutable std::unique_ptr<Multigrid<VectorType>> mg;

  mutable std::unique_ptr<PreconditionMG<dim, VectorType, MGTransferType>>
    preconditioner;
};



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
      : n_stages(n_stages)
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
  };



  /**
   * A parallel IRK implementation.
   */
  class IRK : public IRKBase
  {
  public:
    IRK(const MPI_Comm                        comm,
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
                                             time_step,
                                             op,
                                             block_preconditioner,
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

      op.vmult(tmp, solution, 0.0, 1.0);

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
                             -time_step);
                  else // proceed with off-diagonals
                    op.vmult_add(dst.block(i), src.block(k), A_inv(i, k), 0.0);
                }
          }
        else
          {
            VectorType tmp;
            tmp.reinit(src.block(0));
            for (unsigned int i = 0; i < n_stages; ++i)
              op.vmult(dst.block(i), src.block(i), 0.0, -time_step);

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
                     const double                          time_step,
                     const MassLaplaceOperator &           op,
                     const PreconditionerBase<VectorType> &preconditioner,
                     double &                              time_bc,
                     double &                              time_solver)
        : n_max_iterations(100)
        , abs_tolerance(1e-6)
        , cut_off_tolerance(1e-12)
        , n_stages(d_vec.size())
        , d_vec(d_vec)
        , T_mat(T)
        , T_mat_inv(T_inv)
        , tau(time_step)
        , op(op)
        , time_bc(time_bc)
        , time_solver(time_solver)
        , n_iterations(0)
      {
        preconditioners.resize(n_stages);

        for (unsigned int i = 0; i < n_stages; ++i)
          {
            op.reinit(d_vec[i], -tau);

            preconditioners[i] = preconditioner.clone();
            preconditioners[i]->reinit();
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

            op.reinit(d_vec[i], -tau);

            solver.solve(op,
                         tmp_vectors.block(i),
                         dst.block(i),
                         *preconditioners[i]);

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

      const MassLaplaceOperator &op;
      std::vector<std::unique_ptr<const PreconditionerBase<VectorType>>>
        preconditioners;

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

    IRKStageParallel(const MPI_Comm             comm_global,
                     const MPI_Comm             comm_row,
                     const unsigned int         n_stages,
                     const bool                 do_reduce_number_of_vmults,
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
          this->system_matrix =
            std::make_unique<SystemMatrix>(comm_row,
                                           do_reduce_number_of_vmults,
                                           A_inv,
                                           time_step,
                                           op,
                                           time_system_vmult);
          this->preconditioner =
            std::make_unique<Preconditioner>(comm_row,
                                             d_vec,
                                             T,
                                             T_inv,
                                             time_step,
                                             op,
                                             block_preconditioner,
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
      op.vmult(tmp, solution, 0.0, 1.0);
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
      SystemMatrix(const MPI_Comm &comm_row,
                   const bool      do_reduce_number_of_vmults,
                   const FullMatrix<typename VectorType::value_type> &A_inv,
                   const double                                       time_step,
                   const MassLaplaceOperator &                        op,
                   double &                                           time)
        : my_stage(Utilities::MPI::this_mpi_process(comm_row))
        , do_reduce_number_of_vmults(do_reduce_number_of_vmults)
        , A_inv(A_inv)
        , time_step(time_step)
        , op(op)
        , time(time)
      {}

      void
      vmult(ReshapedVectorType &dst, const ReshapedVectorType &src) const
      {
        const auto time = std::chrono::system_clock::now();

        ReshapedVectorType temp;
        temp.reinit(src);

        if (do_reduce_number_of_vmults == false)
          {
            matrix_vector_rol_operation<VectorType>(
              dst,
              src,
              [this,
               &temp](const auto i, const auto j, auto &dst, const auto &src) {
                if (i == j)
                  op.vmult(static_cast<VectorType &>(dst),
                           static_cast<const VectorType &>(src),
                           A_inv(i, j),
                           -time_step);
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
                     -time_step);
            op.vmult(static_cast<VectorType &>(temp),
                     static_cast<const VectorType &>(src),
                     1.0,
                     0.0);

            matrix_vector_rol_operation<VectorType>(
              dst,
              temp,
              [this,
               &temp](const auto i, const auto j, auto &dst, const auto &src) {
                dst.add(A_inv(i, j), src);
              });
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
    };

    class Preconditioner
    {
    public:
      Preconditioner(const MPI_Comm &                               comm_row,
                     const Vector<typename VectorType::value_type> &d_vec,
                     const FullMatrix<typename VectorType::value_type> &T,
                     const FullMatrix<typename VectorType::value_type> &T_inv,
                     const double                          time_step,
                     const MassLaplaceOperator &           op,
                     const PreconditionerBase<VectorType> &preconditioners,
                     double &                              time_bc,
                     double &                              time_solver)
        : n_max_iterations(100)
        , abs_tolerance(1e-6)
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
      {
        op.reinit(d_vec[my_stage], -tau);
        preconditioners.reinit();
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

        op.reinit(d_vec[my_stage], -tau);

        solver.solve(op,
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

    unsigned int irk_stages                 = 3;
    bool         do_reduce_number_of_vmults = true;

    std::string operator_type             = "MatrixBased";
    std::string block_preconditioner_type = "AMG";

    bool do_row_major = true;
    int  padding      = -1; // -1: no padding; 0: use sm;
                            // else valid: padding > irk_stages

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

      prm.add_parameter("OperatorType",
                        operator_type,
                        "",
                        Patterns::Selection("MatrixBased|MatrixFree"));
      prm.add_parameter("BlockPreconditionerType",
                        block_preconditioner_type,
                        "",
                        Patterns::Selection("AMG|GMG"));

      prm.add_parameter("DoRowMajor", do_row_major);
      prm.add_parameter("Padding", padding);

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
      std::unique_ptr<MassLaplaceOperator> mass_laplace_operator;

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
          mg_triangulations = MGTransferGlobalCoarseningTools::
            create_geometric_coarsening_sequence(triangulation);

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
            params.irk_stages,
            params.do_reduce_number_of_vmults,
            *mass_laplace_operator,
            *preconditioner,
            evaluate_rhs_function);
      else
        Assert(false, ExcNotImplemented());

      mass_laplace_operator->initialize_dof_vector(solution);
      mass_laplace_operator->initialize_dof_vector(system_rhs);

      double       time            = 0.0;
      unsigned int timestep_number = 0;

      VectorTools::interpolate(dof_handler, AnalyticalSolution(), solution);

      output_results(time, timestep_number);

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

          AssertThrow(size_x <= Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),
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
              ((params.padding == 0) ?
                 Utilities::MPI::n_procs_of_sm(MPI_COMM_WORLD) :
                 params.padding);

          MPI_Comm comm_global =
            Utilities::MPI::create_rectangular_comm(MPI_COMM_WORLD,
                                                    size_x,
                                                    padding);

          const unsigned int size_v =
            Utilities::MPI::n_mpi_processes(comm_global) / size_x;

          if (comm_global != MPI_COMM_NULL)
            {
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
                1 + Utilities::needed_digits(
                      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));

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
