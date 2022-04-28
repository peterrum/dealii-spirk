#pragma once

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/matrix_creator.h>

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

  virtual const SparseMatrixType &
  get_system_matrix(const MPI_Comm comm) const = 0;

  virtual bool
  supports_sub_communicator() const = 0;

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

    MatrixCreator::create_mass_matrix<dim, dim, SparseMatrixType>(
      dof_handler, quadrature, mass_matrix, nullptr, constraints);
    MatrixCreator::create_laplace_matrix<dim, dim, SparseMatrixType>(
      dof_handler, quadrature, laplace_matrix, nullptr, constraints);

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

  using MassLaplaceOperator::vmult;

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    dst = 0.0; // TODO
    this->vmult_add(dst, src);
  }

  void
  vmult_add(VectorType &dst, const VectorType &src) const override
  {
    tmp.reinit(src);

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
    else if (laplace_matrix_scaling == 1.0)
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
    (void)diagonal;

#ifdef DEAL_II_WITH_TRILINOS
    this->initialize_dof_vector(diagonal);
    const auto &system_matrix = get_system_matrix();
    for (auto entry : system_matrix)
      if (entry.row() == entry.column() && std::abs(entry.value()) > 1e-10)
        diagonal[entry.row()] = 1.0 / entry.value();
#else
    Assert(false, ExcNotImplemented());
#endif
  }

  const SparseMatrixType &
  get_system_matrix() const override
  {
    tmp_matrix.copy_from(laplace_matrix);
    tmp_matrix *= laplace_matrix_scaling;
    tmp_matrix.add(mass_matrix_scaling, mass_matrix);

    return tmp_matrix;
  }

  const SparseMatrixType &
  get_system_matrix(const MPI_Comm comm) const override
  {
    (void)comm;
    return get_system_matrix();
  }

  virtual bool
  supports_sub_communicator() const
  {
    return false;
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

  using MassLaplaceOperator::vmult;

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
    return get_system_matrix(matrix_free.get_task_info().communicator);
  }

  const SparseMatrixType &
  get_system_matrix(const MPI_Comm comm) const override
  {
    if (comm != MPI_COMM_NULL && system_matrix.m() == 0 &&
        system_matrix.n() == 0)
      {
        const auto &dof_handler = this->matrix_free.get_dof_handler();
        TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_dofs(),
                                              comm);
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

  virtual bool
  supports_sub_communicator() const
  {
    return true;
  }

  void
  compute_inverse_diagonal(VectorType &diagonal) const override
  {
    this->initialize_dof_vector(diagonal);

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
              integrator.submit_gradient(laplace_matrix_scaling *
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
          integrator.submit_gradient(laplace_matrix_scaling *
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


class ComplexMassLaplaceOperator : public Subscriptor
{
public:
  using Number = typename VectorType::value_type;

  ComplexMassLaplaceOperator()
    : lambda_re(1.0)
    , lambda_im(1.0)
    , tau(1.0)
  {}

  void
  reinit(const double lambda_re, const double lambda_im, const double tau) const
  {
    this->lambda_re = lambda_re;
    this->lambda_im = lambda_im;
    this->tau       = tau;
  }

  virtual void
  vmult(BlockVectorType &dst, const BlockVectorType &src) const = 0;

protected:
  mutable double lambda_re;
  mutable double lambda_im;
  mutable double tau;
};

template <int dim, typename Number>
class ComplexMassLaplaceOperatorMatrixFree : public ComplexMassLaplaceOperator
{
  using FECellIntegrator = FEEvaluation<dim, -1, 0, 1, Number>;

public:
  ComplexMassLaplaceOperatorMatrixFree(
    const DoFHandler<dim> &          dof_handler,
    const AffineConstraints<Number> &constraints,
    const Quadrature<dim> &          quadrature)
  {
    this->constraints = &constraints;

    typename MatrixFree<dim, Number>::AdditionalData data;
    data.mapping_update_flags = update_values | update_gradients;
    matrix_free.reinit(
      MappingQ1<dim>(), dof_handler, constraints, quadrature, data);
  }

  void
  vmult(BlockVectorType &dst, const BlockVectorType &src) const override
  {
    matrix_free.template cell_loop<BlockVectorType, BlockVectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cells) {
        FECellIntegrator phi_re(matrix_free);
        FECellIntegrator phi_im(matrix_free);

        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi_re.reinit(cell);
            phi_re.gather_evaluate(src.block(0),
                                   EvaluationFlags::values |
                                     EvaluationFlags::gradients);
            phi_im.reinit(cell);
            phi_im.gather_evaluate(src.block(1),
                                   EvaluationFlags::values |
                                     EvaluationFlags::gradients);

            for (const auto q : phi_re.quadrature_point_indices())
              {
                const auto value_re    = phi_re.get_value(q);
                const auto value_im    = phi_im.get_value(q);
                const auto gradient_re = phi_re.get_gradient(q);
                const auto gradient_im = phi_im.get_gradient(q);

                phi_re.submit_value(lambda_re * value_re - lambda_im * value_im,
                                    q);
                phi_re.submit_gradient(gradient_re * tau, q);

                phi_im.submit_value(lambda_im * value_re + lambda_re * value_im,
                                    q);
                phi_im.submit_gradient(gradient_im * tau, q);
              }

            phi_re.integrate_scatter(EvaluationFlags::values |
                                       EvaluationFlags::gradients,
                                     dst.block(0));
            phi_im.integrate_scatter(EvaluationFlags::values |
                                       EvaluationFlags::gradients,
                                     dst.block(1));
          }
      },
      dst,
      src,
      true);
  }

private:
  SmartPointer<const AffineConstraints<Number>> constraints;

  MatrixFree<dim, Number> matrix_free;

  mutable SparseMatrixType system_matrix;
};
