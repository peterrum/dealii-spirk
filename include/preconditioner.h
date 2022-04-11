#pragma once

namespace dealii
{
  /**
   * Coarse grid solver using a preconditioner only. This is a little wrapper,
   * transforming a preconditioner into a coarse grid solver.
   */
  template <class VectorType, class PreconditionerType>
  class MGCoarseGridApplyPreconditioner : public MGCoarseGridBase<VectorType>
  {
  public:
    /**
     * Default constructor.
     */
    MGCoarseGridApplyPreconditioner();

    /**
     * Constructor. Store a pointer to the preconditioner for later use.
     */
    MGCoarseGridApplyPreconditioner(const PreconditionerType &precondition);

    /**
     * Clear the pointer.
     */
    void
    clear();

    /**
     * Initialize new data.
     */
    void
    initialize(const PreconditionerType &precondition);

    /**
     * Implementation of the abstract function.
     */
    virtual void
    operator()(const unsigned int level,
               VectorType &       dst,
               const VectorType & src) const override;

  private:
    /**
     * Reference to the preconditioner.
     */
    SmartPointer<
      const PreconditionerType,
      MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>>
      preconditioner;
  };



  template <class VectorType, class PreconditionerType>
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::
    MGCoarseGridApplyPreconditioner()
    : preconditioner(0, typeid(*this).name())
  {}



  template <class VectorType, class PreconditionerType>
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::
    MGCoarseGridApplyPreconditioner(const PreconditionerType &preconditioner)
    : preconditioner(&preconditioner, typeid(*this).name())
  {}



  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::initialize(
    const PreconditionerType &preconditioner_)
  {
    preconditioner = &preconditioner_;
  }



  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::clear()
  {
    preconditioner = 0;
  }


  namespace internal
  {
    namespace MGCoarseGridApplyPreconditioner
    {
      template <class VectorType,
                class PreconditionerType,
                typename std::enable_if<
                  std::is_same<typename VectorType::value_type, double>::value,
                  VectorType>::type * = nullptr>
      void
      solve(const PreconditionerType preconditioner,
            VectorType &             dst,
            const VectorType &       src)
      {
        // to allow the case that the preconditioner was only set up on a
        // subset of processes
        if (preconditioner != nullptr)
          preconditioner->vmult(dst, src);
      }

      template <class VectorType,
                class PreconditionerType,
                typename std::enable_if<
                  !std::is_same<typename VectorType::value_type, double>::value,
                  VectorType>::type * = nullptr>
      void
      solve(const PreconditionerType preconditioner,
            VectorType &             dst,
            const VectorType &       src)
      {
        LinearAlgebra::distributed::Vector<double> src_;
        LinearAlgebra::distributed::Vector<double> dst_;

        src_ = src;
        dst_ = dst;

        // to allow the case that the preconditioner was only set up on a
        // subset of processes
        if (preconditioner != nullptr)
          preconditioner->vmult(dst_, src_);

        dst = dst_;
      }
    } // namespace MGCoarseGridApplyPreconditioner
  }   // namespace internal


  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::operator()(
    const unsigned int /*level*/,
    VectorType &      dst,
    const VectorType &src) const
  {
    internal::MGCoarseGridApplyPreconditioner::solve(preconditioner, dst, src);
  }
} // namespace dealii


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
  using MGTransferType = MGTransferGlobalCoarsening<dim, VectorType>;

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
  {
    // setup transfer operators
    for (auto l = min_level; l < max_level; ++l)
      transfers[l + 1].reinit(*mg_dof_handlers[l + 1],
                              *mg_dof_handlers[l],
                              *mg_constraints[l + 1],
                              *mg_constraints[l]);

    transfer =
      std::make_unique<MGTransferType>(transfers, [&](const auto l, auto &vec) {
        this->mg_operators[l]->initialize_dof_vector(vec);
      });

    this->sub_comm = create_sub_comm(*mg_dof_handlers[min_level]);
  }

  template <typename MeshType>
  static std::unique_ptr<MPI_Comm, std::function<void(MPI_Comm *)>>
  create_sub_comm(const MeshType &mesh)
  {
    const auto comm     = mesh.get_communicator();
    auto       sub_comm = new MPI_Comm;

    unsigned int cell_counter = 0;

    for (const auto &cell : mesh.active_cell_iterators())
      if (cell->is_locally_owned())
        cell_counter++;

    const unsigned int rank = Utilities::MPI::this_mpi_process(comm);

#if DEBUG
    const auto t = Utilities::MPI::gather(comm, cell_counter);

    if (rank == 0)
      {
        for (const auto tt : t)
          std::cout << tt << " ";
        std::cout << std::endl;
      }
#endif

    const int temp = cell_counter == 0 ? -1 : rank;

    const unsigned int max_rank = Utilities::MPI::max(temp, comm);

    if (max_rank != Utilities::MPI::n_mpi_processes(comm) - 1)
      {
        const bool color = rank <= max_rank;
        MPI_Comm_split(comm, color, rank, sub_comm);

        if (color == false)
          {
            MPI_Comm_free(sub_comm);
            *sub_comm = MPI_COMM_NULL;
          }
      }

    return std::unique_ptr<MPI_Comm, std::function<void(MPI_Comm *)>>(
      sub_comm, [](MPI_Comm *sub_comm) {
        if (*sub_comm != MPI_COMM_NULL)
          MPI_Comm_free(sub_comm);
        delete sub_comm;
      });
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
        smoother_data[level].constraints.copy_from(*mg_constraints[level]);
      }

    mg_smoother.initialize(mg_operators, smoother_data);

    // setup coarse-grid solver
    const auto coarse_comm =
      mg_operators[min_level]->supports_sub_communicator() ?
        *sub_comm :
        mg_dof_handlers[min_level]->get_communicator();
    if (coarse_comm != MPI_COMM_NULL)
      {
        precondition_amg =
          std::make_unique<TrilinosWrappers::PreconditionAMG>();

        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.smoother_sweeps = additional_data.coarse_grid_smoother_sweeps;
        amg_data.n_cycles        = additional_data.coarse_grid_n_cycles;
        amg_data.smoother_type =
          additional_data.coarse_grid_smoother_type.c_str();
        precondition_amg->initialize(
          mg_operators[min_level]->get_system_matrix(coarse_comm), amg_data);
        mg_coarse = std::make_unique<
          MGCoarseGridApplyPreconditioner<VectorType,
                                          TrilinosWrappers::PreconditionAMG>>(
          *precondition_amg);
      }
    else
      {
        mg_coarse = std::make_unique<
          MGCoarseGridApplyPreconditioner<VectorType,
                                          TrilinosWrappers::PreconditionAMG>>();
      }

    // create multigrid algorithm (put level operators, smoothers, transfer
    // operators and smoothers together)
    mg = std::make_unique<Multigrid<VectorType>>(*mg_matrix,
                                                 *mg_coarse,
                                                 *transfer,
                                                 mg_smoother,
                                                 mg_smoother,
                                                 min_level,
                                                 max_level);

    // convert multigrid algorithm to preconditioner
    preconditioner =
      std::make_unique<PreconditionMG<dim, VectorType, MGTransferType>>(
        dof_handler, *mg, *transfer);
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
  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                             VectorType,
                                             SmootherPreconditionerType>;

  const DoFHandler<dim> &dof_handler;

  std::unique_ptr<MPI_Comm, std::function<void(MPI_Comm *)>> sub_comm;

  const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers;
  const MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
                                                              mg_constraints;
  const MGLevelObject<std::shared_ptr<const LevelMatrixType>> mg_operators;

  const unsigned int min_level;
  const unsigned int max_level;

  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
  std::unique_ptr<MGTransferType>                    transfer;

  mutable std::unique_ptr<mg::Matrix<VectorType>> mg_matrix;

  mutable MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>
    mg_smoother;

  mutable std::unique_ptr<TrilinosWrappers::PreconditionAMG> precondition_amg;

  mutable std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  mutable std::unique_ptr<Multigrid<VectorType>> mg;

  mutable std::unique_ptr<PreconditionMG<dim, VectorType, MGTransferType>>
    preconditioner;
};
