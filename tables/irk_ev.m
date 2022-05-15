function irk_ev(s)

  function A = load_matrix(name)
    A = load(strcat(name,s,".txt"));
    A = reshape(A(3:end),A(1),A(2))';
  end
 

  function save_matrix(A, name)
    name;
    A;
    A = [size(A) reshape(A',1, size(A,1) * size(A,2))];
    temp = A';
    file_name = strcat(name,s,".txt");
    dlmwrite(file_name,temp);  
  end 

  function save_vector(A, name)
    name;
    A = diag(A);
    A = [size(A') reshape(A,1, size(A,1) * size(A,2))];
    temp = A';
    file_name = strcat(name,s,".txt");
    dlmwrite(file_name,temp);
  end
  
  A = load_matrix("A");

  A;

  Ainv = A^-1
  save_matrix(Ainv, "A_inv");

  [l,u,p] = lu(sparse(Ainv.'),0);
  Lnew = full(u).';
  Unew = full(l).';

  if true
    [U, T] = schur(Ainv)
    Ainv
    U*T*U'
  end

  if false
    [V,D] = eig(Lnew);
    save_matrix(V, "T")
    save_matrix(V^-1, "T_inv")
    save_vector(D, "D_vec_")
  end
  
  if true
    [V,D] = eig(Ainv);
     Vinv = V^-1;

     [DD,I] = sort(-diag(D*D'));
     

     V = V(:, I);
     Vinv = Vinv(I, : );
     D = D(I,I);

     V
     D

    save_matrix(real(V), "T_re")
    save_matrix(imag(V), "T_im")
    save_matrix(real(Vinv), "T_inv_re")
    save_matrix(imag(Vinv), "T_inv_im")
    save_vector(real(D), "D_vec_re_")
    save_vector(imag(D), "D_vec_im_")
  end
end
