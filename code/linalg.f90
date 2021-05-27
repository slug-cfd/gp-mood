module linalg
  use constants

contains
  function chol(A,n)result(L)

    integer, intent(in)                   :: n
    real(16),dimension(n,n),intent(inout)    :: A
    real(16),dimension(n,n)               :: L
    integer :: i,j,k
    real(16)::S

    L=0._16

    do i = 1,n
      do j = 1,n
        if (A(i,j) /= A(j,i)) then

          print*, 'the cov matrix is not symmetric', i,j
          print*, 'the difference is about', ((A(i,j)-A(j,i))/A(j,i))*100, '%'
          stop


        end if
      end do
    end do

    do i=1,n
      S=0._16
      do k=1,i-1
        S=S+L(i,k)**2
      end do
      L(i,i)=sqrt(A(i,i)-S)
      do j=1+i,n
        S=0._16
        do k=1,i-1
          S=S+L(i,k)*L(j,k)

        end do
        L(j,i)=(A(j,i)-S)/L(i,i)
      end do

    end do
  end function chol

  function reschol(L,b,n)result(x)

    integer, intent(in)                     :: n
    real(16),dimension(n,n)      ,intent(in):: L
    real(16),dimension(n),        intent(in):: b
    real(16),dimension(n)                :: y, x

    real(16),dimension(n,n)           :: LT
    integer                                 ::  i, j
    real(16)                                :: S

    LT=transpose(L)

    !---------------------Ly=b--------------
    do i=1,n
      S=0._16
      do j=1,i-1
        S=S+L(i,j)*y(j)
      end do
      y(i)=(b(i)-S)/L(i,i)
    end do
    !------------------------LTx=y
    do i=n,1,-1
      S=0._16
      do j=i+1,n
        S=S+LT(i,j)*x(j)
      end do
      x(i)=(y(i)-S)/LT(i,i)
    end do

  end function reschol

  function invchol(A,n)result(AM1)

    integer, intent(in)                      :: n
    real(16),dimension(n,n),intent(inout)       :: A
    real(16),dimension(n,n)                  :: AM1, L
    real(16),dimension(n)                    :: ek

    integer                                  :: k

    L = chol(A,n)

    do k = 1,n
      ek    = 0._16
      ek(k) = 1._16

      AM1(k,:) = reschol(L,ek,n)
    end do
  end function invchol

  subroutine QRGIVENS(A,Q,R,n)

    integer, intent(in)                      :: n
    real(16),dimension(n,n),intent(in)       :: A
    real(16),dimension(n,n),intent(out)      :: Q,R


    real(16),dimension(n,n)                  :: G

    Real(16)                                 :: c,s



    integer :: i,j,k

    Q = 0._16

    do i = 1, n
      Q(i,i) = 1._16
    end do

    R = A;

    do j = 1,n
      do i = n,(j+1),-1

        G = 0._16

        do k = 1, n
          G(k,k) = 1._16
        end do

        call givensrotation( R(i-1,j),R(i,j), c, s );

        G(i-1  , i-1:i) = (/c, -s/)
        G(i    , i-1:i) = (/s, c/)


         ! G(i-1:i, i  ) = (/c, -s/);
         ! G(i-1:i, i-1) = (/s, c/);
        R = matmul(transpose(G),R);
        Q = matmul(Q,G);

      end do
    end do

  end subroutine


  subroutine givensrotation(a,b, c, s)

    Real(16), intent(in)  :: a,b
    Real(16), intent(out) :: c,s

    Real(16) :: r
    if (b == 0._16) then
      c = 1._16;
      s = 0._16;
    else
      if (abs(b) > abs(a)) then
        r = a / b;
        s = 1._16 / sqrt(1._16 + r**2);
        c = s*r;
      else
        r = b / a;
        c = 1._16 / sqrt(1._16 + r**2);
        s = c*r;
      end if
    end if

  end subroutine

  function invGinvens(A,n)result(AM1)

    integer, intent(in)                      :: n
    real(16),dimension(n,n),intent(in)       :: A
    real(16),dimension(n,n)                  :: AM1, Q,R,Dm1
    real(16),dimension(n)                    :: ek

    integer                                  :: k


    Dm1 = 0._pr

    do k = 1, n
      Dm1(k,k) = 1._pr/A(k,k)
    end do
     call QRGIVENS(matmul(Dm1,A),Q,R,n)

     print*,'test'
     print*, 'error QR = ',maxval(abs(matmul(Q,R)-matmul(Dm1,A)))

     print*,'A='
     call printmat(A,n)

     print*,'Q='
     call printmat(Q,n)

     print*,'R='
     call printmat(R,n)


    do k = 1,n
      ek    = 0._16
      ek(k) = 1._16

      AM1(k,:) = matmul(Dm1,resGivens_ek(Q,R,k,n))
    end do
  end function invGinvens

  function resGivens(Q,R,b,n)result(x)

    integer, intent(in)                     :: n
    real(16),dimension(n,n)      ,intent(in):: Q,R
    real(16),dimension(n),        intent(in):: b
    real(16),dimension(n)                :: y, x

    integer                                 ::  i, j
    real(16)                                :: S


    y = matmul( transpose(Q),b)

    do i=n,1,-1
      S=0._16
      do j=i+1,n
        S=S+R(i,j)*x(j)
      end do
      x(i)=(y(i)-S)/R(i,i)
    end do


  end function resGivens



  function resGivens_ek(Q,R,k,n)result(x)

    integer, intent(in)                     :: n,k
    real(16),dimension(n,n)      ,intent(in):: Q,R
    real(16),dimension(n)                :: y, x

    integer                                 ::  i, j
    real(16)                                :: S


    y = Q(k,:)

    do i=n,1,-1
      S=0._16
      do j=i+1,n
        S=S+R(i,j)*x(j)
      end do
      x(i)=(y(i)-S)/R(i,i)
    end do


  end function resGivens_ek




    subroutine printmat(A,n)

      integer, intent(in) :: n
      REAl(16), intent(in),dimension(n,n) :: A

      do i = 1, n
        print*, real(A(i,1:n),4)
      end do

    end subroutine

end module
