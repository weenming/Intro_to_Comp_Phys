PROGRAM add_numbers
    implicit none
    REAL :: a, b, c, delta, x1_real, x1_imag, x2_real, x2_imag, i
    PRINT * ,'Type the coefficients a, b and c, separated with commas'
    READ *, a, b, c
    delta = b**2 - 4* a * c
    ! delta>0: real solutions, else complex solutions.
    if (b /= 0 .and. abs(4*a*c / b**2) < 1e-3  .and. delta > 0)THEN
        if (b>0) then
            print*, 'x1 = ', 2*c / (-b-delta**0.5)
            print*, 'x2 = ', (-b-delta**0.5) / (2*a)
        else 
            print*, 'x1 = ', 2*c / (-b+delta**0.5)
            print*, 'x2 = ', (-b+delta**0.5) / (2*a)
        end if
    else
        IF (delta >= 0) THEN
            x1_real = (- b + delta ** 0.5) / (2* a)
            x2_real = (- b - delta ** 0.5) / (2* a)
            PRINT*, 'x1 = ', x1_real
            PRINT*, 'x2 = ', x2_real
        ELSE
            x1_real = -b/2/a
            x1_imag =  (-delta)**0.5/2/a
            x2_real = -b/2/a
            x2_imag =  -(-delta)**0.5/2/a
            PRINT*, 'x1 = ', x1_real, '+', x1_imag,'i' ! combine the real part and imaginary part, and then print
            PRINT*, 'x2 = ', x2_real, '+', x2_imag,'i'
        END IF
    end if
    READ *, i ! have the program waiting for next input to keep the command line window open 
END