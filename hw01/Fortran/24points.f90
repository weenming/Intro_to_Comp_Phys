PROGRAM points24
    IMPLICIT none
    real :: x, goal=24.0, goal1, goal2,result,  x1, x2, x3, x4, cards(4), cards_in_initial_order(4), &
    picked_card1, picked_card2, picked_card3, left_card3(3), left_card2(2), left_card
    integer :: pick4, pick3, pick2, operations1, operations2, operations3, i, break_flag = 0
    character(32) :: pick4_str, pick3_str, pick2_str, left_card_str, x_str
    character(50) :: print_operations=''

    integer :: pick22, operations22_1, operations22_2, operations22_3
    real :: cards22_1(2), cards22_2(2), result22_1, result22_2, result22_3
    print*, 'type the points of 4 cards, separated by commas'
    read*, x1, x2, x3, x4
    
    cards_in_initial_order(1) = x1
    cards_in_initial_order(2) = x2
    cards_in_initial_order(3) = x3
    cards_in_initial_order(4) = x4
    ! first case: f1(f2(f3(x1, x2), x3), x4) == 24
    do pick4= 1, 4 
        ! pick one card out of 4
        ! NOTICE: picked_card, left_card and goal needs to be changed in the embedded loop so different variables are used
        ! pick(5-x), picked_card(4-x), left_card(4-x) and goal(x), x = 1, 2, 3 correspond to the x-th slections respectively
        picked_card3 = cards_in_initial_order(pick4)
        ! delete the pick4-th element in array cards_in_initial_order        
        do i = 1, pick4 - 1
            left_card3(i) = cards_in_initial_order(i)
        end do
        do i = pick4, 3
            left_card3(i) = cards_in_initial_order(i+1)
        end do
        
        do operations1 = 1 , 6 
        ! list all possible results that operation between the picked card and 24 can produce
        ! update goal to goal1, which is one of the results mentioned above
            call list_all_operation_results(picked_card3, goal, operations1, goal1) ! goal1 is the updated goal.

            do pick3 = 1, 3
                ! pick on card out of the remaining 3
                picked_card2 = left_card3(pick3)

                ! delete the pick3-th element in array left_card3        
                do i = 1, pick3 - 1
                    left_card2(i) = left_card3(i)
                end do
                do i = pick3, 2
                    left_card2(i) = left_card3(i+1)
                end do

                do operations2 = 1, 6
                ! list all possible results that operation between the picked card and goal1 can produce
                ! update goal to goal1, which is one of the results mentioned above
                    call list_all_operation_results(picked_card2, goal1, operations2, goal2)
                    do pick2 = 1, 2
                        ! pick one card out of the remaining 2
                        picked_card1 = left_card2(pick2)

                        ! delete the pick2-th element in array left_card2
                        left_card = left_card2(mod(pick2,2) + 1)
                        
                        do operations3 = 1, 6
                        ! list all possible results that operation between the picked card and goal2 can produce
                        ! update goal to goal2, which is one of the results mentioned above
                            call list_all_operation_results(picked_card1, goal2, operations3, result)
                            if (result - left_card < 1e-5 .and. result - left_card >-1e-5) then
                                break_flag = 1
                                exit
                            end if
                        end do
                        if (break_flag == 1) exit
                    end do
                    if (break_flag == 1) exit
                end do                
                if (break_flag == 1) exit
            end do
            if (break_flag == 1) exit
        end do
        if (break_flag == 1) exit
    end do

    ! second case: g(f1(x1, x2), f2(x3, x4)) == 24
    if (break_flag == 0) then
        ! iterate possible combinations (x1x2, x3x4), (x1x3, x2x4), (x1x4, x2x3)
        do pick22 = 2, 4
            cards22_1(1) = cards_in_initial_order(1)
            cards22_1(2) = cards_in_initial_order(pick22)
            select case(pick22)
                case(2)
                    cards22_2(1) = cards_in_initial_order(3)
                    cards22_2(2) = cards_in_initial_order(4)
                case(3)
                    cards22_2(1) = cards_in_initial_order(2)
                    cards22_2(2) = cards_in_initial_order(4)
                case(4)
                    cards22_2(1) = cards_in_initial_order(2)
                    cards22_2(2) = cards_in_initial_order(3)
            end select
            ! iterate possible operations between the first pair
            do operations22_1 = 1, 6
                call list_all_operation_results(cards22_1(1), cards22_1(2), operations22_1, result22_1)
                ! iterate possible operations of the second pair
                do operations22_2 = 1, 6
                    call list_all_operation_results(cards22_2(1), cards22_2(2), operations22_2, result22_2)
                    ! iterate possible operations between the results produced by the first pair and the second pair 
                    do operations22_3 = 1, 6
                        call list_all_operation_results(result22_1, result22_2, operations22_3, result22_3)
                        if (result22_3 - 24 < 1e-5 .and. result22_3 - 24 > -1e-5) then 
                            break_flag = 2
                            exit
                        end if
                    end do
                    if (break_flag == 2) exit
                end do
                if (break_flag == 2) exit
            end do
            if (break_flag == 2) exit
        end do
    end if
    
    ! print results
    if (break_flag == 0) then
        print*, 'no solution.'
    elseif (break_flag == 1) then
        print*, 'solution found'
        write(pick4_str, *) picked_card3
        write(pick3_str, *) picked_card2
        write(pick2_str, *) picked_card1
        write(left_card_str, *) left_card

        call print_inverse_operation(left_card_str, pick2_str, operations3, print_operations)
        call calc_inverse_operation(left_card, picked_card1, operations3, x)
        print*, print_operations, '=', x
        write(x_str, *) x
        call print_inverse_operation(x_str, pick3_str, operations2, print_operations)
        call calc_inverse_operation(x, picked_card2, operations2, x)
        print*, print_operations, '=', x
        write(x_str, *) x
        call print_inverse_operation(x_str, pick4_str, operations1, print_operations)
        call calc_inverse_operation(x, picked_card3, operations1, x)
        print*, print_operations, '=', 24
    else
        print*, 'solution found'
        call print_operation(cards22_1(1), cards22_1(2), operations22_1)
        call print_operation(cards22_2(1), cards22_2(2), operations22_2)
        call print_operation(result22_1, result22_2, operations22_3)
    end if

print*, 'type any word to exit'
read*, i
END PROGRAM points24

SUBROUTINE list_all_operation_results(picked_card, in_goal, operation, out_goal)
    IMPLICIT none
    real :: picked_card, in_goal, out_goal
    integer :: operation
    select case (operation)
        case(1)
            out_goal = in_goal + picked_card
        case(2)
            out_goal = in_goal * picked_card
        case(3)
            out_goal = in_goal - picked_card
        case(4)
            out_goal = in_goal / picked_card
        case(5)
            out_goal = picked_card / in_goal
        case(6)
            out_goal = picked_card - in_goal
    end select
END SUBROUTINE list_all_operation_results

SUBROUTINE print_inverse_operation(a, b, operation, output)
    IMPLICIT none
    character(32) :: a, b
    integer :: operation
    character(50) :: output
    select case (operation)
        case(1)
            output = a // '-' // b
        case(2)
            output = a // '/' // b
        case(3)
            output = a // '+' // b
        case(4)
            output = a // '*' // b
        case(5)
            output = b // '/' // a
        case(6)
            output = b // '-' // a
    end select

END SUBROUTINE

SUBROUTINE calc_inverse_operation(a, b, operation, output)
    IMPLICIT none
    real :: a, b
    integer :: operation
    real :: output
    select case (operation)
        case(1)
            output = a - b
        case(2)
            output = a / b
        case(3)
            output = a + b
        case(4)
            output = a * b
        case(5)
            output = b / a
        case(6)
            output = b - a
    end select
END SUBROUTINE

SUBROUTINE print_operation(a, b, operation)
    IMPLICIT none
    real :: a, b
    integer :: operation
    select case (operation)
        case(1)
            print*, b, '+', a, '=', b+a
        case(2)
            print*, b, '*', a, '=', b*a
        case(3)
            print*, b, '-', a, '=', b-a
        case(4)
            print*, b, '/', a, '=', b/a
        case(5)
            print*, a, '/', b, '=', a/b
        case(6)
            print*, a, '-', b, '=', a-b
    end select
END SUBROUTINE