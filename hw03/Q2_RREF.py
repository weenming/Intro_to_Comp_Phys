from cmath import pi


class matrix:
    def __init__(self, matrix_in=None, size=[0, 0]):
        '''
        Input matrix: a python list, first dim is row, second dim is column
        '''
        self.size = size
        self.matrix = []
        last_row_size = len(matrix_in[0])
        for row in matrix_in:
            self.matrix.append(row)
            assert last_row_size == len(row), 'invalid input: rugged matrix!'
            self.size = [len(matrix_in), last_row_size]
        if self.size[0] == self.size[1]:
            self.is_square = True
        else:
            self.is_square = False
        self.pivot_lower_bound = 1e-10
        self.row_size = self.size[0]
        self.col_size = self.size[1]
        self.no_sol = False

        for i in range(self.row_size):
            for j in range(self.col_size):
                self.matrix[i][j] = float(self.matrix[i][j])

    def print_matrix(self):
        print('matrix:')
        for row in self.matrix:
            print(f'{row}')
        if self.no_sol:
            print('no solution!')

    def get_RREF(self):
        res_mtx = self.matrix.copy()
        # set elements below diagonal
        row_rank = self.row_size
        for row_num in range(self.row_size):
            # when less row than col, determine whether sol exists
            res_mtx = self.swap_pivot(res_mtx, row_num)
            res_mtx[row_num] = [res_mtx[row_num][i] / res_mtx[row_num]
                                [row_num] for i in range(self.col_size)]
            res_mtx = self.set_lower_to_zero(res_mtx, row_num)
            if self.is_all_zero(res_mtx, row_num):
                row_rank = row_num + 1
                if row_rank == self.col_size:
                    self.no_sol = True
                break
        # set elements above diagonal
        for row_num in range(row_rank):
            res_mtx = self.set_upper_to_zero(res_mtx, row_num)
        res = matrix(res_mtx)
        res.no_sol = self.no_sol
        return res

    def set_upper_to_zero(self, matrix, pivot_num):
        res_mtx = matrix.copy()
        pivot_row = matrix[pivot_num]
        for i in range(pivot_num):
            res_mtx[i] = [matrix[i][j] - pivot_row[j] *
                          matrix[i][pivot_num] for j in range(self.col_size)]
        return res_mtx

    def set_lower_to_zero(self, matrix, pivot_row_num):
        res_mtx = matrix.copy()
        pivot_row = matrix[pivot_row_num]
        for i in range(pivot_row_num + 1, self.row_size):
            ratio = matrix[i][pivot_row_num]
            res_mtx[i] = [matrix[i][j] - pivot_row[j] *
                          ratio for j in range(self.col_size)]
        return res_mtx

    def swap_pivot(self, matrix, start_row_num):
        swapped_matrix = matrix.copy()
        for i in range(start_row_num, self.row_size):
            if matrix[i][start_row_num] >= self.pivot_lower_bound:
                tmp = matrix[i]
                swapped_matrix[i] = matrix[start_row_num].copy()
                swapped_matrix[start_row_num] = tmp.copy()
                break
        return swapped_matrix

    def is_all_zero(self, matrix, row_num):
        res = True
        for i in range(row_num + 1, self.row_size):
            for j in range(self.col_size):
                if abs(matrix[i][j]) > self.pivot_lower_bound:
                    res = False
        return res


m = matrix([[2, 8, 4, 2], [2, 5, 1, 5], [4, 10, -1, 1]])
print('test case 1:')
m.print_matrix()
print('RREF:')
m.get_RREF().print_matrix()
print('\n')

m2 = matrix([[1, 0, 1], [2, 5, 10], [3, 900, 4], [5, 91, 900], [6, 6, 6]])
print('test case 2')
m2.print_matrix()
print('RREF:')
if m2.get_RREF() is not None:
    m2.get_RREF().print_matrix()
print('\n')

m3 = matrix([[1, 1, 1], [2, 5, 11], [2, 2, 2], [3, 3, 3], [6, 6, 6]])
print('test case 3')
m3.print_matrix()
print('RREF:')
if m3.get_RREF() is not None:
    m3.get_RREF().print_matrix()
print('\n')
