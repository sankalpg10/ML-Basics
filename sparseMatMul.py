def sparseMul(mat1,mat2):
    def compress_matrix(matrix):
        rows, cols = len(matrix), len(matrix[0])

        compressed_matrix = [[] for r in range(rows)]  # cause we wanna save values for each row
        for row in range(rows):
            for col in range(cols):

                if matrix[row][col]:
                    compressed_matrix[row].append([matrix[row][col], col])  # [element ,col]

        return compressed_matrix


    m = len(mat1)
    k = len(mat1[0])
    n = len(mat2[0])
    
    mat1_c = compress_matrix(mat1)
    mat2_c = compress_matrix(mat2)

    ans = [[0] * n for _ in range(m)]  # becz output = mxn dimensions

    for mat1_row in range(m):
        # to go over all non zero elements for current row for mat1

        for element1, mat1_col in mat1_c[mat1_row]:

            # for the ans[mat1_row][mat2_col] element, we will have to multiple element1 with all non zero values for (mat1_row)th row in mat2

            for element2, mat2_col in mat2_c[mat1_col]:
                ans[mat1_row][mat2_col] += element1 * element2
                # here we arent doing row mat1 * col mat2 for eacg element
                # we are initiatially just adding the the product of element1 with all cols in mat2 for that col in ans one at a time and then we again come to this index to add prodict for next row and same col of mat2 with element from mat1

    return ans