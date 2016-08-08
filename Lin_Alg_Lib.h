#ifndef LIN_ALG_LIB_H
#define LIN_ALG_LIB_H

// This set of routines can be used to perform multiple taks related to numerical linear algebra
// R. Sheehan 21 - 3 - 2014

namespace lin_alg{

	// Matrix Vector Operations

	// R. Sheehan 20 - 2 - 2013
	double inf_norm(double *x, int size); // compute the infinity norm of a vector
	double two_norm(double *x, int size); // compute the two norm of a vector

	double *vector_sum(double *vec1, double*vec2, int size1, int size2); // vector addition

	double *vector_diff(double *vec1, double *vec2, int size1, int size2); // compute the difference between two vectors

	double scalar_product(double *vec1, double *vec2,  int size1, int size2); // scalar product of two vectors

	double *mat_vec_product(double **mat, double *vec, int rows, int columns, int vec_size); // matrix-vector product
	double **mat_mat_sum(double **mat1, int rows1, int columns1, double **mat2, int rows2, int columns2); // matrix-matrix sum
	double **mat_mat_diff(double **mat1, int rows1, int columns1, double **mat2, int rows2, int columns2); // matrix-matrix difference
	double **mat_mat_product(double **mat1, int rows1, int columns1, double **mat2, int rows2, int columns2); // matrix-matrix product

	double **identity_matrix(int n); // order n identiry matrix

	void zero_matrix(double **the_mat, int rows, int cols); // assign all the elements in a matrix to be zero


	// Solution of Systems of Equations

	// Direct Methods

	// 1. Gaussian Elimination
	// C-style MACRO hgas been replaced with a C++ template function
	//static int tmp;
	//#define SWAP(a,b) {tmp=b; b=a; a=tmp;}

	void Gauss_Solve(double **A, double *b, double *x, int size, bool pivoting = true, bool printing = false); // Single call to find solution by gaussian elimination
	void row_reduce(double **A, double *b, double **Aug, int *nrow, int rows, bool pivoting); // This will reduce the augmented to upper triangular form
	void back_substitute(double *x,double **Aug,int *nrow,int rows); // This will perform back-substitution on an augmented matrix

	// 2. LU Decomposition by Doolittle's Algorithm
	void LU_Decompose(double **A, double **L, double **U, int size, bool &error); 

	void LU_Solve(double **A, double *b, double *x, int size, bool &error);

	// 3. Solution of Tri-Diagonal System using Thomas Algorithm
	void TDMA(double *a, double *b, double *c, double *d, double *x, int size); 

	// Iterative Methods

	void jacobi_solve(double **a, double *x, double *b, int n, int max_iter , bool &solved, double tol, double &error); 

	void gauss_seidel_solve(double **a, double *x, double *b, int n, int max_iter, bool &solved, double tol, double &error); 

}


#endif