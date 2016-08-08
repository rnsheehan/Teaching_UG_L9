#ifndef ATTACH_H
#include "Attach.h"
#endif


double *vec_mat_funcs::vector(int size) 
{
	// this function will return a vector that can hold size elements of type double
	// the function dynamically creates an array
	// array indexing starts at i = 1 and ends at i = N
	// R. Sheehan 31 - 1 - 2013

	double *vec_ptr = new (double [size+1]); 

	return vec_ptr; 
}


double **vec_mat_funcs::matrix(int rows, int columns)
{	
	// this function will return a matrix that can hold row*cols elements of type double
	// use dynamic memory allocation to create a matrix
	// row indexing starts at i = 1 and ends at i = rows
	// column indexing starts at j = 1 and ends at j = columns
	// R. Sheehan 31 - 1 - 2013

	double **mat_ptr = new (double *[rows+1] ); 

	for(int i=1; i<=rows; i++){
		mat_ptr[i] = new (double [columns+1]); 
	}

	return mat_ptr; 
}

void vec_mat_funcs::print_vector(double *vec, int size)
{
	// print a vector to the screen
	// R. Sheehan 31 - 1 - 2013

	//cout<<"\nYour vector is\n"; 
	for(int i=1; i<=size; i++){
		cout<<vec[i]<<endl;
	}
	cout<<endl;

}


void vec_mat_funcs::print_matrix(double **mat, int rows, int columns)
{
	// print a matrix to the screen
	// R. Sheehan 31 - 1 - 2013

	//cout<<"\nYour matrix is\n"; 
	for(int i=1; i<=rows; i++){
		for(int j=1; j<=columns; j++)
			cout<<mat[i][j]<<" ";
		cout<<endl;
	}
	cout<<endl;
}