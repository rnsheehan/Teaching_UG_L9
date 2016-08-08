#ifndef ATTACH_H
#include "Attach.h"
#endif

double lin_alg::two_norm(double *x, int size)
{
	// Compute the two norm of the vector x
	
	double s1; 

	s1 = 0.0; 

	for(int i=1; i<=size; i++){
		s1 += DSQR(x[i]); 
	}

	return sqrt(s1); 
}

double lin_alg::inf_norm(double *x, int size)
{
	// The infinity norm is the largest element in the vector by absolute value

	double t1,t2;

	t1=0.0;

	for(int i=1;i<=size;i++){

		if(abs(t2 = x[i]) > abs(t1)){ 
			t1=t2;
		}

	}

	return t1;
}

double *lin_alg::vector_diff(double *vec1, double *vec2, int size1, int size2)
{
	// compute the difference between two vectors

	// Create a vector to store the result of the calculation
	double tmp=0.0; 
	double *diff;

	diff = &tmp; //Initialise the pointer to the memory space

	// The vectors have to be the same size, otherwise you cannot compute their sum
	if(size1 == size2){

		diff = vector(size1); // Allocate the required memory

		// Compute the sum of the vectors vec1 and vec2
		for(int i=1; i<=size1; i++){
			diff[i] = vec1[i] - vec2[i]; 
		}

		return diff; // output the result
	}
	else{

		cout<<"Vector difference is not defined for vectors of unequal length\n"; 

		return diff; // this will return a pointer to zero
	}
}

double *lin_alg::vector_sum(double *vec1, double*vec2, int size1, int size2)
{
	// Compute the sum of the vectors vec1 and vec2
	// R. Sheehan 20 - 2 - 2013

	// Create a vector to store the result of the calculation
	double tmp=0.0; 
	double *sum;

	sum = &tmp; //Initialise the pointer to the memory space

	// The vectors have to be the same size, otherwise you cannot compute their sum
	if(size1 == size2){

		sum = vector(size1); // Allocate the required memory

		// Compute the sum of the vectors vec1 and vec2
		for(int i=1; i<=size1; i++){
			sum[i] = vec1[i] + vec2[i]; 
		}

		return sum; // output the result
	}
	else{

		cout<<"Vector sum not defined for vectors of unequal length\n"; 

		return sum; // this will return a pointer to zero
	}
}

double lin_alg::scalar_product(double *vec1, double*vec2,  int size1, int size2)
{
	// Compute the scalar product of two vectors vec1.vec2
	// R. Sheehan 20 - 2 - 2013

	// Create a variable to store the result of the calculation
	double inner=0.0; 
	
	// The vectors have to be the same size, otherwise you cannot compute their scalar product
	if(size1 == size2){

		for(int i=1; i<=size1; i++){
			inner += (vec1[i]*vec2[i]); 
		}

		return inner; // output the result
	}
	else{

		cout<<"Scalar product not defined for vectors of unequal length\n"; 

		return inner; // this will return a zero

	}
}

double *lin_alg::mat_vec_product(double **mat, double *vec, int rows, int columns, int vec_size)
{
	// Compute the matrix-vector product mat.vec
	// R. Sheehan 20 - 2 - 2013

	// Create a vector to store the result of the calculation
	double tmp=0.0; 
	double *res;

	res = &tmp; // Initialise the pointer to the memory space

	// The number of rows of the matrix must equal the number of elements in the vector, otherwise mat.vec is undefined
	if(rows == vec_size){

		res = vector(vec_size); // Allocate the required memory

		// Compute the sum of the vectors vec1 and vec2
		for(int i=1; i<=rows; i++){

			res[i]=0.0; // Initialise each element to zero 

			for(int j=1; j<=columns; j++){
				res[i] = res[i] + (mat[i][j]*vec[j]); // compute the scalar product of each row with the vector 
			}

		}

		return res; // output the result
	}
	else{

		cout<<"Matrix-Vector product not defined for objects of unequal length\n"; 

		return res; // this will return a pointer to zero

	}
}

double **lin_alg::identity_matrix(int n)
{
	// return the identity matrix of order n

	double **I = matrix(n, n); 

	zero_matrix(I,n,n); 

	for(int i=1; i<=n; i++){
		I[i][i] = 1.0; 
	}

	return I; 
}

void lin_alg::zero_matrix(double **the_mat, int rows, int cols)
{
	// Zero all the elements of the_mat
		
	for(int i=1; i<=rows; i++){
		for(int j=1; j<=cols; j++){
			the_mat[i][j] = 0.0; 
		}
	}
}

double **lin_alg::mat_mat_sum(double **mat1, int rows1, int columns1, double **mat2, int rows2, int columns2)
{
	// matrix-matrix sum	

	// Create a matrix to store the result of the calculation
	double **res;

	// Check that the dimensions are compatible with a matrix.matrix product
	if(rows1 == rows2 && columns1 == columns2){
		
		res = matrix(rows1, columns1); // Allocate the required memory

		// Compute the matrix sum
		for(int i=1; i<=rows1; i++){
			for(int j=1; j<=columns1; j++){
				res[i][j] = mat1[i][j] + mat2[i][j]; 
			}
		}		

		return res; 

	}
	else{

		cout<<"Dimensions not compatible with matrix-matrix sum\n";

		res = matrix(rows1, columns1); // Allocate the required memory

		// Assign all elements in the matrix to zero
		for(int i=1; i<=rows1; i++){
			for(int j=1; j<=columns1; j++){
				res[i][j] = 0.0; 
			}
		}

		return res; 
	}
}

double **lin_alg::mat_mat_diff(double **mat1, int rows1, int columns1, double **mat2, int rows2, int columns2)
{
	// matrix-matrix difference

	// Create a matrix to store the result of the calculation
	double **res;

	// Check that the dimensions are compatible with a matrix.matrix product
	if(rows1 == rows2 && columns1 == columns2){
		
		res = matrix(rows1, columns1); // Allocate the required memory

		// Compute the difference between the matrices
		for(int i=1; i<=rows1; i++){
			for(int j=1; j<=columns1; j++){
				res[i][j] = mat1[i][j] - mat2[i][j]; 
			}
		}		

		return res; 

	}
	else{

		cout<<"Dimensions not compatible with matrix-matrix difference\n";

		res = matrix(rows1, columns1); // Allocate the required memory

		// Assign all elements in the matrix to zero
		for(int i=1; i<=rows1; i++){
			for(int j=1; j<=columns1; j++){
				res[i][j] = 0.0; 
			}
		}

		return res; 
	}
}

double **lin_alg::mat_mat_product(double **mat1, int rows1, int columns1, double **mat2, int rows2, int columns2)
{
	// Compute the matrix-matrix product mat1.mat2
	// R. Sheehan 20 - 2 - 2013

	// Create a matrix to store the result of the calculation
	double **res;

	// Check that the dimensions are compatible with a matrix.matrix product
	if(columns1 == rows2){
		
		res = matrix(rows1, columns2); // Allocate the required memory

		// Assign all elements in the matrix to zero
		for(int i=1; i<=rows1; i++){
			for(int j=1; j<=columns2; j++){
				res[i][j] = 0.0; 
			}
		}

		// Compute the matrix product
		for(int i=1; i<=rows1; i++){

			for(int j=1; j<=columns2; j++){

				res[i][j] = 0.0; // Assign the elements to zero b

				for(int k=1; k<=columns1 ;k++){

					res[i][j]+=mat1[i][k]*mat2[k][j];

				}

			}

		}

		return res; 

	}
	else{

		cout<<"Dimensions not compatible with matrix-matrix product\n";

		res = matrix(rows1, columns2); // Allocate the required memory

		// Assign all elements in the matrix to zero
		for(int i=1; i<=rows1; i++){
			for(int j=1; j<=columns2; j++){
				res[i][j] = 0.0; 
			}
		}

		return res; 
	}
}

// Direct Solvers for the system A.x = b

// 1. Gaussian Elimination

void lin_alg::Gauss_Solve(double **A, double *b, double *x, int size, bool pivoting, bool printing)
{
	// This is a function that performs a row reduction on an augmented matrix followed by 
	// backsubstitution to find the solution of A.x=b
	// R. Sheehan 28 - 2 - 2013

	// the argument pivoting has been defaulted to be set to true
	// so that pivoting is always applied unless otherwise specified

	// Create an array to aid with the simulated row-interchanges
	// new is equivalent to malloc
	int *nrow = new (int [size+1]); 
	
	double **Aug = matrix(size, size+1); // create an array to hold the augmented matrix

	// perform the row reduction
	row_reduce( A, b, Aug, nrow, size, pivoting); 

	if(printing){

		cout<<"Row reduced system\n";
		print_matrix(Aug, size, size+1); 

	}

	// compute x by back-substitution on the row-reduced augmented matrix
	back_substitute(x,Aug,nrow,size); 

	if(printing){

		cout<<"Computed solution\n";
		print_vector(x, size); 

	}

	// Delete memory that is no longer required
	delete[] nrow; 
	delete[] Aug; 
}

void lin_alg::row_reduce(double **A, double *b, double **Aug, int *nrow, int rows, bool pivoting)
{
	// This function takes a square matrix A, rhs vector b and performs
	// a row reduction on the augmented matrix Aug = [A:b]
	// The augmented matrix should be in upper triangular form at the end of the function

	// Use these variables to determine the max pivot element
	int piv_indx;
	double pivot,pivcomp;
	double m; // row multiplier 

	// 1. Use Aug to store the elements of A and b
	for(int i=1; i<=rows; i++){
		for(int j=1; j<=rows; j++){
			Aug[i][j] = A[i][j]; 
		}
	}
	
	// 2. Add the rhs vector to the last column
	for(int i=1; i<=rows; i++){
		Aug[i][rows+1] = b[i]; 
	}

	// 3. Store the index of each of the rows as they are entered in the matrix
	// This will be used to keep track of the row interchanges
	for(int i=1; i<=rows; i++){
		nrow[i] = i; 
	}

	// Main loop	
	for(int i=1; i<=rows-1; i++){	

		// Choose the pivot element
		// Search column i for the largest value by absolute value
		pivot = 0.0; 

		for(int k=i; k <= rows; k++){

			pivcomp = fabs(Aug[ nrow[k] ][i]); 
			
			// This is the search portion
			if(pivcomp > fabs(pivot)){
				pivot = pivcomp; // store the pivot element
				piv_indx = k; 
			}

		}
		
		// simulate a row interchange if required
		if(pivoting && !(nrow[i] == nrow[piv_indx])){
			SWAP(nrow[i], nrow[piv_indx]);
		}

		// Perform the ERO 
		for(int j=i+1; j<=rows; j++){

			m = (Aug[ nrow[j] ][i] / Aug[ nrow[i] ][i]); // Compute the row multiplier, the computer accesses the pivot numer via nrow

			for(int c = 1; c<= rows+1; c++){
				Aug[ nrow[j] ][c] = Aug[ nrow[j] ][c] -( m * Aug[ nrow[i] ][c] ); // perform the ERO
			}
		}

	}
}

void lin_alg::back_substitute(double *x, double **Aug, int *nrow, int rows)
{
	// Solve the system of equations A.x = b, when the system has been reduced to upper triangular form 
	// This implementation of the back substitution is meant to be used as part of the Gaussian elimination algorithm Gauss_Solve

	if(Aug[nrow[rows]][rows]==0.0){
		cout<<"No solution possible\n";
	}
	else{
		double sum=0.0; 
		
		// Compute x[n]
		x[rows]= Aug[ nrow[rows] ][rows+1] / ( Aug[ nrow[rows] ][rows] ); 
		
		// Compute the remaining x[i] by back substitution
		for(int i=rows-1; i>=1; i--){

			for(int j=i+1; j<=rows; j++){

				sum += Aug[ nrow[i] ][j]*x[j];

			}

			x[i] = ( Aug[ nrow[i] ][rows+1] - sum )/( Aug[ nrow[i] ][i] );

			sum=0.0; 
		}
	}
}

// 2. LU Decomposition by Doolittle's Algorithm
void lin_alg::LU_Decompose(double **A, double **L, double **U, int size, bool &error)
{
	// Compute the factors that form the LU decomposition of A
	// Decomposition is computed using Doolittle's algorithm
	// This implementation does not allow for row interchanges
	// For LU Decomposition with partial pivoting see
	// "Numerical Recipes in C" by Press et al
	// R. Sheehan 28 - 2 - 2013

	// Assign the diagonals of L to be unity
	for(int i=1; i<=size; i++){
		L[i][i] = 1.0; 
	}

	// Start the factorisation
	U[1][1] = A[1][1]; 

	if(fabs(U[1][1]) < 1.0e-15){

		cout<<"No solution exists. Algorithm will not proceed\n"; 
		error = true; 

	}
	else{
		// Compute the first column of L and the first row of U
		for(int j=2; j<=size; j++){
			U[1][j] = A[1][j]; 
			L[j][1] = ( A[j][1] / U[1][1] ); 
		}

		// Complete the calculations of L and U
		double usum, lsum; 
		for(int i=2; i<=size-1; i++){

			// Compute the diagonal elements of U
			usum=0.0; 
			for(int k=1; k<=i-1; k++){
				usum = usum + L[i][k]*U[k][i]; 
			}

			U[i][i] = A[i][i]-usum; 

			// Check for non-zero pivots
			if(fabs(U[i][i]) < 1.0e-15){
				// A zero has occurred on the diagonal of U
				// This means that a solution cannot be found
				// by the LU decomposition
				error = true; 
			}

			// Compute the off-diagonal elements of L and U
			for(int j=i+1; j<=size; j++){

				// Compute the doolittle sums
				usum = 0.0;  lsum = 0.0; 
				for(int k=1; k<=i-1; k++){
					usum = usum + L[i][k]*U[k][j]; 
					lsum = lsum + L[j][k]*U[k][i]; 
				}

				// Fill U
				U[i][j] = A[i][j]-usum; 

				// Fill L
				L[j][i] = (A[j][i]-lsum)/U[i][i]; 
			}
		}

		// Fill in the last element of U
		usum=0.0; 
		for(int k=1; k<=size-1; k++){
			usum = usum + L[size][k]*U[k][size]; 
		}

		U[size][size] = A[size][size] - usum; 

	}

}

void lin_alg::LU_Solve(double **A, double *b, double *x, int size, bool &error)
{
	// Solve the system of linear equations A.x=b using the LU Decomposition of A
	// R. Sheehan 28 - 2 - 2013

	double **L = matrix(size,size);
	double **U = matrix(size,size);

	// Compute the lower and upper factors of the LU Decomposition
	LU_Decompose(A,L,U,size,error);

	if(!error){

		double sum;

		// Solve the set of equations L.y = b by forward substitution
		double *y = vector(size); 
		
		y[1] = b[1]; 
		for(int i=2; i<=size; i++){
			sum = 0.0; 
			for(int j=1; j<=i-1; j++){
				sum = sum + L[i][j]*y[j]; 
			}
			y[i] = b[i] - sum; 
		}

		//print_vector(y,size); 

		// Solve the set of equations U.x = y by back-substitution
		if( fabs(U[size][size])<1.0e-15 ){
			// this condition could be prevented by pivoting
			// see "Numerical Recipes in C" by Press et al for implmentation of LU Decomposition with partial pivoting
			cout<<"Solution cannot be found\n";

		}
		else{
			x[size] = y[size] / U[size][size]; 

			for(int i=size-1; i>=1; i--){
				sum = 0.0;

				for(int j=i+1; j<=size; j++){
					sum = sum+U[i][j]*x[j]; 
				}

				x[i] = (y[i]-sum) / U[i][i]; 

			}
		}

		//print_vector(x,size); 

		delete[] y; 

	}
	else{
		cout<<"Solution cannot be computed because LU decomposition was not found\n"; 
	}

	delete[] L; 
	delete[] U; 
}

// 3. Thomas / Tri-Diagonal Matrix Algorithm

void lin_alg::TDMA(double *a, double *b, double *c, double *d, double *x, int size)
{
	// Compute the solution of a tri-diagonal system of equations
	// a = sub-diagonal, b = main-diagonal, c = super-diagonal, d = rhs vector
	// solution is stored in x
	// R. Sheehan 21 - 3 - 2014
	
	double tmp; 
	double *cpr = vector(size); 
	double *dpr = vector(size); 

	//cpr[size] = 0; // not necessary

	// 1. Make the transformations of c to c' and d to d'
	for(int i=1; i<=size; i++){
		
		if(i==1){
			cpr[1] = c[1] / b[1]; 
			
			dpr[1] = d[1] / b[1]; 
		}
		/*else if(i == size){

			tmp = b[i] - ( cpr[i-1] * a[i] );
			
			dpr[i] = ( d[i] - ( dpr[i-1] * a[i] ) ) / tmp; 

		}*/
		else{
			tmp = b[i] - ( cpr[i-1] * a[i] );

			cpr[i] = c[i] / tmp; // this sets cpr[size] = 0 when i = size
			
			dpr[i] = ( d[i] - ( dpr[i-1] * a[i] ) ) / tmp; 
		}
	}

	/*cout<<"The vector c' is \n"; 
	print_vector(cpr, size); 

	cout<<"The vector d' is \n"; 
	print_vector(dpr, size); */

	// 2. Compute the solution using back substitution
	int stp = size; 
	int lasstp; 
	while(stp > 0){
		
		if(stp == size){
			x[stp] = dpr[stp]; 
		}
		else{
			x[stp] = dpr[stp] - ( cpr[stp] * x[lasstp] );
		}

		lasstp = stp; 

		stp--; 
	}

	delete[] cpr; 
	delete[] dpr; 
}

// Iterative Methods

// Gauss-Seidel Iteration

void lin_alg::jacobi_solve(double **a, double *x, double *b, int n, int max_iter, bool &solved, double tol, double &error)
{
	// Solve the system Ax=b by Jacobi's iteration method
	// This has the slowest convergence rate of the iterative techniques
	// In practice you never use this algorithm
	// Always use GS or SOR for iterative solution
	// R. Sheehan 3 - 8 - 2011
	
	bool cgt = false;
	
	double *oldx = vector(n); 
	double *diffx = vector(n); 
		
	int n_iter = 1;
	double bi, mi, err;
		
	while(n_iter < max_iter){

		// store the initial approximation to the solution
		for(int i=1; i<=n; i++){
			oldx[i] = x[i]; 
		}
		
		for(int i=1;i<=n;i++){

			bi=b[i]; mi=a[i][i];

			for(int j=1;j<=n;j++){

				if(j!=i){
					bi-=a[i][j]*oldx[j];
				}

			}

			x[i]=bi/mi;
		}
			
		diffx = vector_diff(x, oldx, n, n); 
		
		err = inf_norm(diffx, n);

		if( n_iter%4 == 0){
			cout<<"iteration "<<n_iter<<" , error = "<<err<<endl;
		}
			
		if(abs(err)<tol){

			cout<<"\nJacobi Iteration Complete\nSolution converged in "<<n_iter<<" iterations\n";
			cout<<"Error = "<<abs(err)<<endl<<endl;
			
			error = abs(err);

			cgt = solved = true;
			
			break;
		}
			
		n_iter++;
	}
		
	if(!cgt){
		cout<<"\nError: Jacobi Iteration\n";
		cout<<"Error: Solution did not converge in "<<max_iter<<" iterations\n\n";
	}

	delete[] oldx; 
	delete[] diffx; 
}

void lin_alg::gauss_seidel_solve(double **a, double *x, double *b, int n, int max_iter, bool &solved, double tol, double &error)
{
	// Solve the system Ax=b by Gauss-Seidel iteration method
	// R. Sheehan 3 - 8 - 2011
	
	bool cgt=false;

	double *oldx = vector(n); 
	double *diffx = vector(n); 
		
	int n_iter = 1;
	double bi, mi, err;
		
	while(n_iter<max_iter){

		// store the initial approximation to the solution
		for(int i=1; i<=n; i++){
			oldx[i] = x[i]; 
		}

		// iteratively compute the solution vector x
		for(int i=1; i<=n; i++){

			bi=b[i]; 
			mi=a[i][i];

			for(int j=1; j<i; j++){

				bi -= a[i][j]*x[j];

			}

			for(int j=i+1; j<=n; j++){

				bi -= a[i][j]*oldx[j];

			}

			x[i] = bi/mi;
		}
			
		// test for convergence
		// compute x - x_{old} difference between the two is the error
		diffx = vector_diff(x, oldx, n, n); 

		// error is measured as the length of the difference vector ||x - x_{old}||_{\infty}
		err = inf_norm(diffx, n); 

		if( n_iter%4 == 0){
			cout<<"iteration "<<n_iter<<" , error = "<<err<<endl;
		}
			
		if(abs(err)<tol){ // solution has converged, stop iterating

			cout<<"\nGauss-Seidel Iteration Complete\nSolution converged in "<<n_iter<<" iterations\n";

			cout<<"Error = "<<abs(err)<<endl<<endl;

			error=abs(err);

			cgt=solved=true;

			break;
		}
			
		n_iter++; // solution has not converged, keep iterating
	}
		
	if(!cgt){ // solution did not converge

		cout<<"\nError: Gauss-Seidel Iteration\n";

		cout<<"Error: Solution did not converge in "<<max_iter<<" iterations\n";

		cout<<"Error = "<<abs(err)<<endl;

		error=abs(err);

		solved=false;
	}

	delete[] oldx; 
	delete[] diffx; 
}