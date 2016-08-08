#ifndef ATTACH_H
#include "Attach.h"
#endif

// forward function declarations
void matrix_algrbra_testing();
void gauss_elimination_testing(); 
void gauss_elimination_timing(); 
void LU_decomposition_testing();
void LU_decomposition_timing(); 
void TDMA_testing();
void iterative_solver_testing(); 

int main()
{
	//matrix_algrbra_testing(); 

	//gauss_elimination_testing();

	//LU_decomposition_testing(); 

	//LU_decomposition_timing();

	//gauss_elimination_timing();

	//TDMA_testing(); 

	//iterative_solver_testing(); 

	cout<<"Press enter to close\n";
	cin.get(); 
	return 0; 
}

void matrix_algrbra_testing()
{
	// It is certainly possible to define and declare matrices by hand inside a C++ program
	// However, in practice this is rarely done because for realistic problems the matrices and 
	// vectors are much too large to be defined by hand
	// For the purposes of illustration I have included this function so you can examine the calls that can be 
	// made to examine certain properties of vectors and matrices

	// Declare a vector and examine its properties

	cout<<"\n\t\tVector Operations\n\n"; 
	int size=6; 

	double *v1 = vector(size); 
	double *v2 = vector(size); 

	v1[1] = 0.5; v1[2] = -7.1; v1[3] = -1.1; v1[4] = -9.7; v1[5] = 0.1; v1[6] = 2.3; 

	v2[1] = -6; v2[2] = 2.3; v2[3] = 0.4; v2[4] = 7.2; v2[5] = 1.3; v2[6] = 5.6; 

	cout<<"The vector v1 is \n"; 
	print_vector(v1,size); 

	cout<<"The length of v1 relative to the 2-norm is "; 
	cout<<two_norm(v1,size)<<endl;
	cout<<"\nThe length of v1 relative to the infinity-norm is "; 
	cout<<abs(inf_norm(v1,size))<<endl<<endl;

	cout<<"The vector v2 is \n"; 
	print_vector(v2,size); 

	cout<<"The length of v2 relative to the 2-norm is "; 
	cout<<two_norm(v2,size)<<endl;
	cout<<"\nThe length of v2 relative to the infinity-norm is "; 
	cout<<abs(inf_norm(v2,size))<<endl<<endl;

	cout<<"The scalar product of v1 and v2 is "<<scalar_product(v1, v2, size, size)<<endl<<endl;
	cout<<"The scalar product is commutative "<<scalar_product(v2, v1, size, size)<<endl<<endl;
	
	double *sum = vector_sum(v1, v2, size, size); 

	cout<<"The sum of vectors v1 and v2 is v1 + v2 \n";
	print_vector(sum, size); 

	double *diff = vector_diff(v1, v2, size, size); 

	cout<<"The difference between vectors v1 and v2 is v1 - v2 \n";
	print_vector(diff, size); 

	cout<<"\t\tMatrix Operations\n"; 

	double **A = matrix(size, size); 

	A[1][1] = 12.0; A[1][2] = -15; A[1][3] = -20; A[1][4] = 17; A[1][5] = 20; A[1][6] = -17;
	A[2][1] = 2.0; A[2][2] = -10; A[2][3] = -17; A[2][4] = 18; A[2][5] = 15; A[2][6] = 14;
	A[3][1] = 16.0; A[3][2] = 16; A[3][3] = 12; A[3][4] = -3; A[3][5] = -4; A[3][6] = -13;
	A[4][1] = -8; A[4][2] = 9; A[4][3] = 9; A[4][4] = -9; A[4][5] = 0.0; A[4][6] = 4;
	A[5][1] = -12; A[5][2] = -12; A[5][3] = 16; A[5][4] = -3; A[5][5] = -2; A[5][6] = 13;
	A[6][1] = 13; A[6][2] = 18; A[6][3] = 18; A[6][4] = 6; A[6][5] = 3; A[6][6] = -9;

	cout<<"\nThe matrix A is \n";
	print_matrix(A, size, size); 

	double **B = matrix(size, size); 

	B[1][1] = 15.0; B[1][2] = 17; B[1][3] = -17; B[1][4] = 16; B[1][5] = -16; B[1][6] = 20;
	B[2][1] = -20.0; B[2][2] = 20; B[2][3] = 18; B[2][4] = -18; B[2][5] = -13; B[2][6] = 12;
	B[3][1] = 10.0; B[3][2] = -16; B[3][3] = -3; B[3][4] = 6; B[3][5] = 10; B[3][6] = -9;
	B[4][1] = -5; B[4][2] = -8; B[4][3] = -10; B[4][4] = -8; B[4][5] = -1; B[4][6] = 3;
	B[5][1] = 8; B[5][2] = 0; B[5][3] = 3; B[5][4] = 15; B[5][5] = -12; B[5][6] = -12;
	B[6][1] = -18; B[6][2] = -15; B[6][3] = -12; B[6][4] = -9; B[6][5] = -11; B[6][6] = 7;

	cout<<"The matrix B is \n";
	print_matrix(B, size, size); 

	double **AB = matrix(size, size); 
	double **BA = matrix(size, size); 

	AB = mat_mat_product(A, size, size, B, size, size); 

	BA = mat_mat_product(B, size, size, A, size, size); 

	cout<<"The matrix product A*B is \n";
	print_matrix(AB, size, size); 

	cout<<"Matrix multiplication is not commutative\n"; 
	cout<<"The matrix product B*A is \n";
	print_matrix(BA, size, size); 

	double *Av = mat_vec_product(A, v1, size, size, size); 

	cout<<"The matrix-vector product is A.v is \n";
	print_vector(Av, size); 

	int rows = 3; 
	int cols = 6; 

	double **C = matrix(rows, cols); 

	C[1][1] = 3; C[1][2] = 0; C[1][3] = 5; C[1][4] = 2; C[1][5] = 5; C[1][6] = 4;
	C[2][1] = 6; C[2][2] = 9; C[2][3] = 5; C[2][4] = 10; C[2][5] = 10; C[2][6] = 2;
	C[3][1] = 5; C[3][2] = 10; C[3][3] = 5; C[3][4] = 7; C[3][5] = 3; C[3][6] = 3;

	cout<<"The matrix C is \n";
	print_matrix(C, rows, cols); 

	cout<<"It is possible to compute the product C*A, but not A*C\n"; 

	double **CA = matrix(cols, size); 

	CA = mat_mat_product(C, rows, cols, A, size, size); 

	cout<<"The matrix product C*A is \n";
	print_matrix(CA, rows, size); 

	double **AC = matrix(cols, size); 

	AC = mat_mat_product(A, size, size, C, rows, cols); 

	delete[] v1;
	delete[] v2; 
	delete[] sum; 
	delete[] diff; 
	delete[] Av; 
	delete[] A;
	delete[] B; 
	delete[] AB;
	delete[] BA; 
	delete[] CA; 
	delete[] AC; 
}

void gauss_elimination_testing()
{
	// In this function we will read in data for a system of equations
	// We will compute the solution of that system using Gaussian Elimination
	
	// Read in the data for the matrix 
	string mat_file = "matrix_m1.txt"; 

	cout<<"\n\tSolution of a system of linear equations by Gaussian elimination\n\n"; 

	int rows = 0, cols = 0; 

	double **A = read_matrix_from_file(mat_file, rows, cols);

	cout<<"The matrix A is \n";
	print_matrix(A, rows, cols); 

	// Read in the data for the vector
	string vec_file = "vector_v1.txt"; 

	double *b = read_vector_from_file(vec_file, rows);

	cout<<"The vector b is \n";
	print_vector(b, rows); 

	double *x_no_pivot = vector(rows); // vector to hold the solution
	double *x_pivot = vector(rows); // vector to hold the solution

	cout<<"\tGaussian Elimination without pivoting\n\n"; 

	Gauss_Solve(A, b, x_no_pivot, rows, false, true); 

	cout<<"\n\tGaussian Elimination with pivoting\n\n"; 

	Gauss_Solve(A, b, x_pivot, rows, true, true); 

	delete[] A; 
	delete[] b; 
	delete[] x_no_pivot; 
	delete[] x_pivot; 
}

void gauss_elimination_timing()
{
	// In this function we will read in data for a ``large'' system of equations
	// We will compute the solution of that system using Gaussian Elimination
	
	// Read in the data for the matrix 
	string mat_file = "matrix_m2.txt"; 

	cout<<"\n\tTiming of Gaussian elimination\n\n"; 

	int rows = 0, cols = 0; 

	double **A = read_matrix_from_file(mat_file, rows, cols);

	cout<<"The size of the system being solved is "<<rows<<" * "<<cols<<"\n\n";

	// Read in the data for the vector
	string vec_file = "vector_v2.txt"; 

	double *b = read_vector_from_file(vec_file, rows);

	double *x = vector(rows); // vector to hold the solution

	// declare your timing variables
	// make sure you have ctime included

	// Since the calculation is done very quickly you have to run the calculation multiple times
	// Determine the time taken for multiple calculations
	// Then the average time per calculation is the total time divided by the number of calculations
	// Total time for a single calculation will depend on implementation of the GE algorithm
	// but in the main it depends on the processor speed
	// Fast processor => fast calculation time
	// processor speed is measured in flops, floating point operations per second
	// more flops = faster, faster  = better

	// CLOCKS_PER_SEC allows you to convert from CPU time to seconds
	// CLOCKS_PER_SEC represents the number of clock-ticks per second your computer performs

	cout<<"Clock-ticks per second on this computer is "<<CLOCKS_PER_SEC<<endl;
	
	clock_t start, finish; 

	int GE_total = 50; 

	start = clock(); // start the clock

	// Perform multiple loops over the Gaussian elimination calculation

	for(int i=1; i<=GE_total; i++){
	
		Gauss_Solve(A, b, x, rows); 

	}

	finish = clock(); // stop the clock

	double total_time = (finish - start) / (static_cast<double>(CLOCKS_PER_SEC)); // compute total time
	double GE_time = total_time / (static_cast<double>(GE_total)); 

	cout<<"Time taken to perform "<<GE_total<<" Gaussian eliminations is "<<total_time<<" seconds\n"; 
	cout<<"Time taken to perform single Gaussian elimination on system is "<<GE_time<<" seconds\n"; 

	delete[] A; 
	delete[] b; 
	delete[] x; 
}

void LU_decomposition_testing()
{
	// In this function we will read in data for a system of equations
	// We will compute the solution of that system using LU Decomposition
	
	cout<<"\n\t\tLU Decomposition of a Matrix\n\n"; 

	// Read in the data for the matrix 
	string mat_file = "matrix_m1.txt"; 

	int rows = 0, cols = 0; 

	bool error = false; 

	double **A = read_matrix_from_file(mat_file, rows, cols);
	double **L = matrix(rows, rows); 
	double **U = matrix(rows, rows); 

	cout<<"The matrix A is \n";
	print_matrix(A, rows, cols); 

	zero_matrix(L, rows, rows); 

	zero_matrix(U, rows, rows); 

	LU_Decompose(A, L, U, rows, error); 

	if(error == false){
		cout<<"The unit lower triangular factor in A is\n";
		print_matrix(L, rows, rows); 

		cout<<"The upper triangular factor in A is\n";
		print_matrix(U, rows, rows); 

		double **LU = mat_mat_product(L, rows, rows, U, rows, rows); 

		cout<<"The matrix product L*U is\n";
		print_matrix(LU, rows, rows); 

		double **diff = mat_mat_diff(A, rows, rows, LU, rows, rows); 

		cout<<"The difference between A and L*U is\n"; 
		print_matrix(diff, rows, rows); 

		delete[] diff; 
		delete[] LU; 
	}

	// Read in the data for the vector
	cout<<"\tSolution of a system of linear equations A.x = b by LU Decomposition\n\n"; 

	string vec_file = "vector_v1.txt"; 

	double *b = read_vector_from_file(vec_file, rows);

	cout<<"The vector b is \n";
	print_vector(b, rows); 

	// Solve the system of equations using LU Decomposition
	double *x = vector(rows); 

	LU_Solve(A, b, x, rows, error); 

	if(error == false){
		cout<<"The solution of the system of equations x is\n";
		print_vector(x, rows); 

		double *Ax = mat_vec_product(A, x, rows, rows, rows); 

		cout<<"The matrix vector product A.x is \n";
		print_vector(Ax, rows); 

		double *diff = vector_diff(Ax, b, rows, rows); 

		cout<<"The vector difference Ax - b is\n"; 
		print_vector(diff, rows); 

		delete[] Ax;
		delete[] diff; 
	}

	delete[] A;
	delete[] L;
	delete[] U; 
	delete[] b; 
	delete[] x;
}

void LU_decomposition_timing()
{
	// In this function we will read in data for a ``large'' system of equations
	// We will compute the solution of that system using LU Decomposition
	
	// Read in the data for the matrix 
	string mat_file = "matrix_m2.txt"; 

	cout<<"\n\tTiming of LU Decomposition\n\n"; 

	int rows = 0, cols = 0; 

	double **A = read_matrix_from_file(mat_file, rows, cols);

	cout<<"The size of the system being solved is "<<rows<<" * "<<cols<<"\n\n";

	// Read in the data for the vector
	string vec_file = "vector_v2.txt"; 

	double *b = read_vector_from_file(vec_file, rows);

	double *x = vector(rows); // vector to hold the solution

	// declare your timing variables
	// make sure you have ctime included

	// Since the calculation is done very quickly you have to run the calculation multiple times
	// Determine the time taken for multiple calculations
	// Then the average time per calculation is the total time divided by the number of calculations
	// Total time for a single calculation will depend on implementation of the GE algorithm
	// but in the main it depends on the processor speed
	// Fast processor => fast calculation time
	// processor speed is measured in flops, floating point operations per second
	// more flops = faster, faster  = better

	// CLOCKS_PER_SEC allows you to convert from CPU time to seconds
	// CLOCKS_PER_SEC represents the number of clock-ticks per second your computer performs

	cout<<"Clock-ticks per second on this computer is "<<CLOCKS_PER_SEC<<endl;
	
	clock_t start, finish; 

	int LU_total = 50; 
	bool error = false; 

	start = clock(); // start the clock

	// Perform multiple loops over the LU Decomposition calculation

	for(int i=1; i<=LU_total; i++){
	
		LU_Solve(A, b, x, rows, error); 

	}

	finish = clock(); // stop the clock

	double total_time = (finish - start) / (static_cast<double>(CLOCKS_PER_SEC)); // compute total time
	double LU_time = total_time / (static_cast<double>(LU_total)); 

	cout<<"Time taken to perform "<<LU_total<<" LU decompositions is "<<total_time<<" seconds\n"; 
	cout<<"Time taken to perform single LU decomposition on system is "<<LU_time<<" seconds\n"; 

	delete[] A; 
	delete[] b; 
	delete[] x; 
}

void TDMA_testing()
{
	// illustration of the Tri-diagonal Matrix Algorithm

	cout<<"\n\t\tTri-Diagonal System of Equations\n\n"; 
	
	int size=6; 

	double *v = vector(size); 

	v[1] = 0.87292; v[2] = -0.00307009; v[3] = -0.0034916; v[4] = -0.00360952; v[5] = -0.00356441; v[6] = 2.15041; 

	double **B = matrix(size, size); 

	B[1][1] = 2.03125;  B[1][2] = -1.125; B[1][3] = 0;  B[1][4] = 0; B[1][5] = 0;  B[1][6] = 0;
	B[2][1] = -0.88889; B[2][2] = 2.02469;  B[2][3] = -1.1111; B[2][4] = 0;  B[2][5] = 0; B[2][6] = 0;
	B[3][1] = 0;  B[3][2] = -0.9; B[3][3] = 2.02;  B[3][4] = -1.1;  B[3][5] = 0;  B[3][6] = 0;
	B[4][1] = 0; B[4][2] = 0;  B[4][3] = -0.909091; B[4][4] = 2.01653;  B[4][5] = -1.09091; B[4][6] = 0;
	B[5][1] = 0;  B[5][2] = 0; B[5][3] = 0;  B[5][4] = -0.916667; B[5][5] = 2.01389;  B[5][6] = -1.083333;
	B[6][1] = 0;  B[6][2] = 0;  B[6][3] = 0; B[6][4] = 0;  B[6][5] = -0.923077; B[6][6] = 2.01183;

	cout<<"Solve the system B.x = v using an TDMA\n\n"; 
	cout<<"The matrix B is \n"; 
	print_matrix(B, size, size); 

	cout<<"The vector v is \n"; 
	print_vector(v, size);

	// Construct vectors to hold the diagonals
	double *a = vector(size); 
	double *b = vector(size); 
	double *c = vector(size); 
	double *d = vector(size); 

	a[1] = 0; c[size] = 0; 

	// Fill the vectors that contain the diagonals
	for(int i=1; i<=size; i++){
		for(int j=1; j<=size; j++){

			if(i == j){ // Main Diagonal
				b[i] = B[i][j]; 
			}
			else if(i == j-1 ){ // Super Diagonal
				c[i] = B[i][j];
			}
			else if(i == j+1){ // Sub Diagonal 
				a[i] = B[i][j]; 
			}
			else{
				// Do nothing
			}
		}
	}

	cout<<"The sub-diagonal is a = \n"; 
	print_vector(a, size); 

	cout<<"The main-diagonal is b = \n"; 
	print_vector(b, size); 
	
	cout<<"The super-diagonal is c = \n"; 
	print_vector(c, size); 

	// Apply the TDMA to solve B.x = v

	double *x = vector(size); 

	for(int i=1; i<=size; i++){
		x[i] = 0.0; 
	}

	TDMA(a, b, c, v, x, size); 

	cout<<"Solution of the system B.x = v using TDMA is\n";
	print_vector(x, size); 

	cout<<"Solution of the system B.x = v using Gaussian elimination is\n";
	double *GE_x = vector(size); 
	Gauss_Solve(B, v, GE_x, size, true, false); 
	print_vector(GE_x, size); 

	cout<<"Difference between the solutions computed by the different methods is\n"; 
	double *diff = vector_diff(x, GE_x, size, size); 
	print_vector(diff, size); 

	delete[] B; 
	delete[] v;
	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d; 
	delete[] x; 
	delete[] GE_x; 
	delete[] diff; 
}

void iterative_solver_testing()
{
	cout<<"\n\t\tIterative Solvers\n\n"; 
	
	int size=6; 

	double *v = vector(size); 

	v[1] = 0; v[2] = 5; v[3] = 0; v[4] = 6; v[5] = -2; v[6] = 6; 

	double **B = matrix(size, size); 

	B[1][1] = 4;  B[1][2] = -1; B[1][3] = 0;  B[1][4] = -1; B[1][5] = 0;  B[1][6] = 0;
	B[2][1] = -1; B[2][2] = 4;  B[2][3] = -1; B[2][4] = 0;  B[2][5] = -1; B[2][6] = 0;
	B[3][1] = 0;  B[3][2] = -1; B[3][3] = 4;  B[3][4] = 0;  B[3][5] = 0;  B[3][6] = -1;
	B[4][1] = -1; B[4][2] = 0;  B[4][3] = -1; B[4][4] = 4;  B[4][5] = -1; B[4][6] = 0;
	B[5][1] = 0;  B[5][2] = -1; B[5][3] = 0;  B[5][4] = -1; B[5][5] = 4;  B[5][6] = -1;
	B[6][1] = 0;  B[6][2] = 0;  B[6][3] = -1; B[6][4] = 0;  B[6][5] = -1; B[6][6] = 4;

	cout<<"Solve the system B.x = v using an iterative method\n\n"; 
	cout<<"The matrix B is \n"; 
	print_matrix(B, size, size); 

	cout<<"The vector v is \n"; 
	print_vector(v, size);

	int max_iter = 500; 
	bool converged = false;
	double err = 0.0; 
	double tolerance = 1.0e-12; 

	double *x = vector(size); 

	// construct an initial approximation to the solution
	for(int j=1; j<=size; j++){
		x[j] = 0.0; 
	}

	cout<<"Jacobi method\n"; 
	jacobi_solve(B, x, v, size, max_iter, converged, tolerance, err); 

	cout<<"The solution of the system according to the Jacobi method is\n"; 
	print_vector(x, size); 

	double *Bx = mat_vec_product(B, x, size, size, size); 
	double *diff = vector_diff(Bx, v, size, size); 

	cout<<"For the Jacobi solution B.x - v = \n";
	print_vector(diff, size); 

	cout<<"||B.x - v||_{infty} = "<<abs(inf_norm(diff, size))<<endl<<endl;

	// construct an initial approximation to the solution for the gauss-seidel method
	for(int j=1; j<=size; j++){
		x[j] = 0.0; 
	}

	converged = false;
	err = 0.0; 

	cout<<"Gauss-Seidel method\n"; 
	gauss_seidel_solve(B, x, v, size, max_iter, converged, tolerance, err); 

	cout<<"The solution of the system according to the Gauss-Seidel method is\n"; 
	print_vector(x, size); 

	Bx = mat_vec_product(B, x, size, size, size); 
	diff = vector_diff(Bx, v, size, size); 

	cout<<"For the Gauss-Seidel solution B.x - v = \n";
	print_vector(diff, size); 

	cout<<"||B.x - v||_{infty} = "<<abs(inf_norm(diff, size))<<endl;

	delete[] v;
	delete[] B; 
	delete[] x; 
	delete[] Bx;
	delete[] diff; 
}