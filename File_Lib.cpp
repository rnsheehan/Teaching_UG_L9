#ifndef ATTACH_H
#include "Attach.h"
#endif

// Definition of functions declared in the namespace file_funcs
// R. Sheehan 31 - 1 - 2013

double *file_funcs::read_vector_from_file(string filename, int &size)
{
	// Use this function to read a vector out of a file and into memory
	// R. Sheehan 31 - 1 - 2013

	// Read the numeric contents of a file into a vector
	// It is assumed that a single column of numbers is stored in the file
	// R. Sheehan 22 - 2 - 2012
	
	ifstream read;
	read.open(filename.c_str(),ios_base::in); // open the file for reading
	
	double arr = 0.0; 
	double *arr_ptr; 

	arr_ptr = &arr; 

	if(read.is_open()){
		// Since you are assuming a single column of numerical data you can use the stream extraction operator
				
		// First item is to count the number of data points in the file, again using stream operators
		size = 0;

		// Count the number of lines in the file
		while(read.ignore(1280,'\n')){
			size++;
		}
		
		read.clear(); // empty the buffer
		read.seekg(0,ios::beg); // move to the start of the file

		arr_ptr = new (double [size+1] ); // create array to hold the data from the file
		
		// loop over the lines and read the data into memory
		for(int i=1; i <= size; i++){
			read>>arr_ptr[i];
		}
		
		read.close(); // close the file holding the data
	}
	else{
		cout<<"Error: could not open "<<filename<<endl;
	}

	return arr_ptr; // return the data as an array of doubles
}

double **file_funcs::read_matrix_from_file(string filename, int &rows, int &columns)
{
	// Read data stored in a file into memory
	// R. Sheehan 23 - 2 - 2012

	ifstream thefile(filename.c_str(),ios_base::in); // open a file for reading

	if(thefile.is_open()){
		// Read the data from the file
		int i,j;
		rows=0;
		columns=0;

		string line;
		string item;

		// Count the number of rows and columns
		while(getline(thefile,line,'\n')){
			rows++;
			istringstream linestream(line);
			if(rows == 1){
				while(getline(linestream,item,',')){
					columns++;
				}
			}
		}

		thefile.clear(); // empty a buffer?
		thefile.seekg(0,ios::beg); // move to the start of the file

		//Matrix<T> thematrix(nrows,ncols,true);
		double **thematrix = matrix(rows,columns); // create a matrix to hold the data being read

		i=1;
		while(getline(thefile,line,'\n')){
			istringstream linestream(line);
			j=1;			
			while(getline(linestream,item,',')){
				thematrix[i][j] = atof(item.c_str());
				j++;
			}			
			i++;
		}

		thefile.close();

		return thematrix;
	}
	else{
		cout<<"Error: could not open "<<filename<<"\n";

		rows = columns = 2; 

		double **thematrix = matrix(rows,columns); // create a matrix to hold the data being read

		return thematrix;
	}
}

void file_funcs::write_vector_to_file(string filename, double *vec, int size)
{
	// Use this function to write a vector to a file
	// R. Sheehan 31 - 1 - 2013

	ofstream write; // create the object used to write the data
	write.open(filename.c_str(), ios_base::out|ios_base::trunc); 

	if(write.is_open()){
		
		for(int i=1; i<=size; i++){
			write<<setprecision(15)<<vec[i]<<endl;
		}

	}
	else{
		cout<<"Could not open "<<filename<<" for writing\n";
	}
}

void file_funcs::write_matrix_to_file(string filename, double **mat, int rows, int columns)
{
	// Write the contents of a matrix to some file

	ofstream write;
	write.open(filename.c_str(),ios_base::out|ios_base::trunc);

	if(!write){
		cout<<"Error: cannot open "<<filename<<endl;
	}
	else{
		for(int i=1;i<=rows;i++){
			for(int j=1;j<=columns;j++)
				write<<mat[i][j]<<" ";
			write<<endl;
		}
		write.close();
	}
}