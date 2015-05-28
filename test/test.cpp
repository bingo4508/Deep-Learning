#include "armadillo"
#include <vector>
#include <deque>
#include <map>
#include <math.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <stdlib.h>

using namespace arma;
using namespace std;

int main()
{
	mat A = randu<mat>(5,10);
	mat C(5,3);

/*	mat B = A;
	B.shed_row(2);
	B.shed_cols(2,4);	
*/
	C.col(0) = A.col(0);
	C.col(1) = A.col(2);
	C.col(2) = A.col(3);

	A.print();
	puts("");
	C.print();
}
