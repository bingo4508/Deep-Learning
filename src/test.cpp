#include "armadillo"
#include <vector>

using namespace arma;
using namespace std;

void test(mat &m){
  for(mat::iterator i=m.begin(); i!=m.end(); ++i)
  {
	  *i = 100;
  }
}

int
main(int argc, char** argv)
  {
  mat A(3,1);
  A(0,0) = 1;
  A(1,0) = 2;
  A(2,0) = 3;

  join_cols(A,1);
  A.print();
 }
