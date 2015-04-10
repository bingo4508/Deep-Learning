g++ train.cpp Net.cpp utility.cpp -o ../bin/train -O2 -I ../lib/armadillo-4.650.4/include -DARMA_DONT_USE_WRAPPER -lblas -llapack

g++ predict.cpp Net.cpp utility.cpp -o ../bin/predict -O2 -I ../lib/armadillo-4.650.4/include -DARMA_DONT_USE_WRAPPER -lblas -llapack
