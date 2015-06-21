# DNN
g++ train-nn.cpp nnet.cpp utility.cpp -o ../bin/train-nn -O2 -I ../lib/armadillo-4.650.4/include -DARMA_DONT_USE_WRAPPER -I/opt/OpenBLAS/include -Wl,-rpath=/opt/OpenBLAS/lib -L/opt/OpenBLAS/lib -lopenblas -llapack
g++ predict-nn.cpp nnet.cpp utility.cpp -o ../bin/predict-nn -O2 -I ../lib/armadillo-4.650.4/include -DARMA_DONT_USE_WRAPPER -I/opt/OpenBLAS/include -Wl,-rpath=/opt/OpenBLAS/lib -L/opt/OpenBLAS/lib -lopenblas -llapack

# RNN
#g++ train-rnn.cpp rnnet.cpp nnet.cpp utility.cpp -o ../bin/train-rnn -O2 -I ../lib/armadillo-4.650.4/include -I ../lib/OpenBLAS/build/include/ -DARMA_DONT_USE_WRAPPER -llapack -L../lib/OpenBLAS/build/lib
#g++ predict-rnn.cpp rnnet.cpp nnet.cpp utility.cpp -o ../bin/predict-rnn -O2 -I ../lib/armadillo-4.650.4/include -I ../lib/OpenBLAS/build/include/ -DARMA_DONT_USE_WRAPPER -llapack -L../lib/OpenBLAS/build/lib
