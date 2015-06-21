SRC=src
BIN=bin
LIB=lib
DATA=data
SCRIPT=script

all: 	
	g++ $(SRC)/train-rnn.cpp $(SRC)/rnnet.cpp $(SRC)/nnet.cpp $(SRC)/utility.cpp -o $(BIN)/train-rnn -O2 -I $(LIB)/armadillo-4.650.4/include -DARMA_DONT_USE_WRAPPER -llapack -lblas
	g++ $(SRC)/predict-rnn.cpp $(SRC)/rnnet.cpp $(SRC)/nnet.cpp $(SRC)/utility.cpp -o $(BIN)/predict-rnn -O2 -I $(LIB)/armadillo-4.650.4/include -DARMA_DONT_USE_WRAPPER -llapack -lblas

run:
	mkdir $(BIN)
	$(BIN)/train-rnn 0.01 0.999 1 200-300-6120 100 5 $(DATA)/train_small model
	$(BIN)/predict-rnn $(DATA)/testing_data.txt $(DATA)/train_small.vec $(DATA)/train_small.class model output
	python $(SCRIPT)/rnn_to_ans.py output final_result.txt
