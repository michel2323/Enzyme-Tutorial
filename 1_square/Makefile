ENZYME_PATH = /disk/mschanen/git/Enzyme/build

all: square

square: square.o
	clang -O3 -Xclang -load -Xclang $(ENZYME_PATH)/Enzyme/ClangEnzyme-21.so -ffast-math $^ -o $@

clean:
	rm -f *.o

%.o: %.c
	clang -O3 -Xclang -load -Xclang $(ENZYME_PATH)/Enzyme/ClangEnzyme-21.so -ffast-math -c $^
