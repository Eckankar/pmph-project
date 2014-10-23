CXX        = nvcc
LIB        =
CXXFLAGS   = -O3 -arch=sm_20 -DWITH_FLOATS=1 -I ../include -D lgWARP5

SOURCES_CPP = ProjectMain.cu ProjHelperFun.cu ProjCoreCuda.cu 
HELPERS     = ProjHelperFun.h ../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ../include/CudaUtilProj.cu.h ProjCoreCUDACores.cu.h
OBJECTS     = ProjectMain.o ProjHelperFun.o  ProjCoreCuda.o
EXECUTABLE  = runproject

default: gpu

.SUFFIXES: .cu .o .h .cu.h

%.o: %.cu $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<


gpu: $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $(OBJECTS)


run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Debug.txt

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)

