CXX := g++ -std=c++17
CXXFLAGS := $(shell root-config --cflags) -fPIC

SRCS := $(wildcard src/*.cpp)
OBJS := $(patsubst %.cpp, bin/%.o, $(notdir $(SRCS)) )
EXES := $(patsubst %.cpp, bin/%.exe, $(notdir $(wildcard main/*.cpp)) )

py:
	python3.10 main/*.py

clean:
	@echo cleaning...
	rm -f $(wildcard bin/*) $(wildcard lib/*)

dump:
	@echo SRCS...[$(SRCS)] [$(OBJS)] [$(EXES)]