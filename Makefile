CC=g++
CFLAGS= -Wall -std=c++11
LDFLAGS=
LIBS := -L/usr/local/lib -lboost_filesystem -lboost_thread -lboost_system -lIlmImf -lIlmThread -lImath -lIex -lfftw3 -ltbb
INCLUDES := -Icpp -I/usr/local/include/OpenEXR -I/home/stephan/torch/install/include
SOURCES := $(wildcard cpp/*.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
EXECUTABLE_1 = ViserionImageIO

.PHONY: depend clean

all:$(EXECUTABLE_1)

OBJS_1 = $(OBJECTS)

$(EXECUTABLE_1): $(OBJS_1)
	$(CC) $(CFLAGS) -shared -rdynamic -o $(EXECUTABLE_1) $(OBJS_1) $(LFLAGS) $(LIBS)

$(OBJECTS): %.o: %.cpp
	$(CC) $(INCLUDES) $(CFLAGS) -c -fPIC $< -o $@

clean:
	$(RM) -f $(OBJECTS) $(wildcard cpp/*.h.gch) $(wildcard *.so) $(NN)
