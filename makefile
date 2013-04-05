include make.inc

.SUFFIXES:
.SUFFIXES: .o .cu .cpp .h

#
#  ------
#  Target
#  ------
#

TGT=ipic-gpu
EXEC=${TGT}.e

#
#  -----------
#  Directories
#  -----------
#

CDA_DIR=${CURDIR}/cuda
SRC_DIR=${CURDIR}/src
OBJ_DIR=${CURDIR}/obj
INC_DIR=${CURDIR}/include

MAIN_DIR=${CURDIR}/main

#
#  -----
#  Files
#  -----
#

CPP_FILES = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(MAIN_DIR)/*.cpp)
CU_FILES  = $(wildcard $(CDA_DIR)/*.cu)

H_FILES   = $(wildcard $(SRC_DIR)/*.h)

#
#  -------
#  Objects
#  -------
#

OBJ_FILES = $(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
CUO_FILES = $(addprefix $(OBJ_DIR)/,$(notdir $(CU_FILES:.cu=.cu.o)))

#
#  ---------------
#  Makefile labels
#  ---------------
#

LIBRARIES=

all : noacc

gpu : src cuda
	@echo " --------------------------------------------------- "
	@echo " Creating GPU executable: "
	@$(NVCC) $(OBJ_FILES) $(CUO_FILES) $(LIRARIES) -o $(EXEC)
	@echo " $(CURDIR)/$(EXEC) "
	@echo " --------------------------------------------------- "

cuda : $(CUO_FILES)
	@echo " --------------------------------------------------- "
	@echo " CUDA files compiled"
	@echo " --------------------------------------------------- "

src : $(OBJ_FILES)
	@echo " --------------------------------------------------- "
	@echo " SRCE files compiled"
	@echo " --------------------------------------------------- "

noacc : src
	@echo " --------------------------------------------------- "
	@echo " Creating CPU executable: "
	@$(CPP) $(OBJ_FILES) $(LIRARIES) -o $(EXEC)
	@echo " $(CURDIR)/$(EXEC) "
	@echo " --------------------------------------------------- "

clean :
	@echo " --------------------------------------------------- "
	@echo " Cleaning the project... "
	@$(RM) $(OBJ_FILES) $(CUO_FILES) $(EXEC)
	@echo " ... Object files and executable deleted."
	@echo " --------------------------------------------------- "

#
#  -----------
#  Compilation
#  -----------
#

INCLUDES=-I$(CURDIR)

$(OBJ_DIR)/%.o : $(MAIN_DIR)/%.cpp $(H_FILES)
	@echo " --------------------------------------------------- "
	@echo " Compiling MAIN file: " $<
	@$(CPP) $(CCFLAGS) $(INCLUDES) $(DACC) -c -o $@ $<
	@echo " MAIN file compiled : " $@
	@touch $@
	@echo " --------------------------------------------------- "

$(OBJ_DIR)/%.cu.o : $(CDA_DIR)/%.cu $(H_FILES)
	@echo " --------------------------------------------------- "
	@echo " CUDA compiling KERNEL file: " $<
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(DACC) -c -o $@ $<
	@echo " CUDA KERNEL file compiled : " $@
	@touch $@
	@echo " --------------------------------------------------- "

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(H_FILES)
	@echo " --------------------------------------------------- "
	@echo " SRCE compiling file: " $<
	@$(CPP) $(CCFLAGS) $(INCLUDES) $(DACC) -c -o $@ $<
	@echo " SRCE file compiled : " $@
	@touch $@
	@echo " --------------------------------------------------- "

