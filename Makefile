VERBOSE=1
ifeq ($(VERBOSE),1)
ECHO := 
else
ECHO := @
endif

# Where is the Altera SDK for OpenCL software?
ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation.)
endif

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
# AOCL_LINK_CONFIG := $(shell aocl link-config )
# ignore -L/opt/altera_pro/16.0/hld/host/arm32/lib
# delete -lalterammdpcie since there is only 32bit version of libalterammdpcie.so
AOCL_LINK_CONFIG := -L/opt/altera_pro/16.0/hld/board/altera_a10socdk/arm32/lib -L/opt/altera_pro/16.0/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -lalterahalmmd -lelf #-lalterammdpcie
#AOCL_COMPILE_CONFIG := $(shell aocl compile-config --arm)
#AOCL_LINK_CONFIG := $(shell aocl link-config --arm)

# Compilation flags
ifeq ($(DEBUG),1)
CXXFLAGS += -g
else
CXXFLAGS += -O0
endif

# Compiler
CXX := g++ -m64 # -m32
#CXX := arm-linux-gnueabihf-g++

# Target
TARGET := host
TARGET_DIR := bin

# Directories
INC_DIRS := ./aocl/common/inc
LIB_DIRS := 

# Files
INCS := $(wildcard )
SRCS := $(wildcard host/src/*.cpp ./aocl/common/src/AOCLUtils/*.cpp)
LIBS := rt

# Make it all!
all : $(TARGET_DIR)/$(TARGET)

# Host executable target.
$(TARGET_DIR)/$(TARGET) : Makefile $(SRCS) $(INCS) $(TARGET_DIR)
	$(ECHO) $(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
			$(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) \
			$(foreach D,$(LIB_DIRS),-L$D) \
			$(foreach L,$(LIBS),-l$L) \
			-o $(TARGET_DIR)/$(TARGET)

$(TARGET_DIR) :
	$(ECHO)mkdir $(TARGET_DIR)

# Standard make targets
clean :
	$(ECHO)rm -f $(TARGET_DIR)/$(TARGET)
distclean :
	$(ECHO)rm -rf $(TARGET_DIR)

.PHONY : all clean
