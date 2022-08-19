# Build the ccq_py python module
#MODULE := f90_crcc
MODULES := cc_loops cc_loops2 crcc_loops mbgf_loops mbpt_loops ccp_matrix ccp_loops eomcc_initial_guess ccp_opt_loops eomcc_active_loops cc_active_loops cct3_loops cc_loops_t4 ccp3_loops mrcc_loops
EXT_SUFFIX := $(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SOURCES := $(addsuffix .f90, $(MODULES))
TARGETS := $(addsuffix $(EXT_SUFFIX), $(MODULES))

#FFLAGS := -fopenmp \
#		-fdefault-integer-8 \
#		-fcheck=all -fbacktrace \
#		-march=native \
#		-fPIC \
#		-shared \
#		-ffixed-line-length-132 \
#		-O3 \
#		-std=f2018
FFLAGS := -std=f2018 -O3 -ffree-line-length-512

LDFLAGS :=

# configuration for local mac OSX
# be careful: -lmkl_intel_ilp64 requires 64-bit integers while -lmkl_intel_lp64 uses 32-bit integers
# 	      Default integer size in Python is 32-bit. Use -lmkl_intel_lp64
#LIBS := -L/Users/harellab/opt/anaconda3/envs/ccpy_dev/lib -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_avx2.2 -lgomp -lpthread -lm -ldl

# configuartion for work computer
#LIBS := -L/anaconda3/envs/ccpy_refactor/lib -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_avx2.1 -lgomp -lpthread -lm -ldl

# configuration for lab computers
LIBS := -L/home2/gururang/.conda/envs/ccpy_dev/lib -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lmkl_avx2 -lgomp -lpthread -lm -ldl

# configuration for personal mac os
#LIBS := -L/opt/anaconda3/envs/ccpy_dev/lib -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_avx2.1 -lgomp -lpthread -lm -ldl

.PHONY: all clean

all: $(TARGETS)

%$(EXT_SUFFIX): %.f90
	f2py $(LIBS) --f90flags="$(FFLAGS)" -c $< -m $*

clean:
	rm $(TARGETS)
