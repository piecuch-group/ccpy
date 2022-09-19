# Build the ccq_py python module
#MODULE := f90_crcc
MODULES := excitations
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

#SRC := cc_loops.f90
LDFLAGS :=
#LIBS := -L/home2/gururang/.conda/envs/ccpy_dev/lib -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lmkl_avx2 -lgomp -lpthread -lm -ldl
LIBS := 
.PHONY: all clean

all: $(TARGETS)

%$(EXT_SUFFIX): %.f90
	f2py $(LIBS) --f90flags="$(FFLAGS)" -c $< -m $*

clean:
	rm $(TARGETS)
