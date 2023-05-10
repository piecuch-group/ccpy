# Build the ccq_py python module
MODULES := cc_loops \
		   cc_loops2 \
		   crcc_loops \
		   ccsdpt_loops \
		   crcc24_loops \
		   crcc24_opt_loops \
		   mbgf_loops \
		   mbpt_loops \
		   eomcc_initial_guess \
		   ccp_loops \
		   ccsdt_p_loops \
		   eomccsdt_p_loops \
		   leftccsdt_p_intermediates \
		   eomcc_active_loops \
		   cc_active_loops \
		   cct3_loops \
		   cc_loops_t4 \
		   ccp3_loops \
		   ccp3_adaptive_loops \
		   clusteranalysis \
		   ecccp3_loops

EXT_SUFFIX := $(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SOURCES := $(addsuffix .f90, $(MODULES))
TARGETS := $(addsuffix $(EXT_SUFFIX), $(MODULES))

# If OpenMP is not available, remove the last two flags, -fopenmp, -lgomp
FFLAGS := -std=f2018 -O3 -ffree-line-length-512 -fopenmp -lgomp

ifeq ($(CONDA_PREFIX),)
	MKLFOLDER := $(MKLROOT)/lib/intel64
else
	MKLFOLDER := $(CONDA_PREFIX)/lib
endif

# configuration for lab computers
# the mkl library specifics are different for different computer. Just don't use this for now...
LIBS :=# -L$(MKLFOLDER) -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

.PHONY: all clean

all: $(TARGETS)

%$(EXT_SUFFIX): %.f90
	echo $(MKLFOLDER)
	f2py $(LIBS) --f90flags="$(FFLAGS)" -c $< -m $*
	#f2py $(LIBS) --quiet --f90flags="$(FFLAGS)" -c $< -m $*

clean:
	rm $(TARGETS)
