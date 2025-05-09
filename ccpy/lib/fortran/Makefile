# DEPRECATED FILE.
#
# This Makfile should be used only for compatibility reasons. Meson is the currently preferred
# build system for ccpy. Please look at the README.md for more information on how to build
# ccpy.

# Build the ccq_py python module
MODULES := cc_loops \
		   cc_loops2 \
		   crcc_loops \
		   cripcc_loops \
		   creacc_loops \
		   ccsdpt_loops \
		   crcc24_loops \
		   crcc24_opt_loops \
		   cc3_loops \
		   hbar_cc3 \
		   mbgf_loops \
		   mbpt_loops \
		   ccsdt_p_loops \
		   ccsdt_p_chol_loops \
		   eomccsdt_p_loops \
		   leftccsdt_p_loops \
		   leftccsdt_p_intermediates \
		   eaeom3_p_loops \
		   eaeom3_p_intermediates \
		   ipeom3_p_loops \
		   ipeom3_p_intermediates \
		   lefteaeom3_p_loops \
		   lefteaeom3_p_intermediates \
		   leftipeom3_p_loops \
		   leftipeom3_p_intermediates \
		   deaeom4_p_loops \
		   hbar_ccsdt_p \
		   eomcc_active_loops \
		   cc_active_loops \
		   cct3_loops \
		   cc_loops_t4 \
		   cc4_loops \
		   ccp3_loops \
		   ccp3_opt_loops \
		   ccp3_adaptive_loops \
		   eaccp3_loops \
		   ipccp3_loops \
		   clusteranalysis \
		   ecccp3_loops \
		   reorder \
		   ccp3_full_correction \
		   ccp3_full_correction_high_mem \
		   vvvv_contraction \
		   eaccp3_correction

EXT_SUFFIX := $(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SOURCES := $(addsuffix .f90, $(MODULES))
TARGETS := $(addsuffix $(EXT_SUFFIX), $(MODULES))

# If OpenMP is not available, remove the last two flags, -fopenmp, -lgomp
# For GitHub CI pipeline, remove OpenMP
FFLAGS := -fbounds-check -std=f2018 -O3 -ffree-line-length-512# -march=native -mtune=native -mavx -fopenmp -lgomp

ifeq ($(CONDA_PREFIX),)

ifeq ($(USE_BLAS),mkl)
	BLASFLAGS := --link-mkl
else ifeq ($(USE_BLAS),openblas)
	BLASFLAGS := --link-openblas64
else
	BLASFLAGS :=
endif

	
else

# Run if inside conda
ifeq ($(USE_BLAS),mkl)
	MKLFOLDER := $(CONDA_PREFIX)/lib
	BLASFLAGS := -L$(MKLFOLDER) -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lmkl_avx2 -lgomp -lpthread -lm -ldl
else
	BLASFLAGS :=
endif

endif

#$(info $$BLASFLAGS is [${BLASFLAGS}])

# configuration for lab computers
LIBS := $(BLASFLAGS)

.PHONY: all clean

all: $(TARGETS)

%$(EXT_SUFFIX): %.f90
	echo $(MKLFOLDER)
	f2py $(LIBS) --f90flags="$(FFLAGS)" -c $< -m $*

clean:
	rm -f $(TARGETS)
