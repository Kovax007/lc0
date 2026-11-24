ifndef EXE
	EXE = $(CURDIR)/lc0
endif

# lc0ex branch: the lc0ex-cuda backend only (the lc0ex runtime is built by
# default on this branch), as the default backend, so OpenBench's option-less
# `bench` and its games run the program carried in the net file
# (Net.lc0ex_programs) for the worker's GPU. No nvcc or cuDNN needed.
BUILD_FLAGS = -Dplain_cuda=false -Dcudnn=false -Ddefault_backend=lc0ex-cuda \
	-Dgtest=false
ifdef EVALFILE
	BUILD_FLAGS += -Dembed=true
endif
# The search the lc0ex engine played (TCEC Cup) unless the test names another.
ifdef SEARCH
	BUILD_FLAGS += -Ddefault_search=$(SEARCH)
else
	BUILD_FLAGS += -Ddefault_search=dag-preview
endif

POST_BUILD_COMMANDS =
ifdef EVALFILE
	POST_BUILD_COMMANDS = \
		cat $(EVALFILE) >> $(EXE); \
		perl -e "printf '%sLc0!', pack('V', -s '$(EVALFILE)')" >> $(EXE)
endif

.PHONY: all
all:
	chmod +x ../build.sh
	../build.sh $(BUILD_FLAGS) && mv ../build/release/lc0 $(EXE)
	$(POST_BUILD_COMMANDS)
