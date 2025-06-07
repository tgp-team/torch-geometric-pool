SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

export PYTHONPATH := ..

help:
	@PYTHONPATH=$(PYTHONPATH) $(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@PYTHONPATH=$(PYTHONPATH) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
