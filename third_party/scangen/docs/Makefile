# Makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= uv run sphinx-build
SPHINXPROJ   = ScANGen
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Build HTML documentation
html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Build PDF documentation (requires LaTeX)
pdf: latexpdf

latexpdf:
	@echo "Building LaTeX files for PDF generation..."
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS)
	@echo "Running LaTeX files through pdflatex..."
	@$(MAKE) -C "$(BUILDDIR)/latex" all-pdf
	@echo
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

# Clean build directory
clean:
	@echo "Cleaning build directory..."
	@rm -rf "$(BUILDDIR)"
	@echo "Build directory cleaned."

# Build documentation with all warnings treated as errors
strict:
	@echo "Building documentation with strict warnings..."
	@$(SPHINXBUILD) -W -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

# Check for broken external links
linkcheck:
	@echo "Checking external links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)
	@echo
	@echo "Link check complete; look for any errors in the above output or in $(BUILDDIR)/linkcheck/output.txt."

# Build documentation and open in browser (requires 'open' command on macOS or 'xdg-open' on Linux)
view: html
	@if command -v open > /dev/null 2>&1; then \
		open "$(BUILDDIR)/html/index.html"; \
	elif command -v xdg-open > /dev/null 2>&1; then \
		xdg-open "$(BUILDDIR)/html/index.html"; \
	else \
		echo "Built HTML documentation is available at: $(BUILDDIR)/html/index.html"; \
	fi

# Quick build for development (only builds changed files)
dev: html

# Build all formats
all: html pdf

# Show build directory contents
show:
	@echo "Build directory contents:"
	@ls -la "$(BUILDDIR)" 2>/dev/null || echo "Build directory does not exist. Run 'make html' first."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
