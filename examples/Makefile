# Example directories
EXAMPLE_DIRS := 0_basic 1_tester

# Pattern rule for examples
$(EXAMPLE_DIRS):
	@echo "Building $@ example..."
	@$(MAKE) -C $@
	@echo "Finished building $@ example."

# Default target
all:
	@for dir in $(EXAMPLE_DIRS); do \
		echo "Building $$dir..."; \
		$(MAKE) -C $$dir; \
		echo "Finished building $$dir.\n"; \
	done
	@echo "Finished building all example.";

# Run targets
run-%:
	@echo "Running $* example..."
	@$(MAKE) -C $* run

# Clean all examples
clean:
	@for dir in $(EXAMPLE_DIRS); do \
		echo "Cleaning $$dir..."; \
		$(MAKE) -C $$dir clean; \
		echo "Finished cleaning $$dir.\n"; \
	done
	@echo "Finished cleaning all example.";

# Help
help:
	@echo "Available targets:"
	@echo "  make <example>     - Build a specific example (e.g., make 0_basic)"
	@echo "  make all           - Build all examples"
	@echo "  make run-<example> - Run a specific example (e.g., make run-0_basic)"
	@echo "  make clean         - Clean all examples"
	@echo "  make help          - Show this help"

.PHONY: $(EXAMPLE_DIRS) all run-% clean help