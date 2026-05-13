#############################################################

.PHONY: default clean install reinstall poc

#############################################################

APP = ln_gamma_fit

#############################################################

default:
	@echo "Available commands: clean, install, reinstall, poc"

#############################################################

clean:
	rm -rf *.egg-info
	rm -rf __pycache__
	rm -rf .venv
	rm -rf uv.lock

install:
	uv sync

reinstall: clean install

#############################################################

poc:
	cd $(APP) && uv run python poc_fit_ln_gamma.py

#############################################################
