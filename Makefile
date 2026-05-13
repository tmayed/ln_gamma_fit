.PHONY: push git_rm_cached

#############################################################

push:
	git add .
	git commit -m "$(m)"
	git push -u origin $(b)

git_rm_cached:
	git ls-files -ci --exclude-standard | git rm --cached -r --pathspec-from-file=-

#############################################################