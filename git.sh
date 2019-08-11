#!/bin/bash

COMM=`git verify-pack -v .git/objects/pack/pack-*.idx | sort -k 3 -g | tail -10 >1.txt`

for value in `cat 1.txt |awk '{print $1}'`
do
	git rev-list --objects --all | grep "${value}" >2.txt
	a=`cat 2.txt |awk '{print $2}'`
	git log --pretty=oneline --branches -- ${a}
	git filter-branch --index-filter "git rm --cached --ignore-unmatch ${a}" -- --all
	git push --force
	rm -Rf .git/refs/original
	rm -Rf .git/logs/
	git gc
	git prune
done
