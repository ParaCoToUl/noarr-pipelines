#!/bin/bash
# packs the current git HEAD (or any other supplied git head) into a tarball
# suitable for CRAN submission. Version comes from git as the latest tag.

NAME=noarr-matrix
PREFIX="noarr-pipelines/bindings/R-bindings"
HEAD="${1:-HEAD}"
VERTAG=0.1
VER=${VERTAG#v}
ARCHIVE=${NAME}_${VER}.tar.gz

TMPDIR=".tmpbuild-$$"

mkdir ${TMPDIR} || exit 1
git archive --format=tar --prefix="${PREFIX}/" "${HEAD}" \
| tar f - \
	--delete "${PREFIX}/pack_cran.sh" \
	--delete "${PREFIX}/README.md" \
| tar xf - -C ${TMPDIR}

git clone https://github.com/ParaCoToUl/noarr-structures.git/ ${TMPDIR}/${PREFIX}/noarr-structures

R --vanilla CMD build ./${TMPDIR}/${PREFIX}/ --no-manual

rm -fr ${TMPDIR}

R --vanilla CMD check --as-cran ${ARCHIVE}
R CMD check --as-cran --use-valgrind ${ARCHIVE}

echo "Did you run valgrind?"
