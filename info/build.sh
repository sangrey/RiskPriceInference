#!/bin/bash
unset MACOSX_DEPLOYMENT_TARGET

${PYTHON} setup.py build --debug install --single-version-externally-managed --record record.txt 

