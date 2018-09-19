#!/bin/bash

if [[ $(hostname -s) = pascal-* ]] || [[ $(hostname -s) = skylake-* ]]; then
    ./scripts/haswell_wikilinks.sh train_tenk foo 01
else
    ./scripts/ultra.sh train foo 01
fi



