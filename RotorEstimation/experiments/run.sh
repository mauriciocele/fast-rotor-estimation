#!/bin/sh
# Builds and runs the experiments of the paper; results go to ../data.
# Usage: EIGEN=/path/to/eigen ./run.sh
set -e
cd "$(dirname "$0")"
EIGEN=${EIGEN:-../../../eigen-3.4.0}
SRC=../../src
LIB="$SRC/FLAE.cpp $SRC/GARotorEstimator.cpp $SRC/ArunSVD.cpp $SRC/Horn.cpp $SRC/GAValkenburg.cpp $SRC/FA3R.cpp $SRC/Davenport.cpp $SRC/Quest.cpp $SRC/QuaternionDirect.cpp $SRC/Multivector.cpp $SRC/BasisBlade.cpp"
c++ -std=c++17 -O3 -DNDEBUG -I"$EIGEN" -I$SRC experiments.cpp $LIB -o experiments
./experiments ../data
for opt in O3 Ofast; do
	c++ -std=c++17 -$opt -DNDEBUG -I"$EIGEN" -I$SRC timing.cpp $LIB -o timing_$opt
	./timing_$opt > ../data/timing_$opt.csv
done
python3 make_tables.py
