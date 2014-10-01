#!/bin/bash
gnuplot <<- EOF
   plot "output_true.txt" with lines, "output_gen.txt" with lines
EOF