//
//  NARMA.cpp
//  ESN
//
//  Created by admin on 01/10/2014.
//  Copyright (c) 2014 AerophobicEmpurplement. All rights reserved.
//

#include "NARMA.h"
#include <cassert>
#include <cmath>

NARMA::NARMA(int length, int order, Matrixd &input, Matrixd &output)
{
    input = Matrixd::Random(2, length);
    
    // Set the bias units to one
    for (int i = 0; i < length; i++)
    {
        input(0, i) = 1;
        input(1, i) = std::abs(input(1, i));
    }
    
    output = 0.1 * Matrixd::Ones(1, length);
    
    // Use a NARMA equation to fill the output sequence
    for (int i = order; i < length; i++)
    {
        output(0, i) = 0.7 * input(1, i - order) + 0.1 + (1 - output(0, i - 1)) * output(0, i - 1);
    }
}