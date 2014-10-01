//
//  NARMA.h
//  ESN
//
//  Created by admin on 01/10/2014.
//  Copyright (c) 2014 AerophobicEmpurplement. All rights reserved.
//

#ifndef __ESN__NARMA__
#define __ESN__NARMA__

#include <iostream>
#include <Eigen/Core>

typedef Eigen::MatrixXd Matrixd;

class NARMA
{
public:
    NARMA(int length, int order, Matrixd &input_seq, Matrixd &output_seq);
};

#endif /* defined(__ESN__NARMA__) */
