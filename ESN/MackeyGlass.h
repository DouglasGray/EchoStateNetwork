//
//  MackeyGlass.h
//  ESN
//
//  Created by admin on 01/10/2014.
//  Copyright (c) 2014 AerophobicEmpurplement. All rights reserved.
//

#ifndef __ESN__MackeyGlass__
#define __ESN__MackeyGlass__

#include <iostream>
#include <Eigen/Core>

typedef Eigen::MatrixXd Matrixd;

class MackeyGlass
{
public:
    MackeyGlass();
    void compute_timeseries(int length, Matrixd& input, Matrixd &output);

private:
    double a;
    double b;
    double n;
    int tau;            // Delay constant
    double x0;          // Initial condition
    double deltat;      // Time step size
    
    double mglass_rk4(double x_t, double x_tau);
    double mglass_eqn(double x_t, double x_tau);
};

#endif /* defined(__ESN__MackeyGlass__) */
