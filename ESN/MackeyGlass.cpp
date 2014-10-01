//
//  MackeyGlass.cpp
//  ESN
//
//  Created by admin on 01/10/2014.
//  Copyright (c) 2014 AerophobicEmpurplement. All rights reserved.
//

#include "MackeyGlass.h"
#include <cmath>
#include <fstream>

MackeyGlass::MackeyGlass()
{
    a = 0.2;
    b = 0.1;
    tau = 17;
    x0 = 1.2;
    deltat = 0.1;
    n = 10.0;
}

void MackeyGlass::compute_timeseries(int length, Matrixd& input, Matrixd &output)
{
    // Initialise input and output
    input = Matrixd::Zero(0, 0);
    output = Matrixd::Zero(1, length);
    
    /* Set the bias units
    for (int i = 0; i < length; i++)
    {
        input(0, i) = 1;
        input(1, i) = i;
    }
    */
    
    std::ifstream in_file("/Users/admin/Documents/Software/Projects/ESN/ESN/MackeyGlass.txt");
    
    if (in_file.fail())
    {
        std::cout << "MackeyGlass::compute_timeseries: File input failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < length; i++)
        in_file >> output(0, i);
    
    /*
    int hist_len = tau / deltat;
    double time = 0;
    int index = 0;
    double x_t = x0;
    double x_tau;
    
    Matrixd history = Matrixd::Zero(1, hist_len);
    
    for (int i = 0; i < length; i++)
    {
        std::cout << x_t << std::endl;
        output(0, i) = x_t;
        input(1, i) = time;
        
        x_tau = (tau == 0) ? 0.0 : history(0, index);
        
        x_t = mglass_rk4(x_t, x_tau);
        
        if (tau != 0)
        {
            history(0, index) = x_t;
            index = (index % (hist_len - 1)) + 1;
        }
        
        time += deltat;
    }
     */
}

double MackeyGlass::mglass_rk4(double x_t, double x_tau)
{
    double k1 = deltat * mglass_eqn(x_t, x_tau);
    double k2 = deltat * mglass_eqn(x_t + 0.5 * k1, x_tau);
    double k3 = deltat * mglass_eqn(x_t + 0.5 * k2, x_tau);
    double k4 = deltat * mglass_eqn(x_t + k3, x_tau);
    
    return (x_t + (k1 / 6) + (k2 / 3) + (k3 / 3) + (k4 / 6));
}

double MackeyGlass::mglass_eqn(double x_t, double x_tau)
{
    return ((-b * x_t) + (a * x_tau) / (1 + std::pow(x_tau, n)));
}