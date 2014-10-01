//
//  ESN.h
//  ESN
//
//  Created by admin on 01/10/2014.
//  Copyright (c) 2014 AerophobicEmpurplement. All rights reserved.
//

#ifndef __ESN__ESN__
#define __ESN__ESN__

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <complex>
#include <vector>
#include "Attributes.h"

typedef Eigen::MatrixXd Matrixd;

class ESN
{
public:
    // Constructor
    ESN(Attributes &atts);
    
    // Train the network
    void train(Matrixd &input, Matrixd &targets, int washout);
    
    // Test the network
    void test(Matrixd &input, Matrixd &generated, int washout);
    
    // Calculate the mean square error between two matrices
    double mse(Matrixd &generated, Matrixd &targets, int washout);
    
private:
    /*-- Parameters --*/
    
    // Network attributes
    int ninput;
    int noutput;
    int ninternal;
    double alpha;
    double beta;
    double density;
    double noise;
    double radius;
    bool is_trained;
    
    // Scaling matrices
    Matrixd iscale;
    Matrixd ishift;
    Matrixd tscale;
    Matrixd tshift;
    Matrixd fscale;
    
    // Weight matrices
    //Eigen::SparseMatrix<double> W;
    Matrixd W;
    Matrixd W_out;
    Matrixd W_in;
    Matrixd W_fb;
    
    /*-- Functions --*/
    
    // Pointers to activation functions
    void (*resf)(Matrixd& input);
    void (*outf)(Matrixd& input);
    void (*inv_outf)(Matrixd& input);
    
    // Specifies the solution method to find the output weights
    void (*solution_method)(Matrixd &X, Matrixd &Y, Matrixd &W_out, double &beta);
    
    // Initialises the reservoir and weight matrices
    void init_matrices();
    double max_eval(Eigen::Matrix<std::complex<double>, -1, 1, 0, -1, 1> &eigenvals);
    
    // State collection functions
    Matrixd compute_state_matrix(Matrixd &inputs, Matrixd &targets, int washout);
    Matrixd compute_teacher_matrix(Matrixd &targets, int washout);
    void compute_internal_state(Matrixd &total_state, Matrixd& internal);
    
    // Check that the total state actually contains valid values
    void error_check(Matrixd &total_state);
    
    // Get covariance for MSE
    Matrixd covariance(Matrixd &X);
};


#endif /* defined(__ESN__ESN__) */
