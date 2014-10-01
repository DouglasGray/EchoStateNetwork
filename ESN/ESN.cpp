//
//  ESN.cpp
//  ESN
//
//  Created by admin on 01/10/2014.
//  Copyright (c) 2014 AerophobicEmpurplement. All rights reserved.
//

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <Eigen/Eigenvalues>
#include <ctime>
#include <cassert>
#include "ESN.h"
#include "Methods.h"
#include "Attributes.h"

typedef Eigen::Triplet<double> T;
typedef Eigen::MatrixXd Matrixd;
typedef boost::random::mt19937 RNG;
typedef boost::random::uniform_real_distribution<double> URD;
typedef boost::random::uniform_int_distribution<int> UID;

// Constructor
ESN::ESN(Attributes &atts)
{
    // Pull the parameters from the attributes structure
    ninput = atts.ninput;
    noutput = atts.noutput;
    ninternal = atts.ninternal;
    alpha = atts.alpha;
    beta = atts.beta;
    density = (atts.density == -1) ? std::min(10.0 / ninternal, 1.0) : atts.density;
    noise = atts.noise;
    radius = atts.radius;
    
    is_trained = false;
    is_predictive = false;
    is_generative = false;
    
    // Set scaling matrices
    iscale = atts.input_scaling * Matrixd::Ones(ninput, 1);
    ishift = atts.input_shift * Matrixd::Ones(ninput, 1);
    tscale = atts.teacher_scaling * Matrixd::Ones(noutput, 1);
    tshift = atts.input_shift * Matrixd::Ones(noutput, 1);
    fscale = atts.feedback_scaling * Matrixd::Ones(noutput, 1);
    
    // Set pointers to the activation functions
    switch (atts.resf)
    {
        case TANH:
            resf = Methods::tanh;
            break;
        case IDENTITY:
            resf = Methods::identity;
            break;
        case SIGMOID:
            resf = Methods::sigmoid;
            break;
        default:
            std::cout << "ESN::ESN: Error assigning reservoir function" << std::endl;
            exit(EXIT_FAILURE);
    }
    
    switch (atts.outf)
    {
        case TANH:
            outf = Methods::tanh;
            inv_outf = Methods::inv_tanh;
            break;
        case IDENTITY:
            outf = Methods::identity;
            inv_outf = Methods::identity;
            break;
        case SIGMOID:
            outf = Methods::sigmoid;
            inv_outf = Methods::inv_sigmoid;
            break;
        default:
            std::cout << "ESN::ESN: Error assigning output function" << std::endl;
            exit(EXIT_FAILURE);
    }
    
    // Set pointer to the solution method
    switch (atts.solution_method)
    {
        case PINV:
            solution_method = Methods::pseudo_inv;
            break;
        case TIKHINOV:
            solution_method = Methods::tikhinov;
            break;
        default:
            std::cout << "ESN::ESN: Error assigning solution method" << std::endl;
            exit(EXIT_FAILURE);
    }
    
    // Do some error checking
    if (ninput < 0)
    {
        std::cout << "ESN::ESN: Number of input units must be specified." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (ninternal < 0)
    {
        std::cout << "ESN::ESN: Number of internal units must be specified." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (noutput < 0)
    {
        std::cout << "ESN::ESN: Number of output units must be specified." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (radius < 0)
    {
        std::cout << "ESN::ESN: Spectral radius must be specified." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Finally, initialise the various weight matrices
    init_matrices();
}

// Weight matrix initialisation
void ESN::init_matrices()
{
    double eig_max = 0;
    int nz;
    int row, col;
    double val;
    Eigen::Matrix<std::complex<double>, -1, 1, 0, -1, 1> E;
    
    // Initialise and seed the random number generators
    RNG eng_int((int)std::time(0) + getpid());
    RNG eng_real((int)std::time(0) + getpid());
    URD urd(-1, 1);
    UID uid(0, ninternal - 1);
    
    Matrixd mat;
    
    // Initialise the sparse internal weights
    std::cout << "Initialising sparse internal weights and setting spectral radius." << std::endl;
    while (!eig_max)
    {
        nz = density * ninternal;
        mat = Matrixd::Zero(ninternal,ninternal);
        for (int i = 0; i < nz; i++)
        {
            row = uid(eng_int);
            col = uid(eng_int);
            val = urd(eng_real);
            mat(row,col) = val;
        }
        Eigen::EigenSolver<Matrixd> eig(mat);
        E = eig.eigenvalues();
        
        eig_max = max_eval(E);
        
        //std::cout << "Found eigenvalues are:" << std::endl << eig.eigenvalues() << std::endl;
        //std::cout << "Max eigenvalue found is:" << std::endl << eig_max << std::endl;
        
        if (!eig_max)
        {
            std::cout << "Failed to converge. Reinitialising..." << std::endl;
            
            // Increment the density by 0.05. At some point the eigenvalues will converge
            // density = std::min(1.0 , density + 0.05);
        }
    }
    
    // Set the spectral radius
    mat /= eig_max;
    mat *= radius;
    
    W = mat;
    //W = mat.sparseView();
    
    std::cout << "Internal weight matrix initialised with final density " << density << "." << std::endl;
    
    // Initialise the input and feedback weights
    W_in = Matrixd::Random(ninternal, ninput);
    W_fb = Matrixd::Random(ninternal, noutput);
}

// Function to locate maximum eigenvalue
double ESN::max_eval(Eigen::Matrix<std::complex<double>, -1, 1, 0, -1, 1> &E)
{
    // Cycle through the column vector of eigenvalues and return the largest
    long r = E.rows();
    double max = 0, val = 0;
    
    for (int i = 0; i < r; i++)
    {
        val = std::abs(E(i,0));
        max = (val > max) ? val : max;
    }
    return max;
}

/*---- Training function ----*/
// inputs = input matrix of size (ninput x timesteps)
// targets = teacher vector of size (noutput x timesteps)

void ESN::train(Matrixd &inputs, Matrixd &targets, int washout)
{
    // Compute state matrix
    Matrixd X = compute_state_matrix(inputs, targets, washout);
    
    // Target matrix
    Matrixd Y = compute_teacher_matrix(targets, washout);
    
    // Solve for W_out
    solution_method(X, Y, W_out, beta);
    
    is_trained = true;
}

void ESN::test(Matrixd &inputs, Matrixd &outputs, int washout)
{
    if (!is_trained)
    {
        std::cout << "ESN::test: Network must be trained before it can be tested." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Set dud targets so that the compute_state function knows that it is testing
    Matrixd targets = Matrixd::Zero(0, 0);
    
    Matrixd X = compute_state_matrix(inputs, targets, washout);
    
    outputs = W_out * X;
    
    // Scale and shift back to its original size
    for (int i = 0; i < (int)outputs.cols(); i++)
    {
        outputs.col(i) -= tshift;
        outputs.col(i).cwiseQuotient(tscale);
    }
}

// Collect the various states for training and testing.
// This function may be used in three different ways:
// 1. If both inputs and targets are specified, then this is a predictive task
// 2. If inputs are empty but targets are specified, then this is a generative task
// 3. If targets are empty but not the inputs, then the network is testing
Matrixd ESN::compute_state_matrix(Matrixd &inputs, Matrixd &targets, int washout)
{
    int ndata = (int)inputs.cols();
    int ntarg = (int)targets.cols();
    bool teacher_forcing;
    int collect_ix = 0;
    
    if (ndata > 0 && ntarg > 0)
    {
        // Predictive task
        if (ndata != ntarg)
        {
            std::cout << "ESN::compute_state_matrix: Input and target matrices must be the same size for training." << std::endl;
            exit(EXIT_FAILURE);
        }
        is_predictive = true;
        teacher_forcing = true;
    }
    else if (ndata == 0)
    {
        // Generative task
        if (ninput != 0)
        {
            std::cout << "ESN::compute_state_matrix: Generative tasks must have zero input units." << std::endl;
            exit(EXIT_FAILURE);
        }
        is_generative = true;
        teacher_forcing = true;
        ndata = ntarg;
    }
    else if (ntarg == 0)
    {
        // Testing
        teacher_forcing = false;
    }
    else
    {
        std::cout << "ESN::compute_state_matrix: HOW1" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    Matrixd X = Matrixd::Zero(ninput + ninternal, ndata - washout);
    Matrixd total_state = Matrixd::Zero(ninput + noutput + ninternal, 1);
    Matrixd in = Matrixd::Zero(ninput, 1);
    Matrixd internal = Matrixd::Zero(ninternal, 1);
    Matrixd out = Matrixd::Zero(noutput, 1);
    Matrixd tmp = Matrixd::Zero(ninternal + ninput, 1);;
    
    for (int i = 0; i < ndata; i++)
    {
        if (teacher_forcing)
        {
            // Only the training for a predictive task requires input
            if (is_predictive)
                in = iscale.cwiseProduct(inputs.col(i)) + ishift;
        }
        else
        {
            // When testing, use an input for prediction.
            // Otherwise set the internal state accordingly for the generative task
            if (is_predictive)
                in = iscale.cwiseProduct(inputs.col(i)) + ishift;
            if (i == 0 && is_generative)
                total_state = x_init;
        }
        
        total_state.middleRows(ninternal, ninput) = in;
        //std::cout << total_state << std::endl;
        
        // Compute the reservoir activations
        compute_internal_state(total_state, internal);
        //std::cout << internal << std::endl;
        
        // If training then apply teacher shift and scaling.
        // Otherwise generate output
        if (teacher_forcing)
            out = tscale.cwiseProduct(targets.col(i)) + tshift;
        else
        {
            tmp << internal,
                   in;
            out = W_out * tmp;
            outf(out);
        }
        //std::cout << out << std::endl;
        
        // Update total state
        total_state << internal,
                       in,
                       out;
        
        // Check for any invalid values (NaN / inf)
        error_check(total_state);
        
        // Accumulate the states if the washout period has been exceeded
        if (i >= washout)
        {
            X.col(collect_ix) << internal,
                                 in;
            collect_ix++;
        }
    }
    
    // Save the last state of the reservoir (used for generative tasks)
    if (teacher_forcing)
        x_init = total_state;
    
    return X;
}

Matrixd ESN::compute_teacher_matrix(Matrixd &targets, int washout)
{
    Matrixd Y = targets.rightCols(targets.cols() - washout);
    Matrixd T = tscale.diagonal() * Y;
    int ndata = (int)Y.cols();
    
    for (int i = 0; i < ndata; i++)
    {
        T.col(i) += tshift;
    }
    
    return T;
}

void ESN::compute_internal_state(Matrixd &total_state, Matrixd &x)
{
    Matrixd x_prev = total_state.topRows(ninternal);
    Matrixd diag = fscale.diagonal();
    Matrixd weights = Matrixd::Zero(ninternal, ninternal + ninput + noutput);
    Matrixd noise_vec = noise * Matrixd::Random(ninternal, 1);
    
    weights << W, W_in, (W_fb * diag);
    weights *= total_state;
    resf(weights);
    
    x = (1 - alpha) * x_prev + alpha * weights;
    
    x += noise_vec;
}

void ESN::error_check(Matrixd &state)
{
    for (int i = 0; i < (int)state.rows(); i++)
        if (std::isnan(state(i, 0)))
        {
            std::cout << "ESN::error_check: Invalid state found in the ";
            if (i < ninternal)
                std::cout << "internal states." << std::endl;
            else if (i >= ninternal && i < ninternal + ninput)
                std::cout << "input values." << std::endl;
            else if (i >= ninternal + ninput)
                std::cout << "target values." << std::endl;
            
            exit(EXIT_FAILURE);
        }
}

double ESN::nrmse(Matrixd &Y_gen, Matrixd &targets, int washout)
{
    double mse;
    double y_min = std::numeric_limits<double>::infinity();
    double y_max = (-1) * std::numeric_limits<double>::infinity();
    double val;
    int ndata = (int)Y_gen.cols();
    
    Matrixd Y_targ = targets.rightCols(targets.cols() - washout);
    
    assert(Y_targ.cols() == Y_gen.cols());
    
    // Find the max and min for the NRMSE
    for (int i = 0; i < ndata; i++)
        for (int j = 0; j < noutput; j++)
        {
            val = Y_targ(j, i);
            y_min = (val < y_min) ? val : y_min;
            y_max = (val > y_max) ? val : y_max;
        }
    
    // Find the difference
    Matrixd diff = Y_gen - Y_targ;
    
    // Square it
    diff = diff.cwiseProduct(diff);
    
    // Take average
    mse = (diff.colwise().sum()).sum() / (ndata * noutput);
    
    // Take square root and normalise
    mse = std::pow(mse, 0.5) / (y_max - y_min);
    
    return mse;
}

Matrixd ESN::covariance(Matrixd &mat)
{
    Matrixd centered = mat.rowwise() - mat.colwise().mean();
    return (centered.adjoint() * centered) / double(mat.rows());
}