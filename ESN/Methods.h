//
//  Methods.h
//  ESN
//
//  Created by admin on 01/10/2014.
//  Copyright (c) 2014 AerophobicEmpurplement. All rights reserved.
//

#ifndef ESN_Methods_h
#define ESN_Methods_h

#include <Eigen/Core>
#include <Eigen/SVD>
#include <limits>
#include <algorithm>
#include <cmath>

typedef Eigen::MatrixXd Matrixd;

class Methods
{
public:
    /*-- Activation functions --*/
    
    // tanh activation function
    inline static void tanh(Eigen::MatrixXd &input)
    {
        long rows = input.rows();
        long cols = input.cols();
        
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                input(i,j) = std::tanh(input(i,j));
            }
    }
    
    // Inverse tanh
    inline static void inv_tanh(Eigen::MatrixXd &input)
    {
        long rows = input.rows();
        long cols = input.cols();
        
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                input(i,j) = std::atanh(input(i,j));
            }
    }
    
    // Identity, do nothing!
    inline static void identity(Eigen::MatrixXd &input) {};
    
    // Sigmoid activation
    inline static void sigmoid(Eigen::MatrixXd &input)
    {
        long rows = input.rows();
        long cols = input.cols();
        
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                input(i,j) = 1.0 / (1.0 + std::exp((-1) * input(i,j)));
            }
    }
    
    // Inverse sigmoid
    inline static void inv_sigmoid(Eigen::MatrixXd &input)
    {
        
        long rows = input.rows();
        long cols = input.cols();
        
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                input(i,j) = (-1) * log((1.0 / input(i,j)) - 1);
            }
    }

    /*-- Weight matrix solution methods --*/
    // X contains collected states, Y contains target data
    
    // Tikhinov regression
    inline static void tikhinov(Matrixd &X, Matrixd &Y, Matrixd &W_out, double &beta)
    {
        Matrixd X_t = X.transpose();
        Matrixd I = Matrixd::Identity(X.rows(), X.rows());
        
        Matrixd cov_mat = X * X_t - beta * I;
        Matrixd p_vec = Y * X_t;
        
        W_out = p_vec * cov_mat.inverse();
        
        //std::cout << W_out << std::endl;
    }
    
    // Pseudo-Inverse
    inline static void pseudo_inv(Matrixd &X, Matrixd &Y, Matrixd &W_out, double &beta)
    {
        Matrixd X_pinv;
        pinv(X, X_pinv);
        
        W_out = Y * X_pinv;
    }
    
private:
    // Calculates the pseudo inverse of a matrix
    // TODO: Check output correct!
    static bool pinv(const Matrixd &a, Matrixd &result)
    {
        if(a.rows() < a.cols())
            return false;
        
        double eps = std::numeric_limits<double>::epsilon();
        
        Eigen::JacobiSVD<Matrixd> svd = a.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        
        double tol = eps * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs().maxCoeff();
        
        result = svd.matrixV()
        * Matrixd((svd.singularValues().array().abs() > tol).select(svd.singularValues().array().inverse(), 0)).asDiagonal()
        * svd.matrixU().adjoint();
        
        return true;
    }
};

#endif
