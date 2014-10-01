//
//  main.cpp
//  ESN
//
//  Created by admin on 30/09/2014.
//  Copyright (c) 2014 AerophobicEmpurplement. All rights reserved.
//

#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include "ESN.h"
#include "NARMA.h"
#include "Attributes.h"
#include "Input.h"

typedef Eigen::MatrixXd Matrixd;

void split_seq(Matrixd &first, Matrixd &second, const Matrixd original, double split);

int main(int argc, char **argv)
{
    // Generate NARMA input sequence
    int length = 1000;
    int order = 3;
    Matrixd input_seq;
    Matrixd output_seq;
    
    NARMA narma(length, order, input_seq, output_seq);
    
    // Split the input and output into two parts, one for training and the other for testing
    double split = 0.6;
    Matrixd input_train, input_test;
    Matrixd output_train, output_test;
    
    split_seq(input_train, input_test, input_seq, split);
    split_seq(output_train, output_test, output_seq, split);
    
    std::cout << output_train << std::endl;
    
    // Gather the attributes
    Attributes atts;
    Input input;
    input.parse(argc, argv, atts);
    
    // Create the network
    int washout = 100;
    Matrixd generated_output;
    ESN esn(atts);
    
    std::cout << "Training the network." << std::endl;
    esn.train(input_train, output_train, washout);
    
    std::cout << "Testing the network." << std::endl;
    esn.test(input_test, generated_output, washout);
    
    // Find the mean squared error
    double mse = esn.mse(generated_output, output_test, washout);
    std::cout << "Mean squared error: " << mse << std::endl;
    
    // Write the generated data and actual test data to file
    std::ofstream out_true("/Users/admin/Documents/Software/Projects/ESN/ESN/output_true.txt");
    std::ofstream out_gen("/Users/admin/Documents/Software/Projects/ESN/ESN/output_gen.txt");
    
    if (out_true.fail() || out_gen.fail())
    {
        std::cout << "main: Failed to open output files." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Get rid of the washout points
    output_test = output_test.rightCols(output_test.cols() - washout);
    
    assert(output_test.cols() == generated_output.cols());
    
    for (int i = 0; i < output_test.cols(); i++)
    {
        out_true << output_test(0, i) << "\n";
        out_gen << generated_output(0, i) << "\n";
    }
    
    out_true.close();
    out_gen.close();
    
    return 0;
}

void split_seq(Matrixd &first, Matrixd &second, const Matrixd original, double split)
{
    int length = (int)original.cols();
    int cut = length * split;
    
    first = original.leftCols(cut);
    second = original.rightCols(length - cut);
}